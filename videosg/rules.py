"""Conservative geometric predicates over sampled panoptic masks."""

import cv2
import numpy as np


def box(mask):
    ys, xs = np.nonzero(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else None


def predicates(subject, obj, subject_class, object_class, settings, history, key, sample):
    """Return observable labels only; these are proxies, not action recognition."""
    a, b = box(subject), box(obj)
    if not a or not b:
        return []
    diagonal = np.hypot(*subject.shape)
    ac = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    bc = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    distance = float(np.hypot(ac[0] - bc[0], ac[1] - bc[1]) / diagonal)
    gap = settings["touching_dilation_px"]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1,) * 2)
    contact = bool(np.any(cv2.dilate(subject.astype(np.uint8), kernel) & obj))
    found = []
    if distance <= settings["near_fraction_of_diagonal"]:
        found.append("near")
    if contact:
        found.append("touching")
    previous = history.get(key)
    if previous and previous["sample"] == sample - 1:
        change = previous["distance"] - distance
        if change >= settings["approaching_delta_fraction"]:
            found.append("approaching")
        if -change >= settings["approaching_delta_fraction"]:
            found.append("moving_away")
        if subject_class == "person" and object_class in settings["portable_classes"]:
            motion = np.hypot(ac[0] - previous["subject_center"][0], ac[1] - previous["subject_center"][1]) / diagonal
            object_motion = np.hypot(bc[0] - previous["object_center"][0], bc[1] - previous["object_center"][1]) / diagonal
            if contact and motion >= settings["holding_min_motion_fraction"] and abs(motion - object_motion) <= settings["holding_motion_tolerance_fraction"]:
                found.append("holding")
    if contact and a[3] <= b[1] + settings["support_overlap_px"] and a[3] >= b[1] - settings["support_overlap_px"]:
        found.append("on")
    if subject_class == "person" and object_class in settings["support_classes"]:
        width, height = a[2] - a[0] + 1, a[3] - a[1] + 1
        lying = contact and width / height >= settings["lying_aspect_ratio"]
        if lying:
            found.append("lying_on")
        if previous and previous["sample"] == sample - 1 and lying:
            drop = (ac[1] - previous["subject_center"][1]) / subject.shape[0]
            if drop >= settings["falling_vertical_fraction"]:
                found.append("falling")
    history[key] = {"sample": sample, "distance": distance, "subject_center": ac, "object_center": bc}
    return found


def region_events(person_mask, regions, history, person_uid, sample):
    """Emit crossings of explicitly configured normalized regions."""
    a = box(person_mask)
    if not a:
        return []
    h, w = person_mask.shape
    center = ((a[0] + a[2]) / (2 * w), (a[1] + a[3]) / (2 * h))
    found = []
    for name, (left, top, right, bottom) in regions.items():
        inside = left <= center[0] <= right and top <= center[1] <= bottom
        key = (person_uid, name)
        previous = history.get(key)
        if previous and previous[0] == sample - 1 and previous[1] != inside:
            found.append(("entering" if inside else "leaving", name))
        history[key] = (sample, inside)
    return found
