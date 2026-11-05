"""Independent event annotations and temporal-overlap agreement."""

from pathlib import Path

from .evaluation import temporal_iou
from .pipeline import read_json, save_json


def consensus(first_path, second_path, output_path, minimum_iou=0.5):
    first, second = read_json(Path(first_path)), read_json(Path(second_path))
    if first["annotator_id"] == second["annotator_id"]:
        raise ValueError("Annotations must come from two distinct people")
    if first["video_sha256"] != second["video_sha256"]:
        raise ValueError("Annotation source hashes differ")
    for source in (first, second):
        for event in source["events"]:
            if event["start_s"] < 0 or event["end_s"] <= event["start_s"]:
                raise ValueError("Invalid annotation interval")
    used, agreed, overlaps = set(), [], []
    for event in first["events"]:
        candidates = [(temporal_iou([event["start_s"], event["end_s"]],
                                    [other["start_s"], other["end_s"]]), index, other)
                      for index, other in enumerate(second["events"])
                      if index not in used and all(event[key] == other[key]
                                                  for key in ("predicate", "subject", "object"))]
        if not candidates:
            continue
        score, index, other = max(candidates)
        if score < minimum_iou:
            continue
        used.add(index)
        overlaps.append(score)
        agreed.append({**{key: event[key] for key in ("predicate", "subject", "object")},
                       "start_s": (event["start_s"] + other["start_s"]) / 2,
                       "end_s": (event["end_s"] + other["end_s"]) / 2})
    result = {"video_sha256": first["video_sha256"], "annotators": [first["annotator_id"], second["annotator_id"]],
              "agreement": {"matched": len(agreed), "first_count": len(first["events"]),
                            "second_count": len(second["events"]),
                            "event_f1": 2 * len(agreed) / (len(first["events"]) + len(second["events"]))
                            if first["events"] or second["events"] else 1.0,
                            "mean_tiou": sum(overlaps) / len(overlaps) if overlaps else None},
              "events": agreed}
    save_json(Path(output_path), result)
    return result
