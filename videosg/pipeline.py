"""A small, video-only panoptic graph and evidence-snippet pipeline."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "8e154c4f946f4fad9aa511a7d706c89c4254cc6eee28fd932e52c32394c5a29f"


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def case_dir(case_id):
    if not case_id or not all(c.isalnum() or c in "_-" for c in case_id):
        raise ValueError("case ID may contain only letters, numbers, _ and -")
    return ROOT / "cases" / case_id / "video"


def config():
    return read_json(ROOT / "config.json")


def pipeline_digest():
    return hashlib.sha256("".join(digest(path) for path in
        [ROOT / "videosg" / name for name in ("pipeline.py", "rules.py", "artifacts.py")]
        + [ROOT / "ontology.json"]).encode()).hexdigest()


def open_video(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    return capture


def gpu_temperature():
    result = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=False)
    return int(result.stdout.splitlines()[0]) if result.returncode == 0 and result.stdout.strip() else None


def download_model():
    """Fetch the recorded internal-use checkpoint and verify its upstream hash."""
    destination = ROOT / "models" / "mask2former-swin-tiny-coco-panoptic"
    destination.mkdir(parents=True, exist_ok=True)
    base = "https://modelscope.cn/models/facebook/mask2former-swin-tiny-coco-panoptic/resolve/master/"
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        target = destination / name
        if target.exists() and (name != "model.safetensors" or digest(target) == MODEL_SHA256):
            continue
        temporary = target.with_suffix(target.suffix + ".download")
        with urlopen(Request(base + name, headers={"User-Agent": "CapstoneVideoSG/0.1"}), timeout=90) as source, open(temporary, "wb") as output:
            shutil.copyfileobj(source, output)
        if name == "model.safetensors" and digest(temporary) != MODEL_SHA256:
            raise RuntimeError("Downloaded model weights do not match upstream SHA-256")
        temporary.replace(target)
    print(f"Model ready: {destination} (SHA-256 {digest(destination / 'model.safetensors')})")


def ingest(case_id, source, start_wallclock=None, group_case=None):
    if group_case:
        case_dir(group_case)  # validate the logical case identifier
    source = Path(source).resolve(strict=True)
    if not source.is_file():
        raise ValueError("Input must be a video file")
    output = case_dir(case_id)
    output.mkdir(parents=True, exist_ok=True)
    capture = open_video(source)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("Video has no usable frame rate")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> (8 * index)) & 255) for index in range(4)).strip("\x00")
    started = time.perf_counter()
    frames = 0
    while capture.grab():
        frames += 1
    capture.release()
    if not frames:
        raise RuntimeError("Video has no decodable frames")
    video_hash = digest(source)
    video = {
        "video_id": f"{case_id}_{video_hash[:16]}", "case_id": group_case or case_id, "path": str(source),
        "sha256": video_hash, "fps": fps, "n_frames": frames,
        "width": width, "height": height, "duration_s": frames / fps,
        "codec": codec,
        "modality": "VIDEO", "decode_seconds": round(time.perf_counter() - started, 3),
    }
    if start_wallclock:
        video["start_wallclock"] = datetime.fromisoformat(start_wallclock).isoformat()
    prior = output / "video.json"
    if prior.exists():
        old = read_json(prior)
        if old["sha256"] != video_hash or old["case_id"] != video["case_id"]:
            raise RuntimeError("This storage ID already belongs to a different video or logical case")
        if start_wallclock and old.get("start_wallclock") != video.get("start_wallclock"):
            raise RuntimeError("Wall-clock provenance cannot be changed after ingestion")
        print(json.dumps(old, indent=2))
        return
    save_json(output / "video.json", video)
    print(json.dumps(video, indent=2))


def mask_iou(left, right):
    intersection = np.count_nonzero(left & right)
    union = np.count_nonzero(left | right)
    return intersection / union if union else 0.0


def touching(left, right, pixels):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1,) * 2)
    return bool(np.any(cv2.dilate(left.astype(np.uint8), kernel) & right))


def aggregate(observations, step, last_frame, gap_samples, min_support, run_id, fps):
    """Merge neighboring sampled observations without inventing unsupported spans."""
    grouped = {}
    for item in observations:
        key = (item["subject"], item["predicate"], item["object"])
        grouped.setdefault(key, []).append((item["frame"], item.get("score", 1.0)))
    result = []
    for (subject, predicate, obj), frames in sorted(grouped.items()):
        spans = [[frames[0]]]
        for frame in frames[1:]:
            if frame[0] - spans[-1][-1][0] <= step * (gap_samples + 1):
                spans[-1].append(frame)
            else:
                spans.append([frame])
        for support in spans:
            if len(support) < (1 if predicate in {"entering", "leaving", "falling"} else min_support):
                continue
            start, end = support[0][0], min(last_frame, support[-1][0] + step - 1)
            support_frames = [frame for frame, _ in support]
            coverage = len(support) / (1 + (support_frames[-1] - support_frames[0]) // step)
            interval_conf = sum(score for _, score in support) / len(support) * coverage
            relation_id = hashlib.sha256(
                f"{run_id}:{subject}:{predicate}:{obj}:{start}:{end}".encode()
            ).hexdigest()[:20]
            result.append({
                "rel_uid": relation_id, "subject": subject, "predicate": predicate,
                "object": obj, "start_frame": start, "end_frame": end,
                "start_s": start / fps, "end_s": (end + 1) / fps,
                "support_frames": support_frames, "interval_conf": interval_conf,
                "run_id": run_id, "method": "geometry",
            })
    return result


def analyze(case_id, force=False):
    from .artifacts import export_case
    from .rules import predicates, region_events
    from huggingface_hub import snapshot_download
    import torch
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    output = case_dir(case_id)
    video = read_json(output / "video.json")
    source = Path(video["path"])
    if digest(source) != video["sha256"]:
        raise RuntimeError("Source video changed since ingestion")
    settings = config()
    forensic_labels = read_json(ROOT / "ontology.json")
    local_model = ROOT / "models" / "mask2former-swin-tiny-coco-panoptic"
    model_dir = local_model if (local_model / "model.safetensors").exists() else Path(snapshot_download(
        repo_id=settings["model"], allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"]
    ))
    weights = model_dir / "model.safetensors"
    if not weights.exists():
        raise RuntimeError("Pinned model weights were not downloaded")
    weights_hash = digest(weights)
    if settings["model"] == "facebook/mask2former-swin-tiny-coco-panoptic" and weights_hash != MODEL_SHA256:
        raise RuntimeError("Model weights do not match the recorded upstream SHA-256")
    pipeline_hash = pipeline_digest()
    run_id = hashlib.sha256(
        (case_id + video["sha256"] + digest(ROOT / "config.json") + weights_hash + pipeline_hash).encode()
    ).hexdigest()[:20]
    graph_file = output / "graph.json"
    if graph_file.exists() and not force:
        existing = read_json(graph_file)
        required = [output / name for name in ("videos.parquet", "tracks.parquet",
                    "relations_frame.parquet", "relations_interval.parquet",
                    "graph_export/cases.csv", "graph_export/videos.csv", "graph_export/runs.csv",
                    "graph_export/tracks.csv", "graph_export/links.csv", "graph_export/relations.csv")]
        required += [Path(track["mask_ref"]) for track in existing.get("tracks", []) if track.get("mask_ref")]
        if existing["run"]["run_id"] == run_id and all(path.exists() for path in required):
            print(f"Run {run_id} already has validated artifacts")
            return
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    capture = open_video(source)
    step = max(1, round(video["fps"] / settings["analysis_fps"]))
    mask_dir = output / "masks" / video["video_id"] / run_id
    mask_dir.mkdir(parents=True, exist_ok=True)
    tracks, previous, observations, rule_history, region_history = {}, {}, [], {}, {}
    next_track = 1
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    sample_count = 0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % step:
                frame_index += 1
                continue
            sample_count += 1
            if device == "cuda" and sample_count % 100 == 0:
                temperature = gpu_temperature()
                print(f"sample {sample_count}, GPU {temperature} C", flush=True)
                if temperature is not None and temperature >= settings["gpu_max_c"]:
                    raise RuntimeError(f"GPU temperature reached {temperature} C")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            scale = min(1.0, settings["max_side"] / max(frame.shape[:2]))
            if scale < 1:
                rgb = cv2.resize(rgb, None, fx=scale, fy=scale)
            inputs = processor(images=rgb, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                prediction = model(**inputs)
            panoptic = processor.post_process_panoptic_segmentation(
                prediction, target_sizes=[rgb.shape[:2]], label_ids_to_fuse=set(),
            )[0]
            labels = panoptic["segmentation"].cpu().numpy()
            segments = []
            for item in panoptic["segments_info"]:
                score = float(item.get("score", 1.0))
                if score < settings["min_segment_score"]:
                    continue
                mask = labels == item["id"]
                if np.any(mask):
                    # COCO-panoptic checkpoint: label IDs 0-79 are things; 80+ are stuff.
                    segments.append((str(model.config.id2label[item["label_id"]]).lower(), mask, score, item["label_id"] < 80))
            candidates = []
            for index, (label, mask, _, _) in enumerate(segments):
                for uid, old in previous.items():
                    if old["class"] == label and sample_count - old["sample"] <= settings["max_track_gap_samples"] + 1:
                        overlap = mask_iou(mask, old["mask"])
                        if overlap >= settings["track_iou"]:
                            candidates.append((overlap, index, uid))
            assigned, used = {}, set()
            for _, index, uid in sorted(candidates, reverse=True):
                if index not in assigned and uid not in used:
                    assigned[index], used = uid, used | {uid}
            track_mask = np.zeros(labels.shape, dtype=np.uint16)
            active, active_scores = {}, {}
            for index, (label, mask, score, is_thing) in enumerate(segments):
                uid = assigned.get(index)
                if uid is None:
                    if next_track > np.iinfo(np.uint16).max:
                        raise RuntimeError("Too many tracks for mask format")
                    uid = next_track
                    next_track += 1
                track_uid = f"{run_id}:{uid}"
                entry = tracks.setdefault(track_uid, {
                    "track_uid": track_uid, "track_id": uid, "video_id": video["video_id"],
                    "class": label, "is_thing": is_thing,
                    "forensic_class": forensic_labels.get(label, "other"),
                    "first_frame": frame_index, "last_frame": frame_index,
                    "score_sum": 0.0, "observations": 0, "run_id": run_id, "modality": "VIDEO",
                })
                entry["last_frame"] = frame_index
                entry["score_sum"] += score
                entry["observations"] += 1
                previous[uid] = {"class": label, "mask": mask, "sample": sample_count}
                active[track_uid] = mask
                active_scores[track_uid] = score
                track_mask[mask] = uid
            if not cv2.imwrite(str(mask_dir / f"{frame_index:08d}.png"), track_mask):
                raise RuntimeError("Could not write track mask")
            ids = sorted(active)
            for index, first in enumerate(ids):
                for second in ids[index + 1:]:
                    if "person" not in (tracks[first]["class"], tracks[second]["class"]):
                        continue
                    subject, obj = first, second
                    if tracks[second]["class"] == "person" and tracks[first]["class"] != "person":
                        subject, obj = second, first
                    words = predicates(active[subject], active[obj], tracks[subject]["class"],
                                       tracks[obj]["class"], settings, rule_history,
                                       (subject, obj), sample_count)
                    observations.extend({"subject": subject, "predicate": word, "object": obj,
                                         "frame": frame_index, "score": min(active_scores[subject], active_scores[obj]),
                                         "run_id": run_id, "method": "geometry"}
                                        for word in words)
            for uid in ids:
                if tracks[uid]["class"] != "person":
                    continue
                for word, region in region_events(active[uid], settings["regions"], region_history, uid, sample_count):
                    observations.append({"subject": uid, "predicate": word, "object": f"{run_id}:region:{region}",
                                         "frame": frame_index, "score": active_scores[uid],
                                         "run_id": run_id, "method": "geometry"})
            frame_index += 1
    finally:
        capture.release()
    for entry in tracks.values():
        entry["mean_score"] = entry.pop("score_sum") / entry.pop("observations")
        entry["mask_ref"] = str(mask_dir / f"{entry['track_id']}.parquet")
    for name in settings["regions"]:
        uid = f"{run_id}:region:{name}"
        tracks[uid] = {"track_uid": uid, "track_id": 0, "video_id": video["video_id"],
                       "class": f"region:{name}", "is_thing": False, "first_frame": 0,
                       "last_frame": video["n_frames"] - 1, "mean_score": 1.0,
                       "run_id": run_id, "modality": "VIDEO", "mask_ref": None}
    relations = aggregate(
        observations, step, video["n_frames"] - 1,
        settings["max_relation_gap_samples"], settings["min_relation_support_samples"], run_id, video["fps"],
    )
    for relation in relations:
        relation["conf"] = min(tracks[relation["subject"]]["mean_score"],
                               tracks[relation["object"]]["mean_score"], relation["interval_conf"])
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    graph = {
        "case_id": video["case_id"], "video": video, "run": {
            "run_id": run_id, "model": settings["model"], "weights_sha256": weights_hash,
            "seg_model": settings["model"], "seg_weights_sha256": weights_hash,
            "tracker": "same-class-mask-iou", "tracker_version": "1",
            "relation_method": "geometry", "host_gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "confidence_method": "mean frame segment score times support coverage; minimum with endpoint means",
            "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
            "config_sha256": digest(ROOT / "config.json"), "pipeline_sha256": pipeline_hash,
            "analysis_fps": video["fps"] / step,
            "sample_step": step, "device": device, "seconds": round(time.perf_counter() - started, 3),
            "sampled_frames": sample_count,
            "git_sha": revision.stdout.strip() if revision.returncode == 0 else None,
        },
        "tracks": list(tracks.values()), "relations": relations,
    }
    export_case(output, graph, observations)
    save_json(graph_file, graph)
    print(json.dumps({"run_id": run_id, "tracks": len(tracks), "relations": len(relations), "seconds": graph["run"]["seconds"]}, indent=2))


TEMPLATES = {
    "near": "near", "touching": "touching", "approaching": "approaching",
}


def load_graph(case_id, uri, user, password):
    from neo4j import GraphDatabase

    graph = read_json(case_dir(case_id) / "graph.json")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
        with driver.session() as session:
            for label, key in (("VIDEO_Case", "case_id"), ("VIDEO_Video", "video_id"),
                               ("VIDEO_Track", "track_uid"), ("VIDEO_Run", "run_id")):
                session.run(f"CREATE CONSTRAINT video_{label.lower()}_{key} IF NOT EXISTS "
                            f"FOR (n:{label}) REQUIRE n.{key} IS UNIQUE").consume()
            for name, statement in (
                ("video_track_class", "FOR (n:VIDEO_Track) ON (n.class)"),
                ("video_source_hash", "FOR (n:VIDEO_Video) ON (n.sha256)"),
                ("video_rel_predicate", "FOR ()-[r:VIDEO_REL]-() ON (r.predicate)"),
            ):
                session.run(f"CREATE INDEX {name} IF NOT EXISTS {statement}").consume()
            with session.begin_transaction() as tx:
                tx.run("MERGE (c:VIDEO_Case {case_id:$id})", id=graph["case_id"]).consume()
                tx.run("MERGE (v:VIDEO_Video {video_id:$id}) SET v += $props WITH v MATCH (c:VIDEO_Case {case_id:$case_id}) MERGE (v)-[:VIDEO_IN_CASE]->(c)", id=graph["video"]["video_id"], props=graph["video"], case_id=graph["case_id"]).consume()
                tx.run("MERGE (r:VIDEO_Run {run_id:$id}) SET r += $props WITH r MATCH (v:VIDEO_Video {video_id:$video_id}) MERGE (r)-[:VIDEO_FOR_VIDEO]->(v)", id=graph["run"]["run_id"], props=graph["run"], video_id=graph["video"]["video_id"]).consume()
                tx.run("UNWIND $tracks AS track MERGE (t:VIDEO_Track {track_uid:track.track_uid}) SET t += track WITH t, track MATCH (v:VIDEO_Video {video_id:track.video_id}), (r:VIDEO_Run {run_id:track.run_id}) MERGE (t)-[:VIDEO_IN_VIDEO]->(v) MERGE (t)-[:VIDEO_PRODUCED_BY]->(r)", tracks=graph["tracks"]).consume()
                tx.run("UNWIND $relations AS relation MATCH (a:VIDEO_Track {track_uid:relation.subject}), (b:VIDEO_Track {track_uid:relation.object}) MERGE (a)-[r:VIDEO_REL {rel_uid:relation.rel_uid}]->(b) SET r += relation", relations=graph["relations"]).consume()
                track_count = tx.run("MATCH (t:VIDEO_Track {run_id:$run_id}) RETURN count(t) AS count", run_id=graph["run"]["run_id"]).single()["count"]
                relation_count = tx.run("MATCH ()-[r:VIDEO_REL {run_id:$run_id}]->() RETURN count(r) AS count", run_id=graph["run"]["run_id"]).single()["count"]
                if (track_count, relation_count) != (len(graph["tracks"]), len(graph["relations"])):
                    raise RuntimeError("Database counts do not match the graph export")
                tx.commit()
    finally:
        driver.close()
    print(f"Loaded {len(graph['tracks'])} tracks and {len(graph['relations'])} relations")


def query_graph(case_id, template, subject_class, object_class, uri, user, password):
    from neo4j import GraphDatabase, READ_ACCESS

    if template not in TEMPLATES:
        raise ValueError(f"Unknown template: {template}")
    graph = read_json(case_dir(case_id) / "graph.json")
    statement = """
        MATCH (a:VIDEO_Track)-[r:VIDEO_REL]->(b:VIDEO_Track)
        WHERE a.run_id = $run_id AND b.run_id = $run_id AND r.run_id = $run_id
          AND r.predicate = $predicate AND a.class = $subject_class
          AND b.class = $object_class
        RETURN r.rel_uid AS rel_uid, r.start_frame AS start_frame,
               r.end_frame AS end_frame, a.track_uid AS subject,
               b.track_uid AS object
        ORDER BY size(r.support_frames) DESC, r.start_frame
        LIMIT 1
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(default_access_mode=READ_ACCESS) as session:
            loaded = session.run("MATCH (r:VIDEO_Run {run_id:$run_id}) RETURN count(r) AS count", run_id=graph["run"]["run_id"]).single()["count"]
            if loaded != 1:
                raise RuntimeError("This run is not loaded exactly once in Neo4j")
            matches = [dict(record) for record in session.run(
                statement, run_id=graph["run"]["run_id"],
                predicate=TEMPLATES[template], subject_class=subject_class,
                object_class=object_class,
            )]
    finally:
        driver.close()
    if not matches:
        print("NO_MATCH")
        return
    for match in matches:
        query_id = hashlib.sha256(
            f"{template}:{subject_class}:{object_class}:{match['rel_uid']}".encode()
        ).hexdigest()[:16]
        path = render(case_id, graph, match, query_id, {
            "template": template, "subject_class": subject_class, "object_class": object_class,
            "cypher": statement.strip(),
            "parameters": {"run_id": graph["run"]["run_id"], "predicate": TEMPLATES[template],
                           "subject_class": subject_class, "object_class": object_class},
        })
        print(path)


def render(case_id, graph, match, query_id, query):
    video = graph["video"]
    if digest(Path(video["path"])) != video["sha256"]:
        raise RuntimeError("Source video no longer matches its provenance hash")
    track_uids = [match["subject"], match["object"]]
    if not all(any(t["track_uid"] == uid for t in graph["tracks"]) for uid in track_uids):
        raise RuntimeError("Query tracks do not resolve to the local graph")
    rel_uids = [uid for uid in (match.get("rel_uid"), match.get("second_rel_uid")) if uid]
    if not all(any(r["rel_uid"] == uid and r["subject"] == match["subject"]
                   and r["object"] == match["object"] for r in graph["relations"]) for uid in rel_uids):
        raise RuntimeError("Query relations do not resolve to the local graph")
    settings = config()
    padding = round(settings["snippet_padding_seconds"] * video["fps"])
    start = max(0, match["start_frame"] - padding)
    end = min(video["n_frames"] - 1, match["end_frame"] + padding)
    destination = case_dir(case_id) / "snippets" / query_id
    destination.mkdir(parents=True, exist_ok=True)
    mp4 = destination / "snippet.mp4"
    temporary = destination / "snippet.tmp.mp4"
    capture = open_video(video["path"])
    for _ in range(start):
        if not capture.grab():
            capture.release()
            raise RuntimeError(f"Source ended before frame {start}")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), video["fps"], (video["width"], video["height"]))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("MP4 encoder unavailable")
    source_track_ids = {track["track_uid"]: track["track_id"] for track in graph["tracks"]}
    wanted = list(dict.fromkeys(source_track_ids[uid] for uid in track_uids))
    mask_ref = Path(next(track["mask_ref"] for track in graph["tracks"] if track["mask_ref"]))
    mask_dir = mask_ref if mask_ref.is_dir() else mask_ref.parent
    step = graph["run"]["sample_step"]
    last_mask = None
    try:
        for frame_index in range(start, end + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Source ended before frame {end}")
            # shortcut: nearest sampled mask is used between analyzed frames; use propagation if motion causes visible drift.
            mask_frame = round(frame_index / step) * step
            mask_path = mask_dir / f"{mask_frame:08d}.png"
            if mask_path.exists():
                last_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if last_mask is not None:
                display_mask = cv2.resize(last_mask, (video["width"], video["height"]), interpolation=cv2.INTER_NEAREST)
                overlay = frame.copy()
                for color, uid in zip(((0, 180, 255), (255, 100, 0)), wanted):
                    overlay[display_mask == uid] = color
                frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
                for uid in wanted:
                    ys, xs = np.nonzero(display_mask == uid)
                    if len(xs):
                        cv2.putText(frame, f"track {uid}", (int(xs.mean()), int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                        cv2.putText(frame, f"track {uid}", (int(xs.mean()), int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            caption = f"{case_id} | {video['video_id']} | {query['template']} | {frame_index / video['fps']:.2f}s"
            if video.get("start_wallclock"):
                caption += " | " + (datetime.fromisoformat(video["start_wallclock"]) +
                                    timedelta(seconds=frame_index / video["fps"])).isoformat(timespec="seconds")
            cv2.putText(frame, caption, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(frame, caption, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            writer.write(frame)
    finally:
        writer.release()
        capture.release()
    if not temporary.exists() or not temporary.stat().st_size:
        raise RuntimeError("Snippet encoder produced no video")
    temporary.replace(mp4)
    sidecar = {
        "query": query, "query_id": query_id, "run_id": graph["run"]["run_id"],
        "rel_uids": rel_uids, "track_uids": track_uids,
        "source_video_sha256": video["sha256"], "source_video_path": video["path"],
        "source_start_frame": start, "source_end_frame": end,
        "relation_start_frame": match["start_frame"], "relation_end_frame": match["end_frame"],
        "padding_seconds": settings["snippet_padding_seconds"],
        "mask_sampling_step": step, "renderer_sha256": digest(Path(__file__)),
        "renderer_git_sha": graph["run"].get("git_sha"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json(destination / "snippet.json", sidecar)
    return mp4


def main():
    local_env = ROOT / ".env"
    if local_env.exists():
        for line in local_env.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                if key in {"NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"}:
                    os.environ.setdefault(key, value)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest_cmd = commands.add_parser("ingest")
    ingest_cmd.add_argument("case_id")
    ingest_cmd.add_argument("source")
    ingest_cmd.add_argument("--start-wallclock", help="Optional ISO-8601 camera start time")
    ingest_cmd.add_argument("--group-case", help="Logical case shared by multiple separately ingested videos")
    analyze_cmd = commands.add_parser("analyze")
    analyze_cmd.add_argument("case_id")
    analyze_cmd.add_argument("--force", action="store_true", help="Recompute for repeatability measurement")
    commands.add_parser("download-model")
    commands.add_parser("download-clip")
    load_cmd = commands.add_parser("load")
    load_cmd.add_argument("case_id")
    query_cmd = commands.add_parser("query")
    query_cmd.add_argument("case_id")
    query_cmd.add_argument("template", choices=TEMPLATES)
    query_cmd.add_argument("subject_class")
    query_cmd.add_argument("object_class")
    ask_cmd = commands.add_parser("ask")
    ask_cmd.add_argument("case_id")
    choice = ask_cmd.add_mutually_exclusive_group(required=True)
    choice.add_argument("--template", choices=("presence", "relation", "holding", "crossing", "approach", "timed_relation", "sequence", "footage_exists"))
    choice.add_argument("--nl")
    ask_cmd.add_argument("--params", default="{}", help="JSON object of template parameters")
    group_cmd = commands.add_parser("ask-case")
    group_cmd.add_argument("case_id", help="Logical case ID shared by ingested videos")
    group_choice = group_cmd.add_mutually_exclusive_group(required=True)
    group_choice.add_argument("--template", choices=("presence", "relation", "holding", "crossing", "approach", "timed_relation", "sequence", "footage_exists"))
    group_choice.add_argument("--nl")
    group_cmd.add_argument("--params", default="{}")
    evaluation_cmd = commands.add_parser("evaluate")
    evaluation_cmd.add_argument("manifest", help="JSON dataset manifest with ground-truth spans")
    evaluation_cmd.add_argument("--skip-clip", action="store_true", help="Metrics smoke check only; not a complete baseline table")
    audit_cmd = commands.add_parser("audit")
    audit_cmd.add_argument("case_id")
    consensus_cmd = commands.add_parser("consensus")
    consensus_cmd.add_argument("first")
    consensus_cmd.add_argument("second")
    consensus_cmd.add_argument("output")
    repeat_cmd = commands.add_parser("repeatability")
    repeat_cmd.add_argument("first_graph")
    repeat_cmd.add_argument("second_graph")
    for command in (load_cmd, query_cmd, ask_cmd, group_cmd):
        command.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
        command.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    args = parser.parse_args()
    if args.command == "ingest":
        ingest(args.case_id, args.source, args.start_wallclock, args.group_case)
    elif args.command == "download-model":
        download_model()
    elif args.command == "download-clip":
        from .evaluation import download_clip
        download_clip()
    elif args.command == "analyze":
        analyze(args.case_id, args.force)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        print(evaluate(args.manifest, use_clip=not args.skip_clip,
                       uri=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
                       user=os.getenv("NEO4J_USER", "neo4j"), password=os.getenv("NEO4J_PASSWORD")))
    elif args.command == "audit":
        from .audit import provenance
        print(json.dumps(provenance(args.case_id, os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
                                    if os.getenv("NEO4J_PASSWORD") else None,
                                    os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD")), indent=2))
    elif args.command == "consensus":
        from .annotations import consensus
        print(json.dumps(consensus(args.first, args.second, args.output)["agreement"], indent=2))
    elif args.command == "repeatability":
        from .audit import repeatability
        print(json.dumps(repeatability(args.first_graph, args.second_graph), indent=2))
    else:
        password = os.getenv("NEO4J_PASSWORD")
        if not password:
            parser.error("Set NEO4J_PASSWORD in the environment")
        if args.command == "load":
            load_graph(args.case_id, args.uri, args.user, password)
        elif args.command == "query":
            query_graph(args.case_id, args.template, args.subject_class, args.object_class, args.uri, args.user, password)
        else:
            from .queries import ask, route_nl
            if args.nl:
                model_config = read_json(ROOT / "models" / "mask2former-swin-tiny-coco-panoptic" / "config.json")
                classes = {str(label).lower() for label in model_config["id2label"].values()}
                classes |= {f"region:{name}" for name in config()["regions"]}
                routed = route_nl(args.nl, classes)
                if routed is None:
                    print("CANNOT_ANSWER")
                    return
                template, params = routed["template"], routed["params"]
            else:
                template, params = args.template, json.loads(args.params)
            if args.command == "ask":
                ask(args.case_id, template, params, args.uri, args.user, password, args.nl)
            else:
                videos = [path.parents[1].name for path in (ROOT / "cases").glob("*/video/video.json")
                          if read_json(path)["case_id"] == args.case_id]
                if not videos:
                    raise RuntimeError(f"No ingested videos in case {args.case_id}")
                paths = [item for storage_id in sorted(videos)
                         for item in ask(storage_id, template, params, args.uri, args.user,
                                         password, args.nl, quiet_no_match=True)]
                if not paths:
                    print("NO_MATCH")
