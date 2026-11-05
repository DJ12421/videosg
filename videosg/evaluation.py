"""Video-level retrieval evaluation; ground truth is read only by metrics."""

import json
import hashlib
import random
import statistics
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from .pipeline import ROOT, case_dir, config, digest, gpu_temperature, read_json, save_json
from .downloads import download_ranges

CLIP_SHA256 = "74813fbcdc750f235c9784c367ca1394d2a5c25eb0aac92761752ac239db7cff"


def temporal_iou(a, b):
    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return overlap / union if union else 0.0


def mock_graph_quality(graph, events, track_map):
    """Identity-aware temporal R@K; volume IoU needs annotated masks."""
    ranked = sorted(graph["relations"], key=lambda row: (-len(row["support_frames"]), row["start_s"]))
    result = {}
    for threshold in (0.1, 0.5):
        for k in (20, 50, 100):
            hits = []
            by_predicate = {}
            for truth in events:
                hit = any(track_map.get(row["subject"]) == truth["subject"]
                          and track_map.get(row["object"]) == truth["object"]
                          and row["predicate"] == truth["predicate"]
                          and temporal_iou([row["start_s"], row["end_s"]],
                                           [truth["start_s"], truth["end_s"]]) >= threshold
                          for row in ranked[:k])
                hits.append(float(hit))
                by_predicate.setdefault(truth["predicate"], []).append(float(hit))
            result[f"R@{k}_tIoU_{threshold}"] = statistics.mean(hits) if hits else None
            result[f"mR@{k}_tIoU_{threshold}"] = statistics.mean(
                statistics.mean(values) for values in by_predicate.values()) if by_predicate else None
    result["volume_IoU"] = None
    result["ranking"] = "support samples descending; source time tie break"
    return result


def image_mask_iou(pred_path, truth_path, pred_id=1, truth_id=1):
    predicted = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
    truth = cv2.imread(str(truth_path), cv2.IMREAD_UNCHANGED)
    if predicted is None or truth is None or predicted.shape != truth.shape:
        raise ValueError("Mask pair is missing or dimensions differ")
    first, second = predicted == pred_id, truth == truth_id
    union = np.count_nonzero(first | second)
    return np.count_nonzero(first & second) / union if union else 1.0


def score_query(predicted, truth):
    """One query's recall and mean best temporal overlap."""
    if not truth:
        return {"abstention": float(not predicted)}
    best = [max((temporal_iou(item, target) for item in predicted), default=0.0) for target in truth]
    result = {"mean_tiou": statistics.mean(best)}
    for limit in (1, 5):
        for threshold in (0.3, 0.5, 0.7):
            hit = sum(any(temporal_iou(item, target) >= threshold for item in predicted[:limit])
                      for target in truth)
            result[f"recall_{limit}_{threshold}"] = hit / len(truth)
    return result


def merge_points(points, step, duration):
    spans = []
    for point in sorted(points):
        if spans and point <= spans[-1][1] + step:
            spans[-1][1] = min(duration, point + step)
        else:
            spans.append([point, min(duration, point + step)])
    return spans


def graph_predictions(graph, query, presence=False):
    video = graph["video"]
    fps = video["fps"]
    tracks = {track["track_uid"]: track for track in graph["tracks"]}
    subject, obj = query["subject_class"], query.get("object_class")
    if presence:
        spans = [[track["first_frame"] / fps, (track["last_frame"] + 1) / fps]
                 for track in tracks.values() if track["class"] == subject]
        if obj:
            spans = [[max(a[0], b[0]), min(a[1], b[1])] for a in spans
                     for b in [[t["first_frame"] / fps, (t["last_frame"] + 1) / fps]
                               for t in tracks.values() if t["class"] == obj]
                     if max(a[0], b[0]) < min(a[1], b[1])]
        return sorted(spans, key=lambda span: span[0])[:5]
    spans = [[r["start_s"], r["end_s"]] for r in graph["relations"]
             if r["predicate"] == query["predicate"]
             and tracks.get(r["subject"], {}).get("class") == subject
             and (obj is None or tracks.get(r["object"], {}).get("class") == obj)]
    return sorted(spans, key=lambda span: span[0])[:5]


def database_predictions(driver, graph, query):
    from neo4j import READ_ACCESS

    statement = """MATCH (a:VIDEO_Track)-[r:VIDEO_REL]->(b:VIDEO_Track)
WHERE a.run_id=$run_id AND b.run_id=$run_id AND r.run_id=$run_id
 AND a.class=$subject_class AND r.predicate=$predicate
 AND ($object_class IS NULL OR b.class=$object_class)
RETURN r.start_s AS start_s, r.end_s AS end_s
ORDER BY r.start_s LIMIT 5"""
    started = time.perf_counter()
    with driver.session(default_access_mode=READ_ACCESS) as session:
        records = session.run(statement, run_id=graph["run"]["run_id"],
                              subject_class=query["subject_class"], object_class=query.get("object_class"),
                              predicate=query["predicate"])
        spans = [[row["start_s"], row["end_s"]] for row in records]
    return spans, time.perf_counter() - started


def random_predictions(duration, typical, seed):
    rng = random.Random(seed)
    length = min(duration, typical)
    return [[start, start + length] for start in
            (rng.uniform(0, max(0, duration - length)) for _ in range(5))]


class ClipBaseline:
    def __init__(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        weights = ROOT / "models" / "laion-clip-vit-b-32" / "model.safetensors"
        if not weights.exists():
            raise RuntimeError("Run `python -m videosg download-clip` before the B2 baseline")
        if digest(weights) != CLIP_SHA256:
            raise RuntimeError("CLIP weights differ from the pinned SHA-256")
        self.model = CLIPModel.from_pretrained(weights.parent, use_safetensors=True)
        self.processor = CLIPProcessor.from_pretrained(weights.parent)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.cached_path = None


    def predict(self, path, question, sample_fps=1):
        from PIL import Image

        if self.cached_path != str(path):
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open {path}")
            fps = capture.get(cv2.CAP_PROP_FPS)
            self.duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps
            step = max(1, round(fps / sample_fps))
            self.sample_step_s = step / fps
            features, self.times = [], []
            frame = 0
            try:
                with self.torch.inference_mode():
                    while True:
                        sampled = frame % step == 0
                        ok, pixels = capture.read() if sampled else (capture.grab(), None)
                        if not ok:
                            break
                        if sampled:
                            image = self.processor(images=Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)),
                                                   return_tensors="pt")["pixel_values"].to(self.device)
                            feature = self.model.get_image_features(pixel_values=image)
                            feature /= feature.norm(dim=-1, keepdim=True)
                            features.append(feature.cpu())
                            self.times.append(frame / fps)
                            if len(features) % 100 == 0:
                                temperature = gpu_temperature() if self.device == "cuda" else None
                                print(f"CLIP sampled {len(features)} frames, GPU {temperature} C", flush=True)
                                if temperature is not None and temperature >= config()["gpu_max_c"]:
                                    raise RuntimeError(f"GPU temperature reached {temperature} C")
                        frame += 1
            finally:
                capture.release()
            self.image_features = self.torch.cat(features).to(self.device) if features else None
            self.cached_path = str(path)
        if self.image_features is None:
            return []
        texts = self.processor(text=[question], return_tensors="pt", padding=True).to(self.device)
        with self.torch.inference_mode():
            text_feature = self.model.get_text_features(**texts)
            text_feature /= text_feature.norm(dim=-1, keepdim=True)
            scores = (self.image_features @ text_feature.T).squeeze(-1).cpu().numpy()
        # Fixed top-quartile selection; never use test labels to choose the threshold.
        cutoff = float(np.quantile(scores, 0.75))
        return merge_points([t for t, score in zip(self.times, scores) if score >= cutoff],
                            self.sample_step_s, self.duration)[:5]


def download_clip():
    """Fetch the MIT-listed CLIP mirror into ignored local model storage."""
    target = ROOT / "models" / "laion-clip-vit-b-32"
    target.mkdir(parents=True, exist_ok=True)
    base = "https://modelscope.cn/models/AI-ModelScope/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/master/"
    for name in ("open_clip_config.json", "model.safetensors"):
        destination = target / name
        if destination.exists():
            continue
        download_ranges(base + name, destination)
    for name in ("config.json", "preprocessor_config.json", "vocab.json", "merges.txt",
                 "tokenizer_config.json", "special_tokens_map.json"):
        destination = target / name
        if destination.exists():
            continue
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with requests.get(base + name, timeout=30) as response:
            response.raise_for_status()
            temporary.write_bytes(response.content)
        temporary.replace(destination)
    weights_hash = digest(target / "model.safetensors")
    if weights_hash != CLIP_SHA256:
        raise RuntimeError("Mirrored CLIP checkpoint does not match the pinned SHA-256")
    save_json(target / "provenance.json", {"source": base, "weights_sha256": weights_hash})
    print(f"CLIP weights SHA-256: {weights_hash}")


def bootstrap_by_video(rows, metric, seed=7, repetitions=1000):
    by_video = {}
    for row in rows:
        if metric in row:
            by_video.setdefault(row["video_id"], []).append(row[metric])
    videos = sorted(by_video)
    if not videos:
        return None
    per_video = [statistics.mean(by_video[video]) for video in videos]
    rng = random.Random(seed)
    samples = sorted(statistics.mean(rng.choices(per_video, k=len(videos))) for _ in range(repetitions))
    return [statistics.mean(per_video), samples[25], samples[974]]


def evaluate(manifest_path, use_clip=True, uri=None, user="neo4j", password=None):
    """Manifest: [{case_id, queries:[{text, subject_class, object_class?, predicate,
    truth:[[start_s,end_s],...], typical_duration_s}]}]. Truth never enters prediction methods.
    """
    entries = read_json(Path(manifest_path))
    datasets = {entry["dataset"] for entry in entries}
    if len(datasets) != 1:
        raise ValueError("Evaluate one dataset per manifest; do not pool datasets")
    clip = ClipBaseline() if use_clip else None
    driver = None
    if password:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri or "bolt://127.0.0.1:7687", auth=(user, password))
        driver.verify_connectivity()
    rows = {name: [] for name in ("B0_random", "B1_presence", "B2_clip", "B3_rules")}
    for entry in entries:
        graph = read_json(case_dir(entry["case_id"]) / "graph.json")
        video = graph["video"]
        if digest(Path(video["path"])) != video["sha256"]:
            raise RuntimeError("Evaluation source hash mismatch")
        if driver:
            with driver.session() as session:
                loaded = session.run("MATCH (run:VIDEO_Run {run_id:$run_id}) RETURN count(run) AS n",
                                     run_id=graph["run"]["run_id"]).single()["n"]
            if loaded != 1:
                raise RuntimeError(f"Evaluation run not loaded exactly once: {entry['case_id']}")
        for index, query in enumerate(entry["queries"]):
            truth = query["truth"]
            typical = query["typical_duration_s"]
            rule_spans, query_latency = (database_predictions(driver, graph, query) if driver else
                                         (graph_predictions(graph, query), None))
            predictions = {
                "B0_random": random_predictions(video["duration_s"], typical,
                                                 int(hashlib.sha256(f"{video['video_id']}:{index}:7".encode()).hexdigest()[:8], 16)),
                "B1_presence": graph_predictions(graph, query, presence=True),
                "B3_rules": rule_spans,
            }
            if clip:
                predictions["B2_clip"] = clip.predict(video["path"], query["text"])
            if "oracle_graph" in entry:
                predictions["B5_oracle"] = graph_predictions(read_json(Path(entry["oracle_graph"])), query)
            for baseline, spans in predictions.items():
                rows.setdefault(baseline, [])
                row = {"video_id": video["video_id"], "question": query["text"],
                       "activity": query.get("activity", query["text"]),
                       "predictions": spans, **score_query(spans, truth)}
                if baseline == "B3_rules" and query_latency is not None:
                    row["query_seconds"] = query_latency
                if baseline == "B3_rules" and query.get("mask_pairs"):
                    row["mask_iou"] = statistics.mean(image_mask_iou(pair["pred"], pair["truth"],
                                                                       pair.get("pred_id", 1), pair.get("truth_id", 1))
                                                       for pair in query["mask_pairs"])
                rows[baseline].append(row)
    output = ROOT / "eval" / Path(manifest_path).stem
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "results.json", rows)
    if driver:
        driver.close()
    metrics = ["recall_1_0.3", "recall_1_0.5", "recall_1_0.7", "recall_5_0.3",
               "recall_5_0.5", "recall_5_0.7", "mean_tiou", "abstention", "mask_iou"]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    lines = [f"# {next(iter(datasets))} retrieval evaluation", "", f"Command: `python -m videosg evaluate {manifest_path}`",
             f"Config SHA-256: `{digest(ROOT / 'config.json')}`",
             f"Git SHA: `{revision.stdout.strip() if revision.returncode == 0 else 'uncommitted'}`",
             f"B3 source: {'Neo4j read-only query' if driver else 'local export (smoke check)' }", "",
             "| Baseline | Metric | Mean | 95% video bootstrap CI |", "|---|---|---:|---:|"]
    if len(entries) < 2:
        lines.insert(6, "Pilot only: a one-video bootstrap interval is degenerate and not a population uncertainty estimate.")
    annotation_revisions = {entry.get("annotation_repo_sha") for entry in entries if entry.get("annotation_repo_sha")}
    if annotation_revisions:
        lines.insert(6, "Annotation repository commit(s): " + ", ".join(sorted(annotation_revisions)))
    for baseline, data in rows.items():
        for metric in metrics:
            value = bootstrap_by_video(data, metric)
            if value:
                lines.append(f"| {baseline} | {metric} | {value[0]:.3f} | [{value[1]:.3f}, {value[2]:.3f}] |")
    activities = sorted({row["activity"] for data in rows.values() for row in data})
    if activities:
        lines += ["", "## Per activity", "", "| Activity | Baseline | Recall@1 tIoU 0.5 | 95% CI |",
                  "|---|---|---:|---:|"]
        for activity in activities:
            for baseline, data in rows.items():
                value = bootstrap_by_video([row for row in data if row["activity"] == activity], "recall_1_0.5")
                if value:
                    lines.append(f"| {activity} | {baseline} | {value[0]:.3f} | [{value[1]:.3f}, {value[2]:.3f}] |")
    latencies = [row["query_seconds"] for row in rows["B3_rules"] if "query_seconds" in row]
    if latencies:
        lines += ["", "## Query cost", "", f"Median: {statistics.median(latencies):.4f} s; "
                  f"p95: {float(np.quantile(latencies, 0.95)):.4f} s."]
    (output / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
