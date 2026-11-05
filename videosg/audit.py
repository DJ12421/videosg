"""Provenance, rendering and repeat-run checks for derived evidence."""

from pathlib import Path
import statistics

import cv2
import numpy as np

from .pipeline import case_dir, digest, read_json


def decoded_frames(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open snippet {path}")
    count = 0
    try:
        while capture.grab():
            count += 1
    finally:
        capture.release()
    return count


def provenance(case_id, uri=None, user="neo4j", password=None):
    if uri and not password:
        raise ValueError("Database password required for graph provenance audit")
    driver = None
    if uri:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri, auth=(user, password))
    folder = case_dir(case_id)
    graph = read_json(folder / "graph.json")
    tracks = {item["track_uid"] for item in graph["tracks"]}
    relations = {item["rel_uid"] for item in graph["relations"]}
    video = graph["video"]
    if digest(Path(video["path"])) != video["sha256"]:
        raise RuntimeError("Source hash mismatch")
    checked = 0
    try:
        for path in (folder / "snippets").glob("*/snippet.json"):
            sidecar = read_json(path)
            if sidecar["run_id"] != graph["run"]["run_id"]:
                continue  # prior pipeline runs remain in the case folder
            if sidecar["source_video_sha256"] != video["sha256"]:
                raise RuntimeError(f"Bad source hash in {path}")
            if not set(sidecar["track_uids"]) <= tracks or not set(sidecar["rel_uids"]) <= relations:
                raise RuntimeError(f"Unresolved graph element in {path}")
            expected = sidecar["source_end_frame"] - sidecar["source_start_frame"] + 1
            if decoded_frames(path.with_name("snippet.mp4")) != expected:
                raise RuntimeError(f"Wrong frame count in {path}")
            if driver:
                with driver.session() as session:
                    found = session.run("""MATCH (v:VIDEO_Video {video_id:$video_id, sha256:$sha}),
 (run:VIDEO_Run {run_id:$run_id})
 RETURN count(DISTINCT v) AS videos, count(DISTINCT run) AS runs""",
                                        video_id=video["video_id"], sha=video["sha256"],
                                        run_id=sidecar["run_id"]).single()
                    if found["videos"] != 1 or found["runs"] != 1:
                        raise RuntimeError(f"Unresolved database video/run in {path}")
                    for uid in sidecar["track_uids"]:
                        if session.run("MATCH (t:VIDEO_Track {track_uid:$uid, run_id:$run_id}) RETURN count(t) AS n",
                                       uid=uid, run_id=sidecar["run_id"]).single()["n"] != 1:
                            raise RuntimeError(f"Unresolved database track {uid}")
                    for uid in sidecar["rel_uids"]:
                        if session.run("MATCH ()-[r:VIDEO_REL {rel_uid:$uid, run_id:$run_id}]->() RETURN count(r) AS n",
                                       uid=uid, run_id=sidecar["run_id"]).single()["n"] != 1:
                            raise RuntimeError(f"Unresolved database relation {uid}")
            checked += 1
    finally:
        if driver:
            driver.close()
    latencies = [read_json(path)["query_seconds"] for path in (folder / "snippets").glob("*/query.json")
                 if "query_seconds" in read_json(path)]
    return {"run_id": graph["run"]["run_id"], "sidecars_checked": checked,
            "provenance_completeness": 1.0 if checked else None, "db_checked": bool(uri),
            "decode_seconds": video.get("decode_seconds"), "analysis_seconds": graph["run"].get("seconds"),
            "query_median_seconds": statistics.median(latencies) if latencies else None,
            "query_p95_seconds": float(np.quantile(latencies, 0.95)) if latencies else None}


def repeatability(first, second):
    a, b = read_json(Path(first)), read_json(Path(second))
    triplets = lambda graph: {(r["subject"].split(":")[-1], r["predicate"], r["object"].split(":")[-1])
                              for r in graph["relations"]}
    left, right = triplets(a), triplets(b)
    union = left | right
    interval_diff = []
    def by_key(graph):
        groups = {}
        for relation in graph["relations"]:
            triple = (relation["subject"].split(":")[-1], relation["predicate"],
                      relation["object"].split(":")[-1])
            groups.setdefault(triple, []).append(relation)
        return {(triple, index): relation for triple, relations in groups.items()
                for index, relation in enumerate(sorted(relations, key=lambda item: item["start_frame"]))}
    aa, bb = by_key(a), by_key(b)
    for key in aa.keys() & bb.keys():
        interval_diff.append(abs(aa[key]["start_s"] - bb[key]["start_s"]))
        interval_diff.append(abs(aa[key]["end_s"] - bb[key]["end_s"]))
    return {"triplet_jaccard": len(left & right) / len(union) if union else 1.0,
            "mean_interval_difference_s": float(np.mean(interval_diff)) if interval_diff else None,
            "matched_interval_boundaries": len(interval_diff)}
