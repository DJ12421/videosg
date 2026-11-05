"""Portable, atomic case artifacts derived from the source and graph export."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .pipeline import read_json, save_json


def parquet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    temporary.replace(path)


def rle(mask):
    """Row-major alternating zero/one run lengths, starting with zeros."""
    values = mask.astype(np.uint8).ravel()
    changes = np.flatnonzero(np.diff(values)) + 1
    edges = np.concatenate(([0], changes, [values.size]))
    counts = np.diff(edges).tolist()
    if values.size and values[0]:
        counts.insert(0, 0)
    return counts


def export_case(folder, graph, observations):
    video, run = graph["video"], graph["run"]
    parquet(folder / "videos.parquet", [video])
    parquet(folder / "tracks.parquet", graph["tracks"])
    parquet(folder / "relations_frame.parquet", observations)
    parquet(folder / "relations_interval.parquet", graph["relations"])
    mask_ref = next((track["mask_ref"] for track in graph["tracks"] if track["mask_ref"]), None)
    mask_dir = Path(mask_ref).parent if mask_ref else None
    if mask_dir:
        tracks = {track["track_id"]: [] for track in graph["tracks"] if track["mask_ref"]}
        for image in sorted(mask_dir.glob("*.png")):
            labels = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
            if labels is None:
                raise RuntimeError(f"Cannot decode mask: {image}")
            for track_id in np.unique(labels):
                if track_id in tracks:
                    tracks[track_id].append({"frame": int(image.stem), "height": labels.shape[0],
                                              "width": labels.shape[1], "counts": rle(labels == track_id)})
        for track_id, rows in tracks.items():
            parquet(mask_dir / f"{track_id}.parquet", rows)
    export = folder / "graph_export"
    export.mkdir(exist_ok=True)
    links = ([{"source": video["video_id"], "target": graph["case_id"], "type": "VIDEO_IN_CASE"},
              {"source": run["run_id"], "target": video["video_id"], "type": "VIDEO_FOR_VIDEO"}]
             + [{"source": track["track_uid"], "target": video["video_id"], "type": "VIDEO_IN_VIDEO"}
                for track in graph["tracks"]]
             + [{"source": track["track_uid"], "target": run["run_id"], "type": "VIDEO_PRODUCED_BY"}
                for track in graph["tracks"]])
    for name, rows in (("cases", [{"case_id": graph["case_id"]}]), ("videos", [video]),
                       ("runs", [run]), ("tracks", graph["tracks"]),
                       ("links", links), ("relations", graph["relations"])):
        path = export / f"{name}.csv"
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value) if isinstance(value, list) else value
                                 for key, value in row.items()})
        temporary.replace(path)
    manifest_path = folder.parent / "RUN_MANIFEST.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    manifest[run["run_id"]] = {
        "source_sha256": video["sha256"], "config_sha256": run["config_sha256"],
        "model": run["model"], "weights_sha256": run["weights_sha256"],
        "pipeline_sha256": run["pipeline_sha256"], "git_sha": run.get("git_sha"),
    }
    save_json(manifest_path, manifest)
