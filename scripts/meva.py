"""Select, fetch and prepare a fixed MEVA evaluation subset (CC BY 4.0)."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import requests
import yaml

from videosg.pipeline import ROOT, case_dir, digest, ingest, read_json, save_json
from videosg.downloads import download_ranges


BASE = ROOT / "data" / "meva"
ANNOTATIONS = ROOT / "data" / "meva-meta" / "annotation" / "DIVA-phase-2" / "MEVA" / "kitware"
MAPPING = ROOT / "eval_config" / "meva_mapping.csv"
BUCKET = "https://mevadata-public-01.s3.amazonaws.com/"


def mapping():
    with MAPPING.open(newline="", encoding="utf-8") as file:
        return {row["activity"]: row for row in csv.DictReader(file)}


def activities(path):
    result = {}
    for item in yaml.safe_load(path.read_text(encoding="utf-8")) or []:
        if "meta" in item:
            continue
        if "act" not in item:
            raise ValueError(f"Unrecognized MEVA annotation in {path}: {item}")
        activity = item["act"]
        if activity.get("src_status") not in (None, "good"):
            continue
        for name in activity["act2"]:
            result.setdefault(name, []).extend(span["tsr0"] for span in activity["timespan"])
    return result


def select(limit=50):
    labels = mapping()
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT / "data" / "meva-meta",
                              capture_output=True, text=True, check=True).stdout.strip()
    candidates = []
    for path in sorted(ANNOTATIONS.rglob("*.activities.yml")):
        found = activities(path)
        relevant = {name: found[name] for name in labels if name in found}
        if relevant:
            candidates.append({"stem": path.name.removesuffix(".activities.yml"),
                               "annotation_path": str(path), "annotation_repo_sha": revision,
                               "activities": relevant})
    if not candidates:
        raise RuntimeError("No MEVA evaluation annotations found")
    selected, covered = [], {name: 0 for name in labels}
    while candidates and len(selected) < limit:
        best = max(candidates, key=lambda row: (sum(1 / (1 + covered[name]) for name in row["activities"]),
                                                 -len(row["stem"]), row["stem"]))
        candidates.remove(best)
        selected.append(best)
        for name in best["activities"]:
            covered[name] += 1
    save_json(BASE / "selection.json", selected)
    print(json.dumps({"selected": len(selected), "activity_clip_counts": covered}, indent=2))


def s3_key(stem):
    day, hour = stem[:10], stem[11:13]
    prefix = f"drops-123-r13/{day}/{hour}/"
    response = requests.get(BUCKET, params={"list-type": "2", "prefix": prefix}, timeout=30)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys = [node.text for node in root.findall("s:Contents/s:Key", ns)]
    matches = [key for key in keys if Path(key).name.startswith(stem + ".") and key.endswith(".avi")]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one S3 video for {stem}, got {len(matches)}")
    return matches[0]


def download(limit=None):
    selected = read_json(BASE / "selection.json")
    for row in selected[:limit]:
        stem = row["stem"]
        target = BASE / "clips" / f"{stem}.avi"
        if not target.exists():
            key = s3_key(stem)
            download_ranges(BUCKET + key, target)
        case_id = "meva_" + hashlib.sha256(stem.encode()).hexdigest()[:12]
        if not (case_dir(case_id) / "video.json").exists():
            ingest(case_id, target)
        row["case_id"] = case_id
        row["source_sha256"] = digest(target)
        print(f"Ready: {stem} {row['source_sha256']}", flush=True)
        save_json(BASE / "selection.json", selected)


def manifest():
    labels = mapping()
    selected = read_json(BASE / "selection.json")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT / "data" / "meva-meta",
                              capture_output=True, text=True, check=True).stdout.strip()
    entries = []
    for row in selected:
        if "case_id" not in row or not (case_dir(row["case_id"]) / "graph.json").exists():
            continue
        graph = read_json(case_dir(row["case_id"]) / "graph.json")
        video = graph["video"]
        if video["sha256"] != row["source_sha256"]:
            raise RuntimeError(f"Source mismatch: {row['stem']}")
        queries = []
        for activity, rule in labels.items():
            spans = [[start / video["fps"], (end + 1) / video["fps"]]
                     for start, end in row["activities"].get(activity, [])]
            queries.append({"text": activity.replace("_", " "), "activity": activity,
                            "subject_class": rule["subject_class"],
                            "object_class": rule["object_class"] or None,
                            "predicate": rule["predicate"],
                            "typical_duration_s": float(rule["typical_duration_s"]),
                            "truth": spans})
        entries.append({"dataset": "MEVA", "case_id": row["case_id"],
                        "annotation_repo_sha": row.get("annotation_repo_sha", revision), "queries": queries})
    output = BASE / "evaluation_manifest.json"
    save_json(output, entries)
    print(f"Wrote {len(entries)} evaluated clips to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select", "download", "manifest"))
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    {"select": lambda: select(args.limit), "download": lambda: download(args.limit),
     "manifest": manifest}[args.command]()
