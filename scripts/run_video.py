"""One command: video file to graph, query result, snippet and provenance audit."""

import argparse
import json
import os
from pathlib import Path

from videosg.audit import provenance
from videosg.pipeline import ROOT, analyze, case_dir, config, digest, ingest, load_graph, read_json
from videosg.queries import ask, route_nl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--case-id", help="Reuse an existing storage ID for the same source")
    parser.add_argument("--question", help="Optional natural-language question; defaults to footage presence")
    args = parser.parse_args()
    source = args.video.resolve(strict=True)
    case_id = args.case_id or "clip_" + digest(source)[:12]
    local_env = ROOT / ".env"
    environment = dict(line.split("=", 1) for line in local_env.read_text(encoding="utf-8").splitlines() if "=" in line) if local_env.exists() else {}
    uri = os.getenv("NEO4J_URI", environment.get("NEO4J_URI", "bolt://127.0.0.1:7687"))
    user = os.getenv("NEO4J_USER", environment.get("NEO4J_USER", "neo4j"))
    password = os.getenv("NEO4J_PASSWORD") or environment.get("NEO4J_PASSWORD")
    if not password:
        parser.error("Set NEO4J_PASSWORD locally before loading the graph")
    ingest(case_id, source)
    analyze(case_id)
    load_graph(case_id, uri, user, password)
    graph = read_json(case_dir(case_id) / "graph.json")
    if args.question:
        model_config = read_json(ROOT / "models" / "mask2former-swin-tiny-coco-panoptic" / "config.json")
        classes = {str(label).lower() for label in model_config["id2label"].values()}
        classes |= {f"region:{name}" for name in config()["regions"]}
        routed = route_nl(args.question, classes)
        if routed is None:
            print(json.dumps({"case_id": case_id, "run_id": graph["run"]["run_id"], "status": "CANNOT_ANSWER"}))
            return
        template, params = routed["template"], routed["params"]
    else:
        detected = {track["class"] for track in graph["tracks"] if not track["class"].startswith("region:")}
        if not detected:
            print(json.dumps({"case_id": case_id, "run_id": graph["run"]["run_id"], "status": "NO_MATCH"}))
            return
        template, params = "footage_exists", {"subject_class": "person" if "person" in detected else sorted(detected)[0]}
    paths = ask(case_id, template, params, uri, user, password, args.question)
    audit = provenance(case_id, uri, user, password) if paths else None
    print(json.dumps({"case_id": case_id, "run_id": graph["run"]["run_id"],
                      "status": "MATCH" if paths else "NO_MATCH", "snippets": [str(path) for path in paths],
                      "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
