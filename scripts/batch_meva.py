"""Run one GPU analysis at a time over the fixed, downloaded MEVA subset."""

import json
import os
from pathlib import Path
import traceback
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "4")

from videosg.pipeline import ROOT, analyze, case_dir, digest, load_graph, pipeline_digest, read_json, save_json


selection = read_json(ROOT / "data" / "meva" / "selection.json")
environment = dict(line.split("=", 1) for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines() if "=" in line)
uri = os.getenv("NEO4J_URI", environment.get("NEO4J_URI", "bolt://127.0.0.1:7687"))
user = os.getenv("NEO4J_USER", environment.get("NEO4J_USER", "neo4j"))
password = os.getenv("NEO4J_PASSWORD") or environment.get("NEO4J_PASSWORD")
if not password:
    raise RuntimeError("Set NEO4J_PASSWORD before the batch")
pid_file = ROOT / "data" / "meva" / "batch.pid"
progress_file = ROOT / "data" / "meva" / "batch_progress.json"
with os.fdopen(os.open(pid_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY), "w", encoding="ascii") as file:
    file.write(str(os.getpid()))
progress = read_json(progress_file) if progress_file.exists() else {}
try:
    for index in range(len(selection)):
        row = selection[index]
        waited = 0
        while not row.get("case_id") and waited < 1800:
            print(f"[{index + 1}/{len(selection)}] waiting for download: {row['stem']}", flush=True)
            time.sleep(15)
            waited += 15
            row = read_json(ROOT / "data" / "meva" / "selection.json")[index]
        case_id = row.get("case_id")
        if not case_id:
            progress[row["stem"]] = {"status": "failed", "error": "download unavailable after 30 minutes"}
            save_json(progress_file, progress)
            continue
        try:
            path = case_dir(case_id) / "graph.json"
            current = read_json(path) if path.exists() else None
            if not current or current["run"]["config_sha256"] != digest(ROOT / "config.json") or current["run"]["pipeline_sha256"] != pipeline_digest():
                print(f"[{index + 1}/{len(selection)}] analyzing {case_id}", flush=True)
                analyze(case_id)
            load_graph(case_id, uri, user, password)
            progress[case_id] = {"status": "complete", "run_id": read_json(path)["run"]["run_id"]}
        except Exception as exc:
            progress[case_id] = {"status": "failed", "error": str(exc)}
            traceback.print_exc()
        save_json(progress_file, progress)
        if progress[case_id]["status"] == "failed" and "GPU temperature" in progress[case_id]["error"]:
            print("Pausing batch because the GPU temperature limit was reached", flush=True)
            break
finally:
    pid_file.unlink(missing_ok=True)
print(json.dumps({"complete": sum(v["status"] == "complete" for v in progress.values()),
                  "failed": sum(v["status"] == "failed" for v in progress.values())}), flush=True)
