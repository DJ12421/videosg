"""Finish the licensed MEVA evaluation after the single-GPU batch exits."""

import json
import hashlib
import os
import subprocess
import time

from scripts.meva import manifest
from neo4j.exceptions import ServiceUnavailable
from videosg.evaluation import evaluate
from videosg.pipeline import ROOT, load_graph, read_json, save_json


base = ROOT / "data" / "meva"
selection = read_json(base / "selection.json")
while (base / "batch.pid").exists():
    progress = read_json(base / "batch_progress.json") if (base / "batch_progress.json").exists() else {}
    print(f"MEVA batch: {sum(row['status'] == 'complete' for row in progress.values())}/{len(selection)} complete", flush=True)
    time.sleep(60)
progress = read_json(base / "batch_progress.json")
environment = dict(line.split("=", 1) for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines() if "=" in line)
password = os.getenv("NEO4J_PASSWORD") or environment.get("NEO4J_PASSWORD")
if not password:
    raise RuntimeError("Set NEO4J_PASSWORD for the Neo4j-backed evaluation")
uri = os.getenv("NEO4J_URI", environment.get("NEO4J_URI", "bolt://127.0.0.1:7687"))
user = os.getenv("NEO4J_USER", environment.get("NEO4J_USER", "neo4j"))
for row in selection:
    case_id = "meva_" + hashlib.sha256(row["stem"].encode()).hexdigest()[:12]
    if progress.get(case_id, {}).get("status") not in {"complete", "analyzed"}:
        raise RuntimeError(f"MEVA analysis incomplete for {case_id}; inspect batch_progress.json")
    while progress[case_id]["status"] == "analyzed":
        try:
            load_graph(case_id, uri, user, password)
        except ServiceUnavailable:
            print(f"Waiting for local Neo4j before loading {case_id}", flush=True)
            time.sleep(60)
        else:
            progress[case_id]["status"] = "complete"
            save_json(base / "batch_progress.json", progress)

manifest()
output = evaluate(base / "evaluation_manifest.json", uri=uri, user=user, password=password)
print(f"Evaluation: {output}", flush=True)

subprocess.run([ROOT / ".venv" / "Scripts" / "python.exe", "-m", "scripts.evidence_samples"], cwd=ROOT, check=True)
subprocess.run([ROOT / ".venv" / "Scripts" / "python.exe", "-m", "scripts.report_cost"], cwd=ROOT, check=True)

from scripts.ablations import main as ablations

ablations()
subprocess.run([ROOT / ".venv" / "Scripts" / "python.exe", "-m", "scripts.model_spike"], cwd=ROOT, check=True)
subprocess.run([ROOT / ".venv" / "Scripts" / "python.exe", "-m", "scripts.figures"], cwd=ROOT, check=True)
demo = read_json(ROOT / "cases" / "demo" / "video" / "video.json")
subprocess.run([ROOT / ".venv" / "Scripts" / "python.exe", "-m", "scripts.run_video",
                demo["path"], "--case-id", "demo", "--question", "Is there footage showing a person?"],
               cwd=ROOT, check=True)
print(json.dumps({"status": "complete", "clips": len(complete)}), flush=True)
