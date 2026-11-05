"""Finish the licensed MEVA evaluation after the single-GPU batch exits."""

import json
import os
import subprocess
import time

from scripts.meva import manifest
from videosg.evaluation import evaluate
from videosg.pipeline import ROOT, read_json


base = ROOT / "data" / "meva"
selection = read_json(base / "selection.json")
while (base / "batch.pid").exists():
    progress = read_json(base / "batch_progress.json") if (base / "batch_progress.json").exists() else {}
    print(f"MEVA batch: {sum(row['status'] == 'complete' for row in progress.values())}/{len(selection)} complete", flush=True)
    time.sleep(60)
progress = read_json(base / "batch_progress.json")
complete = [row for row in progress.values() if row["status"] == "complete"]
if len(complete) != len(selection):
    raise RuntimeError(f"MEVA batch incomplete: {len(complete)}/{len(selection)}; inspect batch_progress.json")

manifest()
environment = dict(line.split("=", 1) for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines() if "=" in line)
password = os.getenv("NEO4J_PASSWORD") or environment.get("NEO4J_PASSWORD")
if not password:
    raise RuntimeError("Set NEO4J_PASSWORD for the Neo4j-backed evaluation")
output = evaluate(base / "evaluation_manifest.json", uri=os.getenv("NEO4J_URI", environment.get("NEO4J_URI", "bolt://127.0.0.1:7687")),
                  user=os.getenv("NEO4J_USER", environment.get("NEO4J_USER", "neo4j")), password=password)
print(f"Evaluation: {output}", flush=True)

subprocess.run([ROOT / ".venv" / "Scripts" / "python.exe", "-m", "scripts.evidence_samples"], cwd=ROOT, check=True)

from scripts.ablations import main as ablations

ablations()
subprocess.run([ROOT / ".venv" / "Scripts" / "python.exe", "-m", "scripts.model_spike"], cwd=ROOT, check=True)
demo = read_json(ROOT / "cases" / "demo" / "video" / "video.json")
subprocess.run([ROOT / ".venv" / "Scripts" / "python.exe", "-m", "scripts.run_video",
                demo["path"], "--case-id", "demo", "--question", "Is there footage showing a person?"],
               cwd=ROOT, check=True)
print(json.dumps({"status": "complete", "clips": len(complete)}), flush=True)
