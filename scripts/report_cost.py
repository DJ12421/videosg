"""Summarize saved MEVA stage costs, local graph size and evidence completeness."""

from pathlib import Path
import statistics
import subprocess
from datetime import datetime, timezone

import numpy as np

from videosg.pipeline import ROOT, case_dir, digest, read_json, save_json


def size(path):
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def summary(values):
    return {"median": statistics.median(values), "p95": float(np.quantile(values, 0.95))}


def main():
    manifest = read_json(ROOT / "data" / "meva" / "evaluation_manifest.json")
    progress = read_json(ROOT / "data" / "meva" / "batch_progress.json")
    evidence = read_json(ROOT / "eval" / "meva_evidence.json")
    results = read_json(ROOT / "eval" / "evaluation_manifest" / "results.json")
    cases = {entry["case_id"] for entry in manifest}
    if len(cases) != 50 or set(progress) != cases or {row["case_id"] for row in evidence} != cases:
        raise RuntimeError("Cost report requires the same 50 cases in manifest, progress and evidence")
    if any(progress[case_id]["status"] != "complete" for case_id in cases):
        raise RuntimeError("Some MEVA cases are not complete")
    if any(row["status"] != "ok" or row["sidecars_checked"] < 1 or not Path(row["snippet"]).exists()
           for row in evidence):
        raise RuntimeError("An evidence snippet was not previously audited or is missing")
    rows = []
    for case_id in sorted(cases):
        graph = read_json(case_dir(case_id) / "graph.json")
        video, run = graph["video"], graph["run"]
        if progress[case_id]["run_id"] != run["run_id"] or video["n_frames"] <= 0:
            raise RuntimeError(f"Run or frame count mismatch: {case_id}")
        rows.append({"case_id": case_id, "run_id": run["run_id"], "source_sha256": video["sha256"],
                     "source_bytes": Path(video["path"]).stat().st_size,
                     "source_frames": video["n_frames"], "sampled_frames": run["sampled_frames"],
                     "decode_seconds": video["decode_seconds"], "analysis_seconds": run["seconds"],
                     "tracks": len(graph["tracks"]), "intervals": len(graph["relations"])})
    latencies = [row["query_seconds"] for row in results["B3_rules"] if "query_seconds" in row]
    if len(latencies) != 5 * len(cases):
        raise RuntimeError("Expected five logged graph queries per evaluated video")
    store = ROOT / "neo4j-local" / "neo4j-community-5.26.30" / "data"
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    report = {"command": "python -m scripts.report_cost", "git_sha": revision,
              "config_sha256": digest(ROOT / "config.json"),
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "clips": len(rows), "queries": len(latencies),
              "evidence_provenance_completeness": len(evidence) / len(cases),
              "source_gib": sum(row["source_bytes"] for row in rows) / 2**30,
              "neo4j_store_gib": size(store / "databases" / "neo4j") / 2**30,
              "neo4j_transaction_gib": size(store / "transactions" / "neo4j") / 2**30,
              "decode_fps": summary([row["source_frames"] / row["decode_seconds"] for row in rows]),
              "analysis_sample_fps": summary([row["sampled_frames"] / row["analysis_seconds"] for row in rows]),
              "analysis_seconds": summary([row["analysis_seconds"] for row in rows]),
              "query_seconds": summary(latencies), "runs": rows}
    output = ROOT / "eval" / "cost_report"
    save_json(output / "results.json", report)
    lines = ["# MEVA stage cost and evidence", "", f"Command: `{report['command']}`",
             f"Git SHA: `{revision}`", f"Config SHA-256: `{report['config_sha256']}`",
             f"Recorded UTC: `{report['recorded_at_utc']}`", "",
             f"Cases: {report['clips']}; logged graph queries: {report['queries']}; "
             f"previously audited evidence snippets: {len(evidence)}/{len(cases)} (100%).", "",
             "| Measure | Median | p95 |", "|---|---:|---:|",
             f"| Original-video decode fps | {report['decode_fps']['median']:.1f} | {report['decode_fps']['p95']:.1f} |",
             f"| Analyzed samples per second | {report['analysis_sample_fps']['median']:.2f} | {report['analysis_sample_fps']['p95']:.2f} |",
             f"| Analysis seconds per video | {report['analysis_seconds']['median']:.1f} | {report['analysis_seconds']['p95']:.1f} |",
             f"| Neo4j query seconds | {report['query_seconds']['median']:.4f} | {report['query_seconds']['p95']:.4f} |",
             "", f"Original MEVA source files: {report['source_gib']:.2f} GiB. "
             f"Local Neo4j database store: {report['neo4j_store_gib']:.2f} GiB; "
             f"transaction logs: {report['neo4j_transaction_gib']:.2f} GiB.", "",
             "Decode timings include clips ingested before the grab-only decoder optimization; "
             "the distribution is a measured mixed-version cost, not a controlled decoder benchmark.",
             "Per-case run IDs, source hashes and raw timings are in `results.json`."]
    (output / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output / "table.md")


if __name__ == "__main__":
    main()
