"""Run the small public clip at three sampling rates and compare graph rules."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from videosg.audit import decoded_frames, repeatability
from videosg.evaluation import graph_predictions, score_query
from videosg.pipeline import ROOT, digest, read_json, save_json


def prepare_workspace(path, analysis_fps):
    shutil.copytree(ROOT / "videosg", path / "videosg", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "ontology.json", path / "ontology.json")
    settings = read_json(ROOT / "config.json")
    settings["analysis_fps"] = analysis_fps
    save_json(path / "config.json", settings)
    source = ROOT / "models" / "mask2former-swin-tiny-coco-panoptic"
    target = path / "models" / source.name
    shutil.copytree(source, target, copy_function=os.link)


def run_variant(path, source, force=False):
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    if not force:
        subprocess.run([python, "-m", "videosg", "ingest", "demo", source], cwd=path, check=True)
    subprocess.run([python, "-m", "videosg", "analyze", "demo", *(["--force"] if force else [])],
                   cwd=path, check=True)
    return path / "cases" / "demo" / "video" / "graph.json"


def padding_sweep(graph_path):
    import videosg.pipeline as pipeline

    graph = read_json(graph_path)
    relation = next(item for item in graph["relations"] if item["predicate"] == "near")
    match = {"start_frame": relation["start_frame"], "end_frame": relation["end_frame"],
             "rel_uid": relation["rel_uid"], "subject": relation["subject"], "object": relation["object"]}
    original = pipeline.config
    results = []
    try:
        for seconds in (0, 2, 4):
            pipeline.config = lambda seconds=seconds: {**original(), "snippet_padding_seconds": seconds}
            mp4 = pipeline.render("demo", graph, match, f"ablation_padding_{seconds}",
                                  {"template": "relation", "predicate": "near"})
            sidecar = read_json(mp4.with_name("snippet.json"))
            frames = decoded_frames(mp4)
            expected = sidecar["source_end_frame"] - sidecar["source_start_frame"] + 1
            if frames != expected:
                raise RuntimeError("Padding sweep snippet has an inaccurate cut")
            results.append({"padding_s": seconds, "frames": frames, "duration_s": frames / graph["video"]["fps"],
                            "start_frame": sidecar["source_start_frame"], "end_frame": sidecar["source_end_frame"],
                            "snippet": str(mp4)})
    finally:
        pipeline.config = original
    return results


def predicate_sweep(manifest):
    rows = []
    for entry in read_json(manifest):
        graph = read_json(ROOT / "cases" / entry["case_id"] / "video" / "graph.json")
        for query in entry["queries"]:
            full = graph_predictions(graph, query)
            rows.append({"video_id": graph["video"]["video_id"], "predicate": query["predicate"],
                         "truth": query["truth"], "full": full})
    report = []
    for predicate in sorted({row["predicate"] for row in rows}):
        control = [score_query(row["full"], row["truth"]).get("recall_1_0.5") for row in rows]
        dropped = [score_query([] if row["predicate"] == predicate else row["full"], row["truth"])
                   .get("recall_1_0.5") for row in rows]
        paired = [(a, b) for a, b in zip(control, dropped) if a is not None and b is not None]
        report.append({"dropped": predicate, "positive_queries": len(paired),
                       "full_recall_1_0.5": sum(a for a, _ in paired) / len(paired) if paired else None,
                       "dropped_recall_1_0.5": sum(b for _, b in paired) / len(paired) if paired else None})
    return report


def main():
    source = read_json(ROOT / "cases" / "demo" / "video" / "video.json")["path"]
    if digest(Path(source)) != read_json(ROOT / "cases" / "demo" / "video" / "video.json")["sha256"]:
        raise RuntimeError("Public demo source hash changed")
    output = ROOT / "eval" / "ablations"
    output.mkdir(parents=True, exist_ok=True)
    fps_rows = []
    with tempfile.TemporaryDirectory(prefix="videosg_ablation_", dir=output) as temporary:
        workspace = Path(temporary)
        for fps in (1, 2, 5):
            variant = workspace / f"fps_{fps}"
            prepare_workspace(variant, fps)
            graph_path = run_variant(variant, source)
            graph = read_json(graph_path)
            fps_rows.append({"analysis_fps": fps, "run_id": graph["run"]["run_id"],
                             "config_sha256": graph["run"]["config_sha256"],
                             "tracks": len(graph["tracks"]),
                             "intervals": len(graph["relations"]), "analysis_seconds": graph["run"]["seconds"],
                             "triplets": len({(r["subject"], r["predicate"], r["object"])
                                              for r in graph["relations"]})})
            if fps == 2:
                first = variant / "first_graph.json"
                shutil.copy2(graph_path, first)
                run_variant(variant, source, force=True)
                repeat = repeatability(first, graph_path)
        # Render in the main demo folder so the provenance sidecars remain inspectable.
    padding = padding_sweep(ROOT / "cases" / "demo" / "video" / "graph.json")
    manifest = ROOT / "data" / "meva" / "evaluation_manifest.json"
    predicate = predicate_sweep(manifest) if manifest.exists() else []
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    report = {"command": "python -m scripts.ablations", "git_sha": revision,
              "config_sha256": digest(ROOT / "config.json"),
              "source_sha256": digest(Path(source)), "fps": fps_rows, "repeatability": repeat,
              "padding": padding, "drop_predicate": predicate}
    save_json(output / "results.json", report)
    lines = ["# Ablations and repeatability", "", f"Command: `{report['command']}`", f"Git SHA: `{revision}`",
             f"Config SHA-256: `{report['config_sha256']}`", f"Source SHA-256: `{report['source_sha256']}`",
             "", "## Analysis sampling rate", "", "| FPS | Tracks | Intervals | Triplets | Analysis seconds |",
             "|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['analysis_fps']} | {r['tracks']} | {r['intervals']} | {r['triplets']} | {r['analysis_seconds']:.1f} |" for r in fps_rows]
    lines += ["", "## Repeat run at 2 FPS", "", json.dumps(repeat, sort_keys=True), "", "## Snippet padding", "",
              "| Padding seconds | Start frame | End frame | Decoded frames | Duration seconds |", "|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['padding_s']} | {r['start_frame']} | {r['end_frame']} | {r['frames']} | {r['duration_s']:.2f} |" for r in padding]
    lines += ["", "## Drop one rule predicate", "", "| Predicate dropped | Positive queries | Full R@1 .5 | Dropped R@1 .5 |",
              "|---|---:|---:|---:|"]
    lines += [f"| {r['dropped']} | {r['positive_queries']} | {r['full_recall_1_0.5'] if r['full_recall_1_0.5'] is not None else 'n/a'} | {r['dropped_recall_1_0.5'] if r['dropped_recall_1_0.5'] is not None else 'n/a'} |" for r in predicate]
    (output / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
