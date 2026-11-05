"""Measure decode and 20-frame panoptic inference on the licensed demo clip."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import subprocess
import time

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from videosg.pipeline import ROOT, MODEL_SHA256, config, digest, gpu_temperature, read_json, save_json


def source_video():
    video = read_json(ROOT / "cases" / "demo" / "video" / "video.json")
    if digest(Path(video["path"])) != video["sha256"]:
        raise RuntimeError("Demo source hash changed")
    return video


def decode(video):
    capture = cv2.VideoCapture(video["path"])
    if not capture.isOpened():
        raise RuntimeError("Cannot open demo video")
    started = time.perf_counter()
    frames = 0
    try:
        while frames < min(100, video["n_frames"]) and capture.read()[0]:
            frames += 1
    finally:
        capture.release()
    if frames != min(100, video["n_frames"]):
        raise RuntimeError("Decode benchmark stopped early")
    seconds = time.perf_counter() - started
    return {"frames": frames, "seconds": seconds, "fps": frames / seconds}


def infer(video):
    settings = config()
    directory = ROOT / "models" / "mask2former-swin-tiny-coco-panoptic"
    if digest(directory / "model.safetensors") != MODEL_SHA256:
        raise RuntimeError("Segmentation checkpoint differs from pinned SHA-256")
    processor = AutoImageProcessor.from_pretrained(directory)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(directory).cuda().eval()
    rows, images = [], []
    for mode, step in (("full_fps", 1), ("analysis_fps", round(video["fps"] / settings["analysis_fps"]))):
        torch.cuda.reset_peak_memory_stats()
        capture = cv2.VideoCapture(video["path"])
        timings, classes = [], Counter()
        try:
            for index in range(20):
                capture.set(cv2.CAP_PROP_POS_FRAMES, index * step)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not decode frame {index * step}")
                scale = min(1.0, settings["max_side"] / max(frame.shape[:2]))
                resized = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
                input_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                inputs = {key: value.cuda() for key, value in processor(images=input_image, return_tensors="pt").items()}
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    prediction = model(**inputs)
                    result = processor.post_process_panoptic_segmentation(
                        prediction, target_sizes=[input_image.shape[:2]], label_ids_to_fuse=set())[0]
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - start)
                labels = result["segmentation"].cpu().numpy()
                for segment in result["segments_info"]:
                    classes[str(model.config.id2label[segment["label_id"]]).lower()] += 1
                if mode == "analysis_fps" and len(images) < 5:
                    overlay = resized.copy()
                    for segment in result["segments_info"]:
                        color = tuple(int(x) for x in np.random.default_rng(segment["id"]).integers(40, 240, size=3))
                        overlay[labels == segment["id"]] = color
                    overlay = cv2.addWeighted(resized, 0.55, overlay, 0.45, 0)
                    images.append(overlay)
        finally:
            capture.release()
        rows.append({"mode": mode, "frames": 20, "mean_inference_seconds": statistics.mean(timings),
                     "median_inference_seconds": statistics.median(timings),
                     "peak_allocated_gpu_mib": torch.cuda.max_memory_allocated() / 2**20,
                     "class_histogram": dict(classes)})
    if gpu_temperature() is not None and gpu_temperature() >= settings["gpu_max_c"]:
        raise RuntimeError("GPU temperature exceeded the configured maximum")
    output = ROOT / "eval" / "model_spike"
    output.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        if not cv2.imwrite(str(output / f"overlay_{index + 1}.png"), image):
            raise RuntimeError("Could not save model overlay")
    height, width = images[0].shape[:2]
    sheet = np.zeros((height, width * len(images), 3), dtype=np.uint8)
    for index, image in enumerate(images):
        sheet[:, index * width:(index + 1) * width] = image
    if not cv2.imwrite(str(output / "contact_sheet.png"), sheet):
        raise RuntimeError("Could not save contact sheet")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-only", action="store_true")
    args = parser.parse_args()
    video = source_video()
    output = ROOT / "eval" / "model_spike"
    output.mkdir(parents=True, exist_ok=True)
    existing = read_json(output / "results.json") if (output / "results.json").exists() else {}
    existing["source_sha256"] = video["sha256"]
    existing["config_sha256"] = digest(ROOT / "config.json")
    existing["git_sha"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         capture_output=True, text=True, check=True).stdout.strip()
    existing["gpu"] = torch.cuda.get_device_name(0)
    existing["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    existing["decode_100"] = decode(video)
    if not args.decode_only:
        existing["inference_20"] = infer(video)
    save_json(output / "results.json", existing)
    (output / "table.md").write_text("# Model and decode spike\n\n"
        f"Command: `python -m scripts.model_spike`\nGit SHA: `{existing['git_sha']}`\n"
        f"Config SHA-256: `{existing['config_sha256']}`\nSource SHA-256: `{video['sha256']}`\n\n"
        f"OpenCV decoded {existing['decode_100']['frames']} frames at {existing['decode_100']['fps']:.1f} fps.\n\n"
        + "```json\n" + json.dumps(existing.get("inference_20", []), indent=2) + "\n```\n", encoding="utf-8")
    print(output / "table.md")


if __name__ == "__main__":
    main()
