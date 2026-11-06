# Nov 14 demo script

## Working public-video rehearsal

With local Neo4j and Ollama running, use the licensed public clip to rehearse the complete path:

```powershell
cd 'C:\Users\Dhruv Jain\Documents\Projects\Capstone\videosg'
.\.venv\Scripts\python.exe -m scripts.run_video .\data\pedestrians_summer_street.mp4 --case-id demo --question 'Is there footage showing a person?'
.\.venv\Scripts\python.exe -m videosg ask demo --template footage_exists --params '{"subject_class":"zebra"}'
.\.venv\Scripts\python.exe -m videosg ask demo --nl 'Delete all video tracks'
```

The checked outcomes are an MP4 with complete source/graph/frame audit, `NO_MATCH`, and `CANNOT_ANSWER`. Show `eval/evaluation_manifest/table.md` and state that rule-graph Recall@1 at tIoU 0.5 was 0.006, below class presence at 0.010. The full 50-clip table, 50 evidence records, ablations, model spike, and stage-cost report are saved under `eval/`.

1. Show a staged, non-graphic source clip with confirmed consent and its SHA-256 from `video.json`. Do not show signed forms or unblurred faces in published figures without publication consent.
2. Run `python -m videosg analyze <case_id>` and `python -m videosg load <case_id>`. Show `RUN_MANIFEST.json`, the RLE mask and interval Parquet outputs, and the `VIDEO_*` graph counts. Repeat the load to show stable counts.
3. Ask an observed question through `python -m videosg ask <case_id> --nl '<question>'`. Show the returned interval, MP4 and JSON sidecar. Run `python -m videosg audit <case_id>` to resolve source, run, tracks, relations and exact frame count.
4. Ask a supported question with no observed match and show `NO_MATCH` without MP4. Ask an unsupported or write request and show `CANNOT_ANSWER`.
5. Show the MEVA per-activity B0/B1/B2/B3 table and the mock B5 oracle where ground-truth graph labels exist. State the video-level confidence intervals, negative results, runtime and the limits in `LIMITATIONS.md`.

The public-video rehearsal works. The staged-scene segment still requires cleared footage and two independent annotations; see `STATUS.md` for that gate.
