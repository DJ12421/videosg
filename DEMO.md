# Nov 14 demo script

## Working public-video rehearsal

With local Neo4j and Ollama running, use the licensed public clip to rehearse the complete path:

```powershell
cd 'C:\Users\Dhruv Jain\Documents\Projects\Capstone\videosg'
.\.venv\Scripts\python.exe -m scripts.run_video .\data\pedestrians_summer_street.mp4 --case-id demo --question 'Is there footage showing a person?'
.\.venv\Scripts\python.exe -m videosg ask demo --template footage_exists --params '{"subject_class":"zebra"}'
.\.venv\Scripts\python.exe -m videosg ask demo --nl 'Delete all video tracks'
```

The checked outcomes are an MP4 with complete source/graph/frame audit, `NO_MATCH`, and `CANNOT_ANSWER`. Show `eval/evaluation_manifest/table.md` and state that confidence-aware rule-graph Recall@1 at tIoU 0.5 was 0.005, below class presence at 0.010 and CLIP at 0.008. Its no-match abstention accuracy was 0.375. The refreshed 50-clip table, 50 audited evidence records, ablations, model spike, and stage-cost report are saved under `eval/`.

## MEVA fallback rehearsal

If consented mock footage is still unavailable, demonstrate the same one-command path on the licensed MEVA clip below. This is a public-dataset fallback, not a completed mock-scene evaluation. Credit MEVA under CC BY 4.0, and describe the observed predicate as person-to-person proximity; it does not establish conversation.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_video '.\data\meva\clips\2018-03-07.11-00-01.11-05-01.school.G421.avi' --case-id meva_d9c9a28fe6ff --question 'When is a person near another person?'
```

The checked run reused graph `991deb480896317c73c1`, returned five snippets, and audited six current-run sidecars against the source hash, graph, Neo4j and decoded frame counts. The first short example is `cases/meva_d9c9a28fe6ff/video/snippets/605b5e57a854c0d6_0/snippet.mp4` with its adjacent `snippet.json`: source frames 0–119 at 30 fps, including two seconds of configured padding after the detected interval. A sampled frame was visually checked for the timestamp, case/video labels and two highlighted track masks. The result is an observed rule match, not a claim that the annotated activity occurred in that exact snippet.

## External staged footage and honest failure example

The pinned Clarity `clip8/cam1.mp4` source is a non-graphic staged phone-snatch camera view. Its original and SHA-256 are documented in `DATA_LICENSES.md`; reproduce the download and two-clip pilot using `README.md` and `eval_config/CLARITY_PILOT.md`. With local Neo4j running, this checked command returns a person-presence snippet and provenance audit without requiring Ollama:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_video .\data\clarity\clip8_cam1.mp4 --case-id clarity_clip8_cam1
.\.venv\Scripts\python.exe -m videosg ask clarity_clip8_cam1 --template relation --params '{"subject_class":"person","object_class":"person","predicate":"near"}'
```

The second query returns `NO_MATCH` and no MP4 even though the agent's visual review marked a near interval. Show it as a false negative, not as proof that no interaction happened. The two-clip provisional rule-graph pilot scored 0.000 Recall@1 at temporal IoU 0.5 on eight agent-reviewed intervals; it is not the required human-annotated mock evaluation.

1. Show a staged, non-graphic source clip with confirmed consent and its SHA-256 from `video.json`. Do not show signed forms or unblurred faces in published figures without publication consent.
2. Run `python -m videosg analyze <case_id>` and `python -m videosg load <case_id>`. Show `RUN_MANIFEST.json`, the RLE mask and interval Parquet outputs, and the `VIDEO_*` graph counts. Repeat the load to show stable counts.
3. Ask an observed question through `python -m videosg ask <case_id> --nl '<question>'`. Show the returned interval, MP4 and JSON sidecar. Run `python -m videosg audit <case_id>` to resolve source, run, tracks, relations and exact frame count.
4. Ask a supported question with no observed match and show `NO_MATCH` without MP4. Ask an unsupported or write request and show `CANNOT_ANSWER`.
5. Show the MEVA per-activity B0/B1/B2/B3 table and the mock B5 oracle where ground-truth graph labels exist. State the video-level confidence intervals, negative results, runtime and the limits in `LIMITATIONS.md`.

The public-video rehearsal works. The staged-scene segment still requires cleared footage and two independent annotations; see `STATUS.md` for that gate.
