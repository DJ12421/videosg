# Status — 2026-11-06

## Confidence schema rerun

- Frame rule outputs now record model-derived scores; intervals record support-adjusted confidence and graph edges take the minimum of interval and endpoint-track confidence. The public clip was reanalyzed and loaded as run `d55f9dd664e6b2dd4989`; its frame/interval Parquet columns and graph edge confidence were checked.
- The earlier 50 MEVA graphs and reports are archived under ignored `data/meva/pre_confidence_graphs/` and `eval/pre_confidence/`. The fixed 50 clips will be reanalyzed under the new pipeline digest, then reloaded and reevaluated. The earlier scores below are historical until that rerun finishes.
- `eval_config/MEVA_DATASET.md` records the selected activity, camera, format and decoded-frame checks. All selected label spans fit within their true decoded video frame counts.

## Previously completed baseline

- MEVA download, analysis, Neo4j load, and the B0/B1/B2/B3 evaluation are complete for all 50 selected clips. `data/meva/batch_progress.json` has 50 `complete` entries, and `eval/meva_evidence.json` has 50 checked snippet entries. No download, GPU batch, or finalizer process is running.
- The full table is `eval/evaluation_manifest/table.md`. At tIoU 0.5, Recall@1 is 0.005 random, 0.010 class presence, 0.008 CLIP, and 0.006 rule graph. The graph does not beat the simple class-presence baseline; keep this negative result in the demo.
- `eval/ablations/table.md` records the 1/2/5 FPS sweep, predicate drops, exact-frame padding sweep, and a repeat run with triplet Jaccard 1.0 and mean interval difference 0.0 s on the public clip. `eval/model_spike/table.md` has the 100-frame decode and two 20-frame inference measurements.
- `eval/cost_report/table.md` summarizes the 50 saved runs: median 1,385.5 source frames/s during ingest, 3.87 analyzed samples/s, 155.3 analysis seconds per video, 0.0197 s median Neo4j query time, and a 0.08 GiB local graph store. All 50 evidence records were previously audited; the report verifies that the same cases occur in the manifest and progress files. Ingest times span two decoder implementations, so this is a measured mixed-version cost.
- The one-command fixed-template and natural-language video paths each returned a snippet with a complete Neo4j/source/frame audit. On the current public run, two sidecars passed the audit. A supported no-match returned `NO_MATCH` and a write request returned `CANNOT_ANSWER`. The earlier 20-question Ollama check passed 20/20. Local Neo4j and Ollama must be running for the live natural-language demo.

## Nov 6 recovery history

- The overnight job reached 14 analyzed clips. Ten loaded to Neo4j; four later loads failed because the locally launched Neo4j server stopped. Nine subsequent clip waits failed after the old downloader stopped. Their original sources and completed graph artifacts were retained.
- The corrected selector confirmed that three annotated candidates have no corresponding public S3 video, preserved the original selection, and chose 50 available clips without changing the pre-committed activity mapping. Download and one-GPU batch jobs have been restarted from saved outputs. The batch now marks graphs `analyzed` while Neo4j is offline; the finalizer will load them before evaluation once the local server is available.
- Neo4j cannot start inside this Codex process because Java's Windows loopback selector fails. The user restarted the approved portable server in a normal PowerShell window; Bolt is reachable on `127.0.0.1:7687`. The batch resumed model analysis at clip 15/50, and the finalizer will reconcile the earlier `analyzed` entries before evaluation.
- A 14-clip Neo4j-backed smoke table in ignored `eval/pilot_14_manifest/` checked manifest construction and B0/B1/B3 queries. At tIoU 0.5 its Recall@1 was 0.003 random, 0.005 presence and 0.006 graph. These very low preliminary scores are not the final result; B2 CLIP was deliberately skipped while the segmentation GPU batch runs.

## Working now

- The existing 20-second licensed Pexels clip completed ingest, panoptic analysis, a local `VIDEO_*` Neo4j load, fixed queries and exact-frame snippets. The most recent checked demo run had 136 tracks and 626 geometric relation intervals. Five current-run sidecars resolved to the original hash, graph elements and Neo4j IDs, with decoded MP4 frame counts matching the sidecars. A second load kept the same 136/626 counts; a no-match query returned no MP4.
- `videosg` now writes videos/tracks/frame relations/interval relations in Parquet, per-track RLE masks, graph CSV, a run manifest and provenance sidecars. There are eight fixed read-only templates and a local Ollama Qwen 2.5 7B router. The 20-question check passed 10/10 answerable, 5/5 unanswerable and 5/5 adversarial prompts. Ten focused unit tests passed.
- Local Neo4j 5.26.30 has four uniqueness constraints and three indexes restricted to the video namespace. No shared team database or schema was available for reconciliation. The user approved this local instance in an earlier turn.
- The MEVA evaluation annotation repository is pinned at commit `421841a75577b697c314e952e585aecbb1b99e17`; a fixed 50-clip selection covers the five pre-committed proxy mappings. The first 300-second clip decoded 9,002 true frames, then produced 541 tracks and 1,191 intervals from 601 sampled frames in 802.561 analysis seconds; GPU temperature stayed below 70 C. It loaded into local Neo4j. This is throughput, not an accuracy claim.
- A one-video MEVA pilot table exists under ignored `eval/`. At temporal IoU 0.5, B3 rule-graph Recall@1 was 0.000, B1 class-presence 0.008, and B2 CLIP 0.008. One-video bootstrap intervals are degenerate; these are diagnostic pilot values, not final dataset results. The rule proxies are weak on the richer MEVA activities. No thresholds were tuned on these evaluation labels.
- LAION CLIP ViT-B/32 weights were downloaded from the recorded mirror and hash-checked (`74813fbcdc750f235c9784c367ca1394d2a5c25eb0aac92761752ac239db7cff`). The Transformers-format model loaded and generated pilot frame-text scores.
- A 100-frame OpenCV decode benchmark on the public clip measured 119.1 fps. An actual natural-language `footage_exists` query returned an MP4; nine sidecars now pass the source/graph/Neo4j/frame audit. All eight templates passed Neo4j `EXPLAIN`.

## Remaining gates

- The full data run and follow-on measurements are complete. The finalizer exited nonzero only at its last optional Ollama-backed query; its earlier evaluation, evidence, ablation, and model-spike outputs are saved and verified individually.
- PVSG evaluation footage has unresolved underlying dataset rights. Its pinned evaluator was inspected, but no PVSG data was downloaded or scored. Follow the handoff's MEVA-plus-mock fallback if clearance remains unresolved.
- Cleared staged mock footage and two independent annotations have not been supplied. Capture/consent templates, consensus scoring and oracle-manifest tooling exist, but the staged demo and mock evaluation cannot be completed or claimed until those external inputs arrive. Signed forms belong outside the repository.
- Known limits: frequent track switches and geometric false positives; mask overlays are nearest-sample approximations; `holding` and `touching` are proxies; region crossing needs manually configured boxes. See `LIMITATIONS.md`.
