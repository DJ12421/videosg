# Status — 2026-11-05

## Working now

- The existing 20-second licensed Pexels clip completed ingest, panoptic analysis, a local `VIDEO_*` Neo4j load, fixed queries and exact-frame snippets. The most recent checked demo run had 136 tracks and 626 geometric relation intervals. Five current-run sidecars resolved to the original hash, graph elements and Neo4j IDs, with decoded MP4 frame counts matching the sidecars. A second load kept the same 136/626 counts; a no-match query returned no MP4.
- `videosg` now writes videos/tracks/frame relations/interval relations in Parquet, per-track RLE masks, graph CSV, a run manifest and provenance sidecars. There are eight fixed read-only templates and a local Ollama Qwen 2.5 7B router. The 20-question check passed 10/10 answerable, 5/5 unanswerable and 5/5 adversarial prompts. Eight focused unit tests passed.
- Local Neo4j 5.26.30 has four uniqueness constraints and three indexes restricted to the video namespace. No shared team database or schema was available for reconciliation. The user approved this local instance in an earlier turn.
- The MEVA evaluation annotation repository is pinned at commit `421841a75577b697c314e952e585aecbb1b99e17`; a fixed 50-clip selection covers the five pre-committed proxy mappings. The first 300-second clip decoded 9,002 true frames, then produced 541 tracks and 1,191 intervals from 601 sampled frames in 802.561 analysis seconds; GPU temperature stayed below 70 C. It loaded into local Neo4j. This is throughput, not an accuracy claim.
- A one-video MEVA pilot table exists under ignored `eval/`. At temporal IoU 0.5, B3 rule-graph Recall@1 was 0.000, B1 class-presence 0.008, and B2 CLIP 0.008. One-video bootstrap intervals are degenerate; these are diagnostic pilot values, not final dataset results. The rule proxies are weak on the richer MEVA activities. No thresholds were tuned on these evaluation labels.
- LAION CLIP ViT-B/32 weights were downloaded from the recorded mirror and hash-checked (`74813fbcdc750f235c9784c367ca1394d2a5c25eb0aac92761752ac239db7cff`). The Transformers-format model loaded and generated pilot frame-text scores.

## Running and remaining gates

- The 50-clip MEVA download/ingest process and one-GPU-at-a-time batch are running in ignored `data/meva/`; `batch.pid`, `batch_progress.json`, and batch logs identify progress. After completion, regenerate the manifest and run the full B0/B1/B2/B3 table with video-level confidence intervals and per-activity results. Re-run analysis for repeatability and document interval differences, provenance, latency and ablations.
- PVSG evaluation footage has unresolved underlying dataset rights. Its pinned evaluator was inspected, but no PVSG data was downloaded or scored. Follow the handoff's MEVA-plus-mock fallback if clearance remains unresolved.
- Cleared staged mock footage and two independent annotations have not been supplied. Capture/consent templates, consensus scoring and oracle-manifest tooling exist, but the staged demo and mock evaluation cannot be completed or claimed until those external inputs arrive. Signed forms belong outside the repository.
- Known limits: frequent track switches and geometric false positives; mask overlays are nearest-sample approximations; `holding` and `touching` are proxies; region crossing needs manually configured boxes. See `LIMITATIONS.md`.
