# Video scene graph pipeline

Research prototype for `VIDEO_PIPELINE_HANDOFF.md`: original video → panoptic masks and tracks → geometric relation intervals → isolated `VIDEO_*` graph in Neo4j → fixed or natural-language query → timestamped MP4 and provenance sidecar → evaluation. It makes no legal or forensic validity claim. Never use real case footage.

## Local setup

Use Python 3.11. The existing `.venv` uses system CUDA PyTorch; a clean environment can install `requirements.lock`. The segmentation weights have an unresolved redistribution licence, so they remain in ignored `models/`. Record any new media or weights in `DATA_LICENSES.md` before downloading.

```powershell
cd 'C:\Users\Dhruv Jain\Documents\Projects\Capstone\videosg'
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m videosg download-model
.\.venv\Scripts\python.exe -m videosg download-clip
```

The approved local Neo4j 5.26.30 instance is in ignored `neo4j-local/`. Start `neo4j.bat console` from that distribution in a normal PowerShell window. Credentials live in ignored `.env`; set `NEO4J_PASSWORD` locally for a new machine. The loader creates uniqueness constraints and indexes **only** for `VIDEO_*` labels and relationships. Review the team schema and obtain approval before pointing this at a shared database.

## Run a video

Use a fresh `case_id` for each video. Original files stay in place and are never edited. `analyze` combines segmentation/tracking, geometric rules, interval aggregation, and graph export; each output stage is retained in the per-case folder.

For several videos in one logical case, give each a distinct storage ID with `ingest camera_a file_a.mp4 --group-case incident_1` and `ingest camera_b file_b.mp4 --group-case incident_1`. Analyze and load each storage ID, then use `ask-case incident_1` with the same `--template` or `--nl` options. Videos remain independent; no cross-camera identity matching is claimed.

```powershell
.\.venv\Scripts\python.exe -m videosg ingest demo .\data\pedestrians_summer_street.mp4
.\.venv\Scripts\python.exe -m videosg analyze demo
.\.venv\Scripts\python.exe -m videosg load demo
.\.venv\Scripts\python.exe -m videosg ask demo --template relation --params '{"subject_class":"person","object_class":"person","predicate":"near"}'
.\.venv\Scripts\python.exe -m videosg ask demo --nl 'When is a person near another person?'
.\.venv\Scripts\python.exe -m videosg audit demo
```

Eight fixed templates are `presence`, `relation`, `holding`, `crossing`, `approach`, `timed_relation`, `sequence`, and `footage_exists`. `ask --nl` uses local Ollama model `qwen2.5-7b-instruct:latest` to choose a template and parameters. Application-owned, read-only Cypher is checked with `EXPLAIN`; the model cannot submit Cypher. An unsupported question yields `CANNOT_ANSWER`; an empty result yields `NO_MATCH` with no MP4.

The case folder contains `videos.parquet`, `tracks.parquet`, per-track RLE masks, frame and interval relation Parquet, graph CSV, `RUN_MANIFEST.json`, snippets and sidecars. `audit` checks source hash, local graph IDs, Neo4j IDs and exact decoded snippet frame counts. The current COCO checkpoint retains raw class labels plus a smaller `forensic_class` view in `ontology.json`. A configured region box is required for `entering` and `leaving`; absent classes and unconfigured regions are unsupported.

## Evaluation

The five MEVA activity-to-rule mappings are fixed in `eval_config/meva_mapping.csv`. The official MEVA evaluation annotations and selected source clips remain in ignored `data/`. The selector chooses 50 clips with a fixed algorithm and preserves both positive and no-match activity questions.

```powershell
.\.venv\Scripts\python.exe -m scripts.meva select --limit 50
.\.venv\Scripts\python.exe -m scripts.meva download --limit 50
.\.venv\Scripts\python.exe -m scripts.batch_meva
.\.venv\Scripts\python.exe -m scripts.meva manifest
.\.venv\Scripts\python.exe -m videosg evaluate .\data\meva\evaluation_manifest.json
.\.venv\Scripts\python.exe -m scripts.check_nl
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m scripts.model_spike
.\.venv\Scripts\python.exe -m scripts.ablations
.\.venv\Scripts\python.exe -m scripts.figures
```

`scripts.batch_meva` runs one GPU analysis at a time, resumes current outputs, records a Python PID and progress in ignored `data/meva/`, and loads each graph into local Neo4j. The evaluator writes B0 seeded random, B1 class presence, B2 CLIP frame/text, and B3 rule-graph tables with video-level bootstrap intervals, per-activity results, abstention and timing. B5 oracle is included when a mock-scene oracle graph is supplied. Mask IoU is reported only when ground-truth masks are supplied. Do not use evaluation annotations to tune thresholds.

`scripts.finalize_meva` can wait for the batch, then run the Neo4j-backed evaluation, ablations and model spike in order. Its logs are in ignored `data/meva/`. The ablation script runs 1/2/5 FPS in isolated temporary copies of the public demo so it does not change the active config or MEVA batch.

PVSG video rights are not yet established. The pinned official evaluator is inspected separately from its dataset; no PVSG video is redistributed or evaluated until those rights and access are cleared. The handoff's documented fallback is MEVA plus consented mock scenes. The stage's rule predicates are geometric proxies: `touching` is mask proximity, `holding` is contact and co-motion, and `near` does not establish conversation.

## Mock scenes and handoff

`mock_scenes/README.md` has the capture and annotation protocol. Do not record without confirmed clearance and consent. Two people annotate independently, then run `python -m videosg consensus first.json second.json consensus.json` and `python -m scripts.prepare_mock scene.json consensus.json` to build an oracle and evaluation manifest. Store signed forms outside this repository; blur identifiable faces in published figures unless publication consent is explicit.

See `STATUS.md` for measured results and incomplete gates. A demo claim requires a cleared staged clip, two annotations, a resolved sidecar, an honest baseline table and a limitations summary.
