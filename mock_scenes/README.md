# Staged scene protocol

Do not record until the human confirms institutional clearance and written consent from every participant. Use only staged, non-graphic actions. Store signed forms outside this repository. Blur identifiable faces in anything published unless publication consent is explicit.

Three proposed scripts are `scene_arrival.json`, `scene_handoff.json`, and `scene_rest.json`. Their times are capture targets, not ground truth; all clearance flags remain false until the human confirms them. They contain no footage or annotations.

Capture checklist:

- Confirm PES/institutional clearance and written consent from every participant before recording; keep forms outside this repository.
- Use a fixed camera at 1080p or higher, fixed frame rate, steady lighting, and a visible clap or clock at the start.
- Use only clearly staged, non-graphic actions and ordinary props. Check that every person and prop stays visible during the intended events.
- Keep each original video unchanged; ingest by absolute path and record its SHA-256. Configure a normalized door region separately for any crossing scene.
- Obtain separate publication consent before showing identifiable faces publicly; otherwise blur them in published figures.

Annotation checklist:

- Have two people watch the original video independently and timestamp observed events without seeing model output, script target times, or each other's file.
- Keep both files. Then compute consensus and agreement; document disagreements rather than replacing either original annotation.
- Identify predicted track IDs independently of relation scores before adding a `track_map`. Report temporal scene-graph quality only when this mapping is available.

For each scene, fill `scene.example.json`: case ID, actors, props, intended observable relations and approximate times. Record on a tripod at fixed frame rate; show a clap or clock at the start. Keep the original recording unchanged. Run `python -m videosg ingest <case_id> <original_path>` and retain its SHA-256.

Two people independently create files in the `annotation.example.json` format without seeing model output or each other's labels. Time is in seconds from the original video's first frame. Each event has `predicate`, `subject`, `object`, `start_s`, and `end_s`. Reconcile disagreements only after both independent files are saved. Run `python -m videosg consensus first.json second.json output.json` to create a consensus file and agreement summary. The resulting JSON can supply evaluation ground truth; do not tune thresholds on the held-out scene.

For identity-aware scene-graph Recall@20/50/100, independently identify visible actors and props in the model's `graph.json`, then add `track_map` to the scene JSON as `{ "predicted_track_uid": "actor_a" }`. Do this without inspecting relation scores. `scripts.prepare_mock` writes temporal R@K and mean R@K if the map is present. Volume IoU remains unavailable without ground-truth masks; do not report it as zero.

The model's COCO ontology covers `knife` and `door-stuff` but cannot recognize many other forensic props (for example firearms). Door crossing still needs an explicit region box in `config.json`. An absent detector class is a documented limitation, not evidence that an event did not occur.
