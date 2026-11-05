# Nov 14 demo script

1. Show a staged, non-graphic source clip with confirmed consent and its SHA-256 from `video.json`. Do not show signed forms or unblurred faces in published figures without publication consent.
2. Run `python -m videosg analyze <case_id>` and `python -m videosg load <case_id>`. Show `RUN_MANIFEST.json`, the RLE mask and interval Parquet outputs, and the `VIDEO_*` graph counts. Repeat the load to show stable counts.
3. Ask an observed question through `python -m videosg ask <case_id> --nl '<question>'`. Show the returned interval, MP4 and JSON sidecar. Run `python -m videosg audit <case_id>` to resolve source, run, tracks, relations and exact frame count.
4. Ask a supported question with no observed match and show `NO_MATCH` without MP4. Ask an unsupported or write request and show `CANNOT_ANSWER`.
5. Show the MEVA per-activity B0/B1/B2/B3 table and the mock B5 oracle where ground-truth graph labels exist. State the video-level confidence intervals, negative results, runtime and the limits in `LIMITATIONS.md`.

This script is prepared for rehearsal. It is not a claim that cleared mock footage, annotation, or final tables already exist; see `STATUS.md` for the current gate.
