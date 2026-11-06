# MEVA evaluation subset audit

The selected 50 clips come from the public MEVA video bucket under its CC BY 4.0 terms. Evaluation-level annotations are from the pinned `MEVA-Data-Repo` commit `421841a75577b697c314e952e585aecbb1b99e17`, `annotation/DIVA-phase-2/MEVA/kitware`. Its README describes KPF evaluation annotations for 769 clips and all 37 ActEV activities. This project maps only the five activities committed in `meva_mapping.csv`; no test labels were used to tune rule thresholds.

The annotation README defines each clip name as `date.start-time.end-time.site.camera`: `site` is the camera's location, not necessarily what it looks at. This subset has 15 distinct site/camera suffixes across bus, school, and hospital sites. It does not establish camera synchronization or cross-view identity.

| Mapped activity | Selected clips carrying it |
|---|---:|
| hand_interacts_with_person | 31 |
| person_picks_up_object | 48 |
| person_carries_heavy_object | 44 |
| person_transfers_object | 41 |
| person_talks_to_person | 50 |

All 50 downloaded files are AVI with observed H.264 codec and 30 fps. Their decoded frame counts range from 9,000 to 9,031 (median 9,003.5); 35 files decode at 1920×1080 and 15 at 1920×1072. The per-video true counts, source SHA-256 values, and IDs are in `data/meva/selection.json` and `cases/<case_id>/video/video.json`. Every selected annotation span was checked against its video's decoded frame count; none exceeded it. `scripts.meva manifest` enforces this check on regeneration.

Three annotated candidate names had no matching public video under the expected S3 prefix. The first selection and the excluded names are preserved in ignored `data/meva/selection_initial.json` and `data/meva/unavailable_clips.json`. The deterministic selector replaced those clips before the final 50-clip evaluation. A query's empty ground-truth list is kept as a no-match question; no clip was excluded because the rule graph failed on it.
