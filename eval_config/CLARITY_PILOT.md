# External staged-scene pilot (provisional)

Source: [Clarity staged crime clips](https://huggingface.co/datasets/waliviqas/clarity-crime-clips), revision `f3811b54144cdf4fbd94ed616f01ac31bc211e52`, CC BY 4.0. Credit: Clarity team, NVIDIA x Red Hat Small Models Hack 2026. The publisher states that on-screen participants agreed to filming and publication. Only the non-graphic, single-camera `clip8/cam1.mp4` and `clip7/cam2.mp4` were used. Source videos and publisher `labels.csv` stay in ignored `data/clarity/`; video SHA-256 values are in `DATA_LICENSES.md` and the agent reviews. The downloaded publisher label file SHA-256 is `87e8d1254339b27606fa5ccbdda59a361210619fae6990b1b141d8b575fe1423`.

The publisher's labels identify a phone-snatch and pickpocket at the **camera-clip level**; they contain no event timestamps and were AI-drafted before team correction. `clarity_clip8_agent_review.json` and `clarity_clip7_agent_review.json` hold four visually reviewed person-to-person intervals per clip. The reviewer was an AI agent, knew the publisher's clip descriptions, and did not inspect model output until after recording these times. These are **provisional labels, not two independent human annotations or ground truth**. Small item transfer/holding was excluded because the item is not clear enough in the reviewed frames. The second clip was selected after the first clip's negative result, so this is not a held-out sample. There are no ground-truth masks.

Run the fixed two-video pilot from the repository root after downloading the two pinned clips and loading both cases into local Neo4j:

```powershell
.\.venv\Scripts\python.exe -m videosg evaluate .\eval_config\clarity_staged_pilot.json
```

The saved run used config SHA-256 `47c4f3a0f929e479d855c800eec570d2ba70e369cac6b071b91146e04caf29ce` and code commit `d3fd19dcab8a38bb122b2a51b8e2675fe86c579a`. Its full B0/B1/B2/B3 table and raw predictions are in ignored `eval/clarity_staged_pilot/`. The two-video bootstrap interval is descriptive only, not a population uncertainty estimate.

| Measure on provisional labels | B0 random | B1 presence | B2 CLIP | B3 rules |
|---|---:|---:|---:|---:|
| Recall@1, temporal IoU ≥ 0.5, eight positive intervals | 0.125 | 0.000 | 0.000 | 0.000 |
| Abstention on two no-match questions | 0.000 | 1.000 | 0.000 | 1.000 |

The `clip8/cam1.mp4` graph has 19 tracks and 18 relation intervals; `clip7/cam2.mp4` has 28 tracks and seven intervals. Neither graph has a person-to-person `near` interval. Clip8's sole person-to-person `approaching` interval is 4.50–7.01 s, after the reviewed approach at 3.3–4.2 s. No thresholds were changed based on these clips. The result is a negative extraction finding, not evidence that the staged activities were absent.

The clip8 fixed `approach` query rendered an MP4 with two highlighted person masks and timestamps. Its sidecar passed source-hash, graph-ID, Neo4j-ID and decoded-frame checks. A person-to-person `near` query returned `NO_MATCH` with no footage. The one-command path without an NL question returned a valid person-presence snippet and audited both current-run sidecars. Local Ollama was offline during this pilot; its restart was rejected by the execution policy, so no new NL claim is made here. The earlier public-video NL demo remains separately verified.

This external dataset **does not replace** the handoff's own consented mock-scene set or its two independent human annotations. Do not report this pilot as the completed mock-scene evaluation or as evidence of forensic validity.
