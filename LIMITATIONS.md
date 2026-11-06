# Limitations and interpretation

- This is a video-only, single-camera research prototype on staged or licensed public footage. It has no legal or forensic validity and does not infer intent, identity, speech, injury or crime.
- The COCO panoptic checkpoint often misses small or occluded props and can switch track IDs. The weights' redistribution terms remain unclear; weights are kept out of the repository.
- Relation rules are visual proxies. `touching` uses an 8 px mask dilation, `holding` uses contact and co-motion, `on` uses vertical mask support, and `falling` needs an observed downward shift followed by a wide person mask. Camera perspective and occlusion can create false positives. No depth ordering is claimed.
- Edge confidence combines model segment scores with temporal rule support. It is uncalibrated and must not be interpreted as a probability that an event occurred.
- Events are sampled at 2 fps while snippets use the original frames. Nearest sampled masks can drift between analyses. A precise MP4 cut does not imply precise event boundaries.
- A configured region is needed for `entering` and `leaving`. The default config has no region, so those predicates cannot be returned on the current public clip.
- The MEVA mapping deliberately compares coarse geometric proxies with richer annotated activities. A low graph score is a valid negative result; proximity cannot prove conversation and co-motion cannot prove pickup, transfer or object weight.
- On the fixed 50-clip MEVA selection, confidence-aware rule-graph Recall@1 at temporal IoU 0.5 was 0.005 versus 0.010 for class presence and 0.008 for CLIP. These results do not support a retrieval advantage for the rule graph on the mapped MEVA activities. Its no-match abstention accuracy was 0.375, so it can also return unsupported matches.
- PVSG component evaluation is conditional on resolving the source video and annotation rights. Its code licence does not grant rights to the underlying videos. Without clearance, use the documented MEVA-plus-consented-mock fallback and do not present PVSG scores as measured.
- The staged mock-scene demo needs the user's cleared source videos and two independent annotations. Until then, only its tooling can be verified.
