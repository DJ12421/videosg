# Data and model licences

Checked 2026-11-05. Keep source media and model weights outside version control.

| Item | Source | Licence status | Intended use | Redistribute? |
|---|---|---|---|---|
| `pedestrians_summer_street.mp4`, creator K (8 K Lover of aerials and photography) | [Source and attribution](https://github.com/aparsoft/yolo-streamlit-detection-tracking/blob/master/videos/ATTRIBUTION.md), [original Pexels clip](https://www.pexels.com/video/pedestrians-cross-busy-city-street-on-summer-day-38045690/) | [Pexels licence](https://www.pexels.com/legal-pages/license/); free use and modification; do not imply endorsement or portray identifiable people negatively | Neutral public-video demo of people and spatial relations | Yes, under Pexels terms; credit the creator as a courtesy |
| `facebook/mask2former-swin-tiny-coco-panoptic` weights | [Model card](https://huggingface.co/facebook/mask2former-swin-tiny-coco-panoptic), [ModelScope mirror](https://modelscope.cn/models/facebook/mask2former-swin-tiny-coco-panoptic) | Hub says `other`; the research code has an MIT licence, but that does not establish the weights licence. Verify mirrored weight SHA-256 against the [upstream file](https://huggingface.co/facebook/mask2former-swin-tiny-coco-panoptic/blob/main/model.safetensors): `8e154c4f946f4fad9aa511a7d706c89c4254cc6eee28fd932e52c32394c5a29f` | Internal capstone demo only | Unknown: do not redistribute weights |
| MEVA video and annotations | [Official dataset](https://mevadata.org/) and [licence](https://mevadata.org/resources/MEVA-data-license.txt) | CC BY 4.0 stated by the dataset owner; record clip IDs and attribution with every evaluation | Staged activity retrieval evaluation | Only with required attribution and licence terms |
| PVSG test video and annotations | [Official project](https://github.com/LilyDaytoy/OpenPVSG) | Underlying VidOR, Ego4D and EPIC-Kitchens video rights are not established by the code repository licence; access review pending | Component evaluation only if access and rights are cleared | No |
| OpenPVSG evaluator source code | [Official repository](https://github.com/LilyDaytoy/OpenPVSG) | Repository LICENSE is S-Lab License 1.0 with non-commercial terms, separate from dataset rights; inspected pinned commit `b2850b980c272368215a754299986cb904d55820` | Verify metric definition internally | Do not redistribute copied code |
| LAION CLIP ViT-B/32 weights | [Original model card](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K), [ModelScope mirror](https://modelscope.cn/models/AI-ModelScope/CLIP-ViT-B-32-laion2B-s34B-b79K) | MIT listed on the original model repository; downloaded mirrored `model.safetensors` SHA-256 `74813fbcdc750f235c9784c367ca1394d2a5c25eb0aac92761752ac239db7cff` | Internal B2 retrieval baseline on staged/licensed footage | No weights in repository |

If another clip or model is selected, record its source and terms here before downloading it.

