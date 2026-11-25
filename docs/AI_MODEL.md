# Deployed AI model

## OFFIRST v2.1 application model

The Android release configuration selects the v2.1 experimental, DR-only classifier by default.

| Property | Value |
| --- | --- |
| Student encoder | MobileNetV3-Large, KD Run 2 epoch 40 |
| Teacher | Frozen ConvNeXt-Tiny BYOL epoch 20, 768-dimensional features |
| Student initialization | Direct BYOL MobileNetV3-Large epoch 40 |
| Projection | Existing 960 → 768 projection |
| Classifier | Supervised 16-unit MLP for binary DR presence |
| Calibration | Platt calibration included in the FP32 ONNX graph; applied once |
| Input | `fundus_image`, float32 `[batch, 3, 512, 512]`, RGB, NCHW |
| Output | `dr_probability`, float32 `[batch, 1]`, calibrated probability |
| Cutoff | `0.3675977844516245`, inclusive (`>=`) |
| ONNX SHA-256 | `43a87f05df7adc6a231a679721e7f0b11b45a3fe6d1582aa4527ed738fa63b3f` |
| Encoder checkpoint SHA-256 | `8bc239d22ffc4e786820f2151c99380a754bef930e42edea65022e1246f2096d` |

The matching ONNX file is `deployment/offline_dr_screening/assets/offirst_v2_1_experimental.onnx`. Evidence manifests are beside it in `assets/model_v2_1_evidence/`. The application reads the ONNX probability directly and does not recalibrate it in Dart or Kotlin.

## Preprocessing and runtime

The v2.1 Android path uses OpenCV 4.12.0 locally: color decode, BGR-to-RGB conversion, long-edge 512 proxy, foreground mask where the RGB maximum is greater than 15, 5×5 morphological opening, crop when at least 100 mask pixels remain, centered square black padding, sequential 640 then 512 `INTER_AREA` resize, ImageNet normalization, and float32 NCHW packing. OpenCV does not contact a network service.

The v1.1 model remains checked in at `deployment/offline_dr_screening/assets/offirst_dr_model.onnx`; its SHA-256 is `b8fc10191f69faef16f1dc92d08bf9542ee7e26f885767942e28292c5201dd90` and its historical cutoff is `0.48`. Image records keep the model ID and threshold used for that image; legacy images without those fields use the v1.1 fallback values. Reports use per-image metadata.

The saved Python/OpenCV plus ONNX assessment replay covered 141 images, with maximum probability error `1.8341754449702208e-06` and zero decision disagreements. This does not verify the Android OpenCV tensors or Android runtime. No physical-device preprocessing comparison was available for this release.

## Training-method distinction

The deployed v2.1 model follows BYOL and knowledge-distillation lineage. Its MobileNetV3-Large student was distilled in KD Run 2 from a frozen BYOL ConvNeXt-Tiny teacher. KD Run 2 is not explicit Masked InfoNCE contrastive learning. BYOL itself learns from related views without explicit negative samples. Other representation-learning experiments are kept separate in the research directories; they are not components of this ONNX classifier.

## Development evaluation and limitations

The development assessment results were:

| Dataset | Images | Accuracy | Sensitivity | Specificity | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| IDRiD | 72 | 79.17% | 87.23% | 64.00% | 0.874 |
| RFMiD | 69 | 81.16% | 54.55% | 86.21% | 0.875 |

Patient independence is unverified, prior dataset exposure exists, and the datasets use different label protocols. The combined working gates of 85% accuracy, 85% sensitivity, and 80% specificity are not met. RFMiD sensitivity shows a substantial false-negative limitation. These are development results, not independent clinical validation. Read [research results and caveats](RESEARCH_RESULTS.md) and the [final release handoff](../deployment/offline_dr_screening/FINAL_RELEASE_HANDOFF.md).

OFFIRST is an experimental academic prototype, not a certified medical device. Its result must not be used alone for diagnosis, referral, treatment, or other clinical decisions.
