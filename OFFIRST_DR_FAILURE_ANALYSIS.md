# OFFIRST v2.1 DR false-negative analysis

## Executive conclusion

The reported v2.1 RFMiD sensitivity is 54.55% (6/11 positives detected; 5 false negatives) at the fixed development threshold. This is a real aggregate model-performance warning. The available checkout does **not** contain the original RFMiD assessment photographs or a joined per-image v1.1/v2.1 probability/label table, so individual false-negative examples and their probabilities cannot be recovered here. The precise cause of the five errors (label semantics, image/domain shift, model representation/head, calibration, or image-specific factors) is therefore not established.

At the aggregate level, the paired dataset metrics favor different models on the two datasets: v1.1 has higher sensitivity on RFMiD (100.00% vs 54.55%), while v2.1 has higher sensitivity on IDRiD (87.23% vs 57.45%). This reverses any simple claim that the v2.1 Android implementation alone causes the RFMiD misses. It also does not prove v1.1 is generally superior: the assessments are small, development-exposed, not verified patient-independent, and use different labels/protocols across datasets.

## Available matched assessment results

From `docs/RESEARCH_RESULTS.md` and the v2.1 integration evidence:

| Dataset/model | n | Accuracy | Sensitivity | Specificity | AUROC | Confusion matrix TN, FP, FN, TP |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| IDRiD v1.1 | 72 | 70.83% | 57.45% | 96.00% | 0.846 | Not recorded in available summary |
| IDRiD v2.1 | 72 | 79.17% | 87.23% | 64.00% | 0.874 | 16, 9, 6, 41 |
| RFMiD v1.1 | 69 | 86.96% | 100.00% | 84.48% | 0.967 | Not recorded in available summary |
| RFMiD v2.1 | 69 | 81.16% | 54.55% | 86.21% | 0.875 | 50, 8, 5, 6 |

At the RFMiD operating point, v2.1 has 11 DR-positive examples (6 TP + 5 FN), 58 negatives (50 TN + 8 FP). Its sensitivity estimate is based on only 11 positives. The IDRiD v2.1 matrix gives 47 positives (41 TP + 6 FN) and 25 negatives (16 TN + 9 FP). These are aggregate calculations; they do not identify image IDs or allow visual review.

The matched v1.1/v2.1 table comes from the same reported development assessments, so it is more informative than comparing unrelated cohorts, but it is not independent validation. At the saved thresholds v2.1 moves the IDRiD operating point toward sensitivity and more false positives, while on RFMiD it performs worse on both sensitivity and AUROC. Threshold/calibration behavior cannot alone explain a lower RFMiD AUROC; ranking/model-domain issues also warrant investigation. Do not alter either threshold on these same assessment cases.

## Probabilities, calibration, and preprocessing evidence

- v2.1 uses cutoff `0.3675977844516245`, positive at `>=`; its Platt slope/intercept (`0.5428962637583218`, `-0.49103461599559345`) are embedded in the ONNX graph. The app applies neither calibration again nor probability inversion.
- The saved model manifest reports a 141-example Python-versus-exported-ONNX replay with maximum absolute error `1.8341754449702208e-06`, mean error `2.470365740902819e-07`, and zero cutoff disagreements. It validates export parity on saved development outputs, not Python-versus-Android on labeled source images.
- Android preprocessing follows the recorded OpenCV contract. Existing Android tests compare five synthetic tensor fixtures, but these were not executed on a device in this phase. The static APK contains the expected ONNX asset; this does not establish runtime image tensor parity.
- v1.1 uses a distinct preprocessing path and cutoff `0.48`. The aggregate table compares recorded outputs, but source-level matched per-image tensors/probabilities are not available for an end-to-end parity replay.

## Data composition and possible explanations

- IDRiD grading protocol is grade 0 vs grades 1–4 for any DR in the app-evaluation definition. The available official split report gives 413 train and 103 test images, with test counts grade 0: 34, grade 1: 5, grade 2: 32, grade 3: 19, grade 4: 13. Patient IDs/grouping could not be verified in the app assessment summary.
- RFMiD is a multi-disease source collapsed to a DR binary label for this assessment; the app metrics use 69 images, but this checkout's assessment source manifest and image-level label/output join are unavailable. Label prevalence is 11/69 positives (15.9%), making specificity and accuracy dominate accuracy. Source/domain differences from IDRiD may matter.
- The v2.1 head evidence says the head was fit on its development selection data without class weighting, then Platt-calibrated; model selection scores were 84.93% accuracy, 87.10% sensitivity, and 83.33% specificity on that selection context. The manifest explicitly marks use as development only. This is not a substitute for the later RFMiD shift.
- Earlier research records document prior dataset exposure, unavailable patient identifiers for some RFMiD cohorts, and overlap between RFMiD locked-fold samples and representation-learning exposure. Exact train/evaluation image overlap for the 69-image classifier assessment cannot be verified without manifests.
- The KD768 checkpoint and original training payload are not present in this checkout; only the checkpoint digest/provenance and cached representation research artifacts are recorded. This prevents exact training-data/head replay here.
- A small saved-feature IDRiD baseline in `kd_run2/hybrid_dr_improvement/REPORT.md` reports a separate 103-image test: grade 1 2/5 detected, grade 2 28/32, grade 3 17/19, grade 4 12/13; overall confusion matrix TN, FP, FN, TP is 25, 9, 10, 59. This is a distinct logistic probe at 0.50 on cached KD features, not the deployed v2.1 head, and reuses previously scored images. It suggests mild-disease sensitivity may be fragile, but cannot be assigned to the app classifier.

## Error examples and what is missing

No person-identifiable information is needed, but reproducible pseudonymous image IDs, binary labels, v1.1/v2.1 probabilities, tensor/image hashes, source dataset, and exact split are needed. The present research summary supplies only aggregate confusion matrices/metrics. The separate cached IDRiD embeddings/labels and ODIR-5K probe outputs are not the application classifier predictions and cannot stand in for the five RFMiD v2.1 false-negative cases.

No per-image v2.1 false-negative/false-positive probability distribution can be responsibly reported from the files found. No individual example should be fabricated from the synthetic fixtures or from unrelated lesion segmentation/probe artifacts.

## Smallest next diagnostic experiment

1. Restore the exact permitted 69-image RFMiD assessment image manifest/labels and saved v1.1/v2.1 per-image probabilities, with dataset/sample IDs pseudonymized and checksummed. Also restore the 72-image IDRiD assessment bundle and label protocol.
2. Verify counts, label mapping, exact/near-duplicate overlap with model training/calibration/selection, and source patient/site grouping. Preserve the fixed existing thresholds for this audit.
3. On identical images, run pinned Python references for both models; export image and preprocessed tensor SHA-256, uncalibrated logits if accessible, calibrated probability, label, model hash, preprocessing version, and threshold outcome. Where Android is available, compare its tensors and probabilities on the same cases.
4. Generate per-image joined tables and score distributions split by dataset, DR grade/label, camera/site, image quality, and each error category. Review de-identified FP/FN thumbnails only with authorized access.
5. If parity passes but errors remain, treat the cause as classifier/data generalization or label-definition limitation; define a new independent patient/site-held-out cohort before proposing head/calibration experiments. Do not retune threshold on these assessment cases or retrain the encoder without evidence that the representation is limiting.

## Gate status

The 85% accuracy / 85% sensitivity / 80% specificity gates are not established. Existing development assessments miss one or more gates on both IDRiD and RFMiD; v2.1 RFMiD sensitivity is especially low. No performance claim is made from the software, synthetic, or APK checks.
