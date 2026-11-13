# OFFIRST retinal encoder

OFFIRST studies how well a small retinal image encoder can recognize multiple eye diseases from a few labeled fundus photographs. We first train on unlabeled images with BYOL, then distill a ConvNeXt-Tiny teacher into a MobileNetV3-Large student. The student produces a 768-dimensional feature vector. We evaluate it with disease-specific classifiers and similar-image retrieval.

## Training and weights

| Stage | Implementation | Role |
| --- | --- | --- |
| Teacher | [ConvNeXt BYOL v4](encoder_byol/v4/README.md) | Learns from full retinal views and crops centered on the optic disc and macula. Epoch 20 supplies the teacher weights. |
| Student initialization | [MobileNetV3 BYOL](encoder_mnv3_ssl/README.md) | Trains the compact encoder on clean and simulated handheld-camera views. Epoch 40 initializes distillation. |
| Distillation | [KD Run 2](kd_run2/README.md) | The frozen teacher sees a clean view at 384 pixels. The student learns to match its features from the same retinal region at 512 pixels with simulated degradation. |

Use [kd_run2_kd_final.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/kd-run2/kd_run2_kd_final.pth) for the final student encoder. The [checkpoint guide](kd_run2/README.md#weights-github-release-kd-run2) lists the model keys and checksum. Disease prediction requires a trained classification head on top of the encoder.

## What the evaluation shows

On ODIR-5K, with ten labeled positive eyes and ten negatives per training draw, the student's AMD AUROC rose from **0.681 before distillation to 0.772 after distillation**. With all available training labels, both students reached **0.872** clean macro linear-probe AUROC across eight labels. The ODIR labels are derived from per-eye clinical keywords. [ODIR evaluation and saved results](kd_run2/odir5k_eval/)

On IDRiD, the paired confidence intervals included zero for the clean grading comparisons. On RFMiD, 228 of the 384 locked evaluation images had appeared in the student's unlabeled pretraining data, and 308 had appeared in the teacher's. Treat the RFMiD scores as development results. [IDRiD grading report](kd_run2/idrid_grading_eval/REPORT.md) · [RFMiD student comparison](kd_run2/README.md#key-results) · [Teacher split audit](encoder_byol/v4/phase0_rescore/RESULTS.md)

## Reproduce the ODIR evaluation

From the repository root, run the following commands in a Python environment to score the checked-in embeddings on CPU. The evaluator reads the saved eye IDs and writes its summary to `kd_run2/odir5k_eval/results.json`.

```bash
python -m pip install numpy pandas scipy scikit-learn joblib
python kd_run2/odir5k_eval/evaluate.py
```

For training inputs, Kaggle kernel locations, and checkpoint setup, see the [KD reproduction instructions](kd_run2/README.md#reproduce).
