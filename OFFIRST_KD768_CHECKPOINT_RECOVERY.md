# OFFIRST KD768 checkpoint recovery

Repository: Mayank-gitspace/OFFIRST
Branch / starting HEAD: codex/offirst-next-phase / 38a0ba31996eef453ead711b845dc5aa593905c5

## Finding

The exact KD Run 2 final checkpoint is identified and obtainable from the repository's kd-run2 GitHub release, but its binary is not present in the local checkout or scanned project/download locations. The v2.1 manifest records this exact checkpoint SHA, so it is not being selected by architecture similarity:

| Field | Recovered evidence |
|---|---|
| Exact filename | kd_run2_kd_final.pth |
| Release location | [kd-run2 release asset](https://github.com/Mayank-gitspace/OFFIRST/releases/download/kd-run2/kd_run2_kd_final.pth) |
| Recorded size | 15 MB (README rounds the size; exact byte count is not recorded there) |
| SHA-256 | 8bc239d22ffc4e786820f2151c99380a754bef930e42edea65022e1246f2096d |
| v2.1 linkage | deployment/offline_dr_screening/assets/model_v2_1_evidence/model_manifest.json names this same checkpoint_sha256; local v2.1 ONNX SHA-256 is 43a87f05df7adc6a231a679721e7f0b11b45a3fe6d1582aa4527ed738fa63b3f. |
| Local binary status | No kd_run2_kd_final.pth, kd_final.pth, or corresponding .pt checkpoint is in the repo, kd_run2/weights_ds/, Git object history, scanned Downloads/project locations, or inspected local ZIPs. |

The remote reference and manifest establish a precise artifact to recover; they are not a local re-hash or a checkpoint load. Verify the SHA after downloading before use. The checkpoint was not downloaded or loaded during this audit.

## Architecture and lineage

- Student backbone: timm mobilenetv3_large_100, GAPEncoder, 960-D global-average-pooled feature.
- Learned projection: nn.Linear(960, 768, bias=False), followed by L2 normalization. The final 768-D output is what the v2.1 encoder exports.
- Checkpoint payload: student state dict (backbone and projection), epoch, config, and architecture metadata. It is an encoder/projection checkpoint, not the complete v2.1 DR classifier head.
- Teacher: frozen BYOL anatomy-v4 ConvNeXt-T target, epoch 20, 768-D; release asset SHA-256 ed95b4de631bbeee1776775ec8d309313bcdd142de03609ab9c8185e40cde784.
- Student initialization: BYOL MobileNetV3-Large target, epoch 40; release asset SHA-256 79b3a0e1d536446bd3c82dbeef7c3d1ddb3f10acdaad0444dbebe228a6731215.
- KD training: 40 epochs, RFMiD recomputed five-fold assignment, folds 1–3 (1,152 images) used for student training; fold 4 was inner validation/checkpoint selection; fold 0 was a disclosed development evaluation. The training objective combines cosine alignment and masked neighbor-only KL, uses head-only then joint phases, and applies final batch-normalization recalibration.
- The app classifier is separate: the v2.1 ONNX includes the scaler, 16-unit MLP, and Platt calibration. The v2.1 threshold remains 0.3675977844516245; it is not changed here.

## Input and feature preprocessing

KD Run 2 uses the repository's FOV-aware crop_fundus path to construct a fundus square, a 640-pixel cache, and a 512-pixel student input with ImageNet mean/std normalization. During training, student views also receive configured crop/rotation/flip, color, blur/JPEG and camera-degradation augmentation; teacher views use a clean 384-pixel view. The final normalized embedding has dimension 768.

The v2.1 application preprocessing contract is separately recorded in deployment/offline_dr_screening/assets/model_v2_1_evidence/preprocessing.json: OpenCV color decode, RGB conversion, foreground crop/pad, sequential 640 and 512 INTER_AREA resizing, ImageNet normalization, and FP32 NCHW input [B,3,512,512]. This is the deployment graph's recorded contract; it does not prove every cached feature was extracted through a byte-for-byte identical path.

## Checkpoint searches and similar artifacts

- Git branches/history: no checkpoint blob is tracked in the checked-out history or local Git object inventory. The remote repository README links the exact final asset and digest.
- Local repository: no .pt, .pth, .ckpt, or .safetensors checkpoint exists. kd_run2/weights_ds/ contains dataset metadata and a SHA256SUMS file for teacher/student-init inputs, not the final checkpoint.
- Local run outputs: kd_run2/full_out/out/kd_run2/ contains embeddings, history.json, inner_evals.json, embedding_ids.csv, and procrustes_W.npy; no kd_final.pth remains there. The full-run summary and training source document a completed 40-epoch run; kd_run2/WORKFILE.md has a stale “Full run NOT launched” line that conflicts with later results and outputs.
- Kaggle outputs copied locally: full and smoke folders contain feature arrays and logs but not the KD final checkpoint. The private Kaggle weights dataset named in kd_run2/README.md contains the teacher and student initialization, not the final KD result.
- Local C:/Users/doifo/Downloads/OFFIRST_deployment_package.zip (22,076,763 bytes; SHA-256 2179835418b5e1b8e604a6fd69b360bdc7f2f071f0db42b795b20f94d2738971) contains a v1.1 offirst_dr_model.pt TorchScript export (11,884,636 bytes; SHA-256 fe7492fd203e9ab7b646ca1a41f6a39d2f4eff474027a06d239a35a2578c70b9) and the matching v1.1 ONNX. It is not the KD Run 2 checkpoint.
- A separate historical audit inventory under C:/Users/doifo/Downloads/OFFIRST_Accuracy_Audit/ documents offirst_epoch_03.pt (archive-member SHA-256 18595363555f58dd0b0a237c32f7fdc116f7f5caabf65ebb0703838ddea09513) as the exact v1.1 MobileNetV3-Large encoder match. The underlying 1.336-GB results.zip is not present in Downloads; the small audit ZIP contains derived reports, not checkpoint members. This model is a different v1.1 lineage, with a 960-D encoder, and is not a substitute for the 768-D KD student.
- kd_run2/final_contrastive_distilled/OFFIRST_final_contrastive_distilled.ipynb has five cells; its code cells are unexecuted and have no outputs. No checkpoint or feature array for that proposal exists there.

## Recovery action required

Download only the exact linked release asset, verify its SHA-256, inspect the checkpoint config/payload with the documented model code, and confirm the 960→768 projection and BN-recalibration flag. Do not substitute the v1.1 TorchScript model, an MNV3 BYOL checkpoint, a smoke checkpoint, or a similarly shaped feature array.

## Source-data permission constraint

The checkpoint's availability as an OFFIRST release asset grants no additional rights to source datasets or cached features used to train/evaluate it. Maintain the source dataset, version, license/terms, attribution, and exposure record for any reused embeddings or new extraction. The RFMiD and IDRiD papers state CC BY 4.0; the local arrays still require exact source/version and row-provenance checks. Do not redistribute cached dataset-derived arrays or case-level results merely because the model checkpoint itself is downloadable. See `OFFIRST_DATA_LINEAGE_AND_EXPOSURE_MATRIX.csv` for per-source status.
