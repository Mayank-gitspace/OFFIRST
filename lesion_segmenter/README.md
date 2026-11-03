# team_offirst lesion segmenter: v3 and v6

ConvNeXt-T U-Net (ImageNet-pretrained encoder) with 4 sigmoid outputs: MA (microaneurysms), HE (haemorrhages), EX (hard exudates), SE (soft exudates). Trained at 1440 px on the IDRiD segmentation subset. The crop picker scores each blob by its mean probability, applies a per-class `min_score`, and keeps at most one crop per class.

## Checkpoints (GitHub Release `segmenter-v3-v6`)
The files are about 128 MB each, which is over GitHub's 100 MB per-file limit, so they are release assets and are not in git.

| File | Model | SHA-256 |
|---|---|---|
| [v6_best.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/segmenter-v3-v6/v6_best.pth) | v6 best epoch | d3dee7ec0a34c49d4caeb4b7a6ba97d41580078fb6632c13e8a7de350ee8c9c3 |
| [v6_last.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/segmenter-v3-v6/v6_last.pth) | v6 last epoch | 36ed954eeeb515a51a41dbd5eda2c61bdf14c48a56598b38e90f0403aeedbccf |
| [v3_best.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/segmenter-v3-v6/v3_best.pth) | v3 (Kaggle `hires1440` run) best epoch | 860cca2ebdbb6516f8c2fad45e81d6f92d61b1229b73d1d5ac0f317a4630d94a |

## Other trained checkpoints (GitHub Release `segmenter-archive`)
Reference and ablation models; v6 (plus v3's SE head) remains the main candidate.

| File | Model | SHA-256 |
|---|---|---|
| [focal1440_best.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/segmenter-archive/focal1440_best.pth) | early run `focal1440` (1440 px, focal+Dice) best epoch | 193878db7c29933c1aeb3f662df80b72f6a72206fd64acf8319182bd757047b4 |
| [focal1440_last.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/segmenter-archive/focal1440_last.pth) | early run `focal1440` last epoch | d6cddcc583f1a5cecd26e7270e4a4c7be1ccd2788e6af62a6ab557e7bb473010 |
| [quick768_best.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/segmenter-archive/quick768_best.pth) | early run `quick768` (768 px, Dice+BCE) best epoch | 26a6cabe2ded38412cff0aec1fa008fbd956cbc91eb838c6d98e040cfbd633b2 |
| [quick768_last.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/segmenter-archive/quick768_last.pth) | early run `quick768` last epoch | 125be76412469045555773501b72cb4111e7b8c150a96da826cf7792d98ad673 |
| [v3_last.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/segmenter-archive/v3_last.pth) | v3 (Kaggle `hires1440`) last epoch | 1c163099e97dd4d715b8827a78a6b2c199ea7c0ea3d73ab6572f5013d706c43c |
| [v5_best.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/segmenter-archive/v5_best.pth) | v5 (camera-robust) best epoch | 87b72d79915aacdba2c5d58806df2cfcf8d07c534dccecd43c5dc2b5eb2dbd95 |
| [v5_last.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/segmenter-archive/v5_last.pth) | v5 last epoch | 2606efe61a50f4ec7db5d44950c3f84bc83bf46558561c6ee67cc7a3392c3b8f |

Not published: the v4 `retrain_neg` checkpoint (trained with IDRiD grading-set negatives, which breaks our data rules) and local smoke-test checkpoints.

## Results on the official 27 IDRiD segmentation test images (IDRiD_55-81), pixel AUPR at model resolution

| Model | MA | HE | EX | SE | mean MA/HE/EX |
|---|---|---|---|---|---|
| v3 (hires1440) | .469 | .644 | .888 | **.705** | .667 |
| v5 | .431 | .653 | .878 | .105 | .654 |
| **v6** | **.477** | **.698** | .874 | .122 | **.683** |

- **v6 crop picker** (thresholds in `v6/thresholds_v6.json`: MA .95, HE .885, EX .92, SE .5). These were tuned with `max_healthy_rate = 0.10` on 341 RFMiD-train normals and the 27 IDRiD segmentation test images.
  - Healthy RFMiD-train normals that get any crop: **22%**, broken down by camera as 2048x1536 2.2%, 2144x1424 20.5%, and 4288x2848 58.6%.
  - RFMiD-train DR images (n=376) that get any crop: **99.2%**.
  - v6 produces no SE crops.
- **Proposed combination (not final):** take MA/HE/EX from v6 and SE from v3. v6 lost SE (AUPR .122), while v3 keeps it at .705. Per-camera thresholds for the 4288x2848 camera are still an open question.

## Training differences
- **v3:** Kaggle `hires1440` run, Dice+BCE at 1440 px, with no negatives. Config is `v3/config_v3_hires1440.yaml`. The picker thresholds in `v3/config_v3_with_thresholds.yaml` (MA .965, HE .95, EX .94, SE .965) were tuned on IDRiD grading grade-0 images. **They are deprecated under our current data rules; re-tune them on RFMiD-train normals before use.**
- **v6:** 50 RFMiD-train normals used as negatives, with 2 negative patches per normal (down from 8) and SE loss weight 3.0. Config is `v6/config_v6.yaml`; training log is `v6/log.txt`.

## Layout
- `code/`: the current `lesion_detector` source (train, inference, FOV mask, threshold tuning). `code/config.yaml` still holds the v5 thresholds, so use `v6/thresholds_v6.json` for v6.
- `v6/`, `v3/`: configs, thresholds, training history and logs, eval reports, and a snapshot of the code each model was trained with (`train_snapshot/`).
- `eval/`: scripts for the test-27 evaluation, test-27 overlays (`overlays_test27.py`) and the RFMiD DR check.
- `v5/`: v5 config, tuned thresholds, training history/log and code snapshot.
- `early_runs/`: history and logs of the first Kaggle runs (`quick768`, `focal1440`; code = `v3/train_snapshot/`), plus `REPORT.md` comparing them with `hires1440` (v3).
- `kaggle_kernel/`: the self-contained Kaggle training kernel (`team_offirst_lesion_segmenter.py`, current = v6) and its metadata; `history/` holds the v3-focal, v4 (do not use) and v5 kernel versions. Note: the kernel slug in `kernel-metadata.json` was renamed to `team-offirst-lesion-segmenter`; pushing it creates a new Kaggle kernel.
- `demos/`: `v5_before_after/` (v3 vs v5 crop demo), `fov_mask/` (FOV-mask demo and threshold sweep), `deprecated_fixes/` (old threshold experiments on IDRiD grading images; deprecated, see `DEPRECATED.md`).

Scripts use paths relative to the project root (the folder holding `lesion_detector/`, `data/`, `kaggle_runs/`, ...); run them from there.

## Data rules
This repo contains no dataset images. The IDRiD grading set and the RFMiD held-out test set were never used for v6 training or evaluation. IDRiD is used only through its segmentation subset. Note: 59 IDRiD segmentation images are pixel-identical to IDRiD grading images; see the team notes before any few-shot use.
