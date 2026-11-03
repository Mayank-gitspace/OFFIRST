# team_offirst lesion segmenter: v3 and v6

ConvNeXt-T U-Net (ImageNet-pretrained encoder) with 4 sigmoid outputs: MA (microaneurysms), HE (haemorrhages), EX (hard exudates), SE (soft exudates). Trained at 1440 px on the IDRiD segmentation subset. The crop picker scores each blob by its mean probability, applies a per-class `min_score`, and keeps at most one crop per class.

## Checkpoints (GitHub Release `segmenter-v3-v6`)
The files are about 128 MB each, which is over GitHub's 100 MB per-file limit, so they are release assets and are not in git.

| File | Model | SHA-256 |
|---|---|---|
| [v6_best.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/segmenter-v3-v6/v6_best.pth) | v6 best epoch | d3dee7ec0a34c49d4caeb4b7a6ba97d41580078fb6632c13e8a7de350ee8c9c3 |
| [v6_last.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/segmenter-v3-v6/v6_last.pth) | v6 last epoch | 36ed954eeeb515a51a41dbd5eda2c61bdf14c48a56598b38e90f0403aeedbccf |
| [v3_best.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/segmenter-v3-v6/v3_best.pth) | v3 (Kaggle `hires1440` run) best epoch | 860cca2ebdbb6516f8c2fad45e81d6f92d61b1229b73d1d5ac0f317a4630d94a |

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
- `eval/`: scripts for the test-27 evaluation and the RFMiD DR check.

## Data rules
This repo contains no dataset images. The IDRiD grading set and the RFMiD held-out test set were never used for v6 training or evaluation. IDRiD is used only through its segmentation subset. Note: 59 IDRiD segmentation images are pixel-identical to IDRiD grading images; see the team notes before any few-shot use.
