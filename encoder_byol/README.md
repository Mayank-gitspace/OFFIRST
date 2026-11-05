# team_offirst encoder: BYOL self-supervised pretraining (A/B)

BYOL SSL pretraining of `convnext_tiny` (ImageNet init) on RFMiD train (`awsaf49/rfmid-train-dataset`), folds 1-4 only. Fold 0 of `split_5fold_seed42.csv` (from the baseline kernel output) is held out for evaluation only. Two crop modes were trained for 150 epochs each:

- **A `random`**: whole-eye global views + random multi-crop local views.
- **B `anatomy`**: same global views, local views centred on the optic disc and macula (heuristic locator).

Results summary: see [RESULTS.md](RESULTS.md).

## Files
- `team_offirst_encoder_byol.py`: Kaggle script kernel (config YAML at the top). `RUN_MODE: smoke | full | eval`.
- `kernel-metadata.json`: private kernel `jeetmuley/team-offirst-encoder-byol`, T4 GPU, data source `awsaf49/rfmid-train-dataset`, kernel source `jeetmuley/team-offirst-encoder-baseline` (for the split and supervised reference).
- `kernels/random/`, `kernels/anatomy/`: the exact scripts + metadata pushed for the two full runs (`RUN_MODE = "full"`, `crop_modes` = random / anatomy).
- `analyze.py`: prints the comparison tables from the run outputs (`python analyze.py <dir with random/ and anatomy/>`).
- `full_run/<mode>/`: `summary.json` (all evals + references), `report.json`, `history.json` (per-epoch loss, embedding std, kNN AUROC), `run.log`, `kaggle_kernel.log`.
- `full_run/split_used.csv`, `full_run/anatomy_locations.csv`: split and per-image disc/macula locations (identical for both runs).
- `full_run/training_curves.png`: loss, embedding std (collapse check) and kNN AUROC per epoch.

Checkpoints, embedding arrays and contact sheets (dataset images) are not committed.

## Weights (GitHub Release `encoder-byol-v1`)
Each checkpoint is a dict with `online` (backbone+projector+predictor), `target_backbone`, `target_projector`, `epoch`, `crop_mode`, `config`. Use `target_backbone` as the encoder. `best` = epoch-50 checkpoint (the highest kNN mean AUROC among saved checkpoints; kNN peaked at ep 40 for A and ep 60 for B, which were not saved), `last` = epoch 150 (the one evaluated in RESULTS.md). Checksums are in `SHA256SUMS.txt` on the release.

| File | Size | SHA-256 |
|---|---|---|
| [encoder_byol_random_best.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-byol-v1/encoder_byol_random_best.pth) | 244 MB | ecd62634819f1defe44bf10acfc64f77569c2a3c7b9022b8422f8b86a02d7e8a |
| [encoder_byol_random_last.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-byol-v1/encoder_byol_random_last.pth) | 244 MB | 4016517d8ecd86ae9f5fa05eff677b387b23535d2a6e4ff714f692b0de1dc12b |
| [encoder_byol_anatomy_best.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-byol-v1/encoder_byol_anatomy_best.pth) | 244 MB | d94f885fb6e42981258154881c69ede5f74120bd763e9fc49cfe73295f6e7b91 |
| [encoder_byol_anatomy_last.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-byol-v1/encoder_byol_anatomy_last.pth) | 244 MB | 8fd86957f836f937caf425d31811f5fd0d198e7f54d336637ef1e58b79860366 |

- Push: `KAGGLE_API_TOKEN="$KAGGLE_KEY" kaggle kernels push -p kernels/<mode>`
- Citation: RFMiD (Pachade et al., Data 6(2):14, 2021; IEEE DataPort, doi:10.21227/s3g7-st65), CC BY 4.0. BYOL: Grill et al., NeurIPS 2020.

## v4 (anatomy v4 + locked-split re-score)
See [`v4/README.md`](v4/README.md); weights in release `encoder-byol-v4`.
