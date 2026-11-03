# team_offirst encoder baseline

- `team_offirst_encoder_baseline.py`: Kaggle script kernel. The config YAML is at the top. Set `RUN_MODE = "full"` to launch the full run.
- `kernel-metadata.json`: private kernel `jeetmuley/team-offirst-encoder-baseline`, T4 GPU, internet on, data source `awsaf49/rfmid-train-dataset` only.
- `full_run_v2/out/`: metrics from the full Kaggle run (v2): `summary.json`, `run.log`, the 5-fold split, and per-backbone `history.json` + `fewshot_report.json` for `convnext_tiny` and `resnet50`. Checkpoints (`*.pth`), embedding arrays and preview images are not committed.
- `idrid_seg_eval/`: ConvNeXt-T baseline evaluated on the IDRiD segmentation subset vs RFMiD-train fold-0 val normals (`report.md`, `report.json`, plot, per-image CSV, `preds_embs.npz` with probabilities/embeddings, `run_eval.py`). No dataset images are included.
- Data switch: `data.mode: full_image | crops`. Crops mode reads `data.crops_csv`, with columns `crop_id,image_id,path`. Labels and folds come from the parent image, and embeddings are mean-pooled per image.
- Push: `KAGGLE_API_TOKEN="$KAGGLE_KEY" kaggle kernels push -p .`
- Citation: RFMiD (Pachade et al., Data 6(2):14, 2021; IEEE DataPort, doi:10.21227/s3g7-st65), CC BY 4.0.

## Weights (GitHub Release `encoder-baseline-v1`)
Trained checkpoints from the full Kaggle run (v2) are release assets, not in git (some are over GitHub's 100 MB file limit). `best` = best validation epoch (EMA), `last` = final epoch. Checksums are in `SHA256SUMS.txt` on the release.

| File | Size | SHA-256 |
|---|---|---|
| [encoder_convnext_tiny_best.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-baseline-v1/encoder_convnext_tiny_best.pth) | 111 MB | f29ab8b46c0d571de5c22d3b66f42274733e25c419dc19564261109a3778e8a6 |
| [encoder_convnext_tiny_last.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-baseline-v1/encoder_convnext_tiny_last.pth) | 223 MB | ec05e5e4f401390d23fa0884716a7654d440348095bcef71f646068db7a54b03 |
| [encoder_resnet50_best.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-baseline-v1/encoder_resnet50_best.pth) | 95 MB | 9417f4f7b63c8f7a4bfb17fe1ba9c00a932bd850c74d7cdd236baaeafc601232 |
| [encoder_resnet50_last.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/encoder-baseline-v1/encoder_resnet50_last.pth) | 189 MB | 74cbccd5c46cf5a5a2350c99afa1ea64c40fc1657a0754b5ef275f6d4ac47184 |

Scripts expect `best.pth` at `encoder_baseline/full_run_v2/out/<backbone>/best.pth`, so download and rename it there (or set `TOB_ROOT`).

- `demo/run_demo.py`: single-image demo with the ConvNeXt-T baseline (`demo_probs.json` is its sample output).
