# team_offirst encoder baseline

- `team_offirst_encoder_baseline.py`: Kaggle script kernel. The config YAML is at the top. Set `RUN_MODE = "full"` to launch the full run.
- `kernel-metadata.json`: private kernel `jeetmuley/team-offirst-encoder-baseline`, T4 GPU, internet on, data source `awsaf49/rfmid-train-dataset` only.
- `full_run_v2/out/`: metrics from the full Kaggle run (v2): `summary.json`, `run.log`, the 5-fold split, and per-backbone `history.json` + `fewshot_report.json` for `convnext_tiny` and `resnet50`. Checkpoints (`*.pth`), embedding arrays and preview images are not committed.
- `idrid_seg_eval/`: ConvNeXt-T baseline evaluated on the IDRiD segmentation subset vs RFMiD-train fold-0 val normals (`report.md`, `report.json`, plot, per-image CSV, `preds_embs.npz` with probabilities/embeddings, `run_eval.py`). No dataset images are included.
- Data switch: `data.mode: full_image | crops`. Crops mode reads `data.crops_csv`, with columns `crop_id,image_id,path`. Labels and folds come from the parent image, and embeddings are mean-pooled per image.
- Push: `KAGGLE_API_TOKEN="$KAGGLE_KEY" kaggle kernels push -p .`
- Citation: RFMiD (Pachade et al., Data 6(2):14, 2021; IEEE DataPort, doi:10.21227/s3g7-st65), CC BY 4.0.
