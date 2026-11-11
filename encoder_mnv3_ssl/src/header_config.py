# =============================================================================
# team_offirst - Retina encoder: direct small-model BYOL on MobileNetV3-Large (+ frozen eval)
# -----------------------------------------------------------------------------
# Parent: team_offirst_encoder_byol_anatomy_v4.py (ConvNeXt-T BYOL v4). Same data, guards, split, box table,
#   output layout (out/summary.json, out/byol_anatomy_v4/{history,report,epoch_evals}.json, *.pth, embeddings).
# Data: RFMiD Training split ONLY (Kaggle dataset awsaf49/rfmid-train-dataset, 1,920 images). Citation: RFMiD
#   (Pachade et al., Data 6(2):14, 2021; IEEE DataPort, doi:10.21227/s3g7-st65), CC BY 4.0.
# DATA RULES: split_5fold_seed42.csv (baseline). Fold 0 = disclosed development split: never used for SSL or for
#   checkpoint selection. Fold 4 = inner validation fold (labels used ONLY for checkpoint selection; by default
#   fold 4 images are also kept out of SSL, see data.ssl_include_inner_val). No RFMiD Evaluation/Test, no IDRiD,
#   no other retina data, no retina-pretrained weights (ImageNet init via timm only).
# FIXED (Jeet): BYOL (no KD, no disease labels in training), timm mobilenetv3_large_100 ImageNet init, 512 globals,
#   v4 anatomy views (2 globals + 4 locals = 1 disc + 1 macula + 2 random; disc/mac 2.5-3.2 DD; uniform weak
#   colour jitter; camera blur/JPEG/brightness kept) + Visuscout degradation (degrade.py, fitted 2026-11-06,
#   values copied verbatim) inside the view pipeline; VICReg OFF.
# OPEN CHOICES (Argon/Yuri suggestions, NOT decisions; every one is a CONFIG knob, see README 'Choices for Jeet'):
#   degrade.* (which views, p, alpha), cache_size 640, local_size 224, projector/neck, epochs/eval epochs,
#   LR/warmup, inner-val selection metric, multi-region probe off.
# Encoder output = 960-d GAP of the last MobileNetV3 stage (timm's 960->1280 conv_head is NOT used).
# RUN_MODE: 'smoke' (2 epochs) | 'full' | 'eval'
# =============================================================================

RUN_MODE = "__RUN_MODE__"   # "smoke" | "full" | "eval"

CONFIG_YAML = r"""
seed: 42
data:
  cache_size: 640           # fundus-circle crop, resized once into a uint8 memmap (v4: 512; 640 so 512 globals are not upsampled much)
  n_folds: 5
  eval_fold: 0              # disclosed development split: excluded from SSL and from checkpoint selection
  inner_val_fold: 4         # checkpoint selection: probes fit on folds 1-3, scored on this fold
  ssl_include_inner_val: false   # false = SSL on folds 1-3 only (1,152 imgs; fold 4 never seen). true = SSL on 1-4 like v4
  num_workers: 4
views:
  global_size: 512          # FIXED (Jeet)
  global_scale: [0.5, 1.0]
  n_local: 4
  local_size: 224           # suggestion (v4: 160 @ 384 globals; 224/512 ~ 160/384)
  local_scale: [0.25, 0.55]
  ratio: [0.75, 1.3333]
  hflip: 0.5
  vflip: 0.5
  rotate_deg: 30
  jitter_p: 0.8
  brightness: 0.2
  contrast: 0.2
  saturation: 0.15
  hue: 0.03                 # fraction of the hue circle; same jitter on ALL views (incl. disc/macula)
  blur_p: 0.3               # camera-robust, all views (v4 values kept)
  blur_sigma: [0.1, 2.0]
  jpeg_p: 0.5
  jpeg_q: [40, 95]
  cam_bright_p: 0.5
  cam_bright: 0.15
degrade:                    # Visuscout degrader (degrade.py chain, RGB-correct, per view). Suggestions, not decisions.
  enabled: true
  p_global0: 0.0            # global view 0 always clean
  p_global1: 0.85           # global view 1 degraded with this probability
  p_local: 0.6              # each local view independently degraded with this probability
  alpha: [0.0, 1.0]         # severity ~ U(lo, hi); every fitted parameter interpolated identity -> fitted by alpha
  camera_after: true        # v4 camera augs (bright/blur/JPEG) still applied after degradation on degraded views
  emulate_native: true      # blur sigma + noise rescaled so a view ~ degrade(raw image) then resize (degrade.py sigma is px @ raw width/1024)
  jpeg: true                # Visuscout JPEG (q = 100 - 30*alpha) applied at VIEW resolution (harsher than native; README)
  eval_alpha: 1.0           # degraded eval copies: full fitted degradation
  eval_seed: 2026           # fixed per-image noise seed for degraded eval copies
anatomy_v4:
  n_disc: 1
  n_macula: 1
  n_random: 2
  disc_dd: [2.5, 3.2]       # wide only (no tight 1.2-1.5)
  mac_dd: [2.5, 3.2]
  jitter_frac: 0.3          # centre jitter up to 0.3 x crop size
  conf_cutoff: 1.0          # detector top1-top2 margin; below -> anatomy slot becomes random
anatomy:
  p_image: 0.5              # (legacy v1 block, unused by anatomy_v4; kept for config parity)
  crop_dd: 2.5
  dd_frac: 0.12
  jitter_dd: 0.25
  scale_jitter: [0.85, 1.15]
  min_conf: 0.5
model:
  backbone: mobilenetv3_large_100   # FIXED (Jeet), timm ImageNet weights
  pretrained: true
  feature: gap              # 'gap' = 960-d GAP of last stage (timm conv_head dropped) | 'timm_head' = timm pre-logits (1280-d)
  neck_dim: 0               # 0 = probe the GAP directly. >0 = Linear(960->neck_dim) inside the encoder (EMA'd, eval uses its output)
  proj_hidden: 2048
  proj_out: 256
  pred_hidden: 2048
train:
  crop_modes: [anatomy_v4]
  epochs: 40                # suggestion (v4: 120, but best kNN at ep20)
  batch_size: 64            # fallback ladder on OOM: 64 -> 48 -> 48 + grad ckpt -> 32 + grad ckpt -> 16 + grad ckpt
  lr_heads: 1.0e-3          # v4 values kept
  lr_backbone: 2.0e-4
  weight_decay: 0.05        # v4 value kept (Argon suggested 1e-4 for MobileNetV3; not adopted, README)
  warmup_epochs: 4          # v4: 8/120; suggestion 4/40
  tau_base: 0.99            # EMA momentum, cosine 0.99 -> 1.0 over the run
  amp: true
  grad_clip: 3.0
  ckpt_every: 10
  vicreg:
    enabled: false          # FIXED (Jeet): VICReg off
    lambda_var: 10.0
    lambda_cov: 1.0
    loss_weight: 0.1
    gamma: 1.0
monitor:
  knn_every: 5              # kNN on the INNER split (support folds 1-3, query fold 4), clean + degraded query
  knn_k: 20
  collapse_warn: 0.2        # warn if mean per-dim std * sqrt(d) of L2-normed z drops below
  eval_epochs: [20, 30, 40] # inner-val full eval + ckpt; checkpoint selection among these
  select_metric: lp_mean_clean_deg  # inner-val linear probe: mean of clean and clean->degraded mAUROC | lp_clean | knn_clean
  select_encoder: target    # target (EMA) | online
  eval_both_encoders: true  # also inner-eval the other encoder at eval epochs (reported, not used for selection)
eval:
  img_size: 512
  batch_size: 64
  common_min_query_pos: 10  # -> the 12 common labels of the baseline (incl. Disease_Risk)
  rare_min_total_pos: 15
  rare_min_query_pos: 10
  rare_headline: [ODC, ODP, ARMD]   # MH = media haze (not macular hole); per-label results kept for all rare labels
  shots: [1, 5, 10]
  draws: 20
  neg_per_pos: 1
  knn_k: 5
  logreg_C: 1.0
  probe_C_grid: [0.01, 0.1, 1.0]
  degraded_support_probe: true      # also fit probes on degraded support -> degraded query
  multi_region_probe:
    enabled: false          # off: Phase-0 re-score of v4 failed the multi-region gate (keep f_full)
    disc_dd: 2.5
    mac_dd: 2.5
    region_size: 224
    power_norm: true
smoke:
  epochs: 2
  n_train: 0                  # 0 = all SSL images (real epoch time)
  n_query: 96
  knn_every: 1
  eval_epochs: [1, 2]
  common_min_query_pos: 3
  rare_min_total_pos: 8
  rare_min_query_pos: 3
  draws: 3
  time_steps: 6
"""

