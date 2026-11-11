# =============================================================================
# team_offirst - KD Run 2: MobileNetV3-Large student <- frozen BYOL anatomy v4 ConvNeXt-T (ep20 target) teacher
# -----------------------------------------------------------------------------
# Parent: team_offirst_encoder_byol_mnv3.py (MNV3 BYOL SSL arm). Same data guard, labels, split (iterstrat seed 42 =
#   v4/MNV3 in-run split), 640 cache + FOV geometry, Visuscout degrader, eval protocol (probes / few-shot / kNN).
# Data: RFMiD Training split ONLY (awsaf49/rfmid-train-dataset). Weights: teamoffirst/offirst-kd-run2-weights
#   (teacher target backbone + student SSL ep40 target backbone, both trained on RFMiD-train folds only).
# DATA RULES: fold 0 (dev) never used for training, anchor bank or selection. KD trains on folds 1-3 (1,152 imgs).
#   Fold 4 only logged (validation cosine / inner probes). Final model = after epoch 40 + BN recalibration.
# CONFIG (Jeet, 2026-11-10): student timm mobilenetv3_large_100 @512 from SSL ep40 target; head GAP960 ->
#   Linear(960,768,bias=False) -> L2 (KEPT, eval/app use the 768-d output); Procrustes head init; ep1-3 head only
#   (backbone frozen, eval mode); ep4-40 joint; teacher LIVE (no_grad + fp16 autocast) on the SAME view at 384;
#   shared spatial aug hflip/vflip 0.5, rot +-30, RRC 0.80-1.0 / 0.95-1.05; student view degraded p=0.5;
#   loss = (1-cos) + 1.0 * tau_s^2 * KL(p_t || p_s) vs static bank of clean teacher embeddings (folds 1-3).
# RUN_MODE: 'smoke' (2 epochs: 1 head + 1 joint, reduced evals) | 'full'
# =============================================================================

RUN_MODE = "__RUN_MODE__"   # "smoke" | "full"

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
kd:
  teacher_ckpt: teacher_byol_v4_ep20_target_convnext_tiny.pth   # found under /kaggle/input (dataset teamoffirst/offirst-kd-run2-weights)
  student_ckpt: student_init_mnv3_ssl_ep40_target.pth
  teacher_arch: convnext_tiny
  teacher_size: 384
  student_size: 512
  emb_dim: 768
  rrc_scale: [0.80, 1.0]    # mild random resized crop of the fundus square (disc + macula stay in frame)
  rrc_ratio: [0.95, 1.05]
  hflip: 0.5
  vflip: 0.5
  rotate_deg: 30            # = SSL arm / v4 teacher
  p_deg: 0.5                # each image independently degraded for the STUDENT only (teacher always clean)
  colour_jitter: false      # not in the approved config (teacher must see the same pixels); knob kept
  camera_augs: false        # v4 camera blur/JPEG/bright: off (same reason); knob kept
  head_epochs: 3            # epochs 1-3: head only, backbone frozen + eval mode
  head_lr: 1.0e-3
  head_wd: 0.0
  lr_backbone: 1.5e-4
  lr_backbone_start: 1.5e-5
  bb_warmup_epochs: 3       # linear warmup of the backbone LR at the start of the joint phase
  lr_head: 5.0e-4
  lr_min: 1.0e-6
  weight_decay: 1.0e-4      # joint phase; norm/bias params get 0 (SSL-arm convention)
  tau_t: 0.05
  tau_s: 0.10
  lambda_seed: 1.0
  seed_mask_self: true      # Jeet 2026-11-10 18:55: drop each image's own bank entry from p_t/p_s (neighbour-only KL)
  grad_clip: 3.0            # SSL-arm value
  batch_size: 48
  batch_ladder: [[48, false], [48, true], [32, true], [16, true]]   # [bs, grad_ckpt]; first that fits
  bn_recal: true
  inner_eval_epochs: [20, 40]   # fold-4 probes (logged only, never used for selection)
  val_every: 1              # fold-4 clean/degraded validation cosine + KL vs teacher
  ckpt_every: 10
smoke:
  kd_head_epochs: 1
  kd_bb_warmup_epochs: 1
  kd_inner_eval_epochs: [2]
  refs: [ssl960, ssl768p]   # full run: [ssl960, ssl768p, teacher]
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

