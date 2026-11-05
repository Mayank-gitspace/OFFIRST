# BYOL anatomy v4 — config draft (for Sir approval)

Status: **draft only**. No smoke / full run until OK.  
Base: keep **BYOL + ConvNeXt-T + ImageNet** (same SSL family as A/B/v2).  
Sources: Argon lit survey (`/workspace/zeiss/reports/argon_encoder_lit_survey.md`) + check1 numbers.

## Design intent vs v2
| Knob | v2 | **v4** |
|---|---|---|
| Local mix | each local: 40% disc / 30% mac / 30% random (~70% anatomy) | **fixed 1 disc + 1 macula + 2 random** (~50/50) |
| Disc crop size | tight 1.2–1.5 DD **or** wide 2.5–3.0 DD | **wide only 2.5–3.2 DD** (no tight) |
| Macula crop | ~2.3–2.7 DD | **2.5–3.2 DD** (same band) |
| Colour on disc locals | brightness/contrast ≤0.1; **no hue/sat** | **same weak colour jitter as all views** |
| Camera augs | blur 0.3, JPEG 0.5, cam bright 0.5 | **keep** (camera-robust) |
| Epochs | 70 | **120**, eval at **70 and 120** (settle ep confound) |
| Loss | BYOL only | BYOL + light **VICReg** on projector z |
| Eval readout | global GAP only | still train global; add **concat probe** as eval-only experiment |
| Headline few-shot | ODC, ODP, ARMD (no MH) | **same**; plus optional ERM/MHL LOO later |

## Proposed CONFIG block

```yaml
# crop_mode: anatomy_v4
seed: 42
data:
  cache_size: 512
  n_folds: 5
  eval_fold: 0
  num_workers: 4

views:
  global_size: 384
  global_scale: [0.5, 1.0]
  n_local: 4
  local_size: 160
  local_scale: [0.25, 0.55]
  ratio: [0.75, 1.3333]
  hflip: 0.5
  vflip: 0.5
  rotate_deg: 30
  # Uniform weak colour on ALL views (globals + locals + disc/mac) — no disc exemption
  jitter_p: 0.8
  brightness: 0.2
  contrast: 0.2
  saturation: 0.15
  hue: 0.03
  blur_p: 0.3
  blur_sigma: [0.1, 2.0]
  jpeg_p: 0.5
  jpeg_q: [40, 95]
  cam_bright_p: 0.5
  cam_bright: 0.15

anatomy_v4:
  n_disc: 1
  n_macula: 1
  n_random: 2
  disc_dd: [2.5, 3.2]
  mac_dd: [2.5, 3.2]
  jitter_frac: 0.3
  conf_cutoff: 1.0

model:
  backbone: convnext_tiny
  pretrained: true
  proj_hidden: 2048
  proj_out: 256
  pred_hidden: 2048

train:
  crop_modes: [anatomy_v4]
  epochs: 120
  batch_size: 64
  lr_heads: 1.0e-3
  lr_backbone: 2.0e-4
  weight_decay: 0.05
  warmup_epochs: 8
  tau_base: 0.99
  amp: true
  grad_clip: 3.0
  ckpt_every: 10
  vicreg:
    enabled: true
    lambda_var: 10.0
    lambda_cov: 1.0
    loss_weight: 0.1

monitor:
  knn_every: 5
  knn_k: 20
  collapse_warn: 0.2
  eval_epochs: [70, 120]

eval:
  img_size: 384
  batch_size: 64
  common_min_query_pos: 10
  rare_min_total_pos: 15
  rare_min_query_pos: 10
  rare_headline: [ODC, ODP, ARMD]
  shots: [1, 5, 10]
  draws: 20
  neg_per_pos: 1
  logreg_C: 1.0
  probe_C_grid: [0.01, 0.1, 1.0]
  multi_region_probe:
    enabled: true
    disc_dd: 2.5
    mac_dd: 2.5
    power_norm: true

smoke:
  epochs: 2
  n_train: 0
  knn_every: 1
```

## Explicitly unchanged
- SSL family: **BYOL**
- Backbone: **convnext_tiny** ImageNet
- Data: RFMiD-train folds 1–4 SSL; fold 0 held out
- Detector: classical CPU cropper; low-conf → random
- No lesion-seg crops, no RETFound, no SupCon on the 12 commons

## Success criteria (full run — separate OK)
- Common linear probe ≥ B (0.911), ideally near v2 (0.921) without colour shortcuts
- Rare headline 1/5/10-shot ≥ B check1 (0.622 / 0.705 / 0.734)
- No collapse; report ep70 vs ep120 on same trajectory

## Out of v4.0 (park)
- 70-ep pure-random twin
- Graham colour-sub aug
- RoI-token matching
- ERM/MHL LOO pitch numbers
