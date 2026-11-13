# OFFIRST — direct small-model SSL: BYOL on MobileNetV3-Large (Run 1)

**Status (2026-11-09, IST):** script written + CPU smoke-tested on the box. **Not pushed to Kaggle or GitHub, no full run started.** Waiting for Jeet's review.

## Files
| path | what |
|---|---|
| `kernels/full/team_offirst_encoder_byol_mnv3.py` | Kaggle script, `RUN_MODE="full"` |
| `kernels/smoke/team_offirst_encoder_byol_mnv3.py` | identical except `RUN_MODE="smoke"` (2 ep, all SSL images, evals at ep 1 and 2) |
| `kernels/{full,smoke}/kernel-metadata.json` | `teamoffirst/team-offirst-encoder-byol-mnv3[-smoke]` — **unpushed**; see "Before pushing" |
| `build_kernel.py`, `src/header_config.py`, `src/tail.py` | builder: parent v4 + replacements → kernels (same convention as v4's `build_kernel.py`) |
| `DIFF_vs_v4.patch` | `diff -u` parent v4 → full kernel |
| `test_degrade_equivalence.py` | checks the in-kernel degrader against `degrade.py` |
| `test_model_paths.py` | neck / grad-ckpt / timm-head / checkpoint round-trip unit checks |
| `bench_views.py` | CPU cost per image of the view pipeline |
| `localtest/` | smoke logs and outputs, test JSONs, example images |
| `parent_team_offirst_encoder_byol_anatomy_v4.py`, `degrade.py` | inputs (unchanged) |

## What changed vs v4 (ConvNeXt-T BYOL v4)
**Kept verbatim:** data guard (incl. IDRiD guard), label loading, split logic (`split_5fold_seed42.csv` from the mounted baseline kernel, else iterstrat recompute), `crop_fundus`, the embedded disc/macula box table (`BOX_B64` byte-identical), anatomy-v4 local-view sampler, v4 colour jitter and camera augs, BYOL loss/EMA/LR/tau schedule code, OOM batch ladder + grad-checkpointing, probe / few-shot / kNN protocols, output layout (`out/summary.json`, `out/run.log`, `out/byol_mnv3_anatomy_v4/{history,report,epoch_evals}.json` (renamed from v4's `byol_anatomy_v4` so outputs never mix with v4; different kernel slug too), `best.pth`, `last.pth`, `epNNN.pth`, `embeddings_{online,target}[_degraded].npy`, `embedding_ids.csv`, `split_used.csv`, `box_table_used.csv`, `local_views_grid.png`).

**Changed (Jeet's fixed decisions):**
- Backbone `timm mobilenetv3_large_100` (ImageNet `ra_in1k`). Encoder output = **960-d GAP of the last stage**; timm's 960→1280 `conv_head`/classifier are removed (timm's `num_classes=0` would otherwise give 1280-d). Backbone 2.97 M params; total online net with heads 6.52 M.
- Globals 512 (cache raised 512→640), eval images 512.
- VICReg off (`train.vicreg.enabled: false`; code path kept).
- **Visuscout degradation inside the view pipeline** (below).

**Changed (new, all config knobs):**
- Per-image geometry saved with the cache (`cache_640_1920_geom.npy`: raw width, raw fundus-square side, FOV centre + radius) — needed for a resolution-consistent, crop-consistent degrader.
- Inner validation: kNN monitor and checkpoint selection now use **support folds 1–3 → query fold 4**. v4's kNN monitor and `best.pth` used fold-0 labels; that is gone. Fold 0 is only evaluated at the end (last online, last target, and the inner-selected checkpoint), labelled `split: dev`.
- Extra eval on every evaluate call: clean probe scored on clean **and** fixed-seed degraded copies of the query fold (same fitted probe), degraded-support→degraded-query probe, few-shot ODC/ODP/ARMD clean→degraded, kNN clean/degraded query, collapse diagnostics (effective rank of centred embeddings; mean paired clean-vs-degraded cosine minus random-pair cosine, raw and mean-centred; clean→degraded top-1 identity retrieval).
- ImageNet-init reference = the same MobileNetV3 960-d GAP at 512, clean + degraded, on both dev and inner splits.
- Multi-region probe disabled by default (Phase-0 v4 gate failed); code kept.
- Non-finite loss stops the run (history saved first).
- Checkpoints also hold `target_neck` and an `arch` dict; `RUN_MODE=eval` evaluates online **and** target of `TOB_CKPT` on dev + inner.
- Local-test-only env overrides (`TOB_BS`, `TOB_EVAL_SUB`, `TOB_EVAL_EPOCHS`, `TOB_KNN_EVERY`, `TOB_REFS_IN_SMOKE`, `TOB_GLOBAL_SIZE`/`LOCAL_SIZE`/`CACHE_SIZE`/`EVAL_SIZE`); none are set on Kaggle.

### How the degrader is applied (and why it isn't `degrade.py` called directly)
Per view: geometry (v4 affine) → v4 colour jitter → *[with prob p: Visuscout degrade at severity α]* → v4 camera augs (bright/blur/JPEG).
The degrade step re-implements `degrade.py`'s chain (CRF → vignette → defocus blur → gains → noise → FOV mask → JPEG) with the **fitted values copied verbatim** (asserted against `degrade.py` defaults in the test). Three problems with calling `degrade.py` on our views were fixed (Argon flagged the first two):
1. **Channel order:** `degrade.py` is BGR; the cache is RGB. Called on RGB it swaps the red/blue gains — test MAE vs the reference 10.6–22.3 (8-bit) instead of 0.11–0.12.
2. **Vignette geometry:** `degrade.py` centres the vignette on the FOV of whatever image it gets, so on a local crop it would draw a ring around the crop. Here the FOV centre/radius of the full image is mapped through the crop's affine, so a crop gets the rim falloff of its true position.
3. **Blur scale:** `degrade.py` blurs by `1.15 × raw_width/1024` raw pixels (that's how it was fitted, on raw RFMiD). Here σ is converted to view pixels using the image's raw width, raw fundus-square side and the crop's scale (`emulate_native: true`). Applying `degrade.py`'s rule to the view itself (Argon's code: `w/1024` with w = view width) under-blurs: Sobel-mean on image 1741 = 9.41 vs reference 8.15 (ours 8.10).

**Severity α:** each fitted parameter is interpolated linearly from identity (α=0) to the fit (α=1): γ 1→1.05, toe 0→0.02, shoulder 1→0.92, exposure 1→1.12, vignette 0→0.16, blur 0→1.15, gains 1→(0.85, 1.05, 1.15), noise 0→1.4, JPEG q 100→70. I chose this over pixel-blending clean↔degraded because blending a sharp and a blurred image is not a weaker blur (Argon made the same point). Light leak is not implemented (fitted value 0 = off; asserted).

**Not exact:** the view is warped from the 640 cache before blur/noise (degrade.py blurs at raw res, then we downsample); the order difference is what the 0.11–0.21 MAE measures. Noise is divided by raw-px-per-view-px (iid noise averaging), which makes it ~0.4–0.6/255 at view res — negligible. **JPEG q70 is applied at view resolution**, which is harsher than JPEG at native res then downsampling (what a real 1536-px Visuscout JPEG goes through). Knob `degrade.jpeg`.

## Choices for Jeet (Argon/Yuri suggestions — NOT decisions; all are CONFIG knobs)
| knob | default in script | source / why | alternatives |
|---|---|---|---|
| `degrade.p_global0` / `p_global1` / `p_local` | 0.0 / 0.85 / 0.6 | Argon: keep one global always clean so every image has a clean BYOL view; the clean/degraded pair is predicted in both directions by the symmetric loss | p_global1 1.0; degrade both globals |
| `degrade.alpha` | U(0, 1) | your brief said U(0,1); **Argon proposed U(0.35, 1)** (fewer near-clean "degraded" views). Smoke mean α ≈ 0.5 | [0.35, 1.0] |
| severity mechanism | parameter interpolation identity→fit | see above | pixel blend (not implemented) |
| Argon's ±10 % per-parameter jitter, 15 % random green leak | **not implemented** | would change/extend fitted values; you said don't | — |
| `degrade.camera_after` | true | your "keep camera blur/JPEG/bright"; degraded views get both (double JPEG possible) | false = Argon's (camera augs skipped on degraded views) |
| `degrade.emulate_native` | true | resolution-consistent blur/noise (verified) | false = degrade.py rule on the view (under-blurs) |
| `degrade.jpeg` | true (q 100−30α at view res) | fitted chain includes JPEG | false; harsher than native JPEG |
| `degrade.eval_alpha` / `eval_seed` | 1.0 / 2026 | degraded eval copies = full fitted degradation, fixed noise seed per image | — |
| degrader placement | CPU DataLoader workers on the 640 cache | Argon. Measured on the box (1 core): **112 ms/img** for the 6 views with the degrader vs **73 ms/img** without; degraded eval copy 26 ms/img | GPU torch re-implementation (not done; JPEG is the hard part) |
| `data.cache_size` | 640 | Argon; 512 globals come from RRC boxes of 0.5–1.0 of the area of a 640 square (≈452–640 px wide), so mostly mild down- or small up-sampling | 768 (+44 % memory: 3.4 GB) |
| `views.local_size` | 224 | Argon; 224/512 ≈ 160/384 of v4; MobileNetV3's native ImageNet size | 160, 192 |
| projector / predictor | 960→2048→256 / 256→2048→256 | v4 sizes kept | — |
| `model.neck_dim` | 0 (probe the 960-d GAP) | your brief/Argon: optional 768-d export neck. If >0, a Linear(960→768) sits inside the encoder (EMA'd), eval uses its output; tested | 768 |
| `model.feature` | gap (960-d) | brief | timm_head (1280-d, timm pre-logits; tested) |
| `train.epochs` / `monitor.eval_epochs` | 40 / [20, 30, 40] | Argon (v4 peaked at ep20 of 120). **Caveat:** ep20 of a 40-ep cosine run is not the same LR/τ state as ep20 of v4's 120-ep run | 60–80 |
| `train.batch_size` | 64 (ladder 64→48→48+ckpt→32+ckpt→16+ckpt) | Argon / v4 ladder. **T4 memory not measured** (no GPU here); MobileNetV3 activations at 512 are much smaller than ConvNeXt-T's, so 64 probably fits — unverified | — |
| `train.lr_backbone` / `lr_heads` / `weight_decay` | 2e-4 / 1e-3 / 0.05 (v4) | I kept v4's values. **Argon suggested lr_bb 3e-4, wd 1e-4** for MobileNetV3; not adopted (with ~720 steps, AdamW wd 0.05 × lr 2e-4 shrinks weights ≈1 % total) | Argon's |
| `train.warmup_epochs` | 4 | v4 8/120 → Argon 4/40 | 3 |
| `train.tau_base` | 0.99 cosine → 1 | v4 | — (only ~720 steps at bs 64 (18/epoch × 40); τ-cycle is short) |
| BN | as-is (online BN in train mode, target BN in train mode, buffers copied online→target as v4) | brief | — |
| `data.ssl_include_inner_val` | **false: SSL on folds 1–3 only (1,152 images)** | My literal reading of "train on folds 1–3, inner-validate on fold 4". **Please confirm** — this means 25 % fewer SSL images than v4 (1,536), so it's not apples-to-apples with v4/KD runs. `true` = SSL on folds 1–4 like v4 (Argon's code), with fold 4 seen unlabelled by SSL but its labels still only used for selection | true |
| `monitor.select_metric` / `select_encoder` | mean of inner-val clean LP and clean→degraded LP mAUROC / target (EMA) | Yuri suggestion (robustness is the point of this run; target = v4 RESULTS protocol) | lp_clean, knn_clean; online |
| `monitor.eval_both_encoders` | true | report online too at eval epochs (not used for selection) | false (saves ~2 evals) |
| `eval.degraded_support_probe` | true | extra deg→deg probe | false (halves probe time) |
| `eval.multi_region_probe.enabled` | false | Phase-0 v4 gate failed | true |

## Before pushing (things only Jeet can confirm)
- **Kernel ids:** I used `teamoffirst/team-offirst-encoder-byol-mnv3[-smoke]` and `kernel_sources: teamoffirst/team-offirst-encoder-baseline`. v4's metadata files say `jeetmuley/...`, while the v4 run lived at `kaggle.com/code/teamoffirst/...` (run via the web UI). **The baseline kernel source matters:** it mounts `split_5fold_seed42.csv`. Without it the script recomputes the split with iterstrat (seed 42). That should reproduce the split, but it hasn't been checked here. The smoke log prints `split from ...`.
- On Kaggle: GPU T4, Internet on (timm weight download), dataset `awsaf49/rfmid-train-dataset` only — **do not attach IDRiD** (the data guard will abort if it is mounted).
- Suggested order: Kaggle smoke kernel first (2 epochs on all 1,152 SSL images → real img/s, T4 peak memory, batch-probe result), then full.

## Smoke results (box CPU — NOT representative of a T4)
All numbers are from runs on this box (CPU, fp32, torch 2.14.1+cpu, timm 1.0.15). Timings are **not** representative of a T4.

**Smoke run on the final build** (`kernels/smoke` script, `localtest/smoke.log`, outputs in `localtest/out/byol_mnv3_anatomy_v4/`). Settings: 32 SSL images from folds 1–3, bs 4, 8 steps/epoch, 2 epochs, `EVAL_SUB` = 150 query images per fold. Wall time 882 s, exit 0.
- Cache: 640 px, built in 47.8 s on the first run and reused on the rerun. Degrader geometry: median raw width/side 1.537, median FOV radius 0.499 of side. Low-confidence box fraction: 0.205 overall and 0.206 in the SSL folds.
- ImageNet-init reference, dev split: LP 0.813, c→deg 0.802, deg→deg 0.813, few-shot k=1/5/10 0.519/0.523/0.547, erank 88.1. On the inner split: LP 0.874, c→deg 0.863.
- Epoch 1: loss 1.0753, std_z 0.884, std_h 0.643, inner kNN 0.808 (degraded query 0.778), erank 106.8. Degraded fraction of views: g0 0.00, g1 0.88, locals 0.59, mean α 0.55. Local anchors: random 0.50, disc 0.20, macula 0.20, fallback 0.11.
- Epoch 2: loss 0.8602, std_z 0.917, std_h 0.660, inner kNN 0.819 (degraded query 0.782), erank 111.9.
- Inner-val selection (target encoder, `lp_mean_clean_deg`): epoch 1 scored 0.8742 (LP 0.891, c→deg 0.857). Epoch 2 scored LP 0.886, c→deg 0.835. Epoch 1 was written to `best.pth`.
- Final dev results:
  - Target: LP 0.837, c→deg 0.770, deg→deg 0.814.
  - Online: LP 0.826, c→deg 0.740.
  - Selected epoch-1 target: LP 0.833, c→deg 0.782, deg→deg 0.833.
- Files written: `best.pth` / `last.pth` / `ep001.pth` (48 MB each), `embeddings_{online,target,selected_target}[_degraded].npy`, `embedding_ids.csv`, `report.json`, `epoch_evals.json`, `history.json`, `local_views_grid.png`, `degrade_eval_examples.png`. A rerun reproduced the earlier pre-rename smoke numbers exactly (deterministic seeds).

**Eval mode** (`RUN_MODE=eval` on the smoke `best.pth`, all 1,920 rows, no `EVAL_SUB`; `localtest/eval_mode.log`). 3,337 s, exit 0. Each full-size evaluate took 250–700 s on CPU.

| encoder | split | LP clean | LP c→deg | LP deg→deg | few-shot k1/5/10 | kNN clean/deg | erank |
|---|---|---|---|---|---|---|---|
| ImageNet init | dev | 0.886 | 0.865 | 0.886 | 0.543/0.562/0.618 | 0.846/0.806 | 178.6 |
| ImageNet init | inner | 0.907 | 0.872 | — | — | — | — |
| ckpt online | dev | 0.901 | 0.854 | 0.892 | 0.562/0.599/0.645 | — | 226.2 |
| ckpt online | inner | 0.897 | 0.828 | 0.901 | 0.556/0.597/0.647 | 0.851/0.761 | 227.7 |
| ckpt target | dev | 0.900 | 0.853 | 0.892 | 0.565/0.601/0.648 | 0.861/0.803 | 226.8 |
| ckpt target | inner | 0.900 | 0.827 | 0.900 | 0.556/0.600/0.648 | 0.853/0.773 | 228.5 |

This only proves the eval path works end to end. The checkpoint saw 32 images for 1 epoch, so the differences from ImageNet are not evidence of anything.

**Unit tests and benchmarks**
- Degrader equivalence (`test_degrade_equivalence.py`, images 1, 1494, 1741). Mean absolute error in 8-bit units inside the FOV, compared with `degrade.py` applied at raw resolution:
  - Ours: 0.108–0.119 on full images, 0.19–0.21 on rotated 224 local crops.
  - Naive blur rule: 0.24–0.42.
  - `degrade.py` run directly on the RGB cache (BGR mix-up): 10.6–22.3.
  - Sobel mean on image 1741: clean 11.36, reference 8.15, ours 8.10.
- Model paths (`test_model_paths.py`): tested default gap960, neck768 (768-d output), grad-checkpointing and timm_head (1280-d). In all four, losses were finite and falling, the EMA target moved, and a checkpoint round-trip gave equal outputs.
- View cost (`bench_views.py`, 1 core, 32 images): 111.9 ms/img with the degrader, 73.3 ms/img for v4-equivalent views, 119.1 ms/img with every view degraded. `degrade_eval` at 512 costs 26.1 ms/img; a clean resize costs 1.9 ms/img.

## Not tested
- Anything on GPU: CUDA/AMP fp16, the bs-64 OOM ladder and batch probe on a T4, and real img/s with 4 workers.
- The supervised-baseline reference (c). It needs the baseline kernel's embeddings mounted and was skipped in the smoke.
- `eval.multi_region_probe.enabled: true` (off by default).
- `data.ssl_include_inner_val: true`.
- A full 40-epoch run, plus the eval_epochs 20/30/40 path at full scale. Only epochs 1 and 2 were exercised in the smoke.
- The Kaggle mount paths and metadata (ids, `kernel_sources`). Nothing has been pushed.

## Estimated Kaggle runtime (ESTIMATE, not measured)
- No GPU number exists yet. The script's own `ESTIMATE` line is extrapolated from the CPU smoke (3.9 h) and is **meaningless for a T4**.
- My rough guess: data loading is probably the bottleneck. On one box core the views cost 112 ms/img. With 4 Kaggle workers and similar-speed cores, the ceiling is ~30–35 img/s → ~35 s/epoch for 1,152 images → ~25 min for 40 epochs. Add cache build (v4 took 251 s at 512 px), 8 kNN monitors, and ~11 full evaluate calls. On the box CPU each full-size evaluate took ~5 min. v4's took 43–90 s on Kaggle; ours does about twice the probe work. **Total guess: ~1–1.5 h.** The Kaggle smoke run will replace this with measured `img_per_s` and `eval_s`.

## Risks / unverified
- **No GPU test:** AMP fp16 with hard-swish, T4 memory at bs 64, real throughput, and the batch probe were not exercised (the box has no CUDA). Only CPU fp32 was run.
- **2-epoch smoke on 32 images says nothing about quality.** Loss fell and nothing collapsed, but that's all it shows.
- BYOL with BN on a small dataset (1,152 images, ~18 steps/epoch at bs 64 → ~720 steps total) has only a short EMA horizon. Collapse diagnostics (std_z/std_h, erank, paired−random cosine) are logged every 5 epochs to catch this.
- Degrader realism rests on 24 Visuscout frames. Degraded-copy metrics measure robustness to *our* degrader, not to real handheld images. JPEG at view resolution is harsher than reality.
- `select_metric` includes the degraded probe, so checkpoint choice favours our degrader's notion of robustness.
- v4 `_camera` passes an RGB array to `cv2.imencode` as if it were BGR, so JPEG chroma subsampling treats R as B. I kept this for parity with v4 (camera aug only). The Visuscout JPEG converts correctly.
- Fold-4 inner metrics on 384 images are noisy (rare labels have ~10–20 positives). Differences of ±0.01 between eval epochs are probably noise.
- Few-shot support for the dev split = folds 1–4. With `ssl_include_inner_val: false`, fold 4 was never seen by SSL. That's fine for labels but differs from v4.
