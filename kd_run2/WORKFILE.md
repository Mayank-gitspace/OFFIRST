# OFFIRST KD Run 2 — WORKFILE (all times IST)

MobileNetV3-Large student (init = MNV3 BYOL SSL ep40 target) distilled from the frozen BYOL anatomy v4 ConvNeXt-T
(ep20 target, 768-d). Config approved by Jeet 2026-11-10 (see kernel header). **Full run NOT launched — needs Jeet's OK.**

## Files
| path | what |
|---|---|
| `build_kernel.py` | builds `kernels/{smoke,full}/team_offirst_kd_run2.py` (+ kernel-metadata.json) |
| `src/header_config.py` | header + CONFIG (MNV3 config + new `kd:` block + smoke overrides) |
| `src/kd_tail.py` | KD-specific code (Argon's train_kd.py adapted; see review below) |
| reused verbatim from SSL arm | prefix of `encoder_mnv3_ssl/kernels/full/...py` (data guard, labels, iterstrat split seed 42, 640 cache + FOV geom, box table), and from `encoder_mnv3_ssl/src/tail.py`: torch setup/rows, views + Visuscout DEGRADE BLOCK, EvalDS, norm_gpu, GAPEncoder, std_metric, entire eval section (extract/probes/few-shot/kNN/diagnostics/evaluate) |
| `test_units.py` → `localtest/unit_checks.{log,json}` | CPU unit checks (tiny inputs, 256/192 px) |
| `localtest/teacher_repro.py`, `teacher_locked_lp.py` (+ .log) | teacher reproduction vs phase-0 v4 ep20 numbers |
| `localtest/e2e_smoke.log`, `localtest/e2e_out/` | local CPU end-to-end run of the smoke kernel (48 train imgs, 40 eval rows/fold, bs 8, 256/192 px) |
| `locked_rescore/paired_bootstrap.py` | locked A/B CPU re-score + paired bootstrap (B − A, 1000 it.), decision rule |
| `locked_rescore/selftest_ssl_target_vs_online.json`, `selftest.log` | evaluator self-test (reproduces RESULTS.md locked numbers exactly) |
| `locked_rescore/run_after_full.sh` | the 3 comparisons to run after the full run |
| `weights_ds/` | Kaggle dataset `teamoffirst/offirst-kd-run2-weights` (private): teacher + student-init state dicts, SHA256SUMS |
| `kaggle_smoke_out/` | smoke kernel outputs |
| `argon/` | Argon critique + code reply (base for kd_tail.py / paired_bootstrap.py) |

## Weights plumbing (working way)
Private Kaggle dataset **teamoffirst/offirst-kd-run2-weights** (created 2026-11-10 17:20 via `kaggle datasets create`), mounted via `dataset_sources`:
- `teacher_byol_v4_ep20_target_convnext_tiny.pth` = `target_backbone` of `/workspace/offirst_release_assets_byol_v4/encoder_byol_anatomy_v4_ep020.pth` (bit-identical to `..._best.pth` target, checked). 180 tensors, sha256 ed95b4de…
- `student_init_mnv3_ssl_ep40_target.pth` = `target_backbone` of `encoder_mnv3_ssl/full_out/out/byol_mnv3_anatomy_v4/last.pth` (= best.pth, ep40, checked). 308 tensors, sha256 79b3a0e1…
Not used: kernel_sources of the v4/MNV3 kernels (v4 output is 15 × 243 MB; a mounted `split_5fold_seed42.csv`-like file would also change the split source).

## Argon code review (what was taken / fixed)
Taken: two-stage schedule (head-only → joint), Procrustes `svd(Hsᵀ Zt) → (U Vt)ᵀ`, fp32 loss `(1−cos) + λ·τs²·KL(p_t‖p_s)` vs static clean bank,
stage-2 LR schedule (bb linear warmup 1.5e-5→1.5e-4 over 3 ep then cosine → 1e-6; head cosine 5e-4 → 1e-6 over the stage),
BN recal with `momentum=None`, teacher 384 view = resize of the student's clean 512 view, exact dominance-matrix bootstrap identity.
Fixed / changed:
1. **Preprocessing mismatch (major):** Argon loads raw RFMiD PNGs (rectangular, black borders) and rotates/crops them; the teacher was trained/evaluated on `crop_fundus` squares (512/640 cache). Its anchor-bank loader also squashes the raw rectangle to a square. → Our cached fundus squares + the v4 eval path (cache → INTER_AREA 384) for the bank.
2. **Degrader:** placeholder `degrade(img, rng)` (identity) → our FOV-aware Visuscout chain (`visuscout(x, a, M, row, rng)`, same params as SSL arm, α~U(0,1)), applied after the warp with the view's affine so vignette/blur scale are right.
3. **Checkpoint loading:** `strict=False` + prefix guessing (silently partial; would also fail on our `body.*` keys) → strict loads of explicit state dicts, missing/unexpected logged and checked.
4. Rotation applied to the full rectangular image before the crop → one affine (crop+rot+flips) from the fundus square, rotation about the crop centre (= SSL arm).
5. No grad clipping → clip 3.0 (SSL arm). Weight decay on BN/bias → 0 for norm/bias. Hard-coded `device_type="cuda"` → device-aware.
6. Head-only stage: backbone forward now under `no_grad` (saves memory); verified backbone incl. BN buffers bit-identical after stage-1 steps.
7. Added: OOM batch ladder (48 → 48+ckpt → 32+ckpt → 16+ckpt), fold guards, fold-4 validation cos/KL each epoch, checkpointing, full eval, embeddings export, smoke estimate.
8. **Evaluator:** Argon's LP = unscaled LogisticRegression C=1 (no class weights) and prototype few-shot with its own 100 episodes from folds 1–3 → does not reproduce our reported numbers. Replaced with OUR protocol (StandardScaler + C-grid CV balanced LP on locked folds 1–4; few-shot logreg headline, seeds seed·1000+k·100+d, 20 draws). Self-test reproduces RESULTS.md locked values exactly (SSL ep40 target LP 0.9166, rare 0.5529/0.6068/0.6399, ARMD@10 0.7894; online 0.9156 / 0.6483 / 0.8007).
9. **Bootstrap:** Argon resamples pos/neg independently per class → macro means lose cross-label correlation. Now one joint multiplicity vector over query images per iteration (shared by labels and both models). Added a 2-level variant that also resamples the support draws (query-only bootstrap with fixed supports ignores support-set variance). Added degraded guardrail metrics (LP clean→deg, kNN deg-query).

### Finding to flag before the full run (not changed; config as approved)
With τ_t = 0.05 and each image's own clean embedding in the bank, the teacher distribution is ~one-hot on the image itself:
p_t(self) = **0.996** on clean full images (box, 64-img bank; on Kaggle see smoke `p_t_self`, measured on augmented views). The SEED KL then acts mostly
as instance discrimination ("be nearest to your own clean embedding"), carrying little neighbour structure — the mechanism Argon cited for rare classes.
Also, τ_s ≠ τ_t means KL > 0 even when z_s = z_t (floor 5.2e-4 at identity), so KL pushes towards sharper similarities than cosine alone.
Knob `kd.seed_mask_self: true` drops the self entry (neighbour-only KL).
**Decision (Jeet, 18:55): `seed_mask_self: true`** for smoke + full. Kernels rebuilt 18:56.
Bank stats with the real 1,152-image bank (box, clean full images; `localtest/masked_kl_bank_stats.json`):
self included → p_t max 0.965, ~1.2 effective neighbours; **masked → p_t max 0.44, top-5 mass 0.74, ~23 effective neighbours,
top-1 neighbour has identical label set 46 % (random pair 8 %), ARMD queries put 41 % of p_t on ARMD anchors (prevalence 5.2 %)**.
Masked KL finite (0.043 for a noisy student), gradients finite. Kaggle v1 smoke (unmasked) measured p_t_self = 0.956 on augmented views.

## Local checks (CPU, 2026-11-10) — all PASS
See `localtest/unit_checks.json` and the kernel's own `CHECK` lines (`summary.json["checks"]`).

| check | result |
|---|---|
| teacher strict load (180 tensors, 0 missing/unexpected) | PASS |
| teacher reproduces v4 ep20: box 512-cache path vs phase-0 embeddings, locked fold 0 | PASS, cos min 1.000000 |
| teacher via kernel 640-cache path, locked split (evallib) | PASS: LP 0.9213, rare 0.590/0.688/0.724, ARMD@10 0.942 (phase-0: 0.921, 0.589/0.688/0.725, 0.943) |
| teacher on Kaggle (T4, fp16) vs box fp32, all 1,920 | PASS, cos min 0.99998 |
| student strict load + bit-identical to SSL ep40 target (308 tensors) | PASS; Kaggle SSL960 embeddings = SSL-arm embeddings (cos min 1.00000) |
| shapes 960 → 768, L2 norms | PASS |
| teacher/student same FOV: teacher input vs direct warp of same box at 384 | PASS: MAE 0.019–0.021 (norm. units) vs 5° rotation control 0.075–0.105, h-flip control 0.21–0.39 (box); Kaggle 0.010 vs flip 0.30–0.44 |
| degradation student only (C clean, Sx degraded iff α ≥ 0, p=0 → identical) | PASS |
| Procrustes: orthogonal (|WWᵀ−I| 5e-7), cos random head → Procrustes | PASS: box (64 imgs) −0.004 → 0.825; Kaggle (1,152) −0.019 → 0.756 train, 0.662 fold 4 |
| loss sane: identical inputs cos 0, KL = τ-mismatch floor (unmasked 5.2e-4, masked 4.0e-3); random ≫ | PASS |
| tiny overfit (8 imgs, 25 steps) stage 1 & 2: cos and KL both fall | PASS (masked: cos 0.30→0.004 / 0.68→0.003, KL 0.016→0.0036 / 0.027→0.0035) |
| stage 1 backbone incl. BN buffers bit-identical, head moves | PASS |
| BN recal changes running stats, exactly 50 % degraded, model left in eval | PASS (Kaggle: Δmean 0.0064, rel Δvar 0.016; fold-4 embedding cos pre/post 0.995) |
| eval uses the 768-d output | PASS |
| fold 0 (and fold 4) not in training rows / anchor bank | PASS |
| local CPU end-to-end of the smoke kernel | PASS (`localtest/e2e_smoke.log`) |
| locked evaluator self-test reproduces RESULTS.md | PASS (exact) |
Masked re-run of all unit checks: ALL PASS (`localtest/unit_checks_masked.log`).

## Locked fold 0 positives (locked A/B split, n = 384)
ODC 56, ODP 13, **ARMD 20** (Argon's ~20 estimate correct). Others: MH 64, DN 28, LS 10, TSLN 37, MYA 21, BRVO 14, CRVO 6, CSR 7, RS 9, DR 72, ODE 12, Disease_Risk 303.
Caveat (same as SSL arm): locked fold 0 ≠ in-run fold 0; 228/384 locked-fold-0 images are in KD/SSL training folds 1–3 (label-free) and the teacher saw 308/384.
Paired KD-vs-SSL comparison is still fair (same exposure for both), but absolute numbers are optimistic.

## Kaggle runs
- Weights dataset: https://www.kaggle.com/datasets/teamoffirst/offirst-kd-run2-weights (private).
- **Smoke v1 (UNMASKED, superseded)** https://www.kaggle.com/code/teamoffirst/team-offirst-kd-run2-smoke version 1: pushed 17:50, QUEUED ~70 min, ran ~19:00–19:26, COMPLETE, wall 1,620 s, no errors/NaN. Outputs `kaggle_smoke_out_v1_unmasked/`.
  T4, torch 2.11.0+cu130, timm 1.0.30. bs 48, **no grad ckpt**, probe peak 5.69 GiB (train peak 5.66 GiB). 24 steps/epoch.
  Throughput: head stage 71.4 img/s wall (GPU 138), joint 65.6 img/s wall (GPU 87) → 16–18 s/epoch (+~10 s fold-4 val).
  Losses (SMOKE-ONLY): ep1 head cos 0.306→0.302 (first step→mean), KL 0.0334→0.0307; ep2 joint cos 0.290, KL 0.0298; val fold-4 cos clean/deg 0.668/0.635 → 0.681/0.648.
  SMOKE-ONLY dev (in-run fold 0, 3 draws): SSL960 LP 0.903, rare 0.555/0.591/0.645; SSL768-Procrustes 0.905, 0.555/0.591/0.645; KD768 after 2 ep 0.919, c→deg 0.905, rare 0.601/0.643/0.707, kNN 0.864/0.856.
  SMOKE-ONLY locked paired bootstrap KD768(v1, 2 ep) − SSL960: LP +0.010, rare 5/10 +0.044/+0.044, ARMD@10 0.789→0.853 (+0.063, P 1.00), degraded guardrails ≥ −0.005 → rule would pass (`locked_rescore/SMOKE_ONLY_kd768v1_vs_ssl960.*`). Not a result.
- GPU queue: v1 queued ~70 min; a trivial nvidia-smi probe kernel (team-offirst-gpu-queue-probe) pushed 18:49 never left QUEUED; pushing v2 at 19:01 failed with "Maximum batch GPU session count of 2 reached" (queued sessions count). Probe deleted 19:50 (requested). No CLI cancel exists.
- **Smoke v2 (MASKED)** version 2 pushed 19:26, queued.
- **Smoke v2 (MASKED)** completed (wall 1,719 s), all CHECKS pass, no NaN. p_t max/top5 0.44/0.74 (neighbour structure, vs self 0.956 in v1), KL 0.0225→0.0216, cos 0.301→0.290, 60–65 img/s, peak 5.69 GiB bs48 no ckpt. SMOKE-ONLY dev KD768: LP 0.917, rare 0.601/0.641/0.707 (≈ v1). Outputs `kaggle_smoke_out_v2_masked/`.
- **FULL run** https://www.kaggle.com/code/teamoffirst/team-offirst-kd-run2 version 1 pushed 21:27 IST 2026-11-10 (masked, 40 ep, approved by Jeet), RUNNING at push. Est ~1.6 h + setup.
- After it completes: pull outputs to full_out/, then run `locked_rescore/run_after_full.sh` (paired bootstrap vs SSL ep40, decision rule).

## FULL run RESULT (collected 2026-11-10 ~23:45 IST)
Kaggle COMPLETE (wall 2,706 s). Outputs full_out/. Locked paired bootstrap KD768 vs SSL ep40 960 (locked_rescore/kd768_vs_ssl960.*):
LP common 0.9166→0.9316 (+0.015, P .999); LP c→deg 0.9082→0.9285 (+0.020); kNN deg 0.8758→0.8893 (+0.0135, P .878, CI crosses 0);
rare 1/5/10 0.553/0.607/0.640 → 0.602/0.693/0.728 (+0.049/+0.086/+0.088, P 1.0); rare mean 5/10 +0.087; ARMD@10 0.789→0.949 (+0.159, P 1.0).
Vs SSL 768-Procrustes: same picture. Sanity (box SSL vs kernel SSL) identical. Verdict: KEEP KD (all criteria pass).
Caveat: locked fold 0 overlaps training folds (228/384) and teacher saw 308/384; ARMD n=20 positives.

## IDRiD Disease Grading cross-domain eval (2026-11-11, box CPU, eval only)
- Jeet asked for this: "No test on idrid grading set".
- **Data:** IDRiD B. Disease Grading from Kaggle `aaryapatel98/indian-diabetic-retinopathy-image-dataset` (official mirror). Official 413 train / 103 test; counts and label distribution verified.
- **Method:** frozen encoders with the exact kernel preprocessing and degrader (exec'd from kernel source). Box fp32 vs Kaggle embeddings: cosine ≥ 0.99988.
- **Probe:** balanced logreg, C chosen by CV on train. Tasks: DR5, DME3, RDR, and kNN RDR. Paired bootstrap 1000.
- **Results:**
  - Clean: KD−SSL960 shows no detectable difference; every CI includes 0. DR5 AUROC is −0.010, RDR AUROC +0.009, DME AUROC +0.009.
  - Clean probe→degraded test: RDR AUROC is +0.022, CI [+0.001, +0.049], P 0.98 (vs SSL768p: +0.030). This is a hint only, since about 17 metrics were tested with no correction. DR5 QWK is −0.071, and its CI includes 0.
  - Teacher reference: better on DME and RDR, worse on DR5.
- **Caveats:**
  - n=103, with DR grade 1 at only 5 test images.
  - C=0.01 (grid edge) was selected almost everywhere; whether to extend the grid is an open decision.
  - The IDRiD camera matches 277 RFMiD Kowa images, and both datasets are from Nanded, so the domain shift is only partial.
- **Files:** `idrid_grading_eval/` (REPORT.md, results.json, extract.py, evaluate.py, embeddings/).
