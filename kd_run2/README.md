# KD Run 2: MobileNetV3 student distilled from the BYOL v4 ConvNeXt-T teacher (+ cross-domain evals)

Asymmetric KD: frozen teacher = BYOL anatomy v4 ConvNeXt-T (ep20 target, 768-d) sees the **clean** view at 384; student = MobileNetV3-Large initialised from the SSL ep40 target (`encoder_mnv3_ssl/`), with a 960→768 head, sees the same view at 512 with the portable-camera degrader (`kd_imaging/`). Loss = (1 − cos) + λ·τ²·KL against a static clean anchor bank (self-entry masked → neighbour-only KL), two stages (head-only → joint), Procrustes head init, BN recalibration at the end. 40 epochs on one Kaggle T4 (45 min wall). Trained on RFMiD folds 1–3.

- `WORKFILE.md` — design, Argon review fixes, all unit checks, Kaggle runs, results log (source of truth).
- `kernels/{full,smoke}` — Kaggle kernels (`build_kernel.py` + `src/header_config.py` + `src/kd_tail.py`); `kernels/probe` — GPU-queue probe.
- `test_units.py`, `localtest/` — CPU unit checks, teacher reproduction, masked-KL bank stats, local end-to-end smoke.
- `full_out/` — full-run logs, summary, history, embeddings (`kd768`, `kd960`, `ssl_ep40_960`, `ssl_ep40_768procrustes`, `teacher`, each clean + degraded) and `procrustes_W.npy`.
- `kaggle_smoke_out_v1_unmasked/`, `kaggle_smoke_out_v2_masked/` — smoke runs (SMOKE-ONLY numbers).
- `locked_rescore/` — locked A/B CPU re-score + paired bootstrap (`paired_bootstrap.py`, `run_after_full.sh`).
- `idrid_grading_eval/` — IDRiD Disease Grading cross-domain eval (`REPORT.md`, `results.json`, `extract.py`, `evaluate.py`, embeddings).
- `odir5k_eval/` — ODIR-5K cross-domain eval (`build_labels.py`, `extract.py`, `evaluate.py`, `results.json`, per-eye labels, embeddings, `samples/` with 5 example montages + `samples.md`).
- `weights_ds/` — SHA256SUMS + Kaggle dataset metadata for the teacher / student-init weights (weights themselves in the release).
- `argon/` — external review prompt/replies.

## Key results
**RFMiD, locked A/B split, paired bootstrap KD768 − SSL ep40 960** (`locked_rescore/kd768_vs_ssl960.*`):
LP 0.917 → **0.932** (+0.015, P 0.999); LP clean→deg 0.908 → 0.929 (+0.020); rare ODC/ODP/ARMD 1/5/10 0.553/0.607/0.640 → **0.602/0.693/0.728** (+0.049/+0.086/+0.088, P 1.0); ARMD@10 0.789 → **0.949**; kNN deg-query +0.014 (CI crosses 0). Same picture vs SSL 768-Procrustes. Verdict: keep KD. Caveat: locked fold 0 overlaps label-free training folds (228/384); ARMD n = 20.

**IDRiD grading (n = 103 test, frozen encoders, linear probe)** (`idrid_grading_eval/REPORT.md`): clean — no detectable KD−SSL difference (DR5 AUROC 0.814 vs 0.824, RDR 0.925 vs 0.915, DME 0.893 vs 0.885; all CIs include 0). Clean probe → degraded test: RDR AUROC +0.022 [+0.001, +0.049] (hint only, ~17 metrics uncorrected).

**ODIR-5K (1,982 test eyes / 1,015 patients, patient-level split, per-eye keyword labels)** (`odir5k_eval/results.json`):

| metric | KD768 | SSL960 | SSL768p | Teacher (ref) |
|---|---|---|---|---|
| LP macro, clean | 0.872 | 0.872 | 0.868 | 0.877 |
| LP macro, clean→deg | **0.860** | 0.848 | 0.848 | 0.859 |
| LP macro, deg→deg | 0.871 | 0.869 | 0.866 | 0.863 |
| AMD 1/5/10-shot | **0.622/0.721/0.772** | 0.591/0.642/0.681 | = SSL960 | 0.644/0.761/0.806 |
| Glaucoma 1/5/10-shot | 0.625/0.718/0.773 | 0.621/0.694/0.735 | = SSL960 | 0.649/0.747/0.799 |
| Myopia 1/5/10-shot | 0.855/0.974/0.985 | 0.817/0.944/0.972 | = SSL960 | 0.900/0.985/0.992 |
| kNN AMD clean / deg-query | **0.826 / 0.817** | 0.760 / 0.725 | 0.767 / 0.716 | 0.871 / 0.842 |

Few-shot and kNN gains have P(Δ>0) ≥ 0.99. Takeaway: KD does not help with plenty of labels (full probes tie) but consistently helps label-scarce few-shot and similar-case retrieval, incl. on degraded queries; the student closes ~60–75 % of the SSL→teacher few-shot gap. Caveats: ODIR labels are keyword-derived (noisy); degradation is simulated; many metrics, no multiplicity correction. Nearest ODIR↔RFMiD cosine < 0.94 (no duplicate overlap found).

## Reproduce
- Kaggle (T4): attach `awsaf49/rfmid-train-dataset` + a dataset holding the two `weights_ds` files from the release (teacher, student init); push `kernels/full`. Then `bash locked_rescore/run_after_full.sh`.
- Cross-domain evals (CPU, eval only, run from the repo root with the release weights placed at `kd_run2/weights_ds/` and `kd_run2/full_out/out/kd_run2/`): `python kd_run2/idrid_grading_eval/extract.py && python kd_run2/idrid_grading_eval/evaluate.py`; for ODIR-5K `build_labels.py` → `extract.py` → `evaluate.py` → `samples/make_samples.py`. Dataset images are **not** in this repo: IDRiD from Kaggle `aaryapatel98/indian-diabetic-retinopathy-image-dataset` (B. Disease Grading); ODIR-5K from its public release. Point the scripts' data paths at your copies.
- Box-specific absolute paths in logs are shown as `<workdir>/...` (= repo root layout).

## Weights (GitHub Release `kd-run2`)
Student checkpoints hold `student` (state dict incl. 960→768 head), `epoch`, `config`. **Recommended: `kd_run2_kd_final.pth`** (ep40, after BN recalibration; the 768-d output is what all evals above use). `last_pre_recal` = ep40 before BN recal; `ep010`–`ep030` intermediate (pre-recal). `smoke_v2_*` = Kaggle 2-epoch smoke, `localtest_*` = CPU e2e smoke (not for use). `teacher_byol_v4_ep20_target_convnext_tiny.pth` = `target_backbone` of `encoder_byol_anatomy_v4_ep020.pth`; `student_init_mnv3_ssl_ep40_target.pth` = `target_backbone` of `encoder_mnv3_ssl` ep40.

| file | size | sha256 |
|---|---|---|
| [kd_run2_ep010.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/kd-run2/kd_run2_ep010.pth) | 15 MB | dbc5ea54505bea1f13e06c9f29996ad55a882caa23e561a5e40740e59aa271cf |
| [kd_run2_ep020.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/kd-run2/kd_run2_ep020.pth) | 15 MB | 78fd0509fbd232ebecaa795da43c3e70a5f3997fe4b9249059ba86812bc9bd73 |
| [kd_run2_ep030.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/kd-run2/kd_run2_ep030.pth) | 15 MB | e1bb0d8a0e323fa73614c7d9c39f52bfc7df0949a6dbfed051918b2d3241d9b4 |
| [kd_run2_kd_final.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/kd-run2/kd_run2_kd_final.pth) | 15 MB | 8bc239d22ffc4e786820f2151c99380a754bef930e42edea65022e1246f2096d |
| [kd_run2_last_pre_recal.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/kd-run2/kd_run2_last_pre_recal.pth) | 15 MB | 15dac9bff759e51ea369e4e8b657603480c19fb06fcd9b6afc33e28effbe2314 |
| [kd_run2_localtest_kd_final.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/kd-run2/kd_run2_localtest_kd_final.pth) | 15 MB | 2854f220cd93a75fdb2ba79d2a429cf04fcc57345027b51b4823af172d443066 |
| [kd_run2_localtest_last_pre_recal.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/kd-run2/kd_run2_localtest_last_pre_recal.pth) | 15 MB | cd11b633dcb06d6968f8b7bb3cdb24b7b95b34ee823808f1a40a5b87b3295064 |
| [kd_run2_smoke_v2_kd_final.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/kd-run2/kd_run2_smoke_v2_kd_final.pth) | 15 MB | 8e40b00490fc3979e31d2ad45967adfa89ec594078615d5a78111cd159beed48 |
| [kd_run2_smoke_v2_last_pre_recal.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/kd-run2/kd_run2_smoke_v2_last_pre_recal.pth) | 15 MB | b487a2167685edd24a3d93a456b7c97edd215164a88753f56d0c34342a3ae729 |
| [student_init_mnv3_ssl_ep40_target.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/kd-run2/student_init_mnv3_ssl_ep40_target.pth) | 12 MB | 79b3a0e1d536446bd3c82dbeef7c3d1ddb3f10acdaad0444dbebe228a6731215 |
| [teacher_byol_v4_ep20_target_convnext_tiny.pth](https://github.com/Mayank-gitspace/OFFIRST/releases/download/kd-run2/teacher_byol_v4_ep20_target_convnext_tiny.pth) | 111 MB | ed95b4de631bbeee1776775ec8d309313bcdd142de03609ab9c8185e40cde784 |
