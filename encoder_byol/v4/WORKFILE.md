# OFFIRST BYOL encoder — workfile (v4 locked)

**Locked:** 2026-11-05 (Asia/Calcutta) · Jeet: “last version”  
**Status:** full Kaggle COMPLETE · see RESULTS.md  
**Family:** BYOL + ConvNeXt-Tiny (ImageNet) — do not switch SSL family

## Paths
- Config draft: `CONFIG_DRAFT.md` (this folder)
- This workfile: `WORKFILE.md`
- Argon survey: `/workspace/zeiss/reports/argon_encoder_lit_survey.md`
- Prompt to Argon: `/workspace/zeiss/reports/argon_encoder_lit_prompt.txt`
- Parent kernel: `/workspace/zeiss/encoder_byol/team_offirst_encoder_byol.py`
- v2 reference: `/workspace/zeiss/encoder_byol/v2/`
- Detector / boxes: `/workspace/zeiss/cropper_extension` (+ v2 `box_table.csv` / stage1)

## Locked v4 knobs (vs v2)
| Knob | v4 |
|---|---|
| Locals | 1 disc + 1 macula + 2 random (fixed slots) |
| Disc / mac DD | **2.5–3.2 only** (no tight 1.2–1.5) |
| Colour | uniform weak jitter all views: b/c 0.2, sat 0.15, hue 0.03, p=0.8 |
| Camera augs | blur_p 0.3, jpeg_p 0.5 q 40–95, cam_bright_p 0.5 |
| Epochs | 120; forced eval @ 70 & 120 |
| Loss | BYOL + VICReg var/cov on projector z (λ_var=10, λ_cov=1, w=0.1) |
| Warmup | 8 ep; knn_every 5; ckpt_every 10 |
| Headline few-shot | ODC, ODP, ARMD (**MH excluded** — media haze) |
| Eval extra | multi-region concat probe (global+disc+mac), eval-only |
| Data | RFMiD-train folds 1–4 SSL; fold 0 held out; ImageNet OK; no RETFound |

## Success bar (full run)
- Probe mAUROC ≥ B (0.911), ideally near v2 (0.921) without colour shortcuts
- Rare 1/5/10 ≥ B check1: **0.622 / 0.705 / 0.734**
- No collapse; report ep70 vs ep120

## Reference papers (cite only after verify)
Core method
- Grill et al. (2020). *Bootstrap Your Own Latent*. NeurIPS. arXiv:2006.07733 — BYOL; colour-jitter importance.
- Caron et al. (2020). *SwAV*. NeurIPS. arXiv:2006.09882 — multi-crop.
- Caron et al. (2021). *DINO*. ICCV — local↔global semantics.
- Bardes et al. (2022). *VICReg*. ICLR. arXiv:2105.04906 — variance/covariance terms.
- Bardes et al. (2022). *VICRegL*. NeurIPS — local features (optional later RoI match).

Fundus / medical SSL & anatomy
- Azizi et al. (2021). *Big Self-Supervised Models…* ICCV. arXiv:2101.05224 — ImageNet→medical SSL.
- Pachade et al. (2021). *RFMiD*. MDPI Data 6(2):14 — dataset; disc/macula disease share (~40% pitch).
- Li et al. (2021). Rotation-oriented collaborative SSL for retinal diagnosis. IEEE TMI.
- Xiao et al. (2021). *LooC* — what should not be contrastive. ICLR. arXiv:2008.05659 — colour vs invariance.
- Kang et al. (2020). Decoupling representation and classifier for long-tailed recognition. ICLR. arXiv:1910.09217 — eval readout / few-shot.
- Graham (2015). Kaggle DR 1st place — local average colour subtraction (parked aug).
- Zhou et al. (2023). *RETFound*. Nature — **do not use** (MAE/ViT; data scale; frozen few-shot weak for us).
- Khosla et al. (2020). *SupCon*. NeurIPS. arXiv:2004.11362 — **do not** SupCon on 12 commons (supervision collapse).

Unverified from Argon (do not cite until checked)
- arXiv:2608.26686; some Holmberg/Hervella/Taleb/Yellapragada attributions — see survey caveats.

## Explicit non-goals (v4)
- No MAE / RETFound / DINOv2 ViT on N=1536
- No SupCon on common labels
- No tight DD crops under GAP-aligned BYOL
- No heavy trained disc/lesion segmenter
- No ZEISS name in public OFFIRST repo commits

## Run log
| When | What | Result |
|---|---|---|
| 2026-11-05 | Config locked by Jeet | this file |
| 2026-11-05 15:55 IST | Kernel built (`build_kernel.py` → `kernels/{smoke,full}`), Argon VICReg snippet integrated, `test_vicreg.py` | PASS |
| 2026-11-05 15:58 IST | Local CPU smoke (16 imgs, 2 ep, bs 4) — `SMOKE.md`, `localtest/` | runs end-to-end; anchors random .50 / disc .22 / mac .22 / fallback .06; VICReg var .65 cov 2–2.5; ep1 + final + multi-region evals written |
| | Kaggle GPU smoke | pending (needs OK) |
| | Full Kaggle | pending (after smoke) |

| 2026-11-05 16:20 IST | Full Kaggle Save & Run All v1 | started RUNNING — https://www.kaggle.com/code/teamoffirst/team-offirst-encoder-byol-anatomy-v4 |
| 2026-11-05 18:04 IST | Live kaggle kernels status | still **RUNNING** (no output artifacts yet; collect routine owns poll) |
| 2026-11-05 19:50 IST | Full Kaggle v1 **COMPLETED** (wall 3.24 h, bs48+grad-ckpt after OOM probe) | outputs in `v4/full_out/` (summary.json, run.log, ckpts ep10–110/last/best, embeddings). ep120 target LP 0.913, rare 0.553/0.640/0.685; ep70 online LP 0.916, rare 0.558/0.652/0.699; best ep20 LP 0.928, rare 0.608/0.686/0.742. Rare FAILS B bar at ep70/120. Summary: `full_out/RESULTS_SUMMARY.md` |
| 2026-11-05 19:45 IST | Full Kaggle COMPLETE · outputs in `full_out/` · RESULTS.md | target probe **0.913** (≥B 0.911, <v2 0.921); rare 1/5/10 **0.553/0.640/0.686** (misses B); ep70>ep120; best_ep20 online probe 0.928 |
| 2026-11-05 22:38 IST | Phase 0 re-score of ep20 on locked split_5fold_seed42 (CPU, box) — `phase0_rescore/RESULTS.md` | ep20 target LP **0.921**, rare **0.589/0.688/0.725** (B 0.622/0.705/0.734); ARMD 10-shot **0.943 vs B 0.969** (−0.026, paired CI [−0.049,−0.008]) → gate says **RUN v4.1**. L2-mean multi-region fails gate (DN −0.033, ARMD down) → keep f_full |
