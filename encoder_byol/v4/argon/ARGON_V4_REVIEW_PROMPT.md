# Argon review request — BYOL anatomy v4 results (ZEISS retina encoder)

You are Argon. Think at the **highest / deepest reasoning level**. Be concrete and opinionated. We will paste your full answer back to Jeet.

## Context (locked decisions)
- SSL family: BYOL + ConvNeXt-T + ImageNet init on RFMiD-train only (~1536 imgs). Fold 0 held out for eval.
- No lesion-seg crops. Anatomy views = optic disc + macula + random. MH in RFMiD = media haze (not macular hole).
- IDRiD grading is test-only (after dedupe); never for training/probe/threshold.
- Pitch: ~40% of RFMiD's 45 categories defined by disc/macula changes → steer SSL views there while keeping random for vascular/peripheral.
- v4 locked as last planned encoder version, but results may force a short v4.1.

## What we ran (v4)
- Views: fixed 1 disc + 1 macula + 2 random (wide DD 2.5–3.2 only); uniform weak colour on all views; camera blur/JPEG kept.
- Loss: BYOL + light VICReg on projector (λ_var=10, λ_cov=1, weight 0.1).
- Schedule: 120 epochs, eval @70 and 120; batch planned 64 → OOM → 48 + grad checkpoint. T4, ~3.2h, 17.4 img/s.
- Actual view mix logged: ~40% anatomy (20% disc / 20% mac), 50% random, 10% fallback.
- No collapse (std_z≈0.99 at end). BYOL loss 1.31 → 0.019. kNN peaked ep20 (~0.911) then ~0.89.
- VICReg: var 0.60→0.218, cov 0.52→~1.94 (flattened early).

## Headline results
LP = linear-probe mean AUROC on common diseases. Rare = mean of ODC/ODP/ARMD few-shot logreg (no MH), 1/5/10-shot.

| Run | LP mAUROC | DiseaseRisk | Rare 1/5/10 |
|---|---|---|---|
| Supervised ConvNeXt-T | 0.962 | 0.967 | 0.715 / 0.830 / 0.859 |
| ImageNet (A/B split) | 0.900 | 0.932 | 0.542 / 0.572 / 0.611 |
| ImageNet (v4 run) | 0.887 | 0.931 | 0.556 / 0.570 / 0.621 |
| A random crops | 0.898 | 0.927 | 0.617 / 0.701 / 0.730 |
| B anatomy v1 | 0.911 | 0.930 | 0.622 / 0.705 / 0.734 |
| v2 anatomy | 0.921 | 0.940 | 0.598 / 0.679 / 0.707 |
| **v4 ep20 (best kNN)** | **0.928** | **0.950** | 0.608 / 0.686 / **0.742** |
| v4 ep70 online | 0.916 | 0.922 | 0.558 / 0.652 / 0.699 |
| v4 ep120 target-EMA | 0.913 | 0.918 | 0.553 / 0.640 / 0.685 |

Longer training hurt: ep20 > ep70 > ep120 on almost every metric.

## Group means (LP)
- Optical disc (ODC/ODP/ODE): Sup 0.921 | B 0.811 | v2 0.819 | **v4 ep20 0.868** | v4 ep120 0.852
- Macular (ARMD/MYA/TSLN): Sup 0.983 | B 0.975 | v2 0.979 | v4 ep20 0.963 | v4 ep120 0.956
- Vascular (DR/BRVO/DN): Sup 0.973 | B 0.938 | v2 0.943 | v4 ep20 0.936 | v4 ep120 0.914
- Media haze (MH): Sup 0.969 | B 0.975 | v2 0.977 | v4 ep20 0.978 | v4 ep120 0.962

## Disease-level highlights vs B
Wins: ODP LP +0.09, ODE LP +0.04 and few-shot up a lot, ODC LP +0.03, DR slightly up.
Losses: ARMD few-shot −0.15 to −0.18 at ep120 (even ep20 −0.06 at 10-shot) — main reason rare headline fails; DN LP −0.06 late; BRVO/TSLN softer.
Multi-region (disc+mac concat) helps disc (ODP→~0.80) but hurts DN; overall wash.

## Protocol caveats
- v4 recomputed iterstrat fold 0 instead of reusing A/B `split_5fold_seed42.csv` → ImageNet LP 0.887 vs 0.900; LS dropped from common mean; ~±0.01 noise vs A/B/v2.
- Supervised saw all labels → rare few-shot is a ceiling, not fair SSL rival.
- ODP only 13 test positives (noisy).

## Our working diagnosis
Limiting factor = small RFMiD-train + overlong schedule (SSL overfit), not backbone. Proposed: re-score ep20–40 on original check1 split; v4.1 with much shorter schedule / early stop; keep more random views for ARMD/DN.

## What we need from you
1. Do you agree with the diagnosis? What are we missing?
2. Rank the top 3 highest-EV next experiments for a ZEISS pitch (deadline ~Nov 15). Be ruthless about compute.
3. Specifically: keep v4 ep20 as the story, or fix ARMD first? Should VICReg stay?
4. How should we talk about “anatomy views help disc but hurt macula” without sounding like we failed?
5. Any literature-backed tweak we skipped that still fits T4 + ~1.5k images?

Answer in clear sections. Numbers first. No filler.
