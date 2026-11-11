New task in this same OFFIRST thread: help me write the training code for the direct small-model SSL run (Run 1 from your survey). Attached: our parent BYOL v4 Kaggle script (parent_team_offirst_encoder_byol_anatomy_v4.py, ConvNeXt-T) and our classical Visuscout degrader (degrade.py, fitted params already as defaults).

DECIDED BY JEET (do not change):
- Method: BYOL (same family as our ConvNeXt-T teacher). No KD in this run, no disease labels in training.
- Backbone: timm MobileNetV3-Large, ImageNet init.
- Global input resolution: 512 (we measured on IDRiD lesion masks that median microaneurysm diameter is ~1.9 px at 384 vs ~2.6 px at 512; 2.3% vs 0.5% go sub-pixel).
- Views: reuse v4 anatomy recipe (2 globals + 4 locals: 1 disc + 1 macula + 2 random, disc/mac 2.5–3.2 DD, uniform weak colour jitter, camera blur/JPEG/bright kept) AND add the Visuscout degradation from degrade.py into the view pipeline.
- VICReg OFF (we killed it in v4).
- Runs on Kaggle (account teamoffirst), T4 x2 (or single T4), Internet on, same RFMiD train dataset and same evaluation as parent (5 folds, linear probe mAUROC, few-shot ODC/ODP/ARMD, kNN monitor).

OPEN CHOICES — propose a default for each, say why, and mark it clearly as YOUR suggestion (Jeet decides):
1. How degradation enters: which views get degraded (e.g. online vs target branch asymmetry, per-view probability), and continuous severity (e.g. alpha ~ U(0,1) blend between clean and fully degraded, or per-parameter jitter around fitted values).
2. Speed: degrade.py is numpy/OpenCV at native resolution. Should it run on the cached square (cache_size must rise from 512 to e.g. 640/768 for 512 globals) in DataLoader workers, or be re-implemented in torch on GPU? Give the T4 throughput consequence honestly (estimate, not a promise).
3. Local crop size for MobileNetV3 at 512 globals (parent used 160 locals with 384 globals).
4. Projector/predictor sizes for a 960-d MobileNetV3 GAP (parent: 2048 hidden, 256 out) and where the 768-d deployable linear layer lives (or whether to just probe the 960-d GAP).
5. Optimizer/LR/batch/epochs/EMA tau for MobileNetV3 + BYOL (BN, not LN; hardswish). Note: in v4 longer training hurt; ep20 was best for ConvNeXt-T. Suggest epoch budget and eval checkpoints. Also BN handling (SyncBN / shuffle-BN not needed for BYOL? say).
6. Inner validation: probes/early-stopping on folds 1–4 inner split; fold 0 is a disclosed development split only. IDRiD must not be touched.
7. Add an eval that also runs probes on degraded copies of the eval fold (clean vs degraded mAUROC), plus collapse diagnostics (effective rank, paired clean-degraded cosine minus random-pair cosine).

DELIVER:
- A concise design table for choices 1–7 (your suggestions, flagged as such).
- The code: either the full modified script or exact drop-in replacement functions/blocks for the parent (CONFIG block, model class, make_views + degradation, cache size, train_step if changed, extra eval), with filenames/headers. Must run on Kaggle T4 with AMP. Keep the parent's data guards and output format so our existing tooling works.
- A short smoke-test plan (tiny epochs) and the main risks.
Don't claim measured performance or speed you haven't measured; label estimates.
