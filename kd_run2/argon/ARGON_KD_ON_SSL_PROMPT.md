I need a critical, evidence-based review of a planned knowledge-distillation (KD) run before I spend GPU time. Be blunt; tell me if the plan is wrong. Cite papers where you can, and mark clearly what is evidence vs. your heuristic.

CONTEXT (retinal fundus screening, hackathon prototype due Nov 15 2026, ~5 days left; goal = small offline phone model robust to cheap handheld cameras)
- Data: RFMiD (~1,920 train images, multi-label, 5 folds; fold 0 = locked dev/test). Only ~1.1–1.5k unlabeled images for SSL.
- Teacher: ConvNeXt-T trained with BYOL (anatomy-aware crops: optic disc + macula + random locals), epoch-20 target encoder, 768-d. Locked fold-0 scores: linear-probe mean AUROC 0.921 (11 common diseases); rare few-shot (ODC/ODP/ARMD, 1/5/10-shot) 0.589/0.688/0.725; ARMD@10-shot 0.943.
- Anatomy baseline B (supervised reference): LP 0.911, rare 0.622/0.705/0.734, ARMD@10 0.969.
- Student so far: MobileNetV3-Large @512 input, trained directly with BYOL on clean + synthetic Visuscout-degraded views (classical degrader: blur, vignetting, colour cast, noise, JPEG), ImageNet init, folds 1–3 only (1,152 imgs), 40 epochs, batch 48 + grad checkpointing on a Kaggle T4. Best = ep40 target. Locked fold-0: LP 0.917, rare 0.553/0.607/0.640, ARMD@10 0.789. Clearly more robust than ImageNet MNV3 on degraded retrieval, but weak on rare diseases (barely above ImageNet MNV3: 0.543/0.562/0.618).
- Labels are used ONLY for frozen linear/few-shot probes afterwards, never to train the encoder.

PLANNED KD RUN ("KD on top of SSL")
- Student init = the MNV3 SSL ep40 checkpoint above (not ImageNet). Input 512.
- Teacher frozen; its clean-image 768-d embeddings precomputed and cached (teacher at its own 384 input on clean images).
- Student head: GAP -> linear projector to 768-d. Loss = 1 - cosine(student(x), cached_teacher_clean_z). Each batch 50% clean, 50% degraded (degraded copy of same image must match teacher's CLEAN embedding = asymmetric KD). Reason for mixing: MobileNet BatchNorm running stats.
- Same folds 1–3 image pool as the SSL arm, ~40 epochs, short smoke test first.
- Decision rule: keep KD model only if it beats SSL ep40 on the same locked split, especially rare few-shot / ARMD, without losing common LP or degraded robustness. Otherwise ship SSL ep40.

QUESTIONS
1. Is KD-on-top-of-SSL init sound, or will the SSL init bias/limit it vs ImageNet init? Any literature (CompRess, SEED, DisCo, BINGO, etc.) on SSL-pretrained student + embedding distillation?
2. Is pure cosine-to-cached-teacher-embedding the right loss, given our main weakness is rare-disease few-shot? Would relational/similarity-distribution KD (e.g. SEED/CompRess-style over a queue, or RKD) preserve the teacher's fine-grained neighbourhood structure better for rare classes? Cost/benefit given 5 days?
3. Cached teacher embeddings mean the teacher never sees the same augmentation (crop/flip) the student sees. Does that mismatch hurt? Should teacher embeddings be cached for a few fixed augmentations, or should student views be only mild (no random crops) to stay aligned?
4. Resolution mismatch: student 512 vs teacher 384. Any problem?
5. Should the student train on folds 1–3 (match SSL arm) or folds 1–4 (match teacher), given the decision is "does KD add to SSL ep40"?
6. 50/50 clean/degraded mix and ~40 epochs: reasonable? Learning rate / warmup advice when starting from an SSL checkpoint (risk of wrecking features early)? Should we keep a small SSL (BYOL) loss alongside KD?
7. Is the decision rule fair and statistically sensible with only 384 fold-0 images (rare classes have very few positives)? What margin / paired test should count as "KD adds value"?
8. Anything that makes this plan a waste of time, and what you would do instead in 5 days.

Give: (a) verdict on the plan, (b) concrete changes ranked by expected impact vs effort, (c) recommended final config, (d) risks.
