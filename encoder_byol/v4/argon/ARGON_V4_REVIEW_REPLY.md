<!-- Archived Gemini review. Page extraction converted the formulas to plain text. -->

Executive Verdict (Numbers First)

v4 ep20 is your strongest encoder yet—lock it as your anchor checkpoint: Relative to its own split’s ImageNet baseline (0.887 LP / 0.931 DiseaseRisk / 0.621 10-shot), v4 ep20 delivers +0.041 LP mAUROC (0.928), +0.019 DiseaseRisk (0.950), and +0.121 Rare 10-shot (0.742). By contrast, B gained only +0.011 LP (0.900 → 0.911) and v2 gained +0.021 LP (0.900 → 0.921) over their split's ImageNet baseline.

Your 120-epoch schedule is 5x too long for N = 1,536: At batch size 48, one epoch is 32 optimizer steps. ep20 is 640 gradient updates (continual domain adaptation from ImageNet); ep120 is 3,840 updates, driving BYOL loss from 1.31 → 0.019 and causing severe invariance overfitting (LP 0.928 → 0.913, DiseaseRisk 0.950 → 0.918).

Compute reality: At 3.2h for 120 epochs (1.6 min/epoch on T4), a 25-epoch run takes 40 minutes. You have plenty of compute before Nov 15 if you stop training past epoch 30.

## 1. Diagnosis Review: What You Got Right and What You Are Missing

Your diagnosis that small N (1,536 images) + an overlong schedule causes SSL overfitting is directionally right, but it misses four mechanistic bugs explaining why ODC/ODP/ODE surged (0.811 → 0.868) while ARMD few-shot (−0.06 at ep20, −0.18 at ep120) and late DN (−0.06) degraded:

**The Cross-Anatomy False-Positive Trap (Disc ↔ Macula Alignment):**
In v4, each image produces 1 disc + 1 macula + 2 wide random views. Whenever BYOL pulls the disc view and the macula view of the same eye toward a shared representation, the predictor must discard information present in one region but absent in the other. Because focal macular drusen and RPE mottling (ARMD) are invisible inside an optic disc crop, aligning z_disc with z_macula actively trains the encoder to be invariant to focal macular lesions. The only features shared between a disc crop and a macula crop of the same eye are global choroidal pigmentation (TSLN/MYA), background illumination, and camera sensor artifacts.

**Shortcut Learning via Weak Color + Preserved Camera Blur/JPEG:**
Contrastive and self-distillation methods rapidly exploit color histograms and imaging artifacts when color distortion is weak. On 1,536 fundus images, keeping camera blur/JPEG while applying uniform weak color jitter allows the network after ~600 steps (ep20) to minimize BYOL loss (→ 0.019) by memorizing patient/camera chromatic signatures rather than pathology semantics.

**Projector/Predictor Warmup vs. Backbone Overwriting:**
You initialized the ConvNeXt-T backbone from ImageNet (0.887 LP, where mid-level texture filters already detect drusen/exudates reasonably well), while the BYOL projector and predictor initialized randomly. For the first ~20 epochs, gradients align the projector to retinal geometry (boosting Optic Disc from 0.811 → 0.868). Past ep20, backpropagated invariance gradients overwrite ConvNeXt’s fine spatial filters in Stages 3 and 4, eroding ARMD and DN.

**Fold-0 Split Drift Is Hiding Your True Gain:**
Recomputing iterstrat fold 0 lowered the ImageNet LP baseline by −0.013 (0.900 → 0.887) and dropped LS from the common mean. On a delta basis over the exact same split, v4 ep20 (+0.041 LP over ImageNet) is nearly double the gain of v2 (+0.021 LP).

## 2. Top 3 Highest-EV Experiments Before Nov 15 (Ranked by ROI per T4-Hour)

| Rank | Experiment | T4 Wall-Clock | Expected Impact | Goal |
|---|---|---|---|---|
| 1 | Lock Split (split_5fold_seed42.csv) + Multi-Stage Feature Readout (Stage 3 + Stage 4) on v4 ep20 | 15–20 mins (Eval only) | +0.01 LP consistency; +0.04 to +0.07 on ARMD/DN few-shot | Eliminate split noise and recover fine macular texture without retraining |
| 2 | v4.1 Short Run (25–30 ep): Asymmetric Local-to-Global View Routing + Drop VICReg + Freeze Stem | 40–45 mins (1 run) | Retain 0.868 Disc LP while recovering ARMD to B/v2 levels (0.75+ rare 10-shot) | Fix disc ↔ macula feature erasure and stop SSL over-convergence |
| 3 | Dimensionality-Preserving Multi-Region Pooling (768-d Weighted/Max Pool vs. 1536-d Concat) | 15 mins (Eval only) | Fix the disc+mac concat regression on DN while keeping ODP ≈ 0.80 | Give the hackathon judges an inference recipe that wins across both disc and periphery |

### Details on Rank 1 (Do this today — Zero Training Compute)

Re-run linear probe and 1/5/10-shot evaluation for v4 ep20 (and ep10/ep15/ep25/ep30 if saved) on the original split_5fold_seed42.csv so your table is 100% apples-to-apples against A, B, and v2.

Extract features from Stage 3 (384-d) + Stage 4 (768-d) (L2-normalized before concatenation, or tested separately for few-shot ARMD/DN). In ConvNeXt-T, Stage 4 operates at stride 32 (7×7 spatial grid for 224×224 input) with 7×7 depthwise kernels, which smooths out tiny drusen (ARMD) and microaneurysms/nodules (DN). Stage 3 (stride 16, 14×14 grid) preserves the local high-frequency ImageNet filters before the BYOL projector forces global invariance.

### Details on Rank 2 (v4.1 — Single 45-Minute Training Run)

If you run one final training job (v4.1), make exactly four surgical changes:

- Schedule: Train for 30 epochs total, evaluating every 5 epochs (ep5, 10, 15, 20, 25, 30). Set base EMA momentum τ_base = 0.996 annealing to 1.0 at ep30 so the target network evolves conservatively.
- Asymmetric Local-to-Global Pairing (Critical for ARMD):
  - Pass only the 2 wide random views (DD 2.5–3.2) through the EMA target network.
  - Pass the 1 disc + 1 macula + 2 wide random views through the online network, predicting only the wide target views.
  - Never pair disc directly with macula. A wide DD 2.5–3.2 view contains both the optic nerve head and the fovea/macula, so predicting wide from disc and wide from macula teaches part-to-whole context without forcing the macula representation to discard drusen.
- View Ratio: Shift slightly toward wide context: e.g., per step or epoch, keep the two global wide views (DD 2.5–3.2) and sample 1 anatomy view (50% disc / 50% macula across steps, or 1 disc + 1 macula strictly anchored to the wide views).
- Drop VICReg (weight = 0). (See Section 3 below.)

### Details on Rank 3 (Fix Multi-Region Inference in 15 Minutes)

Your note that "Multi-region (disc+mac concat) helps disc (ODP → ~0.80) but hurts DN; overall wash" has a straightforward statistical cause:

Concatenating f_disc + f_mac doubles the feature dimension (768 → 1536) while omitting the peripheral/vascular field where DN and BRVO manifest. In 1/5/10-shot logistic regression, doubling D without L2 normalization or PCA halves the signal-to-noise ratio per coordinate.

Instead of concat(f_disc, f_mac), evaluate L2-normalized feature averaging (0.5 * f_full + 0.25 * f_disc + 0.25 * f_mac) or element-wise max-pooling across the 3 L2-normalized 768-d vectors. That keeps dimensionality at 768-d, preserves peripheral DN features via f_full, and retains the high-resolution f_disc signal for ODP.

## 3. Strategic Calls: v4 ep20 vs. Fixing ARMD, and VICReg Verdict

Keep v4 ep20 as your primary story right now; treat v4.1 purely as an upside shot.
Do not let a single noisy rare class (ARMD few-shot) derail the headline: v4 ep20 achieves 0.928 LP mAUROC, 0.950 DiseaseRisk, 0.868 Optic Disc LP (+0.057 over B), and 0.742 Rare 10-shot (best across all SSL runs). Once re-scored on split_5fold_seed42.csv, if v4 ep20 holds ~0.935+ LP, the narrative is already won. Run v4.1 (45 mins) only as a controlled ablation to show you understood why ep120 and ARMD dipped.

**Kill VICReg immediately.**

Look at your logs: std_z ≈ 0.99 (BYOL’s stop-gradient + predictor + EMA already prevents collapse), while var dropped (0.60 → 0.218) and cov rose and flattened early (0.52 → 1.94).

Why did cov hurt you? Your batch size is 48 (with 4 views per image, that is only 12 to 48 distinct crops per batch), while your projector dimension D is much larger than 48. An empirical covariance matrix computed on B=48 samples has rank at most 47. Penalizing off-diagonal covariance in a low-batch regime forces samples within the same mini-batch—including multiple views of the same eye or eyes sharing macular drusen—to become orthogonal across projector coordinates. That directly corrupts fine-grained cluster geometry for ARMD and DN.

## 4. How to Frame "Anatomy Views Help Disc But Soften Macula" for the Hackathon Pitch

First, check the table: you did not fail on the macula.

At v4 ep20, Macular Group LP is 0.963 (vs. Supervised ceiling 0.983). Considering the fold-0 split shift cost ImageNet −0.013 LP overall, 0.963 vs. B's 0.975 is virtually flat, whereas Optic Disc LP jumped by +0.057 (0.811 → 0.868) and DiseaseRisk reached 0.950.

Frame the finding around two concrete principles that resonate with medical optics/imaging engineers judging the hackathon:

**Solving the Spatial Area Bottleneck (Why Standard SSL Fails on the Optic Nerve Head):**

- In standard 45°–50° fundus photography, the optic disc occupies less than 4% of pixel area, whereas macular/tessellation changes (TSLN, MYA, MH) span broad central/global regions.
- Standard random-crop SSL (Run A) already gets ~0.97 on Macular and Media Haze because random crops naturally sample background/central texture, but it stalls at ~0.80 on Optic Disc diseases (ODC, ODP, ODE) because random crops rarely frame the neuroretinal rim at consistent scale.
- Anatomy-guided SSL (v4 ep20) breaks this ceiling, lifting Optic Disc LP to 0.868 (+5.7% over B, +9.0% on ODP) while maintaining 0.963 on Macular and 0.978 on Media Haze.

**Morphological vs. Textural Invariance (Why Early Stopping at ~650 Steps Matters):**

- Present the ep20 → ep70 → ep120 trajectory as a core scientific finding on small-cohort continual SSL (N ≈ 1.5k):
- Optic disc pathologies (ODC, ODE, ODP) are geometric/structural (cup-to-disc ratio, rim elevation, pallor contour) and benefit from strong multi-view spatial alignment.
- Macular pathologies (ARMD) are micro-textural/chromatic (drusen, RPE mottling) and rely on high-frequency filters already present in the pretrained backbone.
- Stopping at the kNN peak (ep20, ~40 minutes on a single T4) captures the geometric adaptation for the optic disc before prolonged multi-view invariance smooths out micro-textural macular features. For the pitch, this proves that online kNN validation accurately detects the optimal stopping point without downstream labels, cutting compute cost by 6x.

## 5. Three Literature-Backed Tweaks for ConvNeXt-T + Small Fundus SSL (N ≈ 1.5k, 1x T4)

If you do a 40-minute v4.1 run or post-hoc probe refinement, these three techniques from small-data ConvNeXt and medical SSL literature fit your exact constraints:

**Freeze the ConvNeXt Patchify Stem (downsample_layers[0]) + Stage 1 During Short SSL:**
ConvNeXt’s 4×4 stride-4 patchify stem acts like a ViT patch projection. In small-dataset continual SSL (N = 1,536), random projector gradients in the first 5 epochs destabilize early Gabor/chromatic filters in the stem (which are critical for tiny drusen in ARMD and microaneurysms in DR/DN). Freezing the stem and Stage 1 (updating only Stages 2–4 + projector/predictor for 25 epochs) also reduces activation memory, allowing you to increase batch size or remove gradient checkpointing for ~25% faster throughput.

**Keep the First Layer of the MLP Projector (SimCLR v2 / BYOL Few-Shot Trick):**
When evaluating 1/5/10-shot ARMD and ODC/ODP, probe the output of the first linear + BN/LN layer of the projector (or Concat(Backbone_Stage3, Backbone_Stage4)) rather than only the final backbone pooled vector. The backbone layer immediately preceding the projector experiences the strongest invariance bottleneck; stepping one layer earlier (into Stage 3) improves few-shot linear separability on fine-grained medical classes at zero training cost.

**Green-Channel / Contrast-Preserving Jitter Constraint on Macular Crops:**
Drusen (ARMD) and hard exudates/nodules (DN) have peak contrast in the green and yellow-red wavelengths of RGB fundus images. Even "weak" random hue/saturation jitter applied uniformly to a tight macular crop can wash out subtle RPE/drusen contrast, making a drusen macula look identical to a healthy macula under the BYOL loss. For the macular view in v4.1, use geometric augmentation only (scale/flip/rotation) + zero hue/saturation shift (keep color jitter only on the wide random views). This forces the online macular branch to preserve exact chromatic contrast of drusen.
