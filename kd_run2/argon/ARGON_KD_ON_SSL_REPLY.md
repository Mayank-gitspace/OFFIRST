(a) Blunt Verdict on the Plan

Run the distillation—it is the highest-ROI use of GPU time you have left before November 15—but do not run your exact current configuration.

Your diagnosis of the bottleneck is right, and your ConvNeXt-T teacher proves that a 768-d global representation learned with anatomy-aware crops already contains the signal needed for rare diseases (0.725 rare 10-shot, 0.943 ARMD@10). Furthermore, 40 epochs over 1,152–1,536 images will take 12 to 18 minutes total on a Kaggle T4, so compute cost is negligible.

However, four mechanics in your current plan will severely handicap or break the run if left unchanged:

[Math] The 960-to-768 Null-Space Trap: MobileNetV3-Large’s pooled backbone feature is 960-d (or 1280-d after conv_head), which is wider than the teacher's 768-d. If you train GAP (960-d) -> Linear(960, 768) with cosine loss and discard the linear layer at test time to probe the 960-d backbone, 
rank
(
𝑊
)
≤
768
rank(W)≤768 leaves a 192-dimensional null space in the student backbone that receives zero gradient from the KD loss, and non-orthonormal singular values of 
𝑊
W warp cosine distances in 960-d space. You must keep the Linear(960, 768) projection as the final exported layer of the student encoder at inference time.

[Evidence] Random-Head Feature Distortion at Epoch 0 (Kumar et al., ICLR 2022): Unfreezing your converged SSL ep40 backbone (0.917 LP) at epoch 0 while the Linear(960, 768) projector is randomly initialized backpropagates random gradients (
𝑊
rand
⊤
∇
𝐿
W
rand
⊤
	​

∇L) into the backbone, damaging your learned depthwise filters before the head aligns.

[Evidence] Fixed Teacher Cache vs. Spatial Student Crops (Beyer et al., CVPR 2022): Caching one static embedding per image while applying random crops or flips to the student forces the student to map shifted/flipped anatomy (where the macula or optic disc may be displaced or inverted between left/right eye orientation) to an unflipped full-image target.

[Evidence] Paired Clean/Degraded Duplicates in the Same Batch: Putting 24 clean images and 24 degraded copies of the exact same 24 images in a 48-image batch halves your anatomical batch diversity and introduces paired correlation into BatchNorm batch statistics during the forward pass.

Detailed Answers to Questions 1–8
1. Is KD-on-top-of-SSL init sound, or will it bias/limit the student vs. ImageNet init?

[Evidence] Why direct BYOL on MNV3 failed on rare classes: CompRess (Abbasi Koohpayegani et al., NeurIPS 2020), SEED (Fang et al., ICLR 2021), DisCo (Gao et al., ECCV 2022), and BINGO (Xu et al., ICLR 2022) demonstrate that lightweight CNNs (MobileNetV3, ResNet-18, EfficientNet-B0) hit a severe capacity cliff under direct instance-discrimination SSL. Forcing a small encoder to be invariant to heavy augmentations (especially your synthetic Visuscout blur, noise, and vignetting) causes dimensional/subspace collapse of fine-grained features (Jing et al., ICLR 2022). MNV3-L uses its limited capacity to preserve global low-frequency cues (giving 0.917 on common diseases like media haze or tessellation) while collapsing localized high-frequency cues (macular drusen in ARMD, optic disc pits/cupping in ODP/ODC).

[Evidence] SSL init vs. ImageNet init for KD: While SEED and CompRess initialize students from ImageNet (or random) over 100–200 epochs on 1.28M images, two-stage pretraining analyses (TinyMIM, Ren et al., CVPR 2023; Wang et al., CVPR 2023) show that initializing from an in-domain SSL checkpoint accelerates convergence on small datasets because early/mid-level filters are already domain-adapted to retinal vessel geometry and Visuscout color/blur statistics. Because your SSL ep40 achieves 0.917 LP, its filter bank is not rank-collapsed—its penultimate representation just lacks the teacher's fine-grained metric arrangement.

[Heuristic] Verdict on Init: Starting from SSL ep40 is better than ImageNet init for a 40-epoch run on ~1.5k images, provided you align the 960 -> 768 head before unfreezing the backbone (see Q6). Because a 40-epoch KD run takes under 15 minutes on a T4, you can also run an ImageNet-init control arm in 15 minutes as a sanity check.

2. Is pure cosine loss right for rare-disease few-shot, or should you add relational/queue KD (SEED / CompRess / RKD)?

[Evidence] Pointwise Cosine vs. Relational Queue KD: SimReg (Navaneet et al., BMVC 2021, "Regression as a Simple Yet Effective Tool for Self-Supervised Knowledge Distillation") directly benchmarked pointwise L2-normalized cosine regression (
1
−
cos
⁡
(
𝑧
𝑠
,
𝑧
𝑡
)
1−cos(z
s
	​

,z
t
	​

)) against SEED, CompRess, and CRD (Tian et al., ICLR 2020). SimReg showed that direct cosine regression matches or outperforms queue-based distillation on linear and few-shot benchmarks when representations are L2-normalized.

[Evidence] Where relational KD helps on rare classes: SEED (Fang et al., ICLR 2021) and BINGO (Xu et al., ICLR 2022) show that matching a temperature-sharpened similarity distribution over anchor embeddings (
𝜏
𝑡
≪
𝜏
𝑠
τ
t
	​

≪τ
s
	​

, e.g., 
𝜏
𝑡
=
0.05
,
𝜏
𝑠
=
0.1
τ
t
	​

=0.05,τ
s
	​

=0.1) explicitly forces the student to preserve the teacher's local 
𝑘
k-NN neighborhood topology—which is the exact mechanism governing 1/5/10-shot prototypes on rare classes (ODC/ODP/ARMD). Furthermore, under heavy Visuscout degradation where high-frequency pixel evidence is destroyed, pointwise cosine loss regresses uncertain dimensions toward the mean, whereas relational KL loss forces the student to still place the image near its closest clean anatomical neighbors.

[Math] + [Heuristic] The Small-Data Shortcut (Zero-Queue Static Anchor Bank): In SEED and CompRess, a MoCo momentum queue was only necessary because ImageNet has 1.28M images. Your training set is only 
𝑁
=
1
,
152
N=1,152 to 
1
,
536
1,536 images! The entire dataset's clean teacher embeddings 
𝐴
𝑡
∈
𝑅
𝑁
×
768
A
t
	​

∈R
N×768
 occupy just 4.7 MB of VRAM (1536 * 768 * 4 bytes). You do not need a queue. You can compute the exact SEED relational KL loss against all 
𝑁
N training images simultaneously in a single matrix multiplication (z_s @ A_t.T) taking less than 0.1 ms per batch and 6 lines of PyTorch:

𝐿
total
=
(
1
−
cos
⁡
(
𝑧
^
𝑠
,
𝑧
^
𝑡
)
)
⏟
SimReg pointwise alignment
+
𝜆
𝜏
𝑠
2
KL
(
softmax
(
𝑧
^
𝑡
𝐴
𝑡
⊤
𝜏
𝑡
)
∥
softmax
(
𝑧
^
𝑠
𝐴
𝑡
⊤
𝜏
𝑠
)
)
L
total
	​

=
SimReg pointwise alignment
(1−cos(
z
^
s
	​

,
z
^
t
	​

))
	​

	​

+λτ
s
2
	​

KL(softmax(
τ
t
	​

z
^
t
	​

A
t
⊤
	​

	​

)
	​

softmax(
τ
s
	​

z
^
s
	​

A
t
⊤
	​

	​

))

with 
𝜏
𝑡
=
0.05
τ
t
	​

=0.05, 
𝜏
𝑠
=
0.10
τ
s
	​

=0.10, and 
𝜆
=
1.0
λ=1.0. Cost is 5 minutes of coding; benefit is direct supervision on local cluster boundaries for rare few-shot transfer.

3. Does caching a single teacher embedding hurt when the student is augmented?

[Evidence] Yes, significantly: Beyer et al. (CVPR 2022, "Knowledge distillation: A good teacher is patient and consistent") directly compared caching one fixed teacher vector per image ("fixed teacher") against feeding the exact same spatial crop/flip to both teacher and student ("consistent teacher"). Fixed teacher consistently degraded student accuracy because spatial augmentations change the semantic content of the view.

[Heuristic] Why it is acute in fundus imaging: ARMD lives in the macula (~20% of image area); ODC and ODP live in the optic disc (~15% of image area). If the student receives a random crop that clips the macula, or a horizontal flip that mirrors nasal/temporal geometry (turning an OD right eye into an OS left eye), regressing to an unflipped full-image teacher vector penalizes the student for encoding accurate spatial features. Conversely, if you turn off all spatial augmentations on 1,152 images over 40 epochs, MNV3-L will overfit to static pixel coordinates.

[Math] Two Low-Effort Fixes (Pick One):

Option 1 (Best — Online Frozen Teacher in FP16): A frozen ConvNeXt-T (28M params) in torch.no_grad() + torch.cuda.amp.autocast(dtype=torch.float16) uses only ~56 MB of VRAM (no saved activations for backprop) and processes 1,536 images at 
384
×
384
384×384 in ~2.2 seconds per epoch on a T4 (~90 seconds total across 40 epochs). Apply a shared fundus-preserving spatial transform (random scale 0.80–1.0 so both optic disc and macula remain inside the frame, random horizontal/vertical flip, rotation 
±
15
∘
±15
∘
) to get clean_512. Feed F.interpolate(clean_512, 384) to the teacher, and feed clean_512 (or Visuscout-degraded deg_512) to the student.

Option 2 (If you strictly want offline caching): Precompute a Multi-View Cache of 
𝐾
=
8
K=8 deterministic spatial views per image (4 flips/rotations 
×
× 2 scales: 100% and 88% center/FOV crop). For 1,536 images in FP16, 
1536
×
8
×
768
×
2
 bytes
=
18.8
 MB
1536×8×768×2 bytes=18.8 MB total on disk/RAM. During training, sample one of the 8 spatial views, load its exact cached teacher vector, and apply only non-spatial Visuscout degradations (blur, color cast, vignetting, noise, JPEG) online to the student image.

4. Resolution mismatch: Student 512 vs. Teacher 384. Any problem?

[Evidence] No problem—it is an advantage (Touvron et al., NeurIPS 2019, FixRes): Because distillation happens after Global Average Pooling (GAP) in the 768-d vector space, the difference in pre-GAP feature grid size (
16
×
16
16×16 in MNV3 at 512 vs. 
12
×
12
12×12 in ConvNeXt-T at 384) does not create any tensor dimension conflict.

[Heuristic] Architectural reason to keep Student at 512: MobileNetV3-Large starts with a stride-2 
3
×
3
3×3 stem followed by aggressive depthwise separable downsampling, whereas ConvNeXt-T uses 
7
×
7
7×7 depthwise kernels that preserve context better at 384. When your student image is degraded by synthetic Visuscout blur and JPEG compression, small macular drusen (ARMD) and optic disc pits (ODP) need the 
1.78
×
1.78× pixel area of 
512
×
512
512×512 to survive MNV3's early stride-2 convolutions. Just ensure the 512 student view and 384 teacher view cover the exact same spatial field of view (i.e. resize the same crop, do not center-crop one and full-resize the other).

5. Should the student train on Folds 1–3 (1,152 imgs, match SSL arm) or Folds 1–4 (1,536 imgs, match teacher)?

[Heuristic] Do both—it takes 15 minutes per run:

With 5 days left, holding out Fold 4 (25% of your non-test data) from the final model throws away 384 images in an already tiny data regime (
𝑁
=
1
,
152
N=1,152). Moreover, your teacher was already trained on Folds 1–4, so Fold 4 is already in-distribution for the teacher.

Protocol:

Run your smoke test and first full 40-epoch KD run on Folds 1–3 (1,152 imgs). Log the locked Fold-0 scores. This gives you a 100% clean, unconfounded scientific ablation ("Does KD improve SSL ep40 on the exact same 1,152 images?") for your presentation/paper.

Once validated, run the exact same KD recipe on Folds 1–4 (1,536 imgs) to produce the final TFLite weights for the phone prototype.

6. 50/50 clean/degraded mix, 40 epochs, BatchNorm stats, LR/warmup, and keeping BYOL loss?

[Evidence] BatchNorm Mixing Warning (Xie et al., CVPR 2020, AdvProp; Wu & Johnson, ECCV 2021):

Mixing 50% clean and 50% degraded images in a single-branch network forces BatchNorm running statistics (
𝜇
run
,
𝜎
run
2
μ
run
	​

,σ
run
2
	​

) to settle at the mean/variance compromise between clean tabletop cameras and degraded handheld cameras. Since your TFLite phone deployment has a single BN branch and must work on both, a 50/50 domain mixture is the right pragmatic choice.

However, do NOT put [24 clean, 24 degraded copies of the same 24 images] in the same 48-image batch. That cuts unique retinas per batch from 48 to 24 and correlates batch statistics. Instead, sample 48 independent images per batch and apply Visuscout degradation independently with probability 
𝑝
=
0.5
p=0.5.

BN Recalibration Pass (Wu & Johnson, ECCV 2021): Exponential moving average (momentum=0.1) BN stats lag behind changing weights during cosine LR decay. After epoch 40, run 1 extra epoch in train-BN / no-grad mode (model.train(), torch.no_grad(), reset BN running stats, forward pass over 50/50 clean/degraded batches) to lock in exact population BN statistics.

[Evidence] LR / Warmup from SSL Checkpoint (Kumar et al., ICLR 2022, LP-FT):

Never unfreeze a pretrained backbone at epoch 0 while a new linear head is random.

Step 1 (Closed-Form Orthogonal Procrustes Init — takes 3 seconds): Before epoch 1, pass Folds 1–3 clean images through the frozen student (960-d L2-normalized 
𝐻
𝑠
H
s
	​

) and teacher (768-d L2-normalized 
𝑍
𝑡
Z
t
	​

). Initialize the Linear(960, 768, bias=False) weight matrix with the exact SVD orthogonal projection:

𝑈
,
Σ
,
𝑉
⊤
=
SVD
(
𝐻
𝑠
⊤
𝑍
𝑡
)
,
𝑊
init
=
(
𝑈
𝑉
⊤
)
⊤
U,Σ,V
⊤
=SVD(H
s
⊤
	​

Z
t
	​

),W
init
	​

=(UV
⊤
)
⊤

and warm up the head alone for 3 epochs (lr = 1e-3, student backbone frozen with .eval()).

Step 2 (Joint KD for 37 epochs): Unfreeze the student backbone with differential learning rates (AdamW, weight_decay = 1e-4): backbone lr = 1.5e-4 (with 3-epoch linear warmup from 1e-5 and cosine decay to 1e-6), projector head lr = 5e-4. Because you don't need a momentum encoder or BYOL predictor, you can turn off gradient checkpointing, which speeds up training by ~35% on a T4.

[Evidence] Should you keep a small BYOL loss alongside KD?

No. SimReg (Navaneet et al., 2021) and SEED (Fang et al., 2021) show that pure distillation to a frozen SSL teacher matches or beats joint SSL+KD (DisCo) at half the memory and compute. Because the teacher vectors 
𝑧
^
𝑡
z
^
t
	​

 are fixed and well-dispersed on the hypersphere, the student cannot experience representation collapse. Adding MNV3's own BYOL loss re-introduces the exact small-capacity view-invariance pressure that crushed your rare-disease features in the first place.

7. Is the decision rule statistically sensible on 384 Fold-0 images? What margin / paired test should count?

[Math] Standard Error Reality Check on RFMiD Fold 0:

In RFMiD (1,920 images), a 20% split (Fold 0 = 384 images) contains roughly ~20 positive cases of ARMD, ~56 of ODC, and ~13–20 of ODP.

By the Hanley-McNeil AUROC variance approximation, for 
𝑛
+
≈
20
n
+
	​

≈20 positives and 
𝑛
−
≈
364
n
−
	​

≈364 negatives around 
AUROC
≈
0.70
AUROC≈0.70, the unpaired standard error of a single class AUROC is 
𝑆
𝐸
≈
0.068
SE≈0.068 (
±
6.8
±6.8 points!).

However, because you are evaluating both SSL ep40 and KD on the exact same Fold-0 images, their predictions are highly correlated (
𝜌
≈
0.80
–
0.88
ρ≈0.80–0.88). Under a paired DeLong test (DeLong et al., Biometrics 1988) or stratified paired bootstrap, the standard error of the difference 
Δ
AUROC
=
AUROC
KD
−
AUROC
SSL
ΔAUROC=AUROC
KD
	​

−AUROC
SSL
	​

 drops by 
2
(
1
−
𝜌
)
≈
0.55
2(1−ρ)
	​

≈0.55, giving:

Single rare class (e.g., ARMD@10): 
𝑆
𝐸
Δ
≈
0.035
SE
Δ
	​

≈0.035

Mean across the 3 rare classes (ODC/ODP/ARMD): 
𝑆
𝐸
Δ
≈
0.020
SE
Δ
	​

≈0.020

Furthermore, 1-shot and 5-shot evaluations have high support-set sampling variance (
±
0.03
±0.03 depending on which 1 or 5 support images are drawn).

[Heuristic] Statistically Sound Decision Rule:

Lock 100 deterministic support episodes per shot count (
𝑘
∈
{
1
,
5
,
10
}
k∈{1,5,10}), sampled strictly from Folds 1–3 (never Fold 0), using identical random seeds for Teacher, SSL ep40, and KD.

Run a 1,000-iteration stratified paired bootstrap over the 384 Fold-0 query images to compute the distribution of 
Δ
=
Metric
KD
−
Metric
SSL
Δ=Metric
KD
	​

−Metric
SSL
	​

.

Ship KD if and only if:

Primary Rare Gain: Mean 5/10-shot rare AUROC improves by 
Δ
≥
+
0.020
Δ≥+0.020 AND ARMD@10-shot improves by 
Δ
≥
+
0.050
Δ≥+0.050 (closing at least a third of the 0.789 -> 0.943 gap), with paired bootstrap confidence 
𝑃
(
Δ
>
0
)
≥
0.90
P(Δ>0)≥0.90.

Non-Inferiority Guardrails: Common 11-class LP AUROC (0.917) and degraded retrieval top-
𝑘
k / cosine similarity drop by no more than 
−
0.005
−0.005.

No Epoch Peeking on Fold 0: Do not evaluate epochs 5, 10, 15... on Fold 0 and cherry-pick the lucky epoch (which overfits the ~20 ARMD positives). Pick the final cosine-annealed checkpoint (epoch 40) or select the checkpoint using validation cosine loss on Fold 4.

8. Anything that makes this plan a waste of time, and what else to do in 5 days?

[Math] The Hidden Bug That Would Make It a Waste of Time:
As noted in the Verdict, if you train GAP (960-d) -> Linear(960, 768) and throw away the Linear(960, 768) head before running your frozen linear/few-shot probes on the 960-d vector, the run will likely fail or underperform. Keep the Linear(960, 768, bias=False) layer as part of the deployed student encoder so that your mobile model outputs the 768-d L2-normalized embedding directly.

[Heuristic] High-Leverage Bonus Move if Global KD Leaves a Gap on ODC/ODP/ARMD (Multi-Crop / Local-to-Global Alignment):
Why did the ConvNeXt-T teacher reach 0.943 on ARMD and 0.725 on rare classes? Because during BYOL, it saw anatomy-aware crops (optic disc + macula) and learned to pack local macular/disc features into its global vector. If you already have the bounding-box / heuristic cropper that generated the optic disc and macula crops for the teacher, include 20% anatomy-centered crops (optic disc / macula zoomed to 512 for student, 384 for teacher) alongside 80% full-fundus views during KD! That directly forces MNV3's conv filters to resolve fine drusen and cup-to-disc geometry.

(b) Concrete Changes Ranked by Expected Impact vs. Effort
Rank	Change to Planned Run	Expected Impact	Effort	Evidence / Basis
1	Keep Linear(960, 768) in exported student at test time	Critical (avoids 192-d null space)	2 mins	Linear algebra / rank constraint
2	Procrustes SVD init + 3-ep frozen-backbone head warmup	High (prevents ep40 feature thrashing)	10 mins	Kumar et al., ICLR 2022 (LP-FT)
3	Shared spatial view (online FP16 teacher or 8-view cache)	High (fixes crop/flip target mismatch)	15 mins	Beyer et al., CVPR 2022
4	Add Static-Bank SEED KL loss over all 
N teacher vectors	Medium-High (sharpens rare 
k-NN clusters)	10 mins	Fang et al., ICLR 2021 (SEED)
5	Independent 50% degradation per image + 1-ep BN recalibration	Medium (fixes batch diversity & BN lag)	10 mins	Wu & Johnson, ECCV 2021
6	Train final shipped weights on Folds 1–4 (1,536 imgs)	Medium (+33% training data)	15 mins	Sample scaling in low-data KD
(c) Recommended Final Configuration

Architecture & Exported Representation:

Backbone: MobileNetV3-Large initialized from SSL ep40 target encoder (960-d after features + AdaptiveAvgPool2d(1)).

Projection Head (Kept at Inference): nn.Linear(960, 768, bias=False) followed by L2 normalization (F.normalize(z, dim=-1)). Export the entire Backbone + Linear(960, 768) + L2Norm graph to TFLite.

Data & Views:

Ablation Run: Folds 1–3 (1,152 images). Production Run: Folds 1–4 (1,536 images).

Spatial Augmentation (Shared between Teacher & Student): Random resized crop (scale=(0.80, 1.0), ratio=(0.95, 1.05) so both macula and optic disc remain in view), random horizontal flip (p=0.5), random vertical flip (p=0.5), rotation 
±
15
∘
±15
∘
.

Teacher Input: Clean shared spatial view resized to 
384
×
384
384×384, evaluated online in torch.no_grad() + fp16 (or loaded from an 8-view deterministic cache).

Student Input: Shared spatial view at 
512
×
512
512×512, with independent 
𝑝
=
0.5
p=0.5 synthetic Visuscout degradation (no paired duplicates inside the same batch). Batch size = 48 (or 64 with grad checkpointing disabled).

Loss Function (SimReg Cosine + Static-Bank SEED KL):

Python
# z_s: (B, 768) L2-normalized student embeddings
# z_t: (B, 768) L2-normalized online teacher embeddings for current view
# anchor_bank: (N, 768) L2-normalized cached clean full-image teacher embeddings (4.7 MB VRAM)
loss_cos = (1.0 - (z_s * z_t).sum(dim=-1)).mean()

tau_t, tau_s = 0.05, 0.10
with torch.no_grad():
    p_t = F.softmax((z_t @ anchor_bank.T) / tau_t, dim=-1)
log_p_s = F.log_softmax((z_s @ anchor_bank.T) / tau_s, dim=-1)
loss_seed = F.kl_div(log_p_s, p_t, reduction="batchmean") * (tau_s ** 2)

loss = loss_cos + 1.0 * loss_seed


Two-Stage Schedule (40 Epochs Total, ~15 mins on T4):

Pre-Epoch 0 (Closed-Form Procrustes Init): Extract clean full-image student features 
𝐻
𝑠
∈
𝑅
𝑁
×
960
H
s
	​

∈R
N×960
 and teacher features 
𝑍
𝑡
∈
𝑅
𝑁
×
768
Z
t
	​

∈R
N×768
 (both L2-normalized). Set U, _, Vt = torch.linalg.svd(H_s.T @ Z_t, full_matrices=False) and assign head.weight.data.copy_((U @ Vt).T).

Epochs 1–3 (Head Warmup): Freeze MNV3 backbone (backbone.eval(), requires_grad_(False)). Train head only with AdamW (lr=1e-3, weight_decay=0.0).

Epochs 4–40 (Joint KD): Unfreeze MNV3 (backbone.train()). AdamW with parameter groups: backbone lr=1.5e-4 (3-epoch linear warmup from 1.5e-5), head lr=5e-4, cosine decay to 1e-6, weight_decay=1e-4.

Post-Epoch 40 (BN Recalibration): Reset BN running stats, run 1 pass over training loader in model.train() + torch.no_grad() with 50% clean / 50% degraded inputs.

(d) Key Risks & Mitigations

Risk 1: Asymmetric KD hallucinates features on severely obliterated images. If a synthetic Visuscout sample has extreme blur + heavy vignetting that completely erases macular drusen, forcing the student to match a sharp clean teacher embedding causes high gradient variance and memorization.

Mitigation: Cap synthetic degradation severity during KD so lesions are attenuated rather than zeroed out, and use Huber / Smooth-L1 temperature scaling or the loss_seed soft neighborhood target, which allows graceful uncertainty.

Risk 2: TFLite INT8 quantization degrades the final Linear(960, 768) cosine geometry. Post-training INT8 quantization can introduce quantization noise into dense projection layers and Hardswish activations in MobileNetV3-Large.

Mitigation: Export to FP16 TFLite (~10 MB model size, runs natively on mobile GPU/NPU delegates in <25 ms with zero accuracy drop) rather than full INT8, or keep the final Linear(960, 768) layer in FP16/FP32 if quantizing the backbone.

Risk 3: Teacher ceiling on rare classes (0.589 / 0.688 / 0.725). Even perfect distillation cannot vastly exceed the teacher's own 0.725 rare 10-shot score.

Mitigation: At inference time for few-shot evaluation on Fold 0, evaluate concatenated or weighted multi-layer features (or apply L2-normalization + centering + cosine prototype classification as in SimpleShot, Wang et al., 2019—subtracting the training-set mean embedding 
𝑧
ˉ
z
ˉ
 before L2-normalizing consistently adds +2.0 to +4.0 AUROC points on few-shot tasks at zero training cost).

Want the complete PyTorch script for the Procrustes-initialized KD loop and paired bootstrap evaluator?