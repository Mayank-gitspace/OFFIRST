# DR Lesion Segmenter (IDRiD → multi-positive crops), Kaggle-ready

A U-Net with a **ConvNeXt-Tiny** encoder (ImageNet-22k → 1k weights), trained on **IDRiD Segmentation** as a **4-class multi-label** problem
(MA, HE, EX, SE, one sigmoid channel each, so overlapping lesions are allowed). The trained model is then run on
RFMiD (or any fundus folder) to produce **lesion-centred, multi-scale crops** for self-supervised pretraining.

Files: `config.yaml`, `common.py` (shared preprocessing, sliding window, AMP), `model.py`, `train.py`, `inference.py`.

## 1. Notebook
New Notebook → GPU (T4) → Internet ON (pip + ImageNet weights).

## 2. Data: official IDRiD layout
Add an IDRiD dataset that contains **"A. Segmentation"**:
```
A. Segmentation/
  1. Original Images/{a. Training Set, b. Testing Set}/IDRiD_XX.jpg
  2. All Segmentation Groundtruths/{a. Training Set, b. Testing Set}/
       1. Microaneurysms/IDRiD_XX_MA.tif
       2. Haemorrhages/IDRiD_XX_HE.tif
       3. Hard Exudates/IDRiD_XX_EX.tif
       4. Soft Exudates/IDRiD_XX_SE.tif
```
Set `data.idrid_root` to the `A. Segmentation` folder **or any parent of it**, since the loader searches for it.
A missing mask file (common for SE) is treated as all zeros. The loader prints how many are missing.

## 3. Install
```python
!pip install -q segmentation-models-pytorch albumentations timm scikit-learn
```
Tested on: torch 2.x, smp 0.6.0, timm 1.0, albumentations 2.0.8 (the code also guards for albumentations 1.x).

## 4. Train
```python
!ls /kaggle/input   # check the dataset folder name first
!python train.py --config config.yaml --set "data.idrid_root=/kaggle/input/idrid-dataset-lesion-segmentation"
```
Any config key can be overridden: `--set training.epochs=20 data.long_side=1280 training.loss=focal_dice`.

### Run profiles
```python
# 1) QUICK RUN (~sanity check: does it learn at all?) — lower resolution, fewer epochs
# !python train.py --config config.yaml --set data.long_side=768 training.epochs=15

# 2) FULL RUN — default config (long_side 1024, 40 epochs, early stopping 8)
# !python train.py --config config.yaml

# 3) LATER: higher resolution mainly helps microaneurysms (MA). Watch T4 memory/time;
#    drop batch_size to 2 and raise grad_accum_steps to 4 to keep the effective batch at 8.
# !python train.py --config config.yaml --set data.long_side=1440 training.batch_size=2 training.grad_accum_steps=4
```
Check `checkpoints/history.json` for per-lesion AUPR after each run.

What training does:
- **Preprocessing:** crop to the fundus circle, then resize the long side to `data.long_side` (1024). Masks use
  INTER_AREA + `>0`, so microaneurysms survive the downsampling.
- **Patches:** random `patch_size` (512) crops, `patches_per_image` per image per epoch, with `lesion_patch_prob` (70%)
  centred on a lesion pixel.
- **Split:** the official train/test split. `val_source: holdout` (the default) validates on `holdout_n` training images,
  so the official test set stays clean for the final number. `val_source: test` validates on the official test set,
  which is simpler but gives optimistic results.
- **Validation:** full resized images with sliding-window 512 tiles and no augmentation. It uses its own dataset object.
- **Checkpoint selection:** the best **mean per-class AUPR** (sklearn `average_precision_score` on uniformly sampled
  pixels), matching the IDRiD benchmark. Per-class AUPR and Dice are logged every epoch, and `history.json` is saved.
- **`freeze_encoder_epochs`:** the encoder is frozen (with BN in eval mode) for the first N epochs.
- **Loss:** `dice_bce` or `focal_dice` on the 4-channel multi-hot target, with batch-level per-class Dice.
- **AMP:** `torch.amp` (CUDA only). Cosine LR schedule (per epoch).
- **Learning rates:** two AdamW parameter groups: encoder at `training.encoder_lr` (1e-4) and decoder + head at
  `training.lr` (3e-4). Both groups follow the same cosine curve (the ratio is kept). While the encoder is frozen its
  params get no gradients, so AdamW skips them; after unfreezing they train at the lower encoder lr.
- **Gradient accumulation:** `training.grad_accum_steps` (2) micro-batches of `batch_size` (4) per optimizer step,
  so the effective batch is 8. A short final window at the end of an epoch is averaged correctly.
- **Encoder:** `tu-convnext_tiny.fb_in22k_ft_in1k` by default (timm name + pretrained tag), checked against the
  installed smp/timm. Fallbacks: `tu-convnext_tiny`, `tu-efficientnet_b3`, `tu-mobilenetv3_large_100`. If the weights
  download fails, it warns and uses random init (turn Internet ON in the notebook). ConvNeXt has no stride-2 feature
  level (its stem is stride 4); smp 0.6 handles this and the output matches the input size (checked at 512).
  About 31.9M params in total (27.8M encoder).

Outputs: `checkpoints/best.pth`, `last.pth` and `history.json`. The checkpoint stores the encoder name and class order.

## 5. Generate crops on RFMiD
```python
!python inference.py --config config.yaml \
  --input_dir /kaggle/input/rfmid --output_dir /kaggle/working/crops \
  --ckpt /kaggle/working/checkpoints/best.pth
```
Per image: the same fundus crop and resize, then sliding-window 512 tiles and a sigmoid map. Then:
- Connected components are found per class, with **confidence = the max probability inside the component**.
- Candidates are **ranked globally across classes**, then **NMS** (`nms_iou`), then the top `max_lesions_per_image`.
- Each lesion gets `crops_per_lesion` views (`scale_factors`, `jitter`). `crop_size` and `jitter` are in the resized
  (long_side) space, so crops cover the same retinal area on every camera.
- Boxes are **mapped back to original-image pixels** (fundus offset + per-axis scale). Crops are cut from the original image.
- The RNG is seeded (`seed`), so the jitter is reproducible.

`metadata.json` stores one entry per crop: `image_id, lesion_idx, lesion_type, bbox [x1,y1,x2,y2] (original px),
center, confidence, area_resized, scale, view_idx, crop_path`.

## Tips
- **Domain shift:** IDRiD and RFMiD come from different cameras. Look at ~20 RFMiD outputs before you trust
  `conf_threshold: 0.4`, and tune it with `save_prob_maps: true`.
- Use `num_workers: 2` on Kaggle. Outputs in `/kaggle/working/` persist.
- For faster iteration use the quick-run profile above (`long_side 768`, 15 epochs). MA detection suffers below ~1024.
- YAML reads `1e-4` as a string. Write `1.0e-4` (the loader also coerces lr/weight_decay).
