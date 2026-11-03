"""
Training script for the lightweight DR lesion segmenter (IDRiD, 4-class multi-label).

- Official IDRiD layout loader (per-lesion .tif masks, missing mask -> zeros)
- Fundus crop + long-side resize, random 512 patches (~70% lesion-centred)
- Validation on full resized images with sliding-window tiles
- Best checkpoint = mean per-class AUPR; per-class Dice logged
"""
import argparse
import json
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import (load_config, coerce_training_numbers, seed_everything, preprocess_image, preprocess_mask,
                    normalize, sliding_window_predict, autocast_ctx, make_grad_scaler)
from model import build_model, build_loss, build_optimizer, set_encoder_trainable

LESION_FOLDERS = {"MA": "1. Microaneurysms", "HE": "2. Haemorrhages",
                  "EX": "3. Hard Exudates", "SE": "4. Soft Exudates"}
SPLITS = {"train": "a. Training Set", "test": "b. Testing Set"}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


# ----------------------------------------------------------------------------- IDRiD loader

def find_seg_root(root):
    """Accept the 'A. Segmentation' folder or any parent of it."""
    root = Path(root)
    if (root / "1. Original Images").is_dir():
        return root
    for p in sorted(root.rglob("1. Original Images")):
        if p.is_dir() and (p.parent / "2. All Segmentation Groundtruths").is_dir():
            return p.parent
    raise FileNotFoundError(f"Could not find IDRiD 'A. Segmentation' layout under {root}")


def list_idrid(root, split, classes):
    """Returns list of dicts: {id, image, masks: {cls: path or None}}."""
    seg = find_seg_root(root)
    img_dir = seg / "1. Original Images" / SPLITS[split]
    gt_dir = seg / "2. All Segmentation Groundtruths" / SPLITS[split]
    items = []
    for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
        masks = {}
        for c in classes:
            cands = sorted((gt_dir / LESION_FOLDERS[c]).glob(f"{img.stem}_{c}.tif*"))
            masks[c] = cands[0] if cands else None
        items.append({"id": img.stem, "image": img, "masks": masks})
    if not items:
        raise FileNotFoundError(f"No images in {img_dir}")
    return items


def read_binary_mask(path, shape_hw):
    if path is None:
        return np.zeros(shape_hw, np.uint8)
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:  # some IDRiD tifs confuse OpenCV -> try PIL
        from PIL import Image
        m = np.array(Image.open(path))
    if m.ndim == 3:
        m = m.max(axis=2)
    if m.shape[:2] != tuple(shape_hw):
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)


def load_sample(item, classes, long_side, fundus_threshold):
    """Load + preprocess one image and its 4-channel multi-hot mask (H, W, 4)."""
    bgr = cv2.imread(str(item["image"]))
    if bgr is None:
        raise IOError(f"Cannot read {item['image']}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hw = rgb.shape[:2]
    mask = np.stack([read_binary_mask(item["masks"][c], hw) for c in classes], axis=-1)
    img, meta = preprocess_image(rgb, long_side, fundus_threshold)
    mask = preprocess_mask(mask, meta, img.shape[:2])
    return img, mask


class _Base(Dataset):
    def __init__(self, items, cfg):
        d = cfg["data"]
        self.items, self.classes = items, d["lesion_classes"]
        self.long_side, self.thr = d["long_side"], d["fundus_threshold"]
        self.cache = {} if d.get("cache_in_ram", True) else None
        if self.cache is not None:
            for i in tqdm(range(len(items)), desc=f"Caching {type(self).__name__}"):
                self.cache[i] = load_sample(items[i], self.classes, self.long_side, self.thr)

    def get(self, i):
        if self.cache is not None:
            return self.cache[i]
        return load_sample(self.items[i], self.classes, self.long_side, self.thr)


class PatchTrainDataset(_Base):
    """Random patches; `lesion_patch_prob` of them centred on a lesion pixel."""
    def __init__(self, items, cfg, transform):
        super().__init__(items, cfg)
        d = cfg["data"]
        self.patch, self.p_lesion = d["patch_size"], d["lesion_patch_prob"]
        self.per_image = d["patches_per_image"]
        self.transform = transform
        self.lesion_px = {}
        for i in range(len(items)):   # precompute lesion coords (subsampled)
            _, m = self.get(i)
            ys, xs = np.nonzero(m.any(axis=-1))
            if len(ys) > 20000:
                sel = np.random.choice(len(ys), 20000, replace=False)
                ys, xs = ys[sel], xs[sel]
            self.lesion_px[i] = (ys, xs)

    def __len__(self):
        return len(self.items) * self.per_image

    def __getitem__(self, idx):
        i = idx % len(self.items)
        img, mask = self.get(i)
        P = self.patch
        h, w = img.shape[:2]
        if h < P or w < P:   # pad up to patch size
            ph, pw = max(0, P - h), max(0, P - w)
            img = np.pad(img, ((0, ph), (0, pw), (0, 0)))
            mask = np.pad(mask, ((0, ph), (0, pw), (0, 0)))
            h, w = img.shape[:2]
        ys, xs = self.lesion_px[i]
        if len(ys) and random.random() < self.p_lesion:
            k = random.randrange(len(ys))
            cy = ys[k] + random.randint(-P // 4, P // 4)
            cx = xs[k] + random.randint(-P // 4, P // 4)
            y0 = int(np.clip(cy - P // 2, 0, h - P))
            x0 = int(np.clip(cx - P // 2, 0, w - P))
        else:
            y0, x0 = random.randint(0, h - P), random.randint(0, w - P)
        img_p = np.ascontiguousarray(img[y0:y0 + P, x0:x0 + P])
        mask_p = np.ascontiguousarray(mask[y0:y0 + P, x0:x0 + P])
        out = self.transform(image=img_p, mask=mask_p)
        return out["image"], out["mask"].float()   # mask (4, P, P) via transpose_mask


class FullImageValDataset(_Base):
    """Full resized images, no augmentation (separate object from training)."""
    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, mask = self.get(i)
        return normalize(img), torch.from_numpy(mask.transpose(2, 0, 1).copy())


# ----------------------------------------------------------------------------- augmentations

def _try(builders):
    """Return the first transform that builds on this albumentations version."""
    for b in builders:
        try:
            return b()
        except (TypeError, ValueError, AttributeError):
            continue
    return None


def get_train_transform(cfg):
    a = cfg["augment"]
    t = []
    if a.get("geometry", True):
        t += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
              # Affine replaces the deprecated ShiftScaleRotate (works in 1.x and 2.x)
              A.Affine(scale=(0.85, 1.15), translate_percent=(-0.1, 0.1), rotate=(-30, 30), p=0.5)]
    cj = a.get("color_jitter", 0)
    if cj:
        t.append(A.ColorJitter(brightness=cj, contrast=cj, saturation=cj, hue=0.05, p=0.5))
    if a.get("quality_degrade", True):
        noise = _try([lambda: A.GaussNoise(std_range=(0.02, 0.08), p=1),     # >= 1.4.x / 2.x
                      lambda: A.GaussNoise(var_limit=(10.0, 50.0), p=1)])  # older
        comp = _try([lambda: A.ImageCompression(quality_range=(60, 90), p=1),  # 2.x
                     lambda: A.ImageCompression(quality_lower=60, quality_upper=90, p=1)])
        opts = [A.GaussianBlur(blur_limit=(3, 7), p=1)] + [x for x in (noise, comp) if x]
        t.append(A.OneOf(opts, p=0.4))
    t += [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
          ToTensorV2(transpose_mask=True)]
    return A.Compose(t)


# ----------------------------------------------------------------------------- evaluation

@torch.no_grad()
def evaluate(model, ds, cfg, device, amp):
    v = cfg["validation"]
    C = len(cfg["data"]["lesion_classes"])
    rng = np.random.default_rng(cfg.get("seed", 42))
    ys = [[] for _ in range(C)]
    ps = [[] for _ in range(C)]
    inter, psum, tsum = np.zeros(C), np.zeros(C), np.zeros(C)
    model.eval()
    for i in range(len(ds)):
        x, m = ds[i]
        prob = sliding_window_predict(model, x, device, v["tile"], v["overlap"], amp)
        m = m.numpy()
        pred = prob > v["dice_threshold"]
        inter += (pred & (m > 0)).reshape(C, -1).sum(1)
        psum += pred.reshape(C, -1).sum(1)
        tsum += m.reshape(C, -1).sum(1)
        n = prob.shape[1] * prob.shape[2]
        idx = rng.choice(n, size=min(v["pixels_per_image"], n), replace=False)
        for c in range(C):
            ys[c].append(m[c].reshape(-1)[idx])
            ps[c].append(prob[c].reshape(-1)[idx])
    aupr, dice = [], []
    for c in range(C):
        y = np.concatenate(ys[c])
        aupr.append(float(average_precision_score(y, np.concatenate(ps[c]))) if y.any() else float("nan"))
        dice.append(float(2 * inter[c] / (psum[c] + tsum[c])) if (psum[c] + tsum[c]) > 0 else float("nan"))
    return aupr, dice


# ----------------------------------------------------------------------------- main

def train(config_path, overrides=None):
    cfg = load_config(config_path)
    for k, v in (overrides or {}).items():   # "section.key" -> value
        sec, key = k.split(".")
        cfg[sec][key] = v
    coerce_training_numbers(cfg)
    seed_everything(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg["training"]["mixed_precision"]) and device.type == "cuda"
    print(f"Using device: {device}  (AMP: {amp})")

    d = cfg["data"]
    classes = d["lesion_classes"]
    train_items = list_idrid(d["idrid_root"], "train", classes)
    test_items = list_idrid(d["idrid_root"], "test", classes)
    if d.get("val_source", "holdout") == "holdout":
        rnd = random.Random(cfg.get("seed", 42))
        shuffled = train_items[:]
        rnd.shuffle(shuffled)
        n_val = max(1, min(d.get("holdout_n", 6), len(shuffled) - 1))
        val_items, train_items = sorted(shuffled[:n_val], key=lambda x: x["id"]), shuffled[n_val:]
    else:
        val_items = test_items
    print(f"Train imgs: {len(train_items)}  Val imgs: {len(val_items)} "
          f"({d.get('val_source')})  Official test imgs: {len(test_items)}")
    missing = sum(m is None for it in train_items for m in it["masks"].values())
    print(f"Missing lesion masks in train (treated as zeros): {missing}")

    train_ds = PatchTrainDataset(train_items, cfg, get_train_transform(cfg))
    val_ds = FullImageValDataset(val_items, cfg)
    t = cfg["training"]
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, drop_last=False,
                              num_workers=t["num_workers"], pin_memory=device.type == "cuda",
                              persistent_workers=t["num_workers"] > 0,
                              worker_init_fn=lambda w: (np.random.seed(cfg.get("seed", 42) + w),
                                                        random.seed(cfg.get("seed", 42) + w)))

    model, enc = build_model(cfg["model"]["encoder"], len(classes), cfg["model"]["pretrained"],
                             cfg["model"].get("encoder_fallbacks", ()))
    model.to(device)
    print(f"Encoder: {enc}")
    criterion = build_loss(t["loss"])
    n_all = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    print(f"Params: total {n_all / 1e6:.2f}M  (encoder {n_enc / 1e6:.2f}M, decoder+head {(n_all - n_enc) / 1e6:.2f}M)")
    enc_lr = t.get("encoder_lr")
    enc_lr = t["lr"] if enc_lr is None else enc_lr
    accum = t["grad_accum_steps"]
    print(f"LR: encoder {enc_lr:g}  decoder {t['lr']:g}  |  batch {t['batch_size']} x accum {accum} "
          f"= effective {t['batch_size'] * accum}")
    optimizer = build_optimizer(model, t["lr"], enc_lr, t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, t["epochs"]))
    scaler = make_grad_scaler(device, amp)
    freeze_n = int(cfg["model"].get("freeze_encoder_epochs", 0))

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best, patience, history = -1.0, 0, []

    for epoch in range(t["epochs"]):
        frozen = epoch < freeze_n
        set_encoder_trainable(model, not frozen)
        model.train()
        if frozen:
            model.encoder.eval()   # keep ImageNet BN statistics while frozen
        run = 0.0
        n_batches = len(train_loader)
        optimizer.zero_grad(set_to_none=True)
        for step, (imgs, masks) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}{' [enc frozen]' if frozen else ''}")):
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            with autocast_ctx(device, amp):
                logits = model(imgs)
            loss = criterion(logits.float(), masks)   # loss in fp32
            # gradient accumulation: average over the micro-batches in this window
            window = min(accum, n_batches - (step // accum) * accum)
            scaler.scale(loss / window).backward()
            if (step + 1) % accum == 0 or (step + 1) == n_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            run += loss.item()
        lrs = {g.get("name", i): g["lr"] for i, g in enumerate(optimizer.param_groups)}  # lr used this epoch
        scheduler.step()
        run /= max(1, len(train_loader))

        aupr, dice = evaluate(model, val_ds, cfg, device, amp)
        m_aupr = float(np.nanmean(aupr)) if not np.all(np.isnan(aupr)) else 0.0
        fmt = lambda v: "  ".join(f"{c}={x:.3f}" for c, x in zip(classes, v))
        print(f"Epoch {epoch + 1}: train_loss={run:.4f}  mAUPR={m_aupr:.4f}  "
              f"lr(enc/dec)={lrs.get('encoder', 0):.2e}/{lrs.get('decoder', 0):.2e}\n"
              f"  AUPR: {fmt(aupr)}\n  Dice: {fmt(dice)}")
        history.append({"epoch": epoch + 1, "train_loss": run, "mAUPR": m_aupr,
                        "aupr": dict(zip(classes, aupr)), "dice": dict(zip(classes, dice))})

        state = {"model": model.state_dict(), "encoder": enc, "classes": classes,
                 "epoch": epoch + 1, "mAUPR": m_aupr}
        torch.save(state, ckpt_dir / "last.pth")
        if m_aupr > best:
            best, patience = m_aupr, 0
            torch.save(state, ckpt_dir / "best.pth")
            print("  -> saved best")
        else:
            patience += 1
            if patience >= t["early_stopping_patience"]:
                print("Early stopping")
                break

    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. Best mAUPR={best:.4f}  ->  {ckpt_dir / 'best.pth'}")


def _parse_override(s):
    k, v = s.split("=", 1)
    import yaml
    return k, yaml.safe_load(v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--set", nargs="*", default=[],
                        help='overrides, e.g. --set training.epochs=1 data.idrid_root=/path')
    args = parser.parse_args()
    train(args.config, dict(_parse_override(s) for s in args.set))
