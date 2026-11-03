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

from common import (load_config, coerce_training_numbers, seed_everything, preprocess_from_cfg, preprocess_mask,
                    normalize, sliding_window_predict, autocast_ctx, make_grad_scaler, apply_clahe_lab,
                    PREPROC_KEYS)
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


def list_negatives(root, ids, allow_idrid_grading=False):
    """Healthy images used as all-negative samples: every mask = None -> zeros.
    v5: RFMiD TRAINING-split normals (Disease_Risk=0); `root` = the RFMiD 'Training' image folder,
    searched recursively for <id>.png/.jpg. IDRiD grading-set ids are refused unless explicitly allowed."""
    if not ids:
        return []
    ids = [str(i) for i in ids]
    if not allow_idrid_grading and any(i.startswith("IDRiD_") for i in ids):
        raise ValueError("IDRiD grading-set images are not allowed as negatives (data-usage rule)")
    found = {}
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            if p.stem in found and p.stem in ids:
                raise ValueError(f"ambiguous negative id {p.stem}: {found[p.stem]} and {p}")
            found[p.stem] = p
    missing = [i for i in ids if i not in found]
    if missing:
        raise FileNotFoundError(f"{len(missing)} negative images not found under {root}: {missing[:5]}")
    return [{"id": i, "image": found[i], "masks": {}, "negative": True} for i in ids]


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


def load_sample(item, classes, d):
    """Load + preprocess one image and its 4-channel multi-hot mask (H, W, 4).
    Preprocessing = common.preprocess_from_cfg (the same function inference.py uses)."""
    bgr = cv2.imread(str(item["image"]))
    if bgr is None:
        raise IOError(f"Cannot read {item['image']}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hw = rgb.shape[:2]
    mask = np.stack([read_binary_mask(item["masks"].get(c), hw) for c in classes], axis=-1)
    img, meta = preprocess_from_cfg(rgb, d)
    mask = preprocess_mask(mask, meta, img.shape[:2])
    return img, mask


class _Base(Dataset):
    def __init__(self, items, cfg):
        d = cfg["data"]
        self.items, self.classes = items, d["lesion_classes"]
        self.d = d
        self.cache = {} if d.get("cache_in_ram", True) else None
        if self.cache is not None:
            for i in tqdm(range(len(items)), desc=f"Caching {type(self).__name__}"):
                self.cache[i] = load_sample(items[i], self.classes, self.d)

    def get(self, i):
        if self.cache is not None:
            return self.cache[i]
        return load_sample(self.items[i], self.classes, self.d)


class PatchTrainDataset(_Base):
    """Random patches; `lesion_patch_prob` of them centred on a lesion pixel."""
    def __init__(self, items, cfg, transform):
        super().__init__(items, cfg)
        d = cfg["data"]
        self.patch, self.p_lesion = d["patch_size"], d["lesion_patch_prob"]
        self.per_image = d["patches_per_image"]
        # healthy negatives may get fewer patches/epoch so they don't swamp the lesion images
        neg_n = int(d.get("negative_patches_per_image", self.per_image))
        self.index = [i for i, it in enumerate(items)
                      for _ in range(neg_n if it.get("negative") else self.per_image)]
        self.transform = transform
        self.lesion_px, self.fov_px = {}, {}
        for i in range(len(items)):   # precompute lesion coords + retina coords (subsampled)
            img, m = self.get(i)
            for store, sel_mask in ((self.lesion_px, m.any(axis=-1)), (self.fov_px, img.max(axis=-1) > 12)):
                ys, xs = np.nonzero(sel_mask)
                if len(ys) > 20000:
                    sel = np.random.choice(len(ys), 20000, replace=False)
                    ys, xs = ys[sel], xs[sel]
                store[i] = (ys, xs)
        # v5: background patches centred inside the retina (circle crops have black corners)
        self.bg_in_fov = bool(d.get("background_patches_in_fov", False))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i = self.index[idx]
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
        elif self.bg_in_fov and len(self.fov_px[i][0]):
            fy, fx = self.fov_px[i]
            k = random.randrange(len(fy))
            y0 = int(np.clip(fy[k] - P // 2, 0, h - P))
            x0 = int(np.clip(fx[k] - P // 2, 0, w - P))
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


class RandomCLAHE(A.ImageOnlyTransform):
    """CLAHE on LAB-L (same function as the optional preprocessing step) -> CLAHE on/off augmentation."""
    def __init__(self, clip=(1.0, 3.0), p=0.5):
        super().__init__(p=p)
        self.clip = clip

    def apply(self, img, **params):
        return apply_clahe_lab(img, random.uniform(*self.clip), 8, (img.max(axis=-1) > 12).astype(np.uint8))

    def get_transform_init_args_names(self):
        return ("clip",)


def get_camera_robust_transform(cfg):
    """v5 augmentation for camera shift. Applied to EVERY training patch (lesion images and
    RFMiD healthy negatives alike). Image-only ops never touch masks; geometric ops move both."""
    a = cfg["augment"]
    P = cfg["data"]["patch_size"]
    t = [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]
    rrc = _try([lambda: A.RandomResizedCrop(size=(P, P), scale=tuple(a.get("rrc_scale", (0.5, 1.0))),
                                            ratio=(0.8, 1.25), p=a.get("rrc_p", 0.3)),
                lambda: A.RandomResizedCrop(height=P, width=P, scale=tuple(a.get("rrc_scale", (0.5, 1.0))),
                                            ratio=(0.8, 1.25), p=a.get("rrc_p", 0.3))])
    if rrc:
        t.append(rrc)
    t.append(A.Affine(scale=tuple(a.get("scale", (0.8, 1.25))), translate_percent=(-0.05, 0.05),
                      rotate=(-a.get("rotate", 180), a.get("rotate", 180)), p=0.7))
    t.append(RandomCLAHE(clip=(1.0, 3.0), p=a.get("clahe_p", 0.25)))
    cj = a.get("color", {})
    t.append(A.ColorJitter(brightness=cj.get("brightness", 0.25), contrast=cj.get("contrast", 0.25),
                           saturation=cj.get("saturation", 0.2), hue=cj.get("hue", 0.03), p=cj.get("p", 0.8)))
    t.append(A.RandomGamma(gamma_limit=tuple(a.get("gamma", (70, 150))), p=a.get("gamma_p", 0.4)))
    noise = _try([lambda: A.GaussNoise(std_range=(0.01, 0.05), p=1), lambda: A.GaussNoise(var_limit=(5.0, 30.0), p=1)])
    comp = _try([lambda: A.ImageCompression(quality_range=tuple(a.get("jpeg_quality", (30, 90))), p=1),
                 lambda: A.ImageCompression(quality_lower=a.get("jpeg_quality", (30, 90))[0],
                                            quality_upper=a.get("jpeg_quality", (30, 90))[1], p=1)])
    down = _try([lambda: A.Downscale(scale_range=(0.5, 0.9), p=1), lambda: A.Downscale(scale_min=0.5, scale_max=0.9, p=1)])
    opts = [A.GaussianBlur(blur_limit=(3, 7), p=1)] + [x for x in (noise, comp, down) if x]
    t.append(A.OneOf(opts, p=a.get("quality_p", 0.5)))
    t += [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2(transpose_mask=True)]
    return A.Compose(t)


def get_train_transform(cfg):
    a = cfg["augment"]
    if a.get("camera_robust", False):
        return get_camera_robust_transform(cfg)
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


@torch.no_grad()
def evaluate_negatives(model, ds, cfg, device, amp):
    """False-positive check on healthy held-out images (no lesions at all).
    Returns per-class: mean #px > dice_threshold per image, and mean max-prob per image."""
    v = cfg["validation"]
    C = len(cfg["data"]["lesion_classes"])
    fp_px, pmax = np.zeros(C), np.zeros(C)
    model.eval()
    for i in range(len(ds)):
        x, _ = ds[i]
        prob = sliding_window_predict(model, x, device, v["tile"], v["overlap"], amp)
        fp_px += (prob > v["dice_threshold"]).reshape(C, -1).sum(1)
        pmax += prob.reshape(C, -1).max(1)
    n = max(1, len(ds))
    return (fp_px / n).tolist(), (pmax / n).tolist()


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
    neg_root = d.get("negatives_root")
    allow = bool(d.get("allow_idrid_grading_negatives", False))
    neg_train = list_negatives(neg_root, d.get("negative_train_ids") or [], allow)
    neg_val = list_negatives(neg_root, d.get("negative_val_ids") or [], allow)
    excl = set(map(str, d.get("negative_exclude_ids") or []))     # e.g. RFMiD threshold-tuning set
    bad = excl & ({x["id"] for x in neg_train} | {x["id"] for x in neg_val})
    assert not bad, f"negatives overlap the reserved threshold-tuning set: {sorted(bad)[:5]}"
    overlap = {x["id"] for x in neg_train} & {x["id"] for x in neg_val}
    assert not overlap, f"negative train/val overlap: {overlap}"
    train_items = train_items + neg_train
    print(f"Healthy negatives: train {len(neg_train)}  val(FP check) {len(neg_val)}")

    print(f"Preprocessing: { {k: d.get(k) for k in PREPROC_KEYS} }  |  augment: {cfg['augment']}")
    train_ds = PatchTrainDataset(train_items, cfg, get_train_transform(cfg))
    val_ds = FullImageValDataset(val_items, cfg)
    neg_val_ds = FullImageValDataset(neg_val, cfg) if neg_val else None
    sel = cfg["training"].get("select_classes") or classes     # checkpoint-selection classes
    sel_idx = [classes.index(c) for c in sel]
    print(f"Checkpoint selection: mean AUPR over {sel}")
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
        sel_vals = [aupr[k] for k in sel_idx]
        score = float(np.nanmean(sel_vals)) if not np.all(np.isnan(sel_vals)) else 0.0
        neg = None
        if neg_val_ds is not None:
            fp_px, pmax = evaluate_negatives(model, neg_val_ds, cfg, device, amp)
            neg = {"fp_px_per_img": dict(zip(classes, fp_px)), "max_prob": dict(zip(classes, pmax))}
        fmt = lambda v: "  ".join(f"{c}={x:.3f}" for c, x in zip(classes, v))
        print(f"Epoch {epoch + 1}: train_loss={run:.4f}  mAUPR={m_aupr:.4f}  "
              f"lr(enc/dec)={lrs.get('encoder', 0):.2e}/{lrs.get('decoder', 0):.2e}\n"
              f"  AUPR: {fmt(aupr)}\n  Dice: {fmt(dice)}\n  select({'/'.join(sel)})={score:.4f}")
        if neg is not None:
            print(f"  Healthy FP px/img (>{cfg['validation']['dice_threshold']}): {fmt(fp_px)}\n"
                  f"  Healthy mean max-prob: {fmt(pmax)}")
        history.append({"epoch": epoch + 1, "train_loss": run, "mAUPR": m_aupr,
                        "aupr": dict(zip(classes, aupr)), "dice": dict(zip(classes, dice)),
                        "select_score": score, "healthy": neg})

        state = {"model": model.state_dict(), "encoder": enc, "classes": classes,
                 "epoch": epoch + 1, "mAUPR": m_aupr, "select_score": score,
                 "preprocess": {k: d.get(k) for k in PREPROC_KEYS}}   # inference copies this
        torch.save(state, ckpt_dir / "last.pth")
        with open(ckpt_dir / "history.json", "w") as f:   # written every epoch (survives timeouts)
            json.dump(history, f, indent=2)
        if score > best:
            best, patience = score, 0
            torch.save(state, ckpt_dir / "best.pth")
            print("  -> saved best")
        else:
            patience += 1
            if patience >= t["early_stopping_patience"]:
                print("Early stopping")
                break

    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. Best select score ({'/'.join(sel)})={best:.4f}  ->  {ckpt_dir / 'best.pth'}")


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
