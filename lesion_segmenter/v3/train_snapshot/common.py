"""
Shared helpers: config, seeding, fundus crop + resize, normalisation,
sliding-window prediction and AMP helpers. Used by train.py and inference.py so
training and inference see exactly the same preprocessing.
"""
import random
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import yaml

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # YAML 1.1 reads "1e-4" as a string — coerce the numeric training fields.
    coerce_training_numbers(cfg)
    return cfg


def coerce_training_numbers(cfg):
    """Also call after --set overrides (yaml.safe_load("1e-4") is a string)."""
    t = cfg["training"]
    for k in ("lr", "weight_decay", "encoder_lr"):
        if k in t and t[k] is not None:
            t[k] = float(t[k])
    t["grad_accum_steps"] = max(1, int(t.get("grad_accum_steps", 1) or 1))
    return cfg


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- fundus

def fundus_bbox(img_rgb, threshold=15, margin=4):
    """Bounding box (x0, y0, x1, y1) of the fundus circle (largest bright blob)."""
    h, w = img_rgb.shape[:2]
    gray = img_rgb.max(axis=2) if img_rgb.ndim == 3 else img_rgb
    mask = (gray > threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return 0, 0, w, h
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, area = stats[i]
    if area < 0.1 * h * w:          # detection failed -> use the whole image
        return 0, 0, w, h
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(w, x + bw + margin), min(h, y + bh + margin)
    return int(x0), int(y0), int(x1), int(y1)


def preprocess_image(img_rgb, long_side=1024, threshold=15):
    """Crop to fundus, resize long side. Returns (img, meta) where meta maps back."""
    x0, y0, x1, y1 = fundus_bbox(img_rgb, threshold)
    crop = img_rgb[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    scale = long_side / max(ch, cw)
    nw, nh = max(1, round(cw * scale)), max(1, round(ch * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    out = cv2.resize(crop, (nw, nh), interpolation=interp)
    # exact per-axis scale (rounding) for mapping back
    meta = {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "sx": nw / cw, "sy": nh / ch,
            "orig_w": img_rgb.shape[1], "orig_h": img_rgb.shape[0]}
    return out, meta


def preprocess_mask(mask_hwc, meta, out_hw):
    """Apply the same crop/resize to a (H, W, C) binary mask.
    INTER_AREA + '>0' keeps tiny microaneurysms alive after downsampling."""
    x0, y0, x1, y1 = meta["x0"], meta["y0"], meta["x1"], meta["y1"]
    crop = mask_hwc[y0:y1, x0:x1].astype(np.float32)
    out = cv2.resize(crop, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_AREA)
    if out.ndim == 2:
        out = out[..., None]
    return (out > 0).astype(np.uint8)


def to_original_xy(x, y, meta):
    """Map a point from resized-fundus space back to original-image pixels."""
    return meta["x0"] + x / meta["sx"], meta["y0"] + y / meta["sy"]


def normalize(img_rgb_uint8):
    x = img_rgb_uint8.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


# ----------------------------------------------------------------------------- AMP

def autocast_ctx(device, enabled):
    if enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_grad_scaler(device, enabled):
    enabled = bool(enabled and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):      # torch < 2.3
        return torch.cuda.amp.GradScaler(enabled=enabled)


# ----------------------------------------------------------------------------- sliding window

def _starts(length, tile, stride):
    if length <= tile:
        return [0]
    s = list(range(0, length - tile + 1, stride))
    if s[-1] != length - tile:
        s.append(length - tile)
    return s


@torch.no_grad()
def sliding_window_predict(model, img_tensor, device, tile=512, overlap=128,
                           amp=True, batch=4):
    """img_tensor: (3, H, W) normalised. Returns sigmoid probs (C, H, W) float32 numpy."""
    _, H, W = img_tensor.shape
    ph, pw = max(0, tile - H), max(0, tile - W)
    if ph or pw:   # pad small images up to one tile
        img_tensor = torch.nn.functional.pad(img_tensor, (0, pw, 0, ph))
    _, Hp, Wp = img_tensor.shape
    stride = max(1, tile - overlap)
    coords = [(y, x) for y in _starts(Hp, tile, stride) for x in _starts(Wp, tile, stride)]

    acc = None
    cnt = torch.zeros((1, Hp, Wp), dtype=torch.float32)
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        x = torch.stack([img_tensor[:, y:y + tile, x_:x_ + tile] for y, x_ in chunk]).to(device)
        with autocast_ctx(device, amp):
            logits = model(x)
        probs = torch.sigmoid(logits.float()).cpu()
        if acc is None:
            acc = torch.zeros((probs.shape[1], Hp, Wp), dtype=torch.float32)
        for p, (y, x_) in zip(probs, chunk):
            acc[:, y:y + tile, x_:x_ + tile] += p
            cnt[:, y:y + tile, x_:x_ + tile] += 1
    out = (acc / cnt.clamp(min=1))[:, :H, :W]
    return out.numpy()
