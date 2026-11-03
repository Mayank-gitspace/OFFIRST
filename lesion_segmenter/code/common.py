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


def _crop_pad(arr, x0, y0, x1, y1):
    """arr[y0:y1, x0:x1] with zero padding where the box leaves the image."""
    h, w = arr.shape[:2]
    pt, pl, pb, pr = max(0, -y0), max(0, -x0), max(0, y1 - h), max(0, x1 - w)
    if pt or pl or pb or pr:
        pad = ((pt, pb), (pl, pr)) + ((0, 0),) * (arr.ndim - 2)
        arr = np.pad(arr, pad)
        x0, x1, y0, y1 = x0 + pl, x1 + pl, y0 + pt, y1 + pt
    return arr[y0:y1, x0:x1]


def preprocess_mask(mask_hwc, meta, out_hw):
    """Apply the same crop/resize to a (H, W, C) binary mask.
    INTER_AREA + '>0' keeps tiny microaneurysms alive after downsampling.
    Works for both "bbox" and "circle" preprocessing (circle crops may extend past the frame)."""
    x0, y0, x1, y1 = meta["x0"], meta["y0"], meta["x1"], meta["y1"]
    crop = _crop_pad(mask_hwc, x0, y0, x1, y1).astype(np.float32)
    out = cv2.resize(crop, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_AREA)
    if out.ndim == 2:
        out = out[..., None]
    out = (out > 0).astype(np.uint8)
    if "retina" in meta.get("_arrays", {}):
        out *= meta["_arrays"]["retina"][..., None]
    return out


# ----------------------------------------------------------------------------- v5: camera-robust preprocessing

def apply_clahe_lab(img_rgb, clip=2.0, tile=8, mask=None):
    """CLAHE on the L channel of LAB (colour preserved). Pixels outside `mask` stay as they were."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile))).apply(lab[..., 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if mask is not None:
        out = out * mask[..., None]
    return out


def preprocess_circle(img_rgb, diameter=1440, red_threshold=12, clahe=False, clahe_clip=2.0):
    """Camera-independent preprocessing (v5):
    retina mask (red channel) -> square box around the retina circle (diameter = max(bbox w, h),
    so frame-clipped discs such as IDRiD keep their true diameter) -> zero-pad -> resize so the
    circle diameter is `diameter` px -> black outside the retina -> optional CLAHE on LAB-L.
    Output is always diameter x diameter, so every camera gives the same retina scale."""
    from fov_mask import retina_mask
    h, w = img_rgb.shape[:2]
    f = 1024.0 / max(h, w)                       # find the circle on a small copy (fast)
    small = cv2.resize(img_rgb, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
    # adaptive threshold: faint halos / notches / labels outside the disc (IDRiD right-side notch is
    # red ~15-20) must not stretch the circle; 12% of the bright-retina red level ignores them.
    red = small[..., 0]
    bright = red[red > red_threshold]
    thr = max(red_threshold, int(0.12 * np.percentile(bright, 95))) if bright.size else red_threshold
    m = cv2.morphologyEx(retina_mask(small, thr), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    ys, xs = np.nonzero(m)
    if len(xs) < 0.1 * m.size:                   # detection failed -> whole image
        bx0, by0, bx1, by1 = 0, 0, w, h
    else:
        bx0, bx1 = xs.min() / f, (xs.max() + 1) / f
        by0, by1 = ys.min() / f, (ys.max() + 1) / f
    D = max(bx1 - bx0, by1 - by0)
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    x0, y0 = int(round(cx - D / 2)), int(round(cy - D / 2))
    Di = int(round(D))
    x1, y1 = x0 + Di, y0 + Di
    crop = _crop_pad(img_rgb, x0, y0, x1, y1)
    scale = diameter / Di
    out = cv2.resize(crop, (diameter, diameter), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    ret = retina_mask(out, thr)
    out = out * ret[..., None]
    if clahe:
        out = apply_clahe_lab(out, clahe_clip, 8, ret)
    meta = {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "sx": diameter / Di, "sy": diameter / Di,
            "orig_w": w, "orig_h": h, "mode": "circle", "fit_threshold": int(thr), "_arrays": {"retina": ret}}
    return out.astype(np.uint8), meta


PREPROC_KEYS = ("preprocess", "long_side", "fundus_threshold", "fov_diameter", "fov_red_threshold",
                "clahe", "clahe_clip")


def preprocess_from_cfg(img_rgb, d):
    """Single entry point used by train.py AND inference.py (d = cfg['data']).
    preprocess: "bbox" (old: crop fundus bbox, resize long side) | "circle" (v5, camera-robust)."""
    if d.get("preprocess", "bbox") == "circle":
        return preprocess_circle(img_rgb, int(d.get("fov_diameter") or d["long_side"]),
                                 int(d.get("fov_red_threshold", 12)), bool(d.get("clahe", False)),
                                 float(d.get("clahe_clip", 2.0)))
    return preprocess_image(img_rgb, d["long_side"], d["fundus_threshold"])


def json_meta(meta):
    """meta without the numpy arrays (for json dumps)."""
    return {k: v for k, v in meta.items() if k != "_arrays"}


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
