"""
Fix #4 — field-of-view (FOV) edge mask for the crop picker.

The bright retina disc ends in a sharp bright->black rim (plus glare / vignetting),
which the segmenter likes to fire on. We:
  1. build a retina mask from the red channel (threshold, close, fill holes, keep largest blob),
  2. erode it by `edge_margin_frac` x FOV radius -> "safe interior",
  3. multiply the probability maps by the eroded mask before blob extraction,
  4. drop any candidate whose centre lies outside the safe interior.

All work is done in the resized-fundus space (img_r from preprocess_image), i.e. the
same space as the probability maps. Disable with use_fov_mask=False to get the old
behaviour bit-for-bit.
"""
import cv2
import numpy as np


def retina_mask(img_rgb, red_threshold=12, close_frac=0.01):
    """Binary (H, W) uint8 mask of the retina disc from the red channel."""
    h, w = img_rgb.shape[:2]
    red = img_rgb[..., 0] if img_rgb.ndim == 3 else img_rgb
    m = (red > red_threshold).astype(np.uint8)
    k = max(3, int(round(close_frac * max(h, w))) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.ones((h, w), np.uint8)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (labels == i).astype(np.uint8)
    # fill holes: flood the background from the border, everything not reached is inside
    inv = (1 - m).astype(np.uint8)
    n2, lab2 = cv2.connectedComponents(inv, connectivity=4)
    border = set(np.unique(np.concatenate([lab2[0], lab2[-1], lab2[:, 0], lab2[:, -1]])))
    holes = np.isin(lab2, [l for l in range(1, n2) if l not in border])
    m[holes] = 1
    return m


def safe_interior(mask, edge_margin_frac=0.025):
    """Erode the retina mask by edge_margin_frac x equivalent-circle radius.
    Returns (eroded_mask uint8, radius_px, erode_px).
    Note: the image border counts as 'inside' for erosion (cv2 default), so a disc that
    is clipped by the frame (IDRiD top/bottom) is only eroded at its real black rim."""
    r = float(np.sqrt(mask.sum() / np.pi))
    e = int(round(edge_margin_frac * r))
    if e <= 0:
        return mask.copy(), r, 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1))
    return cv2.erode(mask, k), r, e


def apply_fov_mask(probs, img_r, cfg_inf=None, **kw):
    """probs (C, H, W) float -> (masked_probs, info dict). img_r = resized RGB fundus."""
    cfg_inf = cfg_inf or {}
    thr = kw.get("red_threshold", cfg_inf.get("fov_red_threshold", 12))
    frac = kw.get("edge_margin_frac", cfg_inf.get("edge_margin_frac", 0.025))
    m = retina_mask(img_r, thr)
    safe, r, e = safe_interior(m, frac)
    return probs * safe[None].astype(probs.dtype), {"retina": m, "safe": safe,
                                                    "radius_px": r, "erode_px": e}


def filter_candidates(cands, safe):
    """Drop candidates whose centre (cx, cy) is outside the safe interior."""
    H, W = safe.shape
    keep = []
    for c in cands:
        x, y = int(round(c["cx"])), int(round(c["cy"]))
        if 0 <= x < W and 0 <= y < H and safe[y, x]:
            keep.append(c)
    return keep
