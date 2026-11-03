"""
Inference + multi-positive crop extraction.
Run over RFMiD (or any folder) to generate lesion-centred crops.

Pipeline per image (same preprocessing as training):
  fundus crop -> resize long side -> sliding-window 512 tiles -> sigmoid (4, H, W)
  -> per-class connected components (confidence = MAX prob inside the component)
  -> global ranking across classes -> NMS -> top-k lesions -> multi-scale jittered views
  -> boxes mapped back to ORIGINAL image pixels.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from common import (load_config, seed_everything, preprocess_image, normalize,
                    sliding_window_predict, to_original_xy)
from model import build_model

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def load_model(ckpt_path, cfg, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        enc, classes, sd = state["encoder"], state["classes"], state["model"]
    else:  # bare state_dict
        enc, classes, sd = cfg["model"]["encoder"], cfg["data"]["lesion_classes"], state
    model, enc = build_model(enc, len(classes), pretrained=False,
                             fallbacks=cfg["model"].get("encoder_fallbacks", ()))
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, classes


def find_candidates(probs, classes, cfg):
    """All components over all classes, in resized space."""
    inf = cfg["inference"]
    wanted = set(inf.get("lesion_classes", classes))
    cands = []
    for c, name in enumerate(classes):
        if name not in wanted:
            continue
        p = probs[c]
        binary = (p > inf["conf_threshold"]).astype(np.uint8)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < inf["min_area"]:
                continue
            sl = (slice(y, y + h), slice(x, x + w))
            comp = labels[sl] == i
            conf = float(p[sl][comp].max())
            # centre = location of the max-probability pixel (always inside the blob);
            # for large blobs use the centroid if it lies inside the component.
            cx, cy = cents[i]
            if not (0 <= int(cy) - y < h and 0 <= int(cx) - x < w and comp[int(cy) - y, int(cx) - x]):
                yy, xx = np.unravel_index(np.argmax(np.where(comp, p[sl], -1)), comp.shape)
                cx, cy = x + xx, y + yy
            cands.append({"lesion_type": name, "cx": float(cx), "cy": float(cy),
                          "confidence": conf, "area": int(area)})
    cands.sort(key=lambda d: d["confidence"], reverse=True)   # global ranking
    return cands


def square_box(cx, cy, size, W, H):
    """Square box of `size` centred at (cx, cy), shifted (not truncated) to fit in WxH."""
    size = int(round(min(size, W, H)))
    x1 = int(round(np.clip(cx - size / 2, 0, W - size)))
    y1 = int(round(np.clip(cy - size / 2, 0, H - size)))
    return [x1, y1, x1 + size, y1 + size]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms_topk(cands, crop_size, Wr, Hr, iou_thr, k):
    kept = []
    for c in cands:
        box = square_box(c["cx"], c["cy"], crop_size, Wr, Hr)
        if all(iou(box, kk["box_resized"]) < iou_thr for kk in kept):
            kept.append({**c, "box_resized": box})
            if len(kept) >= k:
                break
    return kept


def make_views(lesions, meta, cfg, image_id, rng):
    """Multi-scale jittered views per lesion; boxes in ORIGINAL image coordinates."""
    inf = cfg["inference"]
    W, H = meta["orig_w"], meta["orig_h"]
    s_mean = (meta["sx"] + meta["sy"]) / 2   # resized px per original px
    out = []
    for li, les in enumerate(lesions):
        for v in range(inf["crops_per_lesion"]):
            scale = inf["scale_factors"][v % len(inf["scale_factors"])]
            j = inf["jitter"]
            jx, jy = (rng.integers(-j, j + 1, size=2) if j > 0 else (0, 0))
            ox, oy = to_original_xy(les["cx"] + jx, les["cy"] + jy, meta)
            size_orig = inf["crop_size"] * scale / s_mean
            box = square_box(ox, oy, size_orig, W, H)
            if box[2] - box[0] < 32:
                continue
            cx0, cy0 = to_original_xy(les["cx"], les["cy"], meta)
            out.append({"image_id": image_id, "lesion_idx": li, "lesion_type": les["lesion_type"],
                        "bbox": box, "center": [round(cx0, 1), round(cy0, 1)],
                        "confidence": round(les["confidence"], 4), "area_resized": les["area"],
                        "scale": scale, "view_idx": v})
    return out


def run_inference(config_path, input_dir, output_dir, ckpt, limit=None):
    cfg = load_config(config_path)
    seed = cfg.get("seed", 42)
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg["training"].get("mixed_precision", True)) and device.type == "cuda"
    model, classes = load_model(ckpt, cfg, device)
    inf, d = cfg["inference"], cfg["data"]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    crop_dir, prob_dir = out_path / "images", out_path / "prob_maps"
    if inf.get("save_crop_images", True):
        crop_dir.mkdir(exist_ok=True)
    if inf.get("save_prob_maps", False):
        prob_dir.mkdir(exist_ok=True)

    files = sorted(p for p in Path(input_dir).rglob("*") if p.suffix.lower() in IMG_EXTS)
    if limit:
        files = files[:limit]
    all_meta, n_no_lesion = [], 0
    for img_path in tqdm(files, desc="Inference"):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_r, meta = preprocess_image(rgb, d["long_side"], d["fundus_threshold"])
        probs = sliding_window_predict(model, normalize(img_r), device,
                                       inf["tile"], inf["overlap"], amp)
        Hr, Wr = img_r.shape[:2]
        cands = find_candidates(probs, classes, cfg)
        lesions = nms_topk(cands, inf["crop_size"], Wr, Hr, inf["nms_iou"],
                           inf["max_lesions_per_image"])
        views = make_views(lesions, meta, cfg, img_path.stem, rng)
        n_no_lesion += int(not views)

        if inf.get("save_prob_maps", False):
            np.savez_compressed(prob_dir / f"{img_path.stem}.npz",
                                probs=(probs * 255).astype(np.uint8), meta=json.dumps(meta))
        if inf.get("save_crop_images", True):
            for i, m in enumerate(views):
                x1, y1, x2, y2 = m["bbox"]
                name = f"{img_path.stem}_{m['lesion_type']}_L{m['lesion_idx']}_v{m['view_idx']}.png"
                cv2.imwrite(str(crop_dir / name), bgr[y1:y2, x1:x2])
                m["crop_path"] = str(crop_dir / name)
        all_meta.extend(views)

    with open(out_path / "metadata.json", "w") as f:
        json.dump(all_meta, f, indent=2)
    print(f"Done. {len(all_meta)} crops from {len(files)} images "
          f"({n_no_lesion} images had no lesion above threshold).")
    print(f"Metadata: {out_path / 'metadata.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input_dir", required=True, help="Folder of images (e.g. RFMiD), searched recursively")
    parser.add_argument("--output_dir", default="crops")
    parser.add_argument("--ckpt", default="checkpoints/best.pth")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_inference(args.config, args.input_dir, args.output_dir, args.ckpt, args.limit)
