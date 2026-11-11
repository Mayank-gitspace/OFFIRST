#!/usr/bin/env python3
"""IDRiD lesion-visibility audit under FOV-crop + INTER_AREA resize to 384/512/768.
Measurement only. IDRiD is TEST-ONLY — do not tune degrader/thresholds on these numbers.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(".")
IDRID = ROOT / "data/idrid_seg/A. Segmentation"
OUT = ROOT / "kd_imaging/resolution_audit"
sys.path.insert(0, str(ROOT / "kd_imaging"))
from degrade import DegradeParams, degrade  # noqa: E402

SIZES = [384, 512, 768]
LESION_TYPES = [
    ("MA", "1. Microaneurysms"),
    ("HE", "2. Haemorrhages"),
    ("EX", "3. Hard Exudates"),
    ("SE", "4. Soft Exudates"),
]
CNR_THRESHOLDS = (1.0, 2.0, 3.0)
RING_DILATE = 5


def load_params() -> DegradeParams:
    raw = json.loads((ROOT / "kd_imaging/visuscout_params.json").read_text())["params"]
    keys = {f.name for f in DegradeParams.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return DegradeParams(**{k: v for k, v in raw.items() if k in keys})


def crop_fundus_with_meta(img: np.ndarray, thr: int = 15):
    h, w = img.shape[:2]
    sc = 512.0 / max(h, w)
    small = cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
    mask = (small.max(axis=2) > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    ys, xs = np.where(mask > 0)
    if len(xs) < 100:
        return img.copy(), {"x0": 0, "x1": w, "y0": 0, "y1": h, "m": max(h, w), "pad_y": 0, "pad_x": 0}
    x0, x1 = int(xs.min() / sc), int((xs.max() + 1) / sc)
    y0, y1 = int(ys.min() / sc), int((ys.max() + 1) / sc)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    cropped = img[y0:y1, x0:x1]
    ch, cw = cropped.shape[:2]
    m = max(ch, cw)
    out = np.zeros((m, m, img.shape[2]), img.dtype)
    py, px = (m - ch) // 2, (m - cw) // 2
    out[py : py + ch, px : px + cw] = cropped
    return out, {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "m": m, "pad_y": py, "pad_x": px}


def apply_crop_meta(arr: np.ndarray, meta: dict) -> np.ndarray:
    cropped = arr[meta["y0"] : meta["y1"], meta["x0"] : meta["x1"]]
    ch, cw = cropped.shape[:2]
    m = meta["m"]
    out = np.zeros((m, m), arr.dtype)
    py, px = meta["pad_y"], meta["pad_x"]
    out[py : py + ch, px : px + cw] = cropped
    return out


def resize_image(img: np.ndarray, s: int) -> np.ndarray:
    return cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)


def connected_components(mask01: np.ndarray):
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask01 > 0).astype(np.uint8), connectivity=8)
    out = []
    for lab in range(1, n):
        x, y, w, h, area = stats[lab]
        if area < 1:
            continue
        m = labels[y : y + h, x : x + w] == lab
        out.append((lab, m, int(area), int(x), int(y), int(w), int(h)))
    return out


def eq_diameter(area: float) -> float:
    return float(np.sqrt(4.0 * area / np.pi)) if area > 0 else 0.0


def lesion_region_from_cov(cov: np.ndarray, thr: float = 0.5):
    sel = cov >= thr
    if sel.any():
        return sel
    flat = int(np.argmax(cov))
    yy, xx = divmod(flat, cov.shape[1])
    sel = np.zeros_like(cov, dtype=bool)
    sel[yy, xx] = True
    return sel


def cnr_on_arrays(g: np.ndarray, loc: np.ndarray, ring_r: int) -> float:
    """g: float green window; loc: bool same shape."""
    if loc.sum() == 0:
        return float("nan")
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_r + 1, 2 * ring_r + 1))
    dil = cv2.dilate(loc.astype(np.uint8), kern, iterations=1).astype(bool)
    ring = dil & ~loc & (g > 5)
    if ring.sum() < 5:
        kern2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (ring_r + 3) + 1, 2 * (ring_r + 3) + 1))
        dil2 = cv2.dilate(loc.astype(np.uint8), kern2, iterations=1).astype(bool)
        ring = dil2 & ~loc & (g > 5)
    if ring.sum() < 5:
        return float("nan")
    sd = g[ring].std()
    if sd < 1e-6:
        return float("nan")
    return float(abs(g[loc].mean() - g[ring].mean()) / sd)


def cnr_native(img_bgr: np.ndarray, m_local: np.ndarray, bx: int, by: int, bw: int, bh: int, ring_r: int) -> float:
    pad = ring_r + 4
    y0, y1 = max(0, by - pad), min(img_bgr.shape[0], by + bh + pad)
    x0, x1 = max(0, bx - pad), min(img_bgr.shape[1], bx + bw + pad)
    g = img_bgr[y0:y1, x0:x1, 1].astype(np.float64)
    loc = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    loc[by - y0 : by - y0 + bh, bx - x0 : bx - x0 + bw] = m_local
    return cnr_on_arrays(g, loc, ring_r)


def cnr_at_size(img_s: np.ndarray, m_local: np.ndarray, bx: int, by: int, bw: int, bh: int,
                native: int, s: int, ring_r: int) -> tuple[float, float]:
    """Resize local CC patch consistently with full-frame INTER_AREA scale; return (cnr, eq_diam_px)."""
    # Coverage sum under full-frame INTER_AREA ≈ area * (s/native)^2
    area_s = float(m_local.sum()) * (s / native) ** 2
    diam_s = eq_diameter(area_s)

    # Exact full-frame INTER_AREA pixel overlap, evaluated only near this CC.
    scale = native / s
    ix0 = max(0, int(np.floor(bx / scale)) - ring_r - 4)
    iy0 = max(0, int(np.floor(by / scale)) - ring_r - 4)
    ix1 = min(s, int(np.ceil((bx + bw) / scale)) + ring_r + 4)
    iy1 = min(s, int(np.ceil((by + bh) / scale)) + ring_r + 4)
    def weights(start, stop, src_start, src_len):
        lo = np.arange(start, stop, dtype=float)[:, None] * scale
        hi = lo + scale
        src = np.arange(src_start, src_start + src_len, dtype=float)[None, :]
        return np.maximum(0, np.minimum(hi, src + 1) - np.maximum(lo, src)) / scale
    wx = weights(ix0, ix1, bx, bw)
    wy = weights(iy0, iy1, by, bh)
    cov = wy @ m_local.astype(float) @ wx.T
    loc = lesion_region_from_cov(cov, 0.5)
    g = img_s[iy0:iy1, ix0:ix1, 1].astype(np.float64)
    return cnr_on_arrays(g, loc, ring_r), diam_s


def find_mask(stem: str, lesion_code: str, folder: str) -> Path | None:
    for split in ("a. Training Set", "b. Testing Set"):
        p = IDRID / "2. All Segmentation Groundtruths" / split / folder / f"{stem}_{lesion_code}.tif"
        if p.exists():
            return p
    return None


def find_image(stem: str) -> Path | None:
    for split in ("a. Training Set", "b. Testing Set"):
        p = IDRID / "1. Original Images" / split / f"{stem}.jpg"
        if p.exists():
            return p
    return None


def list_stems():
    stems = []
    for split in ("a. Training Set", "b. Testing Set"):
        for p in sorted((IDRID / "1. Original Images" / split).glob("IDRiD_*.jpg")):
            stems.append(p.stem)
    return stems


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["lesion_type"], r["size"], r["condition"])].append(r)
    out = []
    for (lt, size, cond), items in sorted(groups.items()):
        diams = np.array([x["eq_diam_px"] for x in items], dtype=float)
        cnrs = np.array([x["cnr"] for x in items], dtype=float)
        cnrs_ok = cnrs[np.isfinite(cnrs)]
        n = len(items)
        out.append({
            "lesion_type": lt,
            "size": size,
            "condition": cond,
            "n_lesions": n,
            "median_eq_diam_px": float(np.median(diams)) if n else float("nan"),
            "pct_diam_lt_1px": float(100.0 * (diams < 1.0).mean()) if n else float("nan"),
            "median_cnr": float(np.median(cnrs_ok)) if len(cnrs_ok) else float("nan"),
            "pct_cnr_ge_1": float(100.0 * np.mean(cnrs >= 1.0)) if n else float("nan"),
            "pct_cnr_ge_2": float(100.0 * np.mean(cnrs >= 2.0)) if n else float("nan"),
            "pct_cnr_ge_3": float(100.0 * np.mean(cnrs >= 3.0)) if n else float("nan"),
            "pct_cnr_nan": float(100.0 * np.mean(~np.isfinite(cnrs))) if n else float("nan"),
        })
    return out


def per_image_ma_visible(rows, cnr_thr=2.0):
    by_img = defaultdict(lambda: defaultdict(list))
    images_with_ma = set()
    for r in rows:
        if r["lesion_type"] != "MA":
            continue
        if r["size"] == "original" and r["condition"] == "clean":
            images_with_ma.add(r["image"])
        by_img[r["image"]][(r["size"], r["condition"])].append(r["cnr"])
    out = []
    for size in ["original"] + [str(s) for s in SIZES]:
        for cond in ("clean", "degraded"):
            vis = total = 0
            for img in sorted(images_with_ma):
                cnrs = by_img[img].get((size, cond))
                if not cnrs:
                    continue
                total += 1
                if any(np.isfinite(c) and c >= cnr_thr for c in cnrs):
                    vis += 1
            out.append({
                "size": size,
                "condition": cond,
                "cnr_thr": cnr_thr,
                "n_images_with_ma": total,
                "n_images_with_ge1_visible_ma": vis,
                "frac_images_with_ge1_visible_ma": (vis / total) if total else float("nan"),
            })
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    params = load_params()
    print("DegradeParams:", asdict(params), flush=True)
    stems = list_stems()
    print(f"Images: {len(stems)}", flush=True)

    lesion_rows = []
    fig_candidates = []

    for i, stem in enumerate(stems):
        ipath = find_image(stem)
        if ipath is None:
            continue
        bgr = cv2.imread(str(ipath), cv2.IMREAD_COLOR)
        if bgr is None:
            print("skip unreadable", stem, flush=True)
            continue

        square, meta = crop_fundus_with_meta(bgr)
        deg = degrade(square, DegradeParams(**{**asdict(params), "seed": i}))
        native = square.shape[0]

        imgs_s = {s: {"clean": resize_image(square, s), "degraded": resize_image(deg, s)} for s in SIZES}

        masks = {}
        for code, folder in LESION_TYPES:
            mp = find_mask(stem, code, folder)
            if mp is None:
                continue
            raw = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
            if raw is None:
                continue
            if raw.ndim == 3:
                raw = raw.max(axis=2)
            masks[code] = apply_crop_meta((raw > 0).astype(np.uint8), meta)

        n_cc = 0
        ring0 = max(3, RING_DILATE)
        for code, mask01 in masks.items():
            ccs = connected_components(mask01)
            n_cc += len(ccs)
            for lab, m_local, area, bx, by, bw, bh in ccs:
                diam0 = eq_diameter(area)

                # original clean/degraded
                for cond, img in (("clean", square), ("degraded", deg)):
                    cnr = cnr_native(img, m_local, bx, by, bw, bh, ring0)
                    lesion_rows.append({
                        "image": stem, "lesion_type": code, "lesion_id": int(lab),
                        "size": "original", "condition": cond,
                        "area_px_native": int(area), "eq_diam_native": diam0,
                        "eq_diam_px": diam0, "cnr": cnr,
                    })

                for s in SIZES:
                    ring_s = max(2, int(round(RING_DILATE * (s / native))))
                    for cond in ("clean", "degraded"):
                        cnr, diam_s = cnr_at_size(
                            imgs_s[s][cond], m_local, bx, by, bw, bh, native, s, ring_s
                        )
                        lesion_rows.append({
                            "image": stem, "lesion_type": code, "lesion_id": int(lab),
                            "size": str(s), "condition": cond,
                            "area_px_native": int(area), "eq_diam_native": diam0,
                            "eq_diam_px": diam_s, "cnr": cnr,
                        })

                if code == "MA" and 5 <= area <= 80:
                    cnr0 = cnr_native(square, m_local, bx, by, bw, bh, ring0)
                    if np.isfinite(cnr0) and cnr0 >= 2.0:
                        fig_candidates.append((cnr0, -area, stem, int(lab), by + bh // 2, bx + bw // 2))

        print(f"  processed {i+1}/{len(stems)}  rows={len(lesion_rows)}  ccs={n_cc}  shape={square.shape}", flush=True)

    fields = ["image", "lesion_type", "lesion_id", "size", "condition", "area_px_native", "eq_diam_native", "eq_diam_px", "cnr"]
    with (OUT / "lesion_level.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(lesion_rows)

    summary = summarize(lesion_rows)
    sum_fields = ["lesion_type", "size", "condition", "n_lesions", "median_eq_diam_px", "pct_diam_lt_1px",
                  "median_cnr", "pct_cnr_ge_1", "pct_cnr_ge_2", "pct_cnr_ge_3", "pct_cnr_nan"]
    with (OUT / "summary_by_type_size.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields); w.writeheader(); w.writerows(summary)

    per_img_rows = []
    for thr in CNR_THRESHOLDS:
        per_img_rows.extend(per_image_ma_visible(lesion_rows, cnr_thr=thr))
    with (OUT / "per_image_ma_visible.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_img_rows[0].keys()) if per_img_rows else ["size"])
        w.writeheader(); w.writerows(per_img_rows)

    # Figure
    fig_candidates.sort(reverse=True)
    picked, seen = [], set()
    for item in fig_candidates:
        if item[2] in seen:
            continue
        seen.add(item[2]); picked.append(item)
        if len(picked) >= 3:
            break

    cell, header_h = 96, 28
    cols_labels = ["orig clean", "768c", "512c", "384c", "orig deg", "768d", "512d", "384d"]
    fig_path = OUT / "example_ma_crops.png"
    if picked:
        rows_img = []
        for cnr0, neg_area, stem, lab, cy, cx in picked:
            bgr = cv2.imread(str(find_image(stem)), cv2.IMREAD_COLOR)
            square, meta = crop_fundus_with_meta(bgr)
            seed = stems.index(stem)
            deg_img = degrade(square, DegradeParams(**{**asdict(params), "seed": seed}))
            mp = find_mask(stem, "MA", "1. Microaneurysms")
            raw = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
            if raw.ndim == 3:
                raw = raw.max(axis=2)
            mask01 = apply_crop_meta((raw > 0).astype(np.uint8), meta)
            _, labels = cv2.connectedComponents(mask01, connectivity=8)
            mbool = labels == lab
            row_panels = []
            for cond_img in (square, deg_img):
                for s in [None, 768, 512, 384]:
                    if s is None:
                        half = 48
                        y0, y1 = max(0, cy - half), min(cond_img.shape[0], cy + half)
                        x0, x1 = max(0, cx - half), min(cond_img.shape[1], cx + half)
                        crop = cond_img[y0:y1, x0:x1].copy()
                        local = mbool[y0:y1, x0:x1]
                    else:
                        img_s = resize_image(cond_img, s)
                        cov = cv2.resize(mbool.astype(np.float32), (s, s), interpolation=cv2.INTER_AREA)
                        region = cov >= 0.5
                        if not region.any():
                            flat = int(np.argmax(cov)); yy, xx = divmod(flat, s); region = np.zeros_like(cov, dtype=bool); region[yy, xx] = True
                        ys, xs = np.where(region)
                        cys, cxs = int(ys.mean()), int(xs.mean())
                        half = max(12, int(48 * s / square.shape[0]))
                        y0, y1 = max(0, cys - half), min(s, cys + half)
                        x0, x1 = max(0, cxs - half), min(s, cxs + half)
                        crop = img_s[y0:y1, x0:x1].copy()
                        local = region[y0:y1, x0:x1]
                    cont, _ = cv2.findContours(local.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(crop, cont, -1, (0, 255, 255), 1)
                    row_panels.append(cv2.resize(crop, (cell, cell), interpolation=cv2.INTER_NEAREST))
            strip = np.zeros((header_h, cell * 8, 3), np.uint8)
            for j, labtxt in enumerate(cols_labels):
                cv2.putText(strip, labtxt, (j * cell + 4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            title = np.zeros((header_h, cell * 8, 3), np.uint8)
            cv2.putText(title, f"{stem} MA#{lab} native_area={-neg_area}px CNR0={cnr0:.2f}", (4, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1, cv2.LINE_AA)
            rows_img.append(np.vstack([title, strip, np.hstack(row_panels)]))
        cv2.imwrite(str(fig_path), np.vstack(rows_img))
        print("Wrote", fig_path, flush=True)
    else:
        print("No figure candidates", flush=True)

    def fmt(x, nd=1):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "NA"
        return f"{x:.{nd}f}"

    n_ma = sum(1 for s in stems if find_mask(s, "MA", "1. Microaneurysms"))
    lines = [
        "# IDRiD resolution survival audit",
        "",
        "**Purpose:** measure how well annotated lesions remain spatially and contrast-visible after the OFFIRST FOV-crop + `cv2.INTER_AREA` resize to model input sizes 384 / 512 / 768. Informs whether a phone student (MobileNetV3 / EfficientNet) should use 384 or larger.",
        "",
        "**Not done here:** no model training; no degrader/threshold tuning; no claim that CNR visibility equals model detectability.",
        "",
        "## Data",
        "",
        f"- Source on box: `{IDRID}` (IDRiD Diabetic Retinopathy Segmentation subset, CC-BY-4.0).",
        f"- Images used: **{len(stems)}** (54 train + 27 test JPG originals). MA masks present for {n_ma} images; HE/EX/SE used when present.",
        "- Jeet policy: IDRiD is **TEST-ONLY**. This audit only measures signal visibility; numbers must not be used to choose degrader parameters or CNR thresholds for training.",
        "",
        "## Resize method (matched to pipeline)",
        "",
        "- FOV crop + pad-to-square: same as `crop_fundus` in `encoder_byol/v4/kernels/full/team_offirst_encoder_byol_anatomy_v4.py` (threshold max-channel > 15 on a 512-max preview, morphological open, bbox crop, pad to square).",
        "- Resize: `cv2.resize(..., (S,S), interpolation=cv2.INTER_AREA)` — same as `load_prep` / eval extract helpers.",
        "- Masks: same crop meta; exact full-frame INTER_AREA pixel-overlap weights evaluated on each connected-component patch (no pixel-grid phase approximation); lesion pixels at S = coverage ≥ 0.5 (else single max-coverage pixel). Equivalent diameter uses area scaling `area*(S/native)^2` (matches INTER_AREA mass conservation for an isolated component).",
        "",
        "## Degrader",
        "",
        "- `kd_imaging/degrade.py` with params from `visuscout_params.json` (fitted; JSON exposure=1.04, not the 1.12 code default).",
        "- Order: apply degrader on the **FOV-cropped native square**, then resize (blur sigma scales with width/1024 inside `degrade`). Native-first order follows demo.py, but crop-before-degrade is an audit convention, not an implemented student training pipeline. Demo degrades the uncropped rectangle; this changes spatial calibration slightly.",
        "- Params were **not** modified.",
        "",
        "## Visibility proxy",
        "",
        "- CNR on green channel: `|mean(G_lesion) − mean(G_ring)| / std(G_ring)`, ring = dilated lesion \\\\ lesion (tight window around each lesion).",
        "- **CNR ≥ 2 is an arbitrary heuristic** for 'visible'. Also report ≥1 and ≥3.",
        "- This is a **proxy**, not proof a CNN can or cannot detect the lesion.",
        "",
        "## Caveats",
        "",
        "- Ring radius is 5 native pixels, minimum 2 output pixels. Thus its physical support changes with resolution; resizing also smooths background variance. CNR can rise after blur/downsampling and is NOT a monotonic contrast-survival score. Do not interpret visible fractions as sensitivity or use them to select an input size.",
        "- IDRiD segmentation images are lesion-rich; at-least-one-MA rates saturate and are not mild-DR sensitivity estimates.",
        "- v4 actually caches at 512 then resizes to 384; this audit directly resizes native crops at each size to isolate final resolution, not the exact two-stage cache path.",
        "",
        "- IDRiD camera ≠ RFMiD / Visuscout; absolute rates may not transfer.",
        "- Degrader fit on 24 Visuscout frames — approximate portable look only.",
        "- Compute cost scales ~(S/384)^2 (512≈1.78×, 768≈4× vs 384).",
        "- Test-only usage; do not tune on these outcomes.",
        "",
        "## Summary table (lesion-level)",
        "",
        "| type | size | cond | n | med diam px | % diam<1 | med CNR | % CNR≥1 | % CNR≥2 | % CNR≥3 |",
        "|------|------|------|---|-------------|----------|---------|---------|---------|---------|",
    ]
    for r in summary:
        lines.append(
            f"| {r['lesion_type']} | {r['size']} | {r['condition']} | {r['n_lesions']} | "
            f"{fmt(r['median_eq_diam_px'], 2)} | {fmt(r['pct_diam_lt_1px'], 1)} | {fmt(r['median_cnr'], 2)} | "
            f"{fmt(r['pct_cnr_ge_1'], 1)} | {fmt(r['pct_cnr_ge_2'], 1)} | {fmt(r['pct_cnr_ge_3'], 1)} |"
        )
    lines += [
        "",
        "## Per-image: fraction with ≥1 visible MA (mild-DR relevant)",
        "",
        "| size | cond | CNR thr | n images with MA | frac with ≥1 visible MA |",
        "|------|------|---------|------------------|-------------------------|",
    ]
    for r in per_img_rows:
        lines.append(
            f"| {r['size']} | {r['condition']} | {r['cnr_thr']} | {r['n_images_with_ma']} | "
            f"{fmt(100 * r['frac_images_with_ge1_visible_ma'], 1)}% |"
        )
    lines += [
        "",
        "## Files",
        "",
        "- `run_audit.py` — this script",
        "- `lesion_level.csv` — per-lesion rows",
        "- `summary_by_type_size.csv`",
        "- `per_image_ma_visible.csv`",
        "- `example_ma_crops.png` — example MA crops at original/768/512/384 clean+degraded",
        "- `RESULTS.md` — this file",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines))
    print("Wrote", OUT / "RESULTS.md", flush=True)
    print("Done. Lesion rows:", len(lesion_rows), flush=True)


if __name__ == "__main__":
    main()
