"""Stat vector for fundus images (clean or portable). Fundus-circle aware (ignores black corners).
Usage: python3 stats.py <folder_or_files...> --out stats.json [--csv stats.csv]"""
import cv2, numpy as np, json, sys, argparse, glob, os

BINS = 16; NRAD = 10

def fov_mask(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = (g > 15).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        m = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    return m.astype(bool)

def compute(img, size=512):
    h, w = img.shape[:2]; s = size / max(h, w)
    img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    m = fov_mask(img); f = {}
    b, g, r = [img[..., i][m].astype(np.float32) for i in range(3)]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    for name, ch in (("r", r), ("g", g), ("b", b), ("lum", lum)):
        hst = np.histogram(ch, BINS, (0, 256))[0].astype(float); f[f"hist_{name}"] = (hst / hst.sum()).round(5).tolist()
    gh = np.histogram(g, 256, (0, 256))[0]
    f["green_peak"] = int(np.argmax(cv2.GaussianBlur(gh.astype(np.float32).reshape(1, -1), (1, 9), 0)))
    f["mean_rgb"] = [float(r.mean()), float(g.mean()), float(b.mean())]
    f["ratio_r_g"] = float(r.mean() / (g.mean() + 1e-6)); f["ratio_b_g"] = float(b.mean() / (g.mean() + 1e-6))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    f["mean_L"] = float(lab[..., 0][m].mean() * 100 / 255)
    f["mean_a"] = float(lab[..., 1][m].mean() - 128); f["mean_b"] = float(lab[..., 2][m].mean() - 128)
    # sharpness on green channel, inner (eroded) FOV so rim edge doesn't dominate
    inner = cv2.erode(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    gg = img[..., 1].astype(np.float32)
    sx = cv2.Sobel(gg, cv2.CV_32F, 1, 0); sy = cv2.Sobel(gg, cv2.CV_32F, 0, 1); mag = np.hypot(sx, sy)[inner]
    f["sobel_mean"] = float(mag.mean()); f["sobel_energy"] = float((mag ** 2).mean())
    f["sobel_p95"] = float(np.percentile(mag, 95))  # edge-gradient / crude MTF proxy
    f["lap_var"] = float(cv2.Laplacian(gg, cv2.CV_32F)[inner].var())
    # radial falloff, normalized to center bin
    ys, xs = np.nonzero(m); cy, cx = ys.mean(), xs.mean(); R = np.sqrt(m.sum() / np.pi)
    Y, X = np.indices(m.shape); rad = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2) / R
    lfull = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32); prof = []
    for k in range(NRAD):
        sel = m & (rad >= k / NRAD) & (rad < (k + 1) / NRAD)
        prof.append(float(lfull[sel].mean()) if sel.any() else float("nan"))
    c = prof[0] if prof[0] > 0 else 1.0
    f["radial_profile"] = [round(p / c, 4) for p in prof]; f["radial_edge_ratio"] = f["radial_profile"][-2]
    f["fov_frac"] = float(m.mean())
    return f

def vectorize(f):
    """Flat numeric vector (for distance-matching). Scalars roughly scaled to ~O(1)."""
    v = f["hist_r"] + f["hist_g"] + f["hist_b"] + f["hist_lum"] + f["radial_profile"]
    v += [f["green_peak"] / 255, f["ratio_r_g"], f["ratio_b_g"], f["mean_a"] / 50, f["mean_b"] / 50,
          f["mean_L"] / 100, f["sobel_mean"] / 50, f["sobel_p95"] / 150]
    return np.nan_to_num(np.array(v, np.float32))

SCALARS = ["green_peak", "ratio_r_g", "ratio_b_g", "mean_L", "mean_a", "mean_b", "sobel_mean", "sobel_p95", "lap_var", "radial_edge_ratio"]

def list_images(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for e in ("png", "jpg", "jpeg", "tif", "PNG", "JPG", "JPEG"): out += glob.glob(os.path.join(p, "**", "*." + e), recursive=True)
        else: out.append(p)
    return sorted(set(out))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("paths", nargs="+"); ap.add_argument("--out", default="stats.json"); ap.add_argument("--csv")
    a = ap.parse_args(); res = {}
    for p in list_images(a.paths):
        im = cv2.imread(p)
        if im is not None: res[p] = compute(im)
    json.dump(res, open(a.out, "w"), indent=1)
    if a.csv:
        with open(a.csv, "w") as fh:
            fh.write("path," + ",".join(SCALARS) + "\n")
            for p, f in res.items(): fh.write(p + "," + ",".join(f"{f[k]:.4f}" for k in SCALARS) + "\n")
    print(f"{len(res)} images -> {a.out}")
