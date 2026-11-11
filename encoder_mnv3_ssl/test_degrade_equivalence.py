"""Check the kernel's per-view Visuscout degrader against degrade.py (reference, unchanged).
Reference path: degrade.py on the RAW RFMiD image (BGR, native res) -> same fundus crop -> resize to the 640 cache.
Kernel path:    clean raw -> fundus crop -> 640 cache -> visuscout(alpha=1) on the full square (identity view)
                and on random crops (vs cropping the reference).
Noise off and JPEG off in BOTH paths (stochastic / resolution-specific); everything else = fitted values.
Run: /workspace/kenv/bin/python test_degrade_equivalence.py"""
import sys, math, json, re
from pathlib import Path
import numpy as np, cv2
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, "kd_imaging")
import degrade as REF
from stats import compute
CS = 640
src = (HERE / "kernels/full/team_offirst_encoder_byol_mnv3.py").read_text()
blk = src[src.index("# >>> DEGRADE BLOCK"):src.index("# <<< DEGRADE BLOCK")]
# fitted values in the kernel must equal degrade.py defaults
p0 = REF.DegradeParams()
ns = {"np": np, "cv2": cv2, "math": math, "CS": CS, "CFG": {"degrade": {"emulate_native": True, "jpeg": False, "eval_alpha": 1.0, "eval_seed": 0}}}
exec(blk, ns)
for k, v in ns["FIT"].items(): assert abs(getattr(p0, k) - v) < 1e-12, (k, v, getattr(p0, k))
print("FIT == degrade.py defaults: OK")
ns["FIT"]["noise_sigma"] = 0.0   # test only
img_dir = Path("/workspace/rfmid_train_dl/Training_Set/Training")

def crop_box(img, thr=15):   # = kernel crop_fundus, returning the box
    h, w = img.shape[:2]; sc = 512.0 / max(h, w)
    small = cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
    mask = cv2.morphologyEx((small.max(axis=2) > thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    ys, xs = np.where(mask > 0)
    x0, x1 = int(xs.min() / sc), int((xs.max() + 1) / sc); y0, y1 = int(ys.min() / sc), int((ys.max() + 1) / sc)
    return max(0, y0), min(h, y1), max(0, x0), min(w, x1)
def square(img, b):
    y0, y1, x0, x1 = b; img = img[y0:y1, x0:x1]; h, w = img.shape[:2]; m = max(h, w)
    out = np.zeros((m, m, 3), img.dtype); out[(m - h) // 2:(m - h) // 2 + h, (m - w) // 2:(m - w) // 2 + w] = img; return out
def geom(im, w_raw, side):
    m = cv2.morphologyEx((cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) > 15).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    ys, xs = np.nonzero(m); S = im.shape[0]
    return [w_raw, side, xs.mean() / S, ys.mean() / S, math.sqrt(len(xs) / math.pi) / S]
def fovmae(a, b, m):
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32))[m].mean())
def affine(center, wh, out, ang):
    (cx, cy), (w, h) = center, wh; a = math.radians(ang)
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.]]); Sm = np.diag([out / w, out / h, 1.0])
    Rm = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]); T2 = np.array([[1, 0, out / 2], [0, 1, out / 2], [0, 0, 1.]])
    return (T2 @ Rm @ Sm @ T1)[:2]

ids = {}
for p in sorted(img_dir.glob("*.png"), key=lambda p: int(p.stem)):
    w = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).shape[1] if len(ids) < 3 else None
    if w is not None and w not in ids: ids[w] = p
    if len(ids) >= 3: break
res = []; rng = np.random.RandomState(0)
for w, p in ids.items():
    raw = cv2.imread(str(p), cv2.IMREAD_COLOR); b = crop_box(raw)
    ref_raw = REF.degrade(raw, REF.DegradeParams(noise_sigma=0.0, jpeg_quality=100))              # reference at native res
    sqc = square(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), b); side = sqc.shape[1]
    clean = cv2.resize(sqc, (CS, CS), interpolation=cv2.INTER_AREA)
    ref = cv2.resize(square(cv2.cvtColor(ref_raw, cv2.COLOR_BGR2RGB), b), (CS, CS), interpolation=cv2.INTER_AREA)
    ns["GEOM"] = np.array([geom(clean, raw.shape[1], side)]); I = ns["resize_affine"](CS, CS)
    ours = ns["visuscout"](clean, 1.0, I, 0, np.random.RandomState(0))
    ns["CFG"]["degrade"]["emulate_native"] = False
    ours_naive = ns["visuscout"](clean, 1.0, I, 0, np.random.RandomState(0)); ns["CFG"]["degrade"]["emulate_native"] = True
    # wrong channel order (what calling degrade.py on an RGB cache would do): degrade(RGB-as-BGR)
    wrong = REF.degrade(clean, REF.DegradeParams(noise_sigma=0.0, jpeg_quality=100))
    fov = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY) > 20
    st = lambda x: compute(cv2.cvtColor(x, cv2.COLOR_RGB2BGR))
    s_c, s_r, s_o, s_n, s_w = st(clean), st(ref), st(ours), st(ours_naive), st(wrong)
    # crop consistency: kernel degrader applied on a rotated local crop vs cropping the full-square reference
    crop_mae = []
    for _ in range(4):
        cw = rng.uniform(0.3, 0.6) * CS; c = (rng.uniform(cw / 2, CS - cw / 2), rng.uniform(cw / 2, CS - cw / 2)); out = 224
        M = affine(c, (cw, cw), out, rng.uniform(-30, 30))
        v_clean = cv2.warpAffine(clean, M, (out, out), flags=cv2.INTER_LINEAR)
        v_ours = ns["visuscout"](v_clean, 1.0, M, 0, np.random.RandomState(0))
        v_ref = cv2.warpAffine(ref, M, (out, out), flags=cv2.INTER_LINEAR)
        m = cv2.erode((cv2.cvtColor(v_clean, cv2.COLOR_RGB2GRAY) > 20).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        if m.sum() > 1000: crop_mae.append(fovmae(v_ours, v_ref, m))
    r = {"id": p.stem, "raw_w": w, "raw_side": side,
         "mae_fov_full(ours vs ref)": fovmae(ours, ref, fov), "mae_fov_full(naive-blur vs ref)": fovmae(ours_naive, ref, fov),
         "mae_fov_full(degrade.py on RGB cache vs ref)": fovmae(wrong, ref, fov), "mae_fov_full(clean vs ref)": fovmae(clean, ref, fov),
         "mae_fov_local_crops(ours vs ref crop)": float(np.mean(crop_mae)) if crop_mae else None}
    for k in ["sobel_mean", "ratio_r_g", "ratio_b_g", "mean_L", "radial_edge_ratio"]:
        r[k] = {"clean": round(s_c[k], 3), "ref": round(s_r[k], 3), "ours": round(s_o[k], 3), "naive": round(s_n[k], 3), "rgb_as_bgr": round(s_w[k], 3)}
    res.append(r); print(json.dumps(r, indent=1))
    cv2.imwrite(str(HERE / f"localtest/degrade_equiv_{p.stem}.jpg"),
                cv2.cvtColor(np.concatenate([cv2.resize(x, (320, 320)) for x in (clean, ref, ours, wrong)], 1), cv2.COLOR_RGB2BGR))
json.dump(res, open(HERE / "localtest/degrade_equivalence.json", "w"), indent=1)
