# =============================================================================
# team_offirst - Retina encoder: BYOL self-supervised pretraining (+ frozen eval)
# -----------------------------------------------------------------------------
# Data: RFMiD Training split ONLY (Kaggle dataset awsaf49/rfmid-train-dataset,
#   1,920 images). Citation: RFMiD (Pachade et al., Data 6(2):14, 2021; IEEE
#   DataPort, doi:10.21227/s3g7-st65), CC BY 4.0.
# DATA RULES: same 5-fold split as the supervised baseline (split_5fold_seed42.csv).
#   Fold 0 (384 imgs) is EXCLUDED from SSL entirely; it is used only for eval and
#   label-based monitoring (kNN). No RFMiD Evaluation/Test, no IDRiD, no other
#   retina data, no retina-pretrained weights (ImageNet init via timm only).
# Recipe: BYOL (Grill et al. 2020) on convnext_tiny. 2 global views 384 px
#   (RRC scale 0.5-1.0) + 4 local views 160 px (scale 0.2-0.5, online only).
#   Milder fundus crop scales follow Cheng et al. 2025 (arXiv 2501.02451);
#   multi-crop follows SwAV/DINO.
#   crop_mode 'random' (A) | 'anatomy' (B): in B, on ~50% of images a disc-centred
#   and a macula-centred crop (~2.5 disc diameters wide) replace 2 local views.
#   Disc/fovea are located once on CPU with a hand-written heuristic (brightness +
#   vessel-orientation projections on the green channel, in the spirit of
#   Mahfouz & Fahmy 2010; fovea = darkest low-vessel spot ~2-3 DD from the disc
#   towards the image centre). Low-confidence images fall back to random crops.
# Eval (end of run, or RUN_MODE='eval'): frozen 768-d embeddings ->
#   (a) linear probe: train folds 1-4, test fold 0 (common labels + Disease_Risk)
#   (b) rare non-DR disease few-shot (1/5/10-shot, 20 draws), support folds 1-4,
#       query fold 0
#   (c) the same on the supervised baseline + ImageNet-init embeddings.
# RUN_MODE: 'smoke' (2 epochs, ~256 imgs, both modes) | 'full' | 'eval'
# =============================================================================

RUN_MODE = "smoke"   # "smoke" | "full" | "eval"   <-- full run needs an explicit OK

CONFIG_YAML = r"""
seed: 42
data:
  cache_size: 512           # fundus-circle crop, resized once into a uint8 memmap
  n_folds: 5
  eval_fold: 0              # excluded from SSL; eval/monitor only
  num_workers: 4
views:
  global_size: 384
  global_scale: [0.5, 1.0]
  n_local: 4
  local_size: 160
  local_scale: [0.2, 0.5]
  ratio: [0.75, 1.3333]
  hflip: 0.5
  vflip: 0.5
  rotate_deg: 30
  jitter_p: 0.8
  brightness: 0.2
  contrast: 0.2
  saturation: 0.1
  hue: 0.02                 # fraction of the hue circle
  blur_p: 0.1               # all views
  blur_sigma: [0.1, 2.0]
anatomy:
  p_image: 0.5              # fraction of (confident) images that get disc+macula crops
  crop_dd: 2.5              # crop width in disc diameters
  dd_frac: 0.12             # disc diameter ~ 0.12 x fundus-circle diameter
  jitter_dd: 0.25           # centre jitter (in DD)
  scale_jitter: [0.85, 1.15]
  min_conf: 0.5
model:
  backbone: convnext_tiny
  pretrained: true
  proj_hidden: 2048
  proj_out: 256
  pred_hidden: 2048
train:
  crop_modes: [random, anatomy]   # A then B (one kernel); set one to split runs
  epochs: 150
  batch_size: 64              # fallback ladder on OOM: 64 -> 32 + grad ckpt -> 16 + grad ckpt
  lr_heads: 1.0e-3
  lr_backbone: 2.0e-4
  weight_decay: 0.05
  warmup_epochs: 10
  tau_base: 0.99              # EMA momentum, cosine 0.99 -> 1.0
  amp: true
  grad_clip: 3.0
  ckpt_every: 25
monitor:
  knn_every: 10
  knn_k: 20
  collapse_warn: 0.2          # warn if mean per-dim std * sqrt(d) of L2-normed z drops below
eval:
  img_size: 384
  batch_size: 64
  common_min_query_pos: 10    # -> the 12 common labels of the baseline (incl. Disease_Risk)
  rare_min_total_pos: 15
  rare_min_query_pos: 10
  rare_headline: [ODC, ODP, ARMD, MH]
  shots: [1, 5, 10]
  draws: 20
  neg_per_pos: 1
  knn_k: 5
  logreg_C: 1.0
  probe_C_grid: [0.01, 0.1, 1.0]
smoke:
  epochs: 2
  n_train: 256
  n_query: 96
  knn_every: 1
  common_min_query_pos: 3
  rare_min_total_pos: 8
  rare_min_query_pos: 3
  draws: 3
  time_steps: 6               # extra timed steps at full config for throughput (after warm-up)
"""

import os, sys, json, time, math, random, copy, warnings
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
T0 = time.time()
LOGF = None
def log(*a):
    msg = f"[{time.time()-T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    if LOGF is not None: LOGF.write(msg + "\n"); LOGF.flush()

try:
    import yaml
except ImportError:
    os.system("pip install -q pyyaml"); import yaml
CFG = yaml.safe_load(CONFIG_YAML)
RUN_MODE = os.environ.get("TOB_RUN_MODE", RUN_MODE)
SMOKE = RUN_MODE == "smoke"
if SMOKE:
    s = CFG["smoke"]
    CFG["train"]["epochs"] = s["epochs"]; CFG["train"]["warmup_epochs"] = 0
    CFG["monitor"]["knn_every"] = s["knn_every"]
    for k in ["common_min_query_pos", "rare_min_total_pos", "rare_min_query_pos", "draws"]:
        CFG["eval"][k] = s[k]
if SMOKE and "TOB_SMOKE_N" in os.environ: CFG["smoke"]["n_train"] = int(os.environ["TOB_SMOKE_N"])   # local CPU test only
if SMOKE and "TOB_SMOKE_EPOCHS" in os.environ: CFG["train"]["epochs"] = int(os.environ["TOB_SMOKE_EPOCHS"])
if "TOB_CROP_MODES" in os.environ: CFG["train"]["crop_modes"] = os.environ["TOB_CROP_MODES"].split(",")
if "TOB_WORKERS" in os.environ: CFG["data"]["num_workers"] = int(os.environ["TOB_WORKERS"])
if "TOB_PRETRAINED" in os.environ: CFG["model"]["pretrained"] = os.environ["TOB_PRETRAINED"] == "1"
LOCATE_ONLY = os.environ.get("TOB_LOCATE_ONLY") == "1"   # local CPU: heuristic + contact sheet only

INPUT = Path(os.environ.get("TOB_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("TOB_OUT", "/kaggle/working/out"))
TMP = Path(os.environ.get("TOB_TMP", "/kaggle/temp" if Path("/kaggle").exists() else "/tmp/tob_byol"))
OUT.mkdir(parents=True, exist_ok=True); TMP.mkdir(parents=True, exist_ok=True)
LOGF = open(OUT / "run.log", "a")
log(f"RUN_MODE={RUN_MODE}  input={INPUT}  out={OUT}")

import cv2
cv2.setNumThreads(0)

# ----------------------------------------------------------------------------- data guard
def data_guard():
    bad = ["RFMiD_Testing_Labels.csv", "RFMiD_Validation_Labels.csv", "RFMiD_Evaluation_Labels.csv",
           "B. Disease Grading", "A. Segmentation", "IDRiD_001.jpg", "IDRiD_001test.jpg", "trainLabels.csv"]
    for n in bad:
        if any(INPUT.rglob(n)): raise SystemExit(f"DATA GUARD: forbidden data mounted ({n}).")
    for d in ["Evaluation_Set", "Test_Set", "Validation", "Test"]:
        for p in INPUT.rglob(d):
            if p.is_dir() and any(p.glob("*.png")): raise SystemExit(f"DATA GUARD: eval/test images at {p}")
    for p in INPUT.rglob("*"):
        if "idrid" in p.name.lower() and p.suffix.lower() in (".jpg", ".png", ".tif"):
            raise SystemExit(f"DATA GUARD: IDRiD file mounted ({p})")
data_guard()

# ----------------------------------------------------------------------------- labels / images / split
def find_one(pattern, must=True):
    c = sorted(INPUT.rglob(pattern))
    if not c and must: raise SystemExit(f"{pattern} not found under {INPUT}")
    return c[0] if c else None

lab_csv = find_one("RFMiD_Training_Labels.csv")
df = pd.read_csv(lab_csv); df["ID"] = df["ID"].astype(int)
LABELS = [c for c in df.columns if c != "ID"]
IMG_DIR = next(p.parent for p in sorted(lab_csv.parent.rglob("1.png")))
df["path"] = [str(IMG_DIR / f"{i}.png") for i in df["ID"]]
if not all(os.path.exists(p) for p in df["path"]):
    raise SystemExit("some training PNGs are missing")
log(f"labels {lab_csv} rows={len(df)}  images {IMG_DIR}")

def make_folds(Y, n_folds, seed):
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    except ImportError:
        os.system("pip install -q iterative-stratification")
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    f = -np.ones(len(Y), int)
    for k, (_, va) in enumerate(MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(np.zeros(len(Y)), Y)):
        f[va] = k
    return f

# authoritative split = the baseline's CSV (mounted baseline kernel output or TOB_SPLIT); recompute + compare
split_path = Path(os.environ["TOB_SPLIT"]) if "TOB_SPLIT" in os.environ else find_one("split_5fold_seed42.csv", must=False)
if split_path is not None and split_path.exists():
    sp = pd.read_csv(split_path)[["ID", "fold"]]
    df = df.merge(sp, on="ID", how="left")
    if df["fold"].isna().any(): raise SystemExit("split CSV does not cover all images")
    df["fold"] = df["fold"].astype(int); split_src = str(split_path)
else:
    df["fold"] = make_folds(df[LABELS].values, CFG["data"]["n_folds"], CFG["seed"]); split_src = "recomputed(iterstrat)"
EF = CFG["data"]["eval_fold"]
log(f"split from {split_src}; fold sizes {np.bincount(df.fold).tolist()}; eval fold {EF} (excluded from SSL)")
df[["ID", "fold"]].to_csv(OUT / "split_used.csv", index=False)
Y_ALL = df[LABELS].values.astype(np.float32)
FOLDS = df["fold"].values

# ----------------------------------------------------------------------------- preprocessing + cache (all 1,920)
CS = CFG["data"]["cache_size"]
def crop_fundus(img, thr=15):
    """Same as baseline: crop to the fundus circle by thresholding the black border, pad to square."""
    h, w = img.shape[:2]; sc = 512.0 / max(h, w)
    small = cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
    mask = (small.max(axis=2) > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    ys, xs = np.where(mask > 0)
    if len(xs) < 100: return img
    x0, x1 = int(xs.min() / sc), int((xs.max() + 1) / sc); y0, y1 = int(ys.min() / sc), int((ys.max() + 1) / sc)
    img = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    h, w = img.shape[:2]; m = max(h, w)
    out = np.zeros((m, m, 3), img.dtype)
    out[(m - h) // 2:(m - h) // 2 + h, (m - w) // 2:(m - w) // 2 + w] = img
    return out

def load_prep(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError(f"cannot read {path}")
    return cv2.resize(crop_fundus(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), (CS, CS), interpolation=cv2.INTER_AREA)

TIM = {}
cpath = TMP / f"cache_{CS}_{len(df)}.npy"
t = time.time()
if cpath.exists():
    CACHE = np.load(cpath, mmap_mode="r")
else:
    CACHE = np.lib.format.open_memmap(str(cpath) + ".part.npy", mode="w+", dtype=np.uint8, shape=(len(df), CS, CS, 3))
    with Pool(max(1, CFG["data"]["num_workers"])) as pool:
        for i, im in enumerate(pool.imap(load_prep, df["path"].tolist(), chunksize=4)):
            CACHE[i] = im
            if (i + 1) % 400 == 0: log(f"  cached {i+1}/{len(df)}")
    CACHE.flush(); del CACHE
    os.replace(str(cpath) + ".part.npy", cpath); CACHE = np.load(cpath, mmap_mode="r")
TIM["cache_s"] = time.time() - t
log(f"cache ready {CACHE.shape} in {TIM['cache_s']:.1f}s")

# ----------------------------------------------------------------------------- disc / fovea heuristic (own code)
def _mblur(x, m, sigma):
    """mask-normalised Gaussian blur (ignores pixels outside the fundus)."""
    k = int(6 * sigma) | 1
    num = cv2.GaussianBlur(x * m, (k, k), sigma); den = cv2.GaussianBlur(m, (k, k), sigma)
    return num / np.maximum(den, 1e-3)

def _rank(vals, v):
    return float((vals < v).mean())

def locate_disc_fovea(img):
    """img: RGB uint8 square (fundus-circle crop). Returns dict with disc/fovea (x,y) in px, DD, confidences.
    Disc: (1) Mahfouz&Fahmy-style projections - x from the column profile of vertical-vs-horizontal vessel
    edges (the main arcades leave the disc vertically), y from the row profile of brightness inside a 1-DD
    strip at that x; (2) a 2-D score = z(brightness at disc scale) + z(vessel density); final = best 2-D
    score within 1.5 DD of the projection estimate. Fovea: darkest, low-vessel point 2-3 DD from the disc,
    within +-35 deg of the disc->centre direction."""
    S = img.shape[0]
    m0 = (img.max(2) > 15).astype(np.uint8)
    m0 = cv2.morphologyEx(m0, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    ys, xs = np.nonzero(m0)
    if len(xs) < 1000: return dict(ok=False)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    R = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2
    DD = CFG["anatomy"]["dd_frac"] * 2 * R
    er = max(3, int(0.04 * R))
    m = cv2.erode(m0, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * er + 1, 2 * er + 1))).astype(np.float32)
    g = img[..., 1].astype(np.float32)
    inside = m > 0
    # brightness at disc scale (green, background-flattened by a very wide blur)
    bg = _mblur(g, m, DD * 2.0)
    bright = _mblur(g, m, DD / 4) - 0.5 * bg
    # vessels: black top-hat on green
    kv = int(DD / 4) | 1
    ves = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kv, kv))) * m
    thr = np.percentile(ves[inside], 88)
    vbin = (ves > thr).astype(np.float32)
    vden = _mblur(vbin, m, DD / 2)
    # vessel orientation: vertical vessels -> strong d/dx edges
    vs = cv2.GaussianBlur(ves, (5, 5), 1.5)
    gx = np.abs(cv2.Sobel(vs, cv2.CV_32F, 1, 0, ksize=3)); gy = np.abs(cv2.Sobel(vs, cv2.CV_32F, 0, 1, ksize=3))
    colp = ((gx - gy) * m).sum(0) / np.maximum(m.sum(0), 1)
    colp = cv2.GaussianBlur(colp.reshape(1, -1), (int(DD) | 1, 1), DD / 3).ravel()
    colp[m.sum(0) < 0.2 * S * 0.5] = -1e9
    xa = int(np.argmax(colp))
    x0s, x1s = max(0, int(xa - DD / 2)), min(S, int(xa + DD / 2) + 1)
    rowp = (bright[:, x0s:x1s] * m[:, x0s:x1s]).sum(1) / np.maximum(m[:, x0s:x1s].sum(1), 1)
    rowp[m[:, x0s:x1s].sum(1) < 1] = -1e9
    ya = int(np.argmax(rowp))
    # 2-D score
    def z(a):
        v = a[inside]; med = np.median(v); s = np.percentile(v, 75) - np.percentile(v, 25) + 1e-6
        return (a - med) / s
    score = np.where(inside, z(bright) + 0.7 * z(vden), -1e9)
    yb, xb = np.unravel_index(int(np.argmax(score)), score.shape)
    yy, xx = np.mgrid[0:S, 0:S]
    near = ((xx - xa) ** 2 + (yy - ya) ** 2) <= (1.5 * DD) ** 2
    sc2 = np.where(near, score, -1e9)
    yd, xd = np.unravel_index(int(np.argmax(sc2)), sc2.shape)
    if sc2[yd, xd] < score[yb, xb] - 1.5:   # projection candidate clearly worse -> trust 2-D maximum
        yd, xd = yb, xb
    bvals = bright[inside]; dvals = vden[inside]
    c_b = np.clip((_rank(bvals, bright[yd, xd]) - 0.90) / 0.09, 0, 1)
    c_v = np.clip((_rank(dvals, vden[yd, xd]) - 0.60) / 0.35, 0, 1)
    dab = math.hypot(xa - xb, ya - yb)
    c_a = 1.0 if dab < DD else (0.5 if dab < 2 * DD else 0.0)
    rr = math.hypot(xd - cx, yd - cy) / R
    c_disc = (0.4 * c_b + 0.3 * c_v + 0.3 * c_a) * (0.6 if rr > 0.9 else 1.0)
    # fovea
    dark = _mblur(g, m, DD / 3)
    vx, vy = cx - xd, cy - yd; nv = math.hypot(vx, vy)
    if nv < 0.5 * DD: angs = [0.0, math.pi]          # disc at centre: search both horizontal sides
    else: angs = [math.atan2(vy, vx)]
    rad = np.hypot(xx - xd, yy - yd); ang = np.arctan2(yy - yd, xx - xd)
    best = None
    for a0 in angs:
        da = np.abs((ang - a0 + np.pi) % (2 * np.pi) - np.pi)
        sector = inside & (rad >= 2.0 * DD) & (rad <= 3.0 * DD) & (da <= math.radians(35))
        if sector.sum() < 50: continue
        f = np.where(sector, dark + 30.0 * vden, 1e9)
        yf, xf = np.unravel_index(int(np.argmin(f)), f.shape)
        sv = dark[sector]; contrast = (np.median(sv) - dark[yf, xf]) / (np.std(sv) + 1e-6)
        cand = (f[yf, xf], xf, yf, contrast, _rank(vden[sector], vden[yf, xf]))
        if best is None or cand[0] < best[0]: best = cand
    if best is None:
        xf = yf = -1; c_fov = 0.0
    else:
        _, xf, yf, contrast, vrank = best
        c_fov = min(c_disc, 0.6 * np.clip(contrast / 1.5, 0, 1) + 0.4 * (1 - vrank))
    return dict(ok=True, disc_x=int(xd), disc_y=int(yd), fovea_x=int(xf), fovea_y=int(yf), dd=float(DD),
                disc_conf=float(c_disc), fovea_conf=float(c_fov), cx=float(cx), cy=float(cy), R=float(R),
                proj_x=xa, proj_y=ya, max2d_x=int(xb), max2d_y=int(yb))

def _locate_i(i): 
    r = locate_disc_fovea(np.asarray(CACHE[i])); r["row"] = i; return r

def run_locator():
    t = time.time()
    with Pool(max(1, CFG["data"]["num_workers"])) as pool:
        res = pool.map(_locate_i, range(len(df)), chunksize=8)
    loc = pd.DataFrame(res).sort_values("row").reset_index(drop=True)
    loc.insert(0, "ID", df["ID"].values); loc["fold"] = FOLDS
    mc = CFG["anatomy"]["min_conf"]
    loc["disc_ok"] = loc["disc_conf"] >= mc; loc["fovea_ok"] = loc["fovea_conf"] >= mc
    loc.to_csv(OUT / "anatomy_locations.csv", index=False)
    TIM["locator_s"] = time.time() - t
    st = {"n": int(len(loc)), "disc_ok_rate": float(loc.disc_ok.mean()), "fovea_ok_rate": float(loc.fovea_ok.mean()),
          "both_ok_rate": float((loc.disc_ok & loc.fovea_ok).mean()), "min_conf": mc, "seconds": TIM["locator_s"],
          "disc_ok_rate_normals": float(loc.disc_ok[df.Disease_Risk.values == 0].mean()),
          "disc_ok_rate_diseased": float(loc.disc_ok[df.Disease_Risk.values == 1].mean())}
    log("locator:", json.dumps(st))
    contact_sheet(loc)
    return loc, st

def contact_sheet(loc, n=50, tile=200):
    rng = np.random.RandomState(7)
    rows = rng.choice(np.where(FOLDS != EF)[0], n, replace=False)
    tiles = []
    for i in rows:
        im = cv2.cvtColor(np.ascontiguousarray(CACHE[i]), cv2.COLOR_RGB2BGR).copy(); r = loc.iloc[i]
        if r.ok:
            dd = r.dd; half = int(CFG["anatomy"]["crop_dd"] * dd / 2)
            cd = (0, 255, 0) if r.disc_ok else (0, 165, 255)
            cv2.circle(im, (int(r.disc_x), int(r.disc_y)), int(dd / 2), cd, 2)
            cv2.rectangle(im, (int(r.disc_x) - half, int(r.disc_y) - half), (int(r.disc_x) + half, int(r.disc_y) + half), cd, 1)
            if r.fovea_x >= 0:
                cf = (255, 255, 0) if r.fovea_ok else (0, 0, 255)
                cv2.drawMarker(im, (int(r.fovea_x), int(r.fovea_y)), cf, cv2.MARKER_CROSS, 24, 3)
                cv2.rectangle(im, (int(r.fovea_x) - half, int(r.fovea_y) - half), (int(r.fovea_x) + half, int(r.fovea_y) + half), cf, 1)
        im = cv2.resize(im, (tile, tile), interpolation=cv2.INTER_AREA)
        dz = "DR" if df.DR.values[i] else ("N" if df.Disease_Risk.values[i] == 0 else "D")
        cv2.putText(im, f"{df.ID.values[i]} {dz} d{r.disc_conf:.2f} f{r.fovea_conf:.2f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        tiles.append(im)
    while len(tiles) % 10: tiles.append(np.zeros_like(tiles[0]))
    sheet = np.concatenate([np.concatenate(tiles[k:k + 10], 1) for k in range(0, len(tiles), 10)], 0)
    cv2.imwrite(str(OUT / "anatomy_contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    pd.DataFrame({"ID": df.ID.values[rows]}).to_csv(OUT / "anatomy_contact_sheet_ids.csv", index=False)
    log("contact sheet ->", OUT / "anatomy_contact_sheet.jpg",
        "(green circle/box = disc ok, orange = low conf; cyan cross = fovea ok, red = low conf)")

LOC, LOC_STATS = run_locator()
if LOCATE_ONLY:
    json.dump({"locator": LOC_STATS, "timings": TIM}, open(OUT / "locator_summary.json", "w"), indent=1)
    log("LOCATE_ONLY done"); sys.exit(0)

# ----------------------------------------------------------------------------- torch
import torch, torch.nn as nn, torch.nn.functional as F
try:
    import timm
except ImportError:
    os.system("pip install -q timm"); import timm
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log("device", DEV, torch.cuda.get_device_name(0) if DEV.type == "cuda" else "", "torch", torch.__version__, "timm", timm.__version__)
def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(CFG["seed"])

SSL_ROWS = np.where(FOLDS != EF)[0]; Q_ROWS = np.where(FOLDS == EF)[0]
if SMOKE:
    rng = np.random.RandomState(CFG["seed"])
    SSL_ROWS = np.sort(rng.choice(SSL_ROWS, CFG["smoke"]["n_train"], replace=False))
    Q_ROWS = np.sort(rng.choice(Q_ROWS, CFG["smoke"]["n_query"], replace=False))
assert not set(SSL_ROWS) & set(np.where(FOLDS == EF)[0]), "fold-0 leak into SSL"
N_FULL_SSL = int((FOLDS != EF).sum())
log(f"SSL images {len(SSL_ROWS)} (full run would use {N_FULL_SSL}); eval query (fold {EF}) {len(Q_ROWS)}")

# ----------------------------------------------------------------------------- views
V = CFG["views"]; AN = CFG["anatomy"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255

def _affine(center, wh, out, ang, hf, vf):
    """src->dst affine: crop box (center, w,h) -> out x out, rotated by ang deg, optional flips."""
    (cx, cy), (w, h) = center, wh
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], np.float64)
    Sm = np.diag([out / w, out / h, 1.0])
    a = math.radians(ang); Rm = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    Fm = np.diag([-1.0 if hf else 1.0, -1.0 if vf else 1.0, 1.0])
    T2 = np.array([[1, 0, out / 2], [0, 1, out / 2], [0, 0, 1]], np.float64)
    return (T2 @ Fm @ Rm @ Sm @ T1)[:2]

def _photometric(x, rng):
    if rng.rand() < V["jitter_p"]:
        x = x.astype(np.float32)
        fg = x.max(2) > 5
        ops = rng.permutation(4)
        for o in ops:
            if o == 0: x = x * rng.uniform(1 - V["brightness"], 1 + V["brightness"])
            elif o == 1:
                mu = x[fg].mean() if fg.any() else 0.0
                x = (x - mu) * rng.uniform(1 - V["contrast"], 1 + V["contrast"]) + mu
            elif o == 2:
                gr = x.mean(2, keepdims=True); x = gr + (x - gr) * rng.uniform(1 - V["saturation"], 1 + V["saturation"])
            else:
                hsv = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
                h = hsv[..., 0].astype(np.int16) + int(round(rng.uniform(-V["hue"], V["hue"]) * 180))
                hsv[..., 0] = (h % 180).astype(np.uint8); x = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)
        x[~fg] = 0
        x = np.clip(x, 0, 255).astype(np.uint8)
    if rng.rand() < V["blur_p"]:
        s = rng.uniform(*V["blur_sigma"]); x = cv2.GaussianBlur(x, (0, 0), s)
    return x

def _geom(img, rng, out, center, wh):
    M = _affine(center, wh, out, rng.uniform(-V["rotate_deg"], V["rotate_deg"]), rng.rand() < V["hflip"], rng.rand() < V["vflip"])
    return cv2.warpAffine(img, M, (out, out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def _rrc_box(rng, S, scale):
    for _ in range(10):
        area = S * S * rng.uniform(*scale)
        r = math.exp(rng.uniform(math.log(V["ratio"][0]), math.log(V["ratio"][1])))
        w = math.sqrt(area * r); h = math.sqrt(area / r)
        if w <= S and h <= S:
            return (rng.uniform(w / 2, S - w / 2), rng.uniform(h / 2, S - h / 2)), (w, h)
    return (S / 2, S / 2), (S, S)

def make_views(img, rng, row, mode):
    S = img.shape[0]
    gl = [_photometric(_geom(img, rng, V["global_size"], *_rrc_box(rng, S, V["global_scale"])), rng) for _ in range(2)]
    centers = []
    is_anat = 0
    if mode == "anatomy" and rng.rand() < AN["p_image"]:
        r = LOC_ARR[row]  # disc_x, disc_y, fov_x, fov_y, dd, disc_ok, fov_ok
        if r[5] > 0:
            centers.append((r[0], r[1], r[4])); is_anat = 1
            if r[6] > 0: centers.append((r[2], r[3], r[4])); is_anat = 2
    lo = []
    for (x, y, dd) in centers:
        w = AN["crop_dd"] * dd * rng.uniform(*AN["scale_jitter"])
        c = (x + rng.uniform(-1, 1) * AN["jitter_dd"] * dd, y + rng.uniform(-1, 1) * AN["jitter_dd"] * dd)
        lo.append(_photometric(_geom(img, rng, V["local_size"], c, (w, w)), rng))
    while len(lo) < V["n_local"]:
        lo.append(_photometric(_geom(img, rng, V["local_size"], *_rrc_box(rng, S, V["local_scale"])), rng))
    return np.stack(gl), np.stack(lo), is_anat

LOC_ARR = np.stack([LOC.disc_x.fillna(-1).values, LOC.disc_y.fillna(-1).values, LOC.fovea_x.fillna(-1).values,
                    LOC.fovea_y.fillna(-1).values, LOC.dd.fillna(0).values, LOC.disc_ok.values.astype(float),
                    LOC.fovea_ok.values.astype(float)], 1).astype(np.float64)

class SSLDS(torch.utils.data.Dataset):
    def __init__(self, rows, mode): self.rows = np.asarray(rows); self.mode = mode; self.epoch = 0
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        rng = np.random.RandomState((CFG["seed"] * 1000003 + self.epoch * 7919 + i * 104729 + torch.initial_seed()) % 2**32)
        g, l, a = make_views(np.asarray(CACHE[r]), rng, r, self.mode)
        return torch.from_numpy(g).permute(0, 3, 1, 2), torch.from_numpy(l).permute(0, 3, 1, 2), a

class EvalDS(torch.utils.data.Dataset):
    def __init__(self, rows, size): self.rows = np.asarray(rows); self.size = size
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        im = cv2.resize(np.asarray(CACHE[self.rows[i]]), (self.size, self.size), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(im).permute(2, 0, 1), i

def norm_gpu(x):  # uint8 NCHW -> normalised float, channels_last
    x = x.to(DEV, non_blocking=True).float()
    return ((x - MEAN.to(DEV)) / STD.to(DEV)).contiguous(memory_format=torch.channels_last)

# ----------------------------------------------------------------------------- model
def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Linear(h, o))

class BYOLNet(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        mc = CFG["model"]
        self.backbone = timm.create_model(mc["backbone"], pretrained=pretrained, num_classes=0)
        d = self.backbone.num_features
        self.projector = mlp(d, mc["proj_hidden"], mc["proj_out"])
        self.predictor = mlp(mc["proj_out"], mc["pred_hidden"], mc["proj_out"])
    def project(self, x):
        h = self.backbone(x); return h, self.projector(h)

class Target(nn.Module):
    def __init__(self, online):
        super().__init__()
        self.backbone = copy.deepcopy(online.backbone); self.projector = copy.deepcopy(online.projector)
        for p in self.parameters(): p.requires_grad_(False)
    def forward(self, x): return self.projector(self.backbone(x))

@torch.no_grad()
def ema_update(target, online, tau):
    for tp, op in zip(list(target.backbone.parameters()) + list(target.projector.parameters()),
                      list(online.backbone.parameters()) + list(online.projector.parameters())):
        tp.mul_(tau).add_(op.detach(), alpha=1 - tau)
    for tb, ob in zip(list(target.backbone.buffers()) + list(target.projector.buffers()),
                      list(online.backbone.buffers()) + list(online.projector.buffers())):
        tb.copy_(ob)

def byol_loss(p, z):
    return (2 - 2 * F.cosine_similarity(p.float(), z.float(), dim=-1)).mean()

def param_groups(model):
    tc = CFG["train"]; groups = {}
    for n, p in model.named_parameters():
        part = "backbone" if n.startswith("backbone.") else "heads"
        nowd = p.ndim <= 1 or n.endswith(".bias")
        groups.setdefault((part, nowd), []).append(p)
    return [{"params": ps, "lr": tc["lr_backbone"] if part == "backbone" else tc["lr_heads"], "base_lr": tc["lr_backbone"] if part == "backbone" else tc["lr_heads"],
             "weight_decay": 0.0 if nowd else tc["weight_decay"], "name": f"{part}{'_nowd' if nowd else ''}"} for (part, nowd), ps in groups.items()]

def train_step(online, target, opt, scaler, G, L, tau, amp):
    """G: (B,2,3,H,W) uint8, L: (B,nl,3,h,w) uint8. Returns loss, z (online global proj, detached)."""
    B = G.shape[0]; nl = L.shape[1]
    g = norm_gpu(G.transpose(0, 1).reshape(2 * B, *G.shape[2:]))
    l = norm_gpu(L.transpose(0, 1).reshape(nl * B, *L.shape[2:]))
    with torch.autocast(DEV.type, dtype=torch.float16, enabled=amp):
        hg, zg = online.project(g); pg = online.predictor(zg)
        _, zl = online.project(l); pl = online.predictor(zl)
        with torch.no_grad():
            tg = target(g)
    t1, t2 = tg[:B].detach(), tg[B:].detach()
    p1, p2 = pg[:B], pg[B:]
    terms = [byol_loss(p1, t2), byol_loss(p2, t1)]
    for k in range(nl):
        pk = pl[k * B:(k + 1) * B]; terms += [byol_loss(pk, t1), byol_loss(pk, t2)]
    loss = torch.stack(terms).mean()
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt); gn = torch.nn.utils.clip_grad_norm_(online.parameters(), CFG["train"]["grad_clip"])
    scaler.step(opt); scaler.update()
    ema_update(target, online, tau)
    return loss.detach(), zg[:B].detach().float(), hg[:B].detach().float(), gn

def std_metric(z):
    """mean per-dim std of L2-normalised vectors * sqrt(d): ~1 healthy, -> 0 collapse."""
    zn = F.normalize(z, dim=1); return float(zn.std(0).mean() * math.sqrt(z.shape[1]))

# ----------------------------------------------------------------------------- embeddings + eval
@torch.no_grad()
def extract(backbone, rows, size=None):
    size = size or CFG["eval"]["img_size"]
    dl = torch.utils.data.DataLoader(EvalDS(rows, size), batch_size=CFG["eval"]["batch_size"], shuffle=False,
                                     num_workers=CFG["data"]["num_workers"], pin_memory=DEV.type == "cuda")
    was = backbone.training; backbone.eval(); E = []
    for x, _ in dl:
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
            E.append(backbone(norm_gpu(x)).float().cpu().numpy())
    backbone.train(was)
    E = np.concatenate(E); return (E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)).astype(np.float32)

def label_sets(q_rows, s_rows):
    ec = CFG["eval"]; Yq = Y_ALL[q_rows]; Ys = Y_ALL[s_rows]
    common = [k for k in range(len(LABELS)) if ec["common_min_query_pos"] <= Yq[:, k].sum() < len(q_rows) and 2 <= Ys[:, k].sum() < len(s_rows) - 1]
    rare = [k for k in range(len(LABELS)) if LABELS[k] not in ("Disease_Risk", "DR")
            and (Yq[:, k].sum() + Ys[:, k].sum()) >= ec["rare_min_total_pos"] and Yq[:, k].sum() >= ec["rare_min_query_pos"]
            and Ys[:, k].sum() >= max(ec["shots"])]
    return common, rare

def knn_monitor(Eb, yb, Eq, yq, labs, k):
    sims = Eq @ Eb.T; idx = np.argsort(-sims, 1)[:, :k]; w = np.exp(np.take_along_axis(sims, idx, 1) / 0.07)
    S = np.einsum("qk,qkl->ql", w, yb[idx]) / w.sum(1, keepdims=True)
    au = [roc_auc_score(yq[:, j], S[:, j]) for j in labs]
    dr = LABELS.index("Disease_Risk")
    return float(np.mean(au)), float(((S[:, dr] > 0.5) == (yq[:, dr] > 0.5)).mean()), float(roc_auc_score(yq[:, dr], S[:, dr]))

def linear_probe(Es, ys, Eq, yq, labs):
    sc = StandardScaler().fit(Es); Xs = sc.transform(Es); Xq = sc.transform(Eq)
    out = {}
    from sklearn.model_selection import StratifiedKFold
    for j in labs:
        grid = CFG["eval"]["probe_C_grid"]; best, bestC = -1, grid[len(grid) // 2]   # default = middle of grid
        n_min = int(min(ys[:, j].sum(), len(ys) - ys[:, j].sum())); nspl = min(3, n_min)
        if nspl >= 2:
            for C in grid:
                aucs = []
                for tr, va in StratifiedKFold(nspl, shuffle=True, random_state=0).split(Xs, ys[:, j]):
                    if len(np.unique(ys[va, j])) < 2 or len(np.unique(ys[tr, j])) < 2: continue
                    m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(Xs[tr], ys[tr, j])
                    aucs.append(roc_auc_score(ys[va, j], m.decision_function(Xs[va])))
                a = float(np.nanmean(aucs)) if aucs and not np.all(np.isnan(aucs)) else float("nan")
                if not np.isnan(a) and a > best: best, bestC = a, C
        m = LogisticRegression(C=bestC, class_weight="balanced", max_iter=3000).fit(Xs, ys[:, j])
        out[LABELS[j]] = {"auroc": float(roc_auc_score(yq[:, j], m.decision_function(Xq))), "C": bestC}
    dz = [l for l in out if l != "Disease_Risk"]
    return {"per_label": out, "mean_auroc_common": float(np.mean([out[l]["auroc"] for l in dz])) if dz else float("nan"),
            "disease_risk_auroc": out.get("Disease_Risk", {}).get("auroc", float("nan"))}

def fewshot(Es, ys, Eq, yq, labs, headline):
    ec = CFG["eval"]; rep = {}
    for kshot in ec["shots"]:
        per = {h: {LABELS[j]: [] for j in labs} for h in ["prototype", "logreg", "knn"]}
        for d in range(ec["draws"]):
            rng = np.random.RandomState(CFG["seed"] * 1000 + kshot * 100 + d)   # same draws for every encoder
            for j in labs:
                pos = np.where(ys[:, j] == 1)[0]; neg = np.where(ys[:, j] == 0)[0]
                sp = rng.choice(pos, min(kshot, len(pos)), replace=False); sn = rng.choice(neg, min(kshot * ec["neg_per_pos"], len(neg)), replace=False)
                Xs = np.concatenate([Es[sp], Es[sn]]); y_ = np.r_[np.ones(len(sp)), np.zeros(len(sn))]
                mp = Es[sp].mean(0); mn = Es[sn].mean(0)
                mp /= np.linalg.norm(mp) + 1e-12; mn /= np.linalg.norm(mn) + 1e-12
                s_p = Eq @ mp - Eq @ mn
                s_l = LogisticRegression(C=ec["logreg_C"], class_weight="balanced", max_iter=2000).fit(Xs, y_).decision_function(Eq)
                kk = min(ec["knn_k"], len(Xs)); sims = Eq @ Xs.T; ix = np.argsort(-sims, 1)[:, :kk]
                w = np.take_along_axis(sims, ix, 1) + 1.0; s_k = (w * y_[ix]).sum(1) / (w.sum(1) + 1e-12)
                for h, s in [("prototype", s_p), ("logreg", s_l), ("knn", s_k)]:
                    per[h][LABELS[j]].append(float(roc_auc_score(yq[:, j], s)))
        res = {}
        for h, pl in per.items():
            hl = [l for l in headline if l in pl] or list(pl)
            draws_mean_all = np.mean([pl[l] for l in pl], 0); draws_mean_hl = np.mean([pl[l] for l in hl], 0)
            res[h] = {"headline_labels": hl, "headline_mean": float(draws_mean_hl.mean()), "headline_std_over_draws": float(draws_mean_hl.std()),
                      "all_rare_mean": float(draws_mean_all.mean()), "all_rare_std_over_draws": float(draws_mean_all.std()),
                      "per_label": {l: [float(np.mean(v)), float(np.std(v))] for l, v in pl.items()}}
        rep[f"k={kshot}"] = res
    return rep

def evaluate(Eall_rows, E, tag):
    """E: L2-normed embeddings aligned with Eall_rows (df row indices). Support = folds!=EF rows, query = EF rows."""
    rows = np.asarray(Eall_rows); pos = {r: i for i, r in enumerate(rows)}
    s_rows = [r for r in rows if FOLDS[r] != EF]; q_rows = [r for r in rows if FOLDS[r] == EF]
    Es = E[[pos[r] for r in s_rows]]; Eq = E[[pos[r] for r in q_rows]]; ys = Y_ALL[s_rows]; yq = Y_ALL[q_rows]
    common, rare = label_sets(q_rows, s_rows)
    t = time.time()
    rep = {"tag": tag, "n_support": len(s_rows), "n_query": len(q_rows),
           "common_labels": [LABELS[j] for j in common], "rare_labels": [LABELS[j] for j in rare],
           "rare_counts": {LABELS[j]: {"support_pos": int(ys[:, j].sum()), "query_pos": int(yq[:, j].sum())} for j in rare},
           "linear_probe": linear_probe(Es, ys, Eq, yq, common),
           "fewshot_rare": fewshot(Es, ys, Eq, yq, rare, CFG["eval"]["rare_headline"])}
    rep["eval_s"] = time.time() - t
    lp = rep["linear_probe"]; fs = rep["fewshot_rare"]
    log(f"  EVAL[{tag}] probe mAUROC(common, {len(common)-1} dz) {lp['mean_auroc_common']:.3f}  DiseaseRisk {lp['disease_risk_auroc']:.3f} | "
        + "  ".join(f"{k} logreg {v['logreg']['headline_mean']:.3f}±{v['logreg']['headline_std_over_draws']:.3f}" for k, v in fs.items())
        + f"  ({rep['eval_s']:.0f}s)")
    return rep

def baseline_embeddings():
    """supervised ConvNeXt-T (best.pth) embeddings from the baseline kernel output, if mounted."""
    p = None
    for c in sorted(INPUT.rglob("embeddings.npy")):
        if c.parent.name == "convnext_tiny" and (c.parent / "embedding_ids.csv").exists(): p = c; break
    if p is None and "TOB_BASELINE_EMB" in os.environ: p = Path(os.environ["TOB_BASELINE_EMB"])
    if p is None: return None
    E = np.load(p); ids = pd.read_csv(p.parent / "embedding_ids.csv")["ID"].values
    rowof = {i: r for r, i in enumerate(df.ID.values)}
    return np.array([rowof[i] for i in ids]), E.astype(np.float32), str(p)

# ----------------------------------------------------------------------------- train one crop mode
def lr_factor(step, total, warm):
    if step < warm: return (step + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

def build(bs, ckpt):
    seed_all(CFG["seed"])
    online = BYOLNet(CFG["model"]["pretrained"]).to(DEV).to(memory_format=torch.channels_last)
    if ckpt: online.backbone.set_grad_checkpointing(True)
    target = Target(online).to(DEV); target.train()
    opt = torch.optim.AdamW(param_groups(online), betas=(0.9, 0.999))
    return online, target, opt

def probe_batch(amp):
    """find the largest batch from the ladder that fits on this GPU (one full step on random data)."""
    if DEV.type != "cuda": return 4, False, None
    for bs, ck in [(CFG["train"]["batch_size"], False), (48, False), (48, True), (32, True), (16, True)]:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            online, target, opt = build(bs, ck); scaler = torch.amp.GradScaler("cuda", enabled=amp)
            G = torch.randint(0, 255, (bs, 2, 3, V["global_size"], V["global_size"]), dtype=torch.uint8)
            L = torch.randint(0, 255, (bs, V["n_local"], 3, V["local_size"], V["local_size"]), dtype=torch.uint8)
            for _ in range(2): train_step(online, target, opt, scaler, G, L, 0.99, amp)
            torch.cuda.synchronize(); peak = torch.cuda.max_memory_allocated() / 2**30
            log(f"batch probe: bs={bs} grad_ckpt={ck} fits, peak {peak:.2f} GiB")
            del online, target, opt; torch.cuda.empty_cache(); return bs, ck, peak
        except torch.cuda.OutOfMemoryError:
            log(f"batch probe: bs={bs} grad_ckpt={ck} OOM"); 
            for v in ["online", "target", "opt"]: locals().pop(v, None)
            torch.cuda.empty_cache()
    raise SystemExit("even bs=16 + grad ckpt OOMs")

def run_mode(mode, bs, ck):
    tc = CFG["train"]; amp = tc["amp"] and DEV.type == "cuda"
    od = OUT / f"byol_{mode}"; od.mkdir(parents=True, exist_ok=True)
    online, target, opt = build(bs, ck); scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ds = SSLDS(SSL_ROWS, mode)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=CFG["data"]["num_workers"],
                                     pin_memory=DEV.type == "cuda", persistent_workers=False)
    steps_ep = len(dl); total = steps_ep * tc["epochs"]; warm = steps_ep * tc["warmup_epochs"]
    log(f"=== BYOL crop_mode={mode}: {len(SSL_ROWS)} imgs, bs {bs}, grad_ckpt {ck}, {steps_ep} steps/ep, {tc['epochs']} ep")
    if DEV.type == "cuda": torch.cuda.reset_peak_memory_stats()
    # monitor bank (support = SSL folds, query = eval fold); labels only for monitoring
    mon_s = SSL_ROWS; mon_q = Q_ROWS
    common_mon, _ = label_sets(mon_q, mon_s)
    hist, step = [], 0
    for ep in range(tc["epochs"]):
        ds.epoch = ep; online.train(); target.train()
        t = time.time(); tl, nb, zs, hs, an, tstep = 0.0, 0, [], [], 0, []
        for G, L, a in dl:
            ts = time.time()
            for gp in opt.param_groups: gp["lr"] = gp["base_lr"] * lr_factor(step, total, warm)
            tau = 1 - (1 - tc["tau_base"]) * (math.cos(math.pi * step / max(1, total)) + 1) / 2
            loss, z, h, gn = train_step(online, target, opt, scaler, G, L, tau, amp)
            tl += float(loss); nb += 1; step += 1; an += int((a > 0).sum())
            zs.append(z); hs.append(h)
            if DEV.type == "cuda": torch.cuda.synchronize()
            tstep.append(time.time() - ts)
        t_tr = time.time() - t
        zc = torch.cat(zs); hc = torch.cat(hs)
        row = {"epoch": ep + 1, "loss": tl / max(1, nb), "std_z": std_metric(zc), "std_h": std_metric(hc), "tau_end": tau,
               "lr_bb": opt.param_groups[0]["lr"], "train_s": t_tr, "img_per_s": nb * bs / max(t_tr, 1e-9),
               "gpu_img_per_s_median_step": bs / float(np.median(tstep)) if tstep else float("nan"),
               "anatomy_views_frac": an / max(1, nb * bs)}
        if row["std_z"] < CFG["monitor"]["collapse_warn"]: log(f"  WARNING possible collapse: std_z {row['std_z']:.3f}")
        if row["std_h"] < CFG["monitor"]["collapse_warn"]: log(f"  WARNING backbone std_h {row['std_h']:.3f} < {CFG['monitor']['collapse_warn']} (ep {ep+1})")
        if (ep + 1) % CFG["monitor"]["knn_every"] == 0 or ep + 1 == tc["epochs"]:
            tk = time.time()
            Eb = extract(online.backbone, mon_s); Eq = extract(online.backbone, mon_q)
            row["knn_mauroc"], row["knn_dr_acc"], row["knn_dr_auroc"] = knn_monitor(Eb, Y_ALL[mon_s], Eq, Y_ALL[mon_q], common_mon, CFG["monitor"]["knn_k"])
            row["knn_s"] = time.time() - tk
        hist.append(row)
        log(f"  [{mode}] ep {ep+1}/{tc['epochs']} loss {row['loss']:.4f} std_z {row['std_z']:.3f} std_h {row['std_h']:.3f} "
            f"tau {tau:.4f} {row['img_per_s']:.1f} img/s ({t_tr:.0f}s)"
            + (f" | kNN mAUROC {row['knn_mauroc']:.3f} DR-risk acc {row['knn_dr_acc']:.3f} ({row['knn_s']:.0f}s)" if "knn_mauroc" in row else "")
            + (f" anat {row['anatomy_views_frac']:.2f}" if mode == "anatomy" else ""))
        json.dump(hist, open(od / "history.json", "w"), indent=1)
        if (ep + 1) % tc["ckpt_every"] == 0 or ep + 1 == tc["epochs"]:
            ck_d = {"online": online.state_dict(), "target_backbone": target.backbone.state_dict(),
                    "target_projector": target.projector.state_dict(), "epoch": ep + 1, "crop_mode": mode, "config": CFG}
            torch.save(ck_d, od / ("last.pth" if ep + 1 == tc["epochs"] else f"ep{ep+1:03d}.pth"))
    peak = torch.cuda.max_memory_allocated() / 2**30 if DEV.type == "cuda" else float("nan")
    # throughput at steady state (skip first 2 steps of epoch 1 = cudnn/benchmark warm-up)
    rep = {"crop_mode": mode, "batch_size": bs, "grad_ckpt": ck, "steps_per_epoch": steps_ep, "history": hist, "peak_gpu_gib": peak}
    # eval: online encoder (BYOL convention) + target encoder
    t = time.time()
    ev_rows = np.sort(np.r_[SSL_ROWS, Q_ROWS])
    E_on = extract(online.backbone, ev_rows); np.save(od / "embeddings_online.npy", E_on)
    E_tg = extract(target.backbone, ev_rows); np.save(od / "embeddings_target.npy", E_tg)
    pd.DataFrame({"ID": df.ID.values[ev_rows], "fold": FOLDS[ev_rows]}).to_csv(od / "embedding_ids.csv", index=False)
    rep["eval_online"] = evaluate(ev_rows, E_on, f"byol_{mode}/online")
    rep["eval_target"] = evaluate(ev_rows, E_tg, f"byol_{mode}/target")
    rep["final_eval_s"] = time.time() - t
    json.dump(rep, open(od / "report.json", "w"), indent=1, default=float)
    del online, target, opt; torch.cuda.empty_cache() if DEV.type == "cuda" else None
    return rep

# ----------------------------------------------------------------------------- main
summary = {"run_mode": RUN_MODE, "labels": LABELS, "split_source": split_src, "eval_fold": EF,
           "n_ssl": int(len(SSL_ROWS)), "n_ssl_full": N_FULL_SSL, "n_query": int(len(Q_ROWS)),
           "locator": LOC_STATS, "config": CFG, "modes": {}, "references": {},
           "citation": "RFMiD (Pachade et al., Data 6(2):14, 2021; IEEE DataPort, doi:10.21227/s3g7-st65), CC BY 4.0"}
def dump(): summary["timings"] = TIM; json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=float)

ev_rows_all = np.sort(np.r_[SSL_ROWS, Q_ROWS])
# (c) references: supervised baseline embeddings + ImageNet-init embeddings, same eval
be = baseline_embeddings()
if be is not None:
    brows, bE, bsrc = be; keep = np.isin(brows, ev_rows_all)
    summary["references"]["supervised_baseline"] = evaluate(brows[keep], bE[keep], "supervised_convnext_tiny_best")
    summary["references"]["supervised_baseline"]["source"] = bsrc
    summary["references"]["supervised_baseline"]["note"] = "embeddings at 512 px; this model was TRAINED on all 28 labels of folds 1-4, so rare labels are not unseen for it"
else:
    log("WARNING: supervised baseline embeddings not mounted; skipping reference (c)")
t = time.time()
_m = timm.create_model(CFG["model"]["backbone"], pretrained=CFG["model"]["pretrained"], num_classes=0).to(DEV).to(memory_format=torch.channels_last)
summary["references"]["imagenet_init"] = evaluate(ev_rows_all, extract(_m, ev_rows_all), "imagenet_init_384")
del _m; TIM["ref_eval_s"] = time.time() - t; dump()
if os.environ.get("TOB_STOP_AFTER_REFS") == "1": log("STOP_AFTER_REFS (local test)"); sys.exit(0)

if RUN_MODE == "eval":
    # standalone: evaluate a saved checkpoint (TOB_CKPT=path to last.pth / epNNN.pth)
    ck = torch.load(os.environ["TOB_CKPT"], map_location=DEV, weights_only=False)
    net = BYOLNet(False).to(DEV).to(memory_format=torch.channels_last); net.load_state_dict(ck["online"])
    summary["modes"]["eval_ckpt"] = evaluate(ev_rows_all, extract(net.backbone, ev_rows_all), f"ckpt:{os.environ['TOB_CKPT']}")
    dump(); log("DONE eval"); sys.exit(0)

amp = CFG["train"]["amp"] and DEV.type == "cuda"
bs, ck, probe_peak = probe_batch(amp)
summary["batch"] = {"batch_size": bs, "grad_ckpt": ck, "probe_peak_gib": probe_peak}
if SMOKE and DEV.type == "cuda":
    # timed steps at the real full-run batch on real data (both modes behave the same on GPU)
    online, target, opt = build(bs, ck); scaler = torch.amp.GradScaler("cuda", enabled=amp)
    dsx = SSLDS(SSL_ROWS, "anatomy")
    dlx = torch.utils.data.DataLoader(dsx, batch_size=bs, shuffle=True, drop_last=True, num_workers=CFG["data"]["num_workers"])
    it = iter(dlx); times = []; td = []
    for k in range(min(CFG["smoke"]["time_steps"] + 2, len(dlx))):
        t1 = time.time(); G, L, _ = next(it); td.append(time.time() - t1)
        torch.cuda.synchronize(); t2 = time.time()
        train_step(online, target, opt, scaler, G, L, 0.99, amp); torch.cuda.synchronize(); times.append(time.time() - t2)
    gpu_s = float(np.median(times[2:])) if len(times) > 2 else float(np.median(times))
    TIM["timed_gpu_s_per_step"] = gpu_s; TIM["timed_gpu_img_per_s"] = bs / gpu_s
    TIM["timed_data_wait_s_per_step_median"] = float(np.median(td[2:])) if len(td) > 2 else float("nan")
    log(f"timed: {gpu_s:.3f}s/step GPU at bs {bs} -> {bs/gpu_s:.1f} img/s; data wait median {TIM['timed_data_wait_s_per_step_median']:.3f}s")
    del online, target, opt; torch.cuda.empty_cache()
dump()
for mode in CFG["train"]["crop_modes"]:
    tm = time.time()
    summary["modes"][mode] = run_mode(mode, bs, ck)
    TIM[f"{mode}_total_s"] = time.time() - tm; dump()

# full-run time estimate (from measured throughput)
est = {}
for mode, r in summary["modes"].items():
    h = r["history"]
    ips = max(x["img_per_s"] for x in h)            # wall img/s incl. dataloader (best epoch = warm)
    knn_s = np.mean([x["knn_s"] for x in h if "knn_s" in x]) if any("knn_s" in x for x in h) else 0
    n_knn_imgs = len(SSL_ROWS) + len(Q_ROWS)
    knn_full = knn_s * (N_FULL_SSL + int((FOLDS == EF).sum())) / n_knn_imgs
    ep_s = N_FULL_SSL / ips
    E = 150
    est[mode] = {"wall_img_per_s": ips, "sec_per_epoch_full_data": ep_s, "knn_monitor_s_full": knn_full,
                 "est_150ep_hours": (E * ep_s + (E // CFG["monitor"]["knn_every"]) * knn_full) / 3600}
summary["estimate_full"] = est; TIM["wall_s"] = time.time() - T0; dump()
log("ESTIMATE", json.dumps(est, default=float))
log("DONE", json.dumps(TIM, default=float))
