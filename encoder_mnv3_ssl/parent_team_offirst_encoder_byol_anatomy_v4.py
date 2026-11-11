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
# v2 'anatomy_v2' (run C): 4 local 160 px views, each anchored 40% disc / 30% macula /
#   30% random; disc crop 1.2-1.5 DD (tight) or 2.5-3.0 DD (wide), macula ~2.5 DD,
#   centre jitter <= 0.3 x crop; low-confidence discs (detector top1-top2 margin <
#   cutoff) -> random. Boxes come from the CPU detector (box table embedded below, in
#   fundus-square fractions, so they rescale to any cache size). Disc crops: mild
#   brightness/contrast only (pallor = colour). Camera-robust augs (blur, JPEG 40-95,
#   brightness) on all views. 70 epochs, kNN monitor every 5, best-by-kNN + last ckpt.
# v4 'anatomy_v4' (final): 4 local 160 px views = fixed 1 disc + 1 macula + 2 random (slot order
#   shuffled per image); disc AND macula crops 2.5-3.2 DD only (no tight crops), centre jitter <= 0.3 x crop;
#   low-confidence disc -> that anatomy slot falls back to a random crop. Uniform weak colour jitter on ALL
#   views (b/c 0.2, sat 0.15, hue 0.03, p 0.8; no disc exemption) + camera augs (blur/JPEG/brightness).
#   Loss = BYOL + 0.1 * (10 * VICReg-variance + 1 * VICReg-covariance) on online global projector z
#   (Bardes et al. 2022). 120 epochs, warmup 8, kNN every 5, ckpt every 10, forced eval at ep 70 and 120,
#   eval-only multi-region concat probe [global, disc 2.5 DD, macula 2.5 DD].
# RUN_MODE: 'smoke' (2 epochs) | 'full' | 'eval'
# =============================================================================

RUN_MODE = "full"   # "smoke" | "full" | "eval"

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
  local_scale: [0.25, 0.55]
  ratio: [0.75, 1.3333]
  hflip: 0.5
  vflip: 0.5
  rotate_deg: 30
  jitter_p: 0.8
  brightness: 0.2
  contrast: 0.2
  saturation: 0.15
  hue: 0.03                 # fraction of the hue circle; same jitter on ALL views (incl. disc/macula)
  blur_p: 0.3               # camera-robust, all views
  blur_sigma: [0.1, 2.0]
  jpeg_p: 0.5               # camera-robust, all views
  jpeg_q: [40, 95]
  cam_bright_p: 0.5         # camera-robust brightness gain, all views
  cam_bright: 0.15
anatomy_v4:
  n_disc: 1
  n_macula: 1
  n_random: 2
  disc_dd: [2.5, 3.2]       # wide only (no tight 1.2-1.5)
  mac_dd: [2.5, 3.2]
  jitter_frac: 0.3          # centre jitter up to 0.3 x crop size
  conf_cutoff: 1.0          # detector top1-top2 margin; below -> anatomy slot becomes random
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
  crop_modes: [anatomy_v4]
  epochs: 120
  batch_size: 64              # fallback ladder on OOM: 64 -> 32 + grad ckpt -> 16 + grad ckpt
  lr_heads: 1.0e-3
  lr_backbone: 2.0e-4
  weight_decay: 0.05
  warmup_epochs: 8
  tau_base: 0.99              # EMA momentum, cosine 0.99 -> 1.0 over the run
  amp: true
  grad_clip: 3.0
  ckpt_every: 10
  vicreg:
    enabled: true
    lambda_var: 10.0
    lambda_cov: 1.0
    loss_weight: 0.1
    gamma: 1.0                # target per-dim std
monitor:
  knn_every: 5
  knn_k: 20
  collapse_warn: 0.2          # warn if mean per-dim std * sqrt(d) of L2-normed z drops below
  eval_epochs: [70, 120]      # forced full eval (probe + few-shot + multi-region) + ckpt
eval:
  img_size: 384
  batch_size: 64
  common_min_query_pos: 10    # -> the 12 common labels of the baseline (incl. Disease_Risk)
  rare_min_total_pos: 15
  rare_min_query_pos: 10
  rare_headline: [ODC, ODP, ARMD]   # MH = media haze (not macular hole); per-label results kept for all rare labels
  shots: [1, 5, 10]
  draws: 20
  neg_per_pos: 1
  knn_k: 5
  logreg_C: 1.0
  probe_C_grid: [0.01, 0.1, 1.0]
  multi_region_probe:
    enabled: true
    disc_dd: 2.5
    mac_dd: 2.5
    region_size: 224
    power_norm: true
smoke:
  epochs: 2
  n_train: 0                  # 0 = all SSL images (real epoch time)
  n_query: 96
  knn_every: 1
  eval_epochs: [1]
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
CFG_FULL_EPOCHS = CFG["train"]["epochs"]
RUN_MODE = os.environ.get("TOB_RUN_MODE", RUN_MODE)
SMOKE = RUN_MODE == "smoke"
if SMOKE:
    s = CFG["smoke"]
    CFG["train"]["epochs"] = s["epochs"]; CFG["train"]["warmup_epochs"] = 0
    CFG["monitor"]["knn_every"] = s["knn_every"]; CFG["monitor"]["eval_epochs"] = s["eval_epochs"]
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

# ----------------------------------------------------------------------------- box table (CPU detector, stage 1)
# columns: image_id, disc_cx, disc_cy, disc_diam, mac_cx, mac_cy (fractions of the fundus square = cache frame),
# confidence = top1 - top2 disc-candidate score margin. Produced on the box by stage1_boxes.py (cropper detector).
BOX_B64 = """
H4sIAENF62oC/3y9SZIly64kNve1PLlifbMaSklVkfIHVRxwQu6eUKiax8EJi/sm+W5mWLi7NTBAoVD81//6b//X//w//ut//Od/
/Nf/89//j//+/+rP/49//o//+m//6z//67/5P/gf/99//vv//b//z//6H//zf//3//nk/6R/0q552p995V3tz5x3x59tlLXtzzH3
2P/J/4w9V3mK/c3qaeAnRslzfYzoxX/BzG01G7DHSPXxX1ln7viVu/fx+Yi5On5hz3mW/9R/Rq1rPA0/0VNv/iv73p+PyLMvPjrt
/4x/RstpPfjdeeWJ3z1qai28VK/+GWPv8p9ib5fyfvCT9vYFI3ttM7zVTuXnO8o/9ovtO6Z+N55eWprzY4RNTZn+PWXgN82Z5rM4
U/4ZI+2pufWBfa6MD6+tzm4fXlta7cFrLvsdzX/VHvPyUj1X+/tsU9r2enLy2e15+ZDc+sdD3iGpr2EP6W1NG4I139N+Lf6pzZY/
P6QVn97aqy9M3qv2x18Vk+WvnuwjwzZZ03/V3iP/p9mfc9iQ6k+pc/Nr5+eLzVl9iWx/ZPyqPnKZT8ZfrVUXt1bhC8StNWxFsCRp
7GKf4r/U/sqfP2rJt1W0OV42pNi/23sNTljlkBw2V1ulFX+4/+JhG+Pxo7H61kuMsLUSZ3HYImNNRrP/92QdCfxk3WuEyUqr+Jxx
Z02b/fFkvIMdFv/dzZ71MWCmtX1BRrevyf+UOvJTfNXH9AG9lhRmyqfJHmArvkfCkfVpmbkuXwvbBd8/XoZ9R/4nldTqU/y32dH0
k1lzD0bBNsF6Pxl7yrbDw4kea/jspRImte7GScp+9mff2R6CV9+t6ZelYBbstPBdN05HwzrN/XBR6uQS5Z5bWLrUuLVsvfFetYz5
+EZdjava64pb6jUl+Pr6D16iP36Oc6FhyL2Vy0P4fvmfWu1YP4UHfdD6mHUJm9B+loZj4Z/3Ws2+xOenZj96tuB1hFO7Es+THzc7
kTCiiX/ju7FufOPPQ+rYPhc8vNnWaObyVPxyO0wNv7zZMtXLthotDTeKc6/+VP/Ymmgvmxmxr41iJ7/Dkpj12/YE/LXtb84iPjHM
buduMKtbcBnkPu07fNlr84WwL92fk9XH8pk4y1/nSvYUvzjK9BunptI+n9IyjLoN6c3/edsN8nA67aj5L+09mGozXyvrjI9/arbZ
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
hAuHcXsjOwNT8HkTlHnejZfUMJsGdRPmIplGaLWEFMXhw5xIsNK7WfIRZDxT/pyDw9Q91dlws/EckvKZpEgploZKocZroTvYgohr
mZvvTAjjFr5h55lKX3bjDtsDJANmeakjYme/kro2abbVSMUfkyXzI/IthrDB1xu2y8qOgcchu3Ov9dyC4sUp7Ta/PFUHz6DwUkkJ
nORqtxTTNC9Rhyn3thw/UnHwbmIRf1WufUrPmNOx7GNYG7y0E0r+eoq4Xe4ZonCt+FMYDtBCjTLLjRJ7SlEaGPAPucmoYGacvObv
JNV0PqBdgLjViXs0JrPg1txMgOfpgVLb7NgQrh332sjREar5p8oHOFDxEYcNTgsw669oW8czmeXCzy8hrAxEWgD1mcn8gQBTR1DL
8q7EEpxuocMMGX0y/k91g20WSMM4CrgOC8c+vt1o6h4U2Cpa8LRsDEvD5hZAHQuQBT7/QPXDLrRGJmDN8pyicNRLavLyIJRGWqTx
NHEBmS8GSeVKbU8sQy9mzLqNoR/I66muL27rYfb2Y7HAt5HFW4XXUJ/XQpLVHdfHHZkfBsd2PxclcCISdGyzb7OGKTcPjdh8KSr7
bV874OwzbU2zD6U8TaJDi9s8XQOI4fUX4E+UhRmgHyB7vmJF5UsHVb3ztGDLxmSmgo/gX8hr948LfTUb8LxUXsLTa8ZqnVPo534j
Mod2n9kYFgcy8P6S9PrRu1PcbfbI9pmjJtDXYVwbxRHNzosU6/ZtJpQptAMDDkmMhByV8hNnWYb5ZzbFjgIm0ezql9phaZ/BE0RG
zFewMQ4FwWXhBARWF9HcH/nJYVsIz2Ec0GgA5owktZfcL4rvtFDloek2/4xYUM0xRXsoxKopxaGybebvmxppR3ago+ckai8XtLuL
aieN5cFF0MbPSftCT+HKdgtUbYs8TNDlsZMWvd0ZsVS6MmfbfAByiuxOlrDbCLmgshhy8cYFswFP4U+sI98UvZMT3TtO0+AI2wht
ARbVznoF9k/dtq039LTcYqVOxrZNWbTnh6bmmloVKjx2mzEGzKJDj5yvjxmMOHcCGYDZv7XpNCO2j8T7LoIpi4jMJttOc7toNolM
qdSDA1DG2Qnw0LMdte1jqOPI66+P1S4oNd8Q5F4IfTTyAS2yJh2p9HHTvJRGn/k3tje5rwapIN1Cn79lOzJUAbbNs4T9iiqCYiH2
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
"""
import io, gzip, base64
BOX = pd.read_csv(io.BytesIO(gzip.decompress(base64.b64decode("".join(BOX_B64.split())))))
BOX = BOX.set_index("image_id").loc[df.ID.values].reset_index()
assert len(BOX) == len(df) and (BOX.image_id.values == df.ID.values).all(), "box table / label mismatch"
A2 = CFG["anatomy_v4"]
BOX["low_conf"] = BOX.confidence < A2["conf_cutoff"]
BOX.to_csv(OUT / "box_table_used.csv", index=False)
LOC_STATS = {"n": int(len(BOX)), "conf_cutoff": A2["conf_cutoff"], "low_conf_frac_all": float(BOX.low_conf.mean()),
             "low_conf_frac_ssl": float(BOX.low_conf.values[FOLDS != EF].mean())}
log("box table:", json.dumps(LOC_STATS))
BOX_ARR = BOX[["disc_cx", "disc_cy", "disc_diam", "mac_cx", "mac_cy"]].values.astype(np.float64)
BOX_OK = (~BOX.low_conf.values).astype(bool)
if LOCATE_ONLY: sys.exit(0)

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
    if CFG["smoke"]["n_train"] > 0: SSL_ROWS = np.sort(rng.choice(SSL_ROWS, CFG["smoke"]["n_train"], replace=False))
    pass   # smoke: full eval fold for kNN monitor
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

def _camera(x, rng, bmax):
    """camera-robust augs (all views): brightness gain, gaussian blur, JPEG re-encode."""
    fg = x.max(2) > 5
    if rng.rand() < V["cam_bright_p"]:
        x = np.clip(x.astype(np.float32) * rng.uniform(1 - bmax, 1 + bmax), 0, 255).astype(np.uint8); x[~fg] = 0
    if rng.rand() < V["blur_p"]:
        s = rng.uniform(*V["blur_sigma"]); x = cv2.GaussianBlur(x, (0, 0), s)
    if rng.rand() < V["jpeg_p"]:
        q = int(rng.randint(V["jpeg_q"][0], V["jpeg_q"][1] + 1))
        ok, enc = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok: x = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    return x

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
    return _camera(x, rng, V["cam_bright"])

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

ANCHOR_NAMES = ["random", "disc", "macula", "fallback_random"]
N_ANCH = len(ANCHOR_NAMES)
def anatomy_box(rng, row, S, kind):
    """-> (center, wh, type) for one anatomy slot ('disc' | 'macula'); low-confidence disc -> random crop (type 3)."""
    if not BOX_OK[row]: return _rrc_box(rng, S, V["local_scale"]) + (3,)
    dcx, dcy, dd, mcx, mcy = BOX_ARR[row] * S
    if kind == "disc": w = dd * rng.uniform(*A2["disc_dd"]); cx, cy, ty = dcx, dcy, 1
    else:              w = dd * rng.uniform(*A2["mac_dd"]);  cx, cy, ty = mcx, mcy, 2
    j = A2["jitter_frac"] * w
    return (cx + rng.uniform(-j, j), cy + rng.uniform(-j, j)), (w, w), ty

def make_views(img, rng, row, mode):
    S = img.shape[0]
    gl = [_photometric(_geom(img, rng, V["global_size"], *_rrc_box(rng, S, V["global_scale"])), rng) for _ in range(2)]
    if mode == "anatomy_v4":
        slots = ["disc"] * A2["n_disc"] + ["macula"] * A2["n_macula"] + ["random"] * A2["n_random"]
        assert len(slots) == V["n_local"], "anatomy_v4 slots must equal n_local"
        rng.shuffle(slots)
    else: slots = ["random"] * V["n_local"]
    lo, types = [], []
    for s in slots:
        if s == "random": c, wh, ty = _rrc_box(rng, S, V["local_scale"]) + (0,)
        else: c, wh, ty = anatomy_box(rng, row, S, s)
        lo.append(_photometric(_geom(img, rng, V["local_size"], c, wh), rng)); types.append(ty)   # same colour+camera augs for every view
    return np.stack(gl), np.stack(lo), np.array(types, np.int64)

def crop_grid(path, n_img=12, seed=123):
    """PNG grid: per row one SSL image (with disc/macula marks) + its 4 local views labelled by anchor type."""
    rng0 = np.random.RandomState(seed)
    rows = rng0.choice(SSL_ROWS, n_img, replace=False); T = 200; out = []
    for r in rows:
        img = np.asarray(CACHE[r]); S = img.shape[0]
        ov = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR).copy(); dcx, dcy, dd, mcx, mcy = BOX_ARR[r] * S
        cv2.circle(ov, (int(dcx), int(dcy)), int(dd / 2), (0, 255, 0) if BOX_OK[r] else (0, 0, 255), 2)
        cv2.drawMarker(ov, (int(mcx), int(mcy)), (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
        ov = cv2.resize(ov, (T, T)); cv2.putText(ov, f"ID{df.ID.values[r]} c{BOX.confidence.values[r]:.2f}{'' if BOX_OK[r] else ' LOW'}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        _, lo, ty = make_views(img, np.random.RandomState(seed + int(r)), r, "anatomy_v4")
        tiles = [ov]
        for v, t in zip(lo, ty):
            v = cv2.resize(cv2.cvtColor(v, cv2.COLOR_RGB2BGR), (T, T)).copy()
            cv2.rectangle(v, (0, 0), (T, 18), (0, 0, 0), -1); cv2.putText(v, ANCHOR_NAMES[t], (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            tiles.append(v)
        out.append(np.concatenate(tiles, 1))
    cv2.imwrite(str(path), np.concatenate(out, 0)); log("crop grid ->", path)

class SSLDS(torch.utils.data.Dataset):
    def __init__(self, rows, mode): self.rows = np.asarray(rows); self.mode = mode; self.epoch = 0
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        rng = np.random.RandomState((CFG["seed"] * 1000003 + self.epoch * 7919 + i * 104729 + torch.initial_seed()) % 2**32)
        g, l, a = make_views(np.asarray(CACHE[r]), rng, r, self.mode)
        return torch.from_numpy(g).permute(0, 3, 1, 2), torch.from_numpy(l).permute(0, 3, 1, 2), torch.from_numpy(a)

class EvalDS(torch.utils.data.Dataset):
    def __init__(self, rows, size): self.rows = np.asarray(rows); self.size = size
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        im = cv2.resize(np.asarray(CACHE[self.rows[i]]), (self.size, self.size), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(im).permute(2, 0, 1), i

MR = CFG["eval"]["multi_region_probe"]
class RegionDS(torch.utils.data.Dataset):
    """eval-only: deterministic disc / macula square crop (MR[kind_dd] x DD, no jitter/rotation) resized to region_size.
    Low-confidence detections still use the detector estimate (flagged in report via low_conf_frac)."""
    def __init__(self, rows, kind): self.rows = np.asarray(rows); self.kind = kind
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]; img = np.asarray(CACHE[r]); S = img.shape[0]; out = MR["region_size"]
        dcx, dcy, dd, mcx, mcy = BOX_ARR[r] * S
        cx, cy, w = (dcx, dcy, dd * MR["disc_dd"]) if self.kind == "disc" else (mcx, mcy, dd * MR["mac_dd"])
        M = _affine((cx, cy), (w, w), out, 0.0, False, False)
        im = cv2.warpAffine(img, M, (out, out), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
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

VC = CFG["train"]["vicreg"]
def vicreg_var_cov(z, gamma=1.0, eps=1e-4):
    """VICReg variance + covariance terms (Bardes et al. 2022) on a (B, D) batch; fp32 for AMP safety.
    L_var = mean_j relu(gamma - std_j);  L_cov = sum_{i!=j} cov_ij^2 / D."""
    z = z.float(); B, D = z.shape
    if B <= 1: zero = z.new_zeros(()); return zero, zero
    zc = z - z.mean(0, keepdim=True)
    std = torch.sqrt(zc.var(0) + eps)
    l_var = F.relu(gamma - std).mean()
    cov = (zc.T @ zc) / (B - 1)
    off = cov - torch.diag(torch.diagonal(cov))
    return l_var, off.pow(2).sum() / D

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
    loss_byol = torch.stack(terms).mean()
    if VC["enabled"]:   # online projector z of the two global views, each (B, D); averaged
        vv = [vicreg_var_cov(zg[k * B:(k + 1) * B], VC["gamma"]) for k in range(2)]
        l_var = torch.stack([v[0] for v in vv]).mean(); l_cov = torch.stack([v[1] for v in vv]).mean()
        loss = loss_byol + VC["loss_weight"] * (VC["lambda_var"] * l_var + VC["lambda_cov"] * l_cov)
    else:
        l_var = l_cov = loss_byol.new_zeros(()); loss = loss_byol
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt); gn = torch.nn.utils.clip_grad_norm_(online.parameters(), CFG["train"]["grad_clip"])
    scaler.step(opt); scaler.update()
    ema_update(target, online, tau)
    return loss.detach(), zg[:B].detach().float(), hg[:B].detach().float(), gn, (float(loss_byol), float(l_var), float(l_cov))

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

@torch.no_grad()
def extract_region(backbone, rows, kind):
    dl = torch.utils.data.DataLoader(RegionDS(rows, kind), batch_size=CFG["eval"]["batch_size"], shuffle=False,
                                     num_workers=CFG["data"]["num_workers"], pin_memory=DEV.type == "cuda")
    was = backbone.training; backbone.eval(); E = []
    for x, _ in dl:
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
            E.append(backbone(norm_gpu(x)).float().cpu().numpy())
    backbone.train(was); E = np.concatenate(E)
    return (E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)).astype(np.float32)

def multi_region_eval(backbone, rows, E_glob, tag, od=None):
    """eval-only readout: concat [global, disc, macula] L2-normed embeddings, optional power-norm, re-L2."""
    E = np.concatenate([E_glob, extract_region(backbone, rows, "disc"), extract_region(backbone, rows, "macula")], 1)
    if MR["power_norm"]: E = np.sign(E) * np.sqrt(np.abs(E))
    E = (E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
    if od is not None: np.save(od / f"embeddings_{tag.split('/')[-1]}.npy", E)
    r = evaluate(rows, E, tag); r["low_conf_frac"] = float((~BOX_OK[np.asarray(rows)]).mean()); return r

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
    best_knn, best_ep = -1.0, 0; epoch_evals = {}
    for ep in range(tc["epochs"]):
        ds.epoch = ep; online.train(); target.train()
        t = time.time(); tl, nb, zs, hs, an, tstep = 0.0, 0, [], [], np.zeros(N_ANCH), []
        vcs = []
        for G, L, a in dl:
            ts = time.time()
            for gp in opt.param_groups: gp["lr"] = gp["base_lr"] * lr_factor(step, total, warm)
            tau = 1 - (1 - tc["tau_base"]) * (math.cos(math.pi * step / max(1, total)) + 1) / 2
            loss, z, h, gn, vc = train_step(online, target, opt, scaler, G, L, tau, amp)
            tl += float(loss); nb += 1; step += 1; an += np.bincount(a.numpy().ravel(), minlength=N_ANCH); vcs.append(vc)
            zs.append(z); hs.append(h)
            if DEV.type == "cuda": torch.cuda.synchronize()
            tstep.append(time.time() - ts)
        t_tr = time.time() - t
        zc = torch.cat(zs); hc = torch.cat(hs)
        row = {"epoch": ep + 1, "loss": tl / max(1, nb), "std_z": std_metric(zc), "std_h": std_metric(hc), "tau_end": tau,
               "lr_bb": opt.param_groups[0]["lr"], "train_s": t_tr, "img_per_s": nb * bs / max(t_tr, 1e-9),
               "gpu_img_per_s_median_step": bs / float(np.median(tstep)) if tstep else float("nan"),
               "anchor_frac": {ANCHOR_NAMES[k]: float(an[k] / max(1, an.sum())) for k in range(N_ANCH)},
               "anatomy_views_frac": float(an[1:3].sum() / max(1, an.sum()))}
        vca = np.mean(vcs, 0) if vcs else np.full(3, np.nan)
        row.update({"loss_byol": float(vca[0]), "vicreg_var": float(vca[1]), "vicreg_cov": float(vca[2])})
        if row["std_z"] < CFG["monitor"]["collapse_warn"]: log(f"  WARNING possible collapse: std_z {row['std_z']:.3f}")
        if row["std_h"] < CFG["monitor"]["collapse_warn"]: log(f"  WARNING backbone std_h {row['std_h']:.3f} < {CFG['monitor']['collapse_warn']} (ep {ep+1})")
        if (ep + 1) % CFG["monitor"]["knn_every"] == 0 or ep + 1 == tc["epochs"]:
            tk = time.time()
            Eb = extract(online.backbone, mon_s); Eq = extract(online.backbone, mon_q)
            row["knn_mauroc"], row["knn_dr_acc"], row["knn_dr_auroc"] = knn_monitor(Eb, Y_ALL[mon_s], Eq, Y_ALL[mon_q], common_mon, CFG["monitor"]["knn_k"])
            row["knn_s"] = time.time() - tk
            if row["knn_mauroc"] > best_knn:
                best_knn, best_ep = row["knn_mauroc"], ep + 1
                torch.save({"online": online.state_dict(), "target_backbone": target.backbone.state_dict(),
                            "target_projector": target.projector.state_dict(), "epoch": ep + 1, "knn_mauroc": best_knn,
                            "crop_mode": mode, "config": CFG}, od / "best.pth")
                log(f"  new best kNN {best_knn:.4f} at ep {ep+1} -> best.pth")
        hist.append(row)
        log(f"  [{mode}] ep {ep+1}/{tc['epochs']} loss {row['loss']:.4f} std_z {row['std_z']:.3f} std_h {row['std_h']:.3f} "
            f"(byol {row['loss_byol']:.4f} var {row['vicreg_var']:.3f} cov {row['vicreg_cov']:.3f}) tau {tau:.4f} {row['img_per_s']:.1f} img/s ({t_tr:.0f}s)"
            + (f" | kNN mAUROC {row['knn_mauroc']:.3f} DR-risk acc {row['knn_dr_acc']:.3f} ({row['knn_s']:.0f}s)" if "knn_mauroc" in row else "")
            + f" anat {row['anatomy_views_frac']:.2f} " + " ".join(f"{k[:6]}={v:.2f}" for k, v in row["anchor_frac"].items()))
        json.dump(hist, open(od / "history.json", "w"), indent=1)
        if (ep + 1) % tc["ckpt_every"] == 0 or ep + 1 == tc["epochs"]:
            ck_d = {"online": online.state_dict(), "target_backbone": target.backbone.state_dict(),
                    "target_projector": target.projector.state_dict(), "epoch": ep + 1, "crop_mode": mode, "config": CFG}
            torch.save(ck_d, od / ("last.pth" if ep + 1 == tc["epochs"] else f"ep{ep+1:03d}.pth"))
        if ep + 1 in CFG["monitor"]["eval_epochs"] and ep + 1 != tc["epochs"]:   # forced mid-run eval (final epoch handled below)
            te = time.time(); evr = np.sort(np.r_[SSL_ROWS, Q_ROWS]); Ee = extract(online.backbone, evr)
            torch.save({"online": online.state_dict(), "target_backbone": target.backbone.state_dict(),
                        "target_projector": target.projector.state_dict(), "epoch": ep + 1, "crop_mode": mode, "config": CFG}, od / f"ep{ep+1:03d}.pth")
            ee = {"online": evaluate(evr, Ee, f"byol_{mode}/ep{ep+1}_online")}
            if MR["enabled"]: ee["multi_region_online"] = multi_region_eval(online.backbone, evr, Ee, f"byol_{mode}/ep{ep+1}_multiregion")
            ee["eval_s"] = time.time() - te; epoch_evals[f"ep{ep+1}"] = ee
            json.dump(epoch_evals, open(od / "epoch_evals.json", "w"), indent=1, default=float)
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
    if MR["enabled"]: rep["eval_multi_region_online"] = multi_region_eval(online.backbone, ev_rows, E_on, f"byol_{mode}/multiregion_online", od)
    rep["epoch_evals"] = epoch_evals
    rep["best_knn"] = {"epoch": best_ep, "knn_mauroc": best_knn}
    if best_ep and best_ep != tc["epochs"]:
        bk = torch.load(od / "best.pth", map_location=DEV, weights_only=False); online.load_state_dict(bk["online"])
        E_b = extract(online.backbone, ev_rows); np.save(od / "embeddings_best_online.npy", E_b)
        rep["eval_best_online"] = evaluate(ev_rows, E_b, f"byol_{mode}/best_ep{best_ep}_online")
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
be = baseline_embeddings() if not SMOKE else None
if be is not None:
    brows, bE, bsrc = be; keep = np.isin(brows, ev_rows_all)
    summary["references"]["supervised_baseline"] = evaluate(brows[keep], bE[keep], "supervised_convnext_tiny_best")
    summary["references"]["supervised_baseline"]["source"] = bsrc
    summary["references"]["supervised_baseline"]["note"] = "embeddings at 512 px; this model was TRAINED on all 28 labels of folds 1-4, so rare labels are not unseen for it"
else:
    log("WARNING: supervised baseline embeddings not mounted; skipping reference (c)")
t = time.time()
if not SMOKE:
  _m = timm.create_model(CFG["model"]["backbone"], pretrained=CFG["model"]["pretrained"], num_classes=0).to(DEV).to(memory_format=torch.channels_last)
  summary["references"]["imagenet_init"] = evaluate(ev_rows_all, extract(_m, ev_rows_all), "imagenet_init_384")
  del _m
TIM["ref_eval_s"] = time.time() - t; dump()
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
if False and SMOKE and DEV.type == "cuda":
    # timed steps (skipped since v2: smoke runs full-size epochs) at the real full-run batch on real data (both modes behave the same on GPU)
    online, target, opt = build(bs, ck); scaler = torch.amp.GradScaler("cuda", enabled=amp)
    dsx = SSLDS(SSL_ROWS, "anatomy_v4")
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
    crop_grid(OUT / "local_views_grid.png")
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
    E = CFG_FULL_EPOCHS
    est[mode] = {"wall_img_per_s": ips, "sec_per_epoch_full_data": ep_s, "knn_monitor_s_full": knn_full,
                 "est_full_hours": (E * ep_s + (E // CFG["monitor"]["knn_every"]) * knn_full) / 3600}
summary["estimate_full"] = est; TIM["wall_s"] = time.time() - T0; dump()
log("ESTIMATE", json.dumps(est, default=float))
log("DONE", json.dumps(TIM, default=float))
