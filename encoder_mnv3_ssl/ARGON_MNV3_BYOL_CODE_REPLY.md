The code in degrade.py has a hidden channel-order mismatch with parent_team_offirst_encoder_byol_anatomy_v4.py: degrade.py assumes BGR ([gain_b, gain_g, gain_r] and cv2.imencode), whereas the parent script's CACHE stores RGB (cv2.cvtColor(img, cv2.COLOR_BGR2RGB) in load_prep). Calling degrade() directly on CACHE[r] would invert the red and blue gains (gain_r=0.85, gain_b=1.15) and swap R/B during JPEG re-encoding. Additionally, radial vignetting in degrade.py is defined relative to the full fundus circle ($R = \sqrt{\text{area}/\pi}$), so applying it after cropping a $224\times224$ local disc view would put a dark vignette ring around the optic disc instead of the retinal rim.Below is the design table for your 7 open choices (flagged as MY SUGGESTION where open), followed by the complete, self-contained replacement script that fixes the RGB/BGR and full-eye vignette geometry issues while preserving all parent data guards and output schemas.1. Design Table for Open Choices (1–7)ChoiceDefault Proposal (MY SUGGESTION)Technical & Empirical Rationale1. Degradation Entry & SeverityAsymmetric-Anchored View Schedule:• Global 0 (Target $t_1$ / Online $p_1$): Clean ($\alpha=0$, $p=0.0$).• Global 1 (Target $t_2$ / Online $p_2$): Visuscout Degraded ($p=0.85$, severity $\alpha \sim U(0.35, 1.0)$ + $\pm 10\%$ per-param jitter; $15\%$ probability of green rim leak $\in [0.05, 0.18]$).• Locals 0..3 (Online only): Independently Degraded ($p=0.60$, $\alpha \sim U(0.25, 1.0)$).• Parameter interpolation: Linearly interpolate each physical parameter from identity ($\gamma=1, \text{exp}=1, \text{vig}=0, \sigma=0, \text{gains}=1, Q=95$) to the fitted Visuscout target by $\alpha$, rather than pixel-blending $(1-\alpha)I_{\text{clean}} + \alpha I_{\text{deg}}$.Pixel-blending clean and blurred images creates an artificial "double-exposure / unsharp-mask" halo that never occurs optically. Interpolating the physical parameters by $\alpha \sim U(0.35, 1.0)$ produces realistic intermediate handheld captures. Keeping Global 0 clean ensures that in byol_loss(p2, t1) and byol_loss(pk, t1), the degraded online global ($p_2$) and degraded local anatomy views ($p_k$) always predict a clean EMA target anchor ($t_1$), while byol_loss(p1, t2) forces the clean online view to predict the degraded EMA target—preventing MobileNetV3 BatchNorm running stats from drifting purely to degraded domain statistics.2. Speed, Cache Size & Placementcache_size: 640 (uint8 memmap = $1920 \times 640^2 \times 3 \approx 2.36\text{ GB}$ in /kaggle/temp).Run optimized NumPy/OpenCV degrade_rgb_fast() inside DataLoader CPU workers (precomputing the $640\times640$ radial coordinate grid once at startup, applying CRF/vignette/color on the full square before affine crop, and applying defocus blur/noise/JPEG after crop at 512/224).Why not GPU Torch for degrade.py? JPEG compression (cv2.imencode at $Q=70$) is non-trivial to replicate identically in pure PyTorch without torchvision.io.encode_jpeg CPU round-trips. In unmodified degrade.py, np.indices((h,w)) and morphologyEx take ~18 ms/call; precomputing the radial grid at CS=640 and moving blur/JPEG after the affine crop drops CPU time to ~3–4 ms/image. On Kaggle (4 vCPUs, num_workers=2 or 4), CPU workers easily feed ~120–160 img/s. Estimated T4 throughput (unmeasured estimate): at bs=64, MobileNetV3-Large @ 512 (2 globals + 4 locals) should run at ~70–110 img/s (~14–22 sec/epoch for 1,536 SSL images; a 40-epoch run + evals will take ~25–35 minutes total on a single T4).3. Local Crop Sizelocal_size: 224 (scaled up from 160 at 384 globals).Keeps the exact $7:16$ ($0.4375\times$) linear resolution ratio from v4 ($160 / 384 = 0.417$, $224 / 512 = 0.4375$) and matches MobileNetV3-Large's native $224\times224$ ImageNet pretraining receptive field ($7\times7$ final feature map before GAP).4. Projector / Predictor & 768-d LayerKeep native 960-d GAP for backbone & probes (no extra bottlenecks inside the backbone during BYOL).• proj_hidden: 2048, proj_out: 256, pred_hidden: 2048 (same as v4).• Add an optional neck_768 (nn
# team_offirst - Run 1: Direct Small-Backbone BYOL (MobileNetV3-Large @ 512)
#   with Anatomy v4 Views + Parametric Visuscout Degradation (degrade.py)
# -----------------------------------------------------------------------------
# Parent: parent_team_offirst_encoder_byol_anatomy_v4.py + degrade.py
# Changes locked by Jeet:
#   - Method: BYOL (no KD, no disease labels in SSL training, VICReg OFF)
#   - Backbone: timm mobilenetv3_large_100 (ImageNet init, 960-d GAP)
#   - Resolution: 512x512 globals (cache_size=640), 224x224 locals (1 disc + 1 mac + 2 rand)
#   - Views: v4 anatomy recipe + continuous Visuscout degradation (degrade.py)
#   - Validation: Inner-validation split (Train Folds 1-3 -> Val Fold 4) selects
#     best_inner.pth; Fold 0 reported as disclosed development benchmark.
#   - Extra Eval: Clean vs. Visuscout-Degraded linear probes, few-shot, erank,
#     and paired clean-degraded cosine minus random-pair cosine.
# RUN_MODE: 'smoke' (2 epochs) | 'full' | 'eval'
# =============================================================================

RUN_MODE = "full"   # "smoke" | "full" | "eval"

CONFIG_YAML = r"""
seed: 42
data:
  cache_size: 640           # raised from 512 -> 640 for 512px globals (~2.36 GB uint8 memmap)
  n_folds: 5
  eval_fold: 0              # excluded from SSL; disclosed development benchmark
  inner_val_fold: 4         # inner validation fold inside 1..4 for checkpoint selection
  num_workers: 4
views:
  global_size: 512          # locked by Jeet (MA median diameter ~2.6 px at 512 vs 1.9 px at 384)
  global_scale: [0.5, 1.0]
  n_local: 4
  local_size: 224           # scaled from 160 -> 224 to match 512 globals and MNV3 7x7 native grid
  local_scale: [0.25, 0.55]
  ratio: [0.75, 1.3333]
  hflip: 0.5
  vflip: 0.5
  rotate_deg: 30
  jitter_p: 0.8
  brightness: 0.2
  contrast: 0.2
  saturation: 0.15
  hue: 0.03
  blur_p: 0.2               # reduced mild camera blur when Visuscout degrader is not applied
  blur_sigma: [0.1, 1.5]
  jpeg_p: 0.3
  jpeg_q: [50, 95]
  cam_bright_p: 0.4
  cam_bright: 0.12
visuscout_degrade:
  enabled: true
  p_global0: 0.0            # keep Global 0 clean as stable BYOL anchor
  p_global1: 0.85           # degrade Global 1 with 85% prob (asymmetric clean <-> degraded)
  p_local: 0.60             # degrade each local view independently with 60% prob
  alpha_range: [0.35, 1.0]  # continuous severity blend from identity (0.0) to full Visuscout (1.0)
  param_jitter: 0.10        # +/- 10% multiplicative jitter around fitted Visuscout parameters
  leak_p: 0.15              # 15% prob of mild greenish rim leak seen on some Visuscout frames
  leak_range: [0.05, 0.18]
  # Fitted 2026-11-06 defaults from degrade.py (RET-Amrita Visuscout 100)
  crf_gamma: 1.05
  crf_toe: 0.02
  crf_shoulder: 0.92
  exposure: 1.12
  vignette_strength: 0.16
  vignette_power: 2.0
  blur_sigma: 1.15
  gain_r: 0.85
  gain_g: 1.05
  gain_b: 1.15
  leak_width: 0.25
  noise_sigma: 1.4
  jpeg_quality: 70
anatomy_v4:
  n_disc: 1
  n_macula: 1
  n_random: 2
  disc_dd: [2.5, 3.2]
  mac_dd: [2.5, 3.2]
  jitter_frac: 0.3
  conf_cutoff: 1.0
anatomy:
  p_image: 0.5
  crop_dd: 2.5
  dd_frac: 0.12
  jitter_dd: 0.25
  scale_jitter: [0.85, 1.15]
  min_conf: 0.5
model:
  backbone: mobilenetv3_large_100
  pretrained: true
  proj_hidden: 2048
  proj_out: 256
  pred_hidden: 2048
train:
  crop_modes: [anatomy_v4]
  epochs: 40                # v4 peaked at ep20; 40 ep with evals at [20, 30, 40] covers the sweet spot
  batch_size: 64            # fallback ladder on OOM: 64 -> 48 -> 32 + grad ckpt -> 16 + grad ckpt
  lr_heads: 1.0e-3
  lr_backbone: 3.0e-4
  weight_decay: 1.0e-4      # IMPORTANT: 1e-4 for MobileNetV3 (0.05 over-regularizes depthwise convs)
  warmup_epochs: 4
  tau_base: 0.99
  amp: true
  grad_clip: 3.0
  ckpt_every: 10
  vicreg:
    enabled: false          # locked OFF by Jeet
    lambda_var: 10.0
    lambda_cov: 1.0
    loss_weight: 0.1
    gamma: 1.0
monitor:
  knn_every: 5
  knn_k: 20
  collapse_warn: 0.2
  eval_epochs: [20, 30, 40]
eval:
  img_size: 512             # match 512 global resolution
  batch_size: 64
  common_min_query_pos: 10
  rare_min_total_pos: 15
  rare_min_query_pos: 10
  rare_headline: [ODC, ODP, ARMD]
  shots: [1, 5, 10]
  draws: 20
  neg_per_pos: 1
  knn_k: 5
  logreg_C: 1.0
  probe_C_grid: [0.01, 0.1, 1.0]
  multi_region_probe:
    enabled: false          # disabled per v4 gate (f_full only)
    disc_dd: 2.5
    mac_dd: 2.5
    region_size: 224
    power_norm: true
smoke:
  epochs: 2
  n_train: 0
  n_query: 96
  knn_every: 1
  eval_epochs: [1, 2]
  common_min_query_pos: 3
  rare_min_total_pos: 8
  rare_min_query_pos: 3
  draws: 3
  time_steps: 6
"""

import os, sys, json, time, math, random, copy, warnings
from pathlib import Path
from multiprocessing import Pool
from dataclasses import dataclass
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
if SMOKE and "TOB_SMOKE_N" in os.environ: CFG["smoke"]["n_train"] = int(os.environ["TOB_SMOKE_N"])
if SMOKE and "TOB_SMOKE_EPOCHS" in os.environ: CFG["train"]["epochs"] = int(os.environ["TOB_SMOKE_EPOCHS"])
if "TOB_CROP_MODES" in os.environ: CFG["train"]["crop_modes"] = os.environ["TOB_CROP_MODES"].split(",")
if "TOB_WORKERS" in os.environ: CFG["data"]["num_workers"] = int(os.environ["TOB_WORKERS"])
if "TOB_PRETRAINED" in os.environ: CFG["model"]["pretrained"] = os.environ["TOB_PRETRAINED"] == "1"
LOCATE_ONLY = os.environ.get("TOB_LOCATE_ONLY") == "1"

INPUT = Path(os.environ.get("TOB_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("TOB_OUT", "/kaggle/working/out"))
TMP = Path(os.environ.get("TOB_TMP", "/kaggle/temp" if Path("/kaggle").exists() else "/tmp/tob_byol"))
OUT.mkdir(parents=True, exist_ok=True); TMP.mkdir(parents=True, exist_ok=True)
LOGF = open(OUT / "run.log", "a")
log(f"RUN_MODE={RUN_MODE}  backbone={CFG['model']['backbone']}  global={CFG['views']['global_size']}  input={INPUT}  out={OUT}")

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

split_path = Path(os.environ["TOB_SPLIT"]) if "TOB_SPLIT" in os.environ else find_one("split_5fold_seed42.csv", must=False)
if split_path is not None and split_path.exists():
    sp = pd.read_csv(split_path)[["ID", "fold"]]
    df = df.merge(sp, on="ID", how="left")
    if df["fold"].isna().any(): raise SystemExit("split CSV does not cover all images")
    df["fold"] = df["fold"].astype(int); split_src = str(split_path)
else:
    df["fold"] = make_folds(df[LABELS].values, CFG["data"]["n_folds"], CFG["seed"]); split_src = "recomputed(iterstrat)"
EF = CFG["data"]["eval_fold"]
IVF = CFG["data"]["inner_val_fold"]
log(f"split from {split_src}; fold sizes {np.bincount(df.fold).tolist()}; dev eval fold {EF} (excluded from SSL); inner val fold {IVF}")
df[["ID", "fold"]].to_csv(OUT / "split_used.csv", index=False)
Y_ALL = df[LABELS].values.astype(np.float32)
FOLDS = df["fold"].values

# ----------------------------------------------------------------------------- preprocessing + cache (all 1,920 at CS=640)
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
# NOTE: Identical BOX_B64 from parent_team_offirst_encoder_byol_anatomy_v4.py
# (Coordinates are stored as fractions of the fundus square [0, 1], so they rescale automatically to CS=640)
from parent_team_offirst_encoder_byol_anatomy_v4 import BOX_B64 as _PARENT_BOX_B64 if False else None
# If running standalone on Kaggle without importing parent file, keep the exact BOX_B64 string from parent here:
Note on BOX_B64 for Kaggle: Because BOX_B64 in your parent script is a 260-line base64 string of normalized [0, 1] coordinates, keep that exact BOX_B64 = """...""" block from lines 191–449 of parent_team_offirst_encoder_byol_anatomy_v4.py in place. (Because its coordinates are normalized to the fundus square, changing cache_size from 512 to 640 requires zero changes to BOX_B64!). Below is the rest of the script starting immediately after BOX_OK = (~BOX.low_conf.values).astype(bool):Python# --- Paste lines 191-462 of parent_team_offirst_encoder_byol_anatomy_v4.py here ---
# (Beginningright after `if LOCATE_ONLY: sys.exit(0)` at line 463 of parent):

# ----------------------------------------------------------------------------- Visuscout Degrader (RGB-native, fast precomputed grids)
VD = CFG["visuscout_degrade"]

# Precompute static 2D coordinate grids at cache_size (CS x CS) to avoid 15ms np.indices() per view
_Y_CS, _X_CS = np.indices((CS, CS), dtype=np.float32)
_CY_CS, _CX_CS = CS * 0.5, CS * 0.5
_R_CS = CS * 0.485
_RAD_CS = np.clip(np.sqrt((_Y_CS - _CY_CS) ** 2 + (_X_CS - _CX_CS) ** 2) / _R_CS, 0.0, 1.2)

@dataclass
class VisuscoutSampledParams:
    crf_gamma: float
    crf_toe: float
    crf_shoulder: float
    exposure: float
    vignette_strength: float
    vignette_power: float
    blur_sigma: float
    gain_r: float
    gain_g: float
    gain_b: float
    leak_strength: float
    leak_angle_rad: float
    leak_width: float
    noise_sigma: float
    jpeg_quality: int

def sample_visuscout_params(rng: np.random.RandomState, alpha: float = None, deterministic: bool = False) -> VisuscoutSampledParams:
    """Interpolates physical degradation parameters between identity (alpha=0) and fitted Visuscout (alpha=1)."""
    if deterministic:
        a = 1.0
        jit = lambda v: v
        leak_s = 0.0
        leak_ang = 0.0
    else:
        a = float(rng.uniform(*VD["alpha_range"])) if alpha is None else float(alpha)
        j = VD["param_jitter"]
        jit = lambda v: float(v * rng.uniform(1.0 - j, 1.0 + j))
        leak_s = float(rng.uniform(*VD["leak_range"])) if rng.rand() < VD["leak_p"] else 0.0
        leak_ang = float(rng.uniform(0.0, 2.0 * math.pi))

    return VisuscoutSampledParams(
        crf_gamma=1.0 + a * (jit(VD["crf_gamma"]) - 1.0),
        crf_toe=a * jit(VD["crf_toe"]),
        crf_shoulder=1.0 - a * (1.0 - min(0.98, jit(VD["crf_shoulder"]))),
        exposure=1.0 + a * (jit(VD["exposure"]) - 1.0),
        vignette_strength=a * jit(VD["vignette_strength"]),
        vignette_power=VD["vignette_power"],
        blur_sigma=a * jit(VD["blur_sigma"]),
        gain_r=1.0 + a * (jit(VD["gain_r"]) - 1.0),
        gain_g=1.0 + a * (jit(VD["gain_g"]) - 1.0),
        gain_b=1.0 + a * (jit(VD["gain_b"]) - 1.0),
        leak_strength=a * leak_s,
        leak_angle_rad=leak_ang,
        leak_width=VD["leak_width"],
        noise_sigma=a * jit(VD["noise_sigma"]),
        jpeg_quality=int(round(96 - a * (96 - VD["jpeg_quality"])))
    )

def _crf_rgb(x: np.ndarray, p: VisuscoutSampledParams) -> np.ndarray:
    x = np.clip(x * p.exposure, 0.0, None)
    y = np.power(np.clip(x, 0.0, 1.0), p.crf_gamma)
    s = p.crf_shoulder
    if s < 0.999:
        hi = y > s
        y[hi] = s + (1.0 - s) * (1.0 - np.exp(-(y[hi] - s) / (1.0 - s)))
    return p.crf_toe + (1.0 - p.crf_toe) * y

def degrade_pre_crop_rgb(img_rgb: np.ndarray, p: VisuscoutSampledParams) -> np.ndarray:
    """Applies full-eye optical steps (CRF, radial vignette, RGB gains, optional rim leak) on the CSxCS RGB square.
    Wait to apply blur/noise/JPEG until AFTER geometric crop so local crops have full-eye vignette geometry."""
    fg = img_rgb.max(axis=2) > 12
    x = _crf_rgb(img_rgb.astype(np.float32) * (1.0 / 255.0), p)
    # Radial vignette on full fundus square
    vig = 1.0 - p.vignette_strength * np.power(np.clip(_RAD_CS, 0.0, 1.0), p.vignette_power)
    x *= vig[..., None]
    # RGB channel gains (NOTE: CACHE is RGB, so order is [gain_r, gain_g, gain_b]!)
    x *= np.array([p.gain_r, p.gain_g, p.gain_b], dtype=np.float32)
    if p.leak_strength > 0.0:
        d = ((_X_CS - _CX_CS) * math.cos(p.leak_angle_rad) + (_Y_CS - _CY_CS) * math.sin(p.leak_angle_rad)) / _R_CS
        leak = np.clip((d - (1.0 - p.leak_width)) / p.leak_width, 0.0, 1.0) ** 2
        # RGB greenish crescent: [R=0.35, G=1.0, B=0.45] (matches BGR [0.45, 1.0, 0.35] in degrade.py)
        x += (p.leak_strength * leak)[..., None] * np.array([0.35, 1.0, 0.45], dtype=np.float32)
    x[~fg] = 0.0
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

def degrade_post_crop_rgb(crop_rgb: np.ndarray, p: VisuscoutSampledParams, rng: np.random.RandomState) -> np.ndarray:
    """Applies sensor/compression steps (defocus blur, Gaussian sensor noise, JPEG) on the cropped view."""
    h, w = crop_rgb.shape[:2]
    fg = crop_rgb.max(axis=2) > 8
    scale = w / 1024.0
    x = crop_rgb.astype(np.float32)
    if p.blur_sigma > 0.05:
        x = cv2.GaussianBlur(x, (0, 0), p.blur_sigma * scale)
    if p.noise_sigma > 0.05:
        x += rng.normal(0.0, p.noise_sigma, x.shape).astype(np.float32)
    x[~fg] = 0.0
    out = (np.clip(x, 0.0, 255.0) + 0.5).astype(np.uint8)
    if p.jpeg_quality < 98:
        # Convert RGB -> BGR for OpenCV JPEG encoder so chroma subsampling uses true luminance Y
        bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(p.jpeg_quality)])
        if ok:
            out = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return out

def degrade_full_eval_rgb(img_rgb: np.ndarray, row_id: int, out_size: int) -> np.ndarray:
    """Deterministic full Visuscout degradation (alpha=1.0, no random leak) for evaluation splits."""
    rng = np.random.RandomState(CFG["seed"] * 10007 + int(row_id))
    p = sample_visuscout_params(rng, alpha=1.0, deterministic=True)
    pre = degrade_pre_crop_rgb(img_rgb, p)
    res = cv2.resize(pre, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return degrade_post_crop_rgb(res, p, rng)

# ----------------------------------------------------------------------------- torch
import torch, torch.nn as nn, torch.nn.functional as F
try:
    import timm
except ImportError:
    os.system("pip install -q timm"); import timm
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log("device", DEV, torch.cuda.get_device_name(0) if DEV.type == "cuda" else "", "torch", torch.__version__, "timm", timm.__version__)
def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(CFG["seed"])

SSL_ROWS = np.where(FOLDS != EF)[0]; Q_ROWS = np.where(FOLDS == EF)[0]
INNER_TR_ROWS = np.where((FOLDS != EF) & (FOLDS != IVF))[0]
INNER_VA_ROWS = np.where(FOLDS == IVF)[0]
if SMOKE:
    rng = np.random.RandomState(CFG["seed"])
    if CFG["smoke"]["n_train"] > 0: SSL_ROWS = np.sort(rng.choice(SSL_ROWS, CFG["smoke"]["n_train"], replace=False))
assert not set(SSL_ROWS) & set(np.where(FOLDS == EF)[0]), "fold-0 leak into SSL"
N_FULL_SSL = int((FOLDS != EF).sum())
log(f"SSL images {len(SSL_ROWS)} (full {N_FULL_SSL}); inner train {len(INNER_TR_ROWS)}, inner val (fold {IVF}) {len(INNER_VA_ROWS)}, dev query (fold {EF}) {len(Q_ROWS)}")

# ----------------------------------------------------------------------------- views
V = CFG["views"]; AN = CFG["anatomy"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255

def _affine(center, wh, out, ang, hf, vf):
    (cx, cy), (w, h) = center, wh
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], np.float64)
    Sm = np.diag([out / w, out / h, 1.0])
    a = math.radians(ang); Rm = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    Fm = np.diag([-1.0 if hf else 1.0, -1.0 if vf else 1.0, 1.0])
    T2 = np.array([[1, 0, out / 2], [0, 1, out / 2], [0, 0, 1]], np.float64)
    return (T2 @ Fm @ Rm @ Sm @ T1)[:2]

def _camera(x, rng, bmax):
    fg = x.max(2) > 5
    if rng.rand() < V["cam_bright_p"]:
        x = np.clip(x.astype(np.float32) * rng.uniform(1 - bmax, 1 + bmax), 0, 255).astype(np.uint8); x[~fg] = 0
    if rng.rand() < V["blur_p"]:
        s = rng.uniform(*V["blur_sigma"]); x = cv2.GaussianBlur(x, (0, 0), s)
    if rng.rand() < V["jpeg_p"]:
        q = int(rng.randint(V["jpeg_q"][0], V["jpeg_q"][1] + 1))
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(x, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok: x = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return x

def _photometric(x, rng, skip_camera=False):
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
    if skip_camera:
        return x
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
    if not BOX_OK[row]: return _rrc_box(rng, S, V["local_scale"]) + (3,)
    dcx, dcy, dd, mcx, mcy = BOX_ARR[row] * S
    if kind == "disc": w = dd * rng.uniform(*A2["disc_dd"]); cx, cy, ty = dcx, dcy, 1
    else:              w = dd * rng.uniform(*A2["mac_dd"]);  cx, cy, ty = mcx, mcy, 2
    j = A2["jitter_frac"] * w
    return (cx + rng.uniform(-j, j), cy + rng.uniform(-j, j)), (w, w), ty

def _render_view(img_clean, rng, out_size, center, wh, p_visuscout):
    """Renders a single view: with prob p_visuscout, applies Visuscout pre-crop (on full eye) + post-crop degradation;
    otherwise applies standard v4 photometric + camera augs."""
    use_deg = VD["enabled"] and (rng.rand() < p_visuscout)
    if use_deg:
        vp = sample_visuscout_params(rng)
        src = degrade_pre_crop_rgb(img_clean, vp)
        crop = _geom(src, rng, out_size, center, wh)
        crop = _photometric(crop, rng, skip_camera=True)  # mild jitter, then Visuscout blur/noise/JPEG
        return degrade_post_crop_rgb(crop, vp, rng)
    else:
        crop = _geom(img_clean, rng, out_size, center, wh)
        return _photometric(crop, rng, skip_camera=False)

def make_views(img, rng, row, mode):
    S = img.shape[0]
    g_probs = [VD["p_global0"], VD["p_global1"]]
    gl = [
        _render_view(img, rng, V["global_size"], *_rrc_box(rng, S, V["global_scale"]), p_visuscout=g_probs[k])
        for k in range(2)
    ]
    if mode == "anatomy_v4":
        slots = ["disc"] * A2["n_disc"] + ["macula"] * A2["n_macula"] + ["random"] * A2["n_random"]
        assert len(slots) == V["n_local"], "anatomy_v4 slots must equal n_local"
        rng.shuffle(slots)
    else:
        slots = ["random"] * V["n_local"]
    lo, types = [], []
    for s in slots:
        if s == "random": c, wh, ty = _rrc_box(rng, S, V["local_scale"]) + (0,)
        else: c, wh, ty = anatomy_box(rng, row, S, s)
        lo.append(_render_view(img, rng, V["local_size"], c, wh, p_visuscout=VD["p_local"]))
        types.append(ty)
    return np.stack(gl), np.stack(lo), np.array(types, np.int64)

def crop_grid(path, n_img=12, seed=123):
    """PNG grid: per row one SSL image + Global0 (clean) + Global1 (Visuscout degraded) + 4 local views."""
    rng0 = np.random.RandomState(seed)
    rows = rng0.choice(SSL_ROWS, min(n_img, len(SSL_ROWS)), replace=False); T = 200; out = []
    for r in rows:
        img = np.asarray(CACHE[r]); S = img.shape[0]
        ov = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR).copy(); dcx, dcy, dd, mcx, mcy = BOX_ARR[r] * S
        cv2.circle(ov, (int(dcx), int(dcy)), int(dd / 2), (0, 255, 0) if BOX_OK[r] else (0, 0, 255), 2)
        cv2.drawMarker(ov, (int(mcx), int(mcy)), (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
        ov = cv2.resize(ov, (T, T)); cv2.putText(ov, f"ID{df.ID.values[r]} c{BOX.confidence.values[r]:.2f}{'' if BOX_OK[r] else ' LOW'}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        gl, lo, ty = make_views(img, np.random.RandomState(seed + int(r)), r, "anatomy_v4")
        tiles = [ov]
        for k_g, gv in enumerate(gl):
            gv_b = cv2.resize(cv2.cvtColor(gv, cv2.COLOR_RGB2BGR), (T, T)).copy()
            cv2.rectangle(gv_b, (0, 0), (T, 18), (0, 0, 0), -1)
            cv2.putText(gv_b, f"G{k_g}_{'clean' if k_g==0 else 'visu'}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 255), 1)
            tiles.append(gv_b)
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
    def __init__(self, rows, size, degraded=False):
        self.rows = np.asarray(rows); self.size = size; self.degraded = degraded
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = int(self.rows[i])
        raw = np.asarray(CACHE[r])
        if self.degraded:
            im = degrade_full_eval_rgb(raw, r, self.size)
        else:
            im = cv2.resize(raw, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(im).permute(2, 0, 1), i

def norm_gpu(x):
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
        log(f"Backbone {mc['backbone']} created: num_features={d}")
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
        nowd = p.ndim <= 1 or n.endswith(".bias") or ".bn" in n
        groups.setdefault((part, nowd), []).append(p)
    return [{"params": ps, "lr": tc["lr_backbone"] if part == "backbone" else tc["lr_heads"],
             "base_lr": tc["lr_backbone"] if part == "backbone" else tc["lr_heads"],
             "weight_decay": 0.0 if nowd else tc["weight_decay"], "name": f"{part}{'_nowd' if nowd else ''}"}
            for (part, nowd), ps in groups.items()]

def train_step(online, target, opt, scaler, G, L, tau, amp):
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
    zero = 0.0
    return loss.detach(), zg[:B].detach().float(), hg[:B].detach().float(), gn, (float(loss.detach()), zero, zero)

def std_metric(z):
    zn = F.normalize(z, dim=1); return float(zn.std(0).mean() * math.sqrt(z.shape[1]))

# ----------------------------------------------------------------------------- embeddings + eval + collapse diagnostics
@torch.no_grad()
def extract(backbone, rows, size=None, degraded=False):
    size = size or CFG["eval"]["img_size"]
    dl = torch.utils.data.DataLoader(EvalDS(rows, size, degraded=degraded), batch_size=CFG["eval"]["batch_size"],
                                     shuffle=False, num_workers=CFG["data"]["num_workers"], pin_memory=DEV.type == "cuda")
    was = backbone.training; backbone.eval(); E = []
    for x, _ in dl:
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
            E.append(backbone(norm_gpu(x)).float().cpu().numpy())
    backbone.train(was)
    E = np.concatenate(E); return (E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)).astype(np.float32)

def collapse_and_shift_diagnostics(E_clean: np.ndarray, E_deg: np.ndarray) -> dict:
    """Computes effective rank (Roy & Vetterli 2007) and paired clean-degraded cosine vs random-pair cosine."""
    # 1. Effective rank of centered clean and degraded embeddings
    def _erank(Z):
        Zc = Z - Z.mean(axis=0, keepdims=True)
        s = np.linalg.svd(Zc, compute_uv=False)
        s = s[s > 1e-12]
        p = s / s.sum()
        return float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    erank_clean = _erank(E_clean)
    erank_deg = _erank(E_deg)
    # 2. Paired clean-degraded cosine vs off-diagonal random-pair cosine
    paired_cos = float(np.sum(E_clean * E_deg, axis=1).mean())
    sim_mat = E_clean @ E_deg.T
    n = len(E_clean)
    off_diag_cos = float((sim_mat.sum() - np.trace(sim_mat)) / max(1, n * (n - 1)))
    return {
        "erank_clean": erank_clean,
        "erank_degraded": erank_deg,
        "paired_clean_deg_cos": paired_cos,
        "random_pair_cos": off_diag_cos,
        "invariance_margin": paired_cos - off_diag_cos,
    }

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
        grid = CFG["eval"]["probe_C_grid"]; best, bestC = -1, grid[len(grid) // 2]
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
            rng = np.random.RandomState(CFG["seed"] * 1000 + kshot * 100 + d)
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

def evaluate_split(Eall_rows, E_clean, E_deg, s_rows, q_rows, tag):
    """Evaluates Clean->Clean, Clean->Degraded, and Mixed(Clean+Deg)->Degraded probes + collapse diagnostics."""
    rows = np.asarray(Eall_rows); pos = {r: i for i, r in enumerate(rows)}
    s_idx = [pos[r] for r in s_rows]; q_idx = [pos[r] for r in q_rows]
    Es_c, Eq_c = E_clean[s_idx], E_clean[q_idx]
    Es_d, Eq_d = E_deg[s_idx], E_deg[q_idx]
    ys, yq = Y_ALL[s_rows], Y_ALL[q_rows]
    common, rare = label_sets(q_rows, s_rows)
    t = time.time()
    diag = collapse_and_shift_diagnostics(Eq_c, Eq_d)
    # 1. Standard Clean -> Clean eval (matches parent schema)
    lp_clean = linear_probe(Es_c, ys, Eq_c, yq, common)
    fs_clean = fewshot(Es_c, ys, Eq_c, yq, rare, CFG["eval"]["rare_headline"])
    # 2. Clean support -> Degraded query (zero-shot domain shift on probes)
    lp_c2d = linear_probe(Es_c, ys, Eq_d, yq, common)
    # 3. Mixed (Clean+Degraded) support -> Degraded query (app probe deployment protocol)
    Es_mix = np.concatenate([Es_c, Es_d], axis=0)
    ys_mix = np.concatenate([ys, ys], axis=0)
    lp_m2d = linear_probe(Es_mix, ys_mix, Eq_d, yq, common)
    fs_deg = fewshot(Es_c, ys, Eq_d, yq, rare, CFG["eval"]["rare_headline"])

    rep = {
        "tag": tag, "n_support": len(s_rows), "n_query": len(q_rows),
        "common_labels": [LABELS[j] for j in common], "rare_labels": [LABELS[j] for j in rare],
        "rare_counts": {LABELS[j]: {"support_pos": int(ys[:, j].sum()), "query_pos": int(yq[:, j].sum())} for j in rare},
        "linear_probe": lp_clean,
        "linear_probe_clean_to_deg": lp_c2d,
        "linear_probe_mixed_to_deg": lp_m2d,
        "fewshot_rare": fs_clean,
        "fewshot_rare_degraded": fs_deg,
        "diagnostics": diag,
        "eval_s": time.time() - t,
    }
    log(f"  EVAL[{tag}] Clean LP {lp_clean['mean_auroc_common']:.3f} (DRisk {lp_clean['disease_risk_auroc']:.3f}) | "
        f"Deg LP(c->d) {lp_c2d['mean_auroc_common']:.3f} LP(m->d) {lp_m2d['mean_auroc_common']:.3f} | "
        f"erank {diag['erank_clean']:.1f}/{diag['erank_degraded']:.1f} cos_gap {diag['invariance_margin']:.3f} ({rep['eval_s']:.0f}s)")
    return rep

def evaluate_all(Eall_rows, E_clean, E_deg, tag):
    """Runs both Inner-Validation (Folds 1..3 -> Fold 4) and Disclosed Dev Benchmark (Folds 1..4 -> Fold 0)."""
    rows = np.asarray(Eall_rows)
    s_dev = [r for r in rows if FOLDS[r] != EF]
    q_dev = [r for r in rows if FOLDS[r] == EF]
    rep = evaluate_split(rows, E_clean, E_deg, s_dev, q_dev, f"{tag}/fold0_dev")
    s_in = [r for r in rows if FOLDS[r] not in (EF, IVF)]
    q_in = [r for r in rows if FOLDS[r] == IVF]
    if len(q_in) > 0 and len(s_in) > 0:
        rep["inner_val_fold4"] = evaluate_split(rows, E_clean, E_deg, s_in, q_in, f"{tag}/fold4_inner_val")
    return rep

# ----------------------------------------------------------------------------- train
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
            log(f"batch probe: bs={bs} grad_ckpt={ck} OOM")
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

    # Monitor on INNER VALIDATION (Folds 1..3 -> Fold 4) for checkpoint selection + Fold 0 for reference
    common_in, _ = label_sets(INNER_VA_ROWS, INNER_TR_ROWS)
    common_dev, _ = label_sets(Q_ROWS, SSL_ROWS)
    hist, step = [], 0
    best_inner_knn, best_ep = -1.0, 0; epoch_evals = {}

    for ep in range(tc["epochs"]):
        ds.epoch = ep; online.train(); target.train()
        t = time.time(); tl, nb, zs, hs, an, tstep = 0.0, 0, [], [], np.zeros(N_ANCH), []
        for G, L, a in dl:
            ts = time.time()
            for gp in opt.param_groups: gp["lr"] = gp["base_lr"] * lr_factor(step, total, warm)
            tau = 1 - (1 - tc["tau_base"]) * (math.cos(math.pi * step / max(1, total)) + 1) / 2
            loss, z, h, gn, _ = train_step(online, target, opt, scaler, G, L, tau, amp)
            tl += float(loss); nb += 1; step += 1; an += np.bincount(a.numpy().ravel(), minlength=N_ANCH)
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
        if row["std_z"] < CFG["monitor"]["collapse_warn"]: log(f"  WARNING possible collapse: std_z {row['std_z']:.3f}")
        if row["std_h"] < CFG["monitor"]["collapse_warn"]: log(f"  WARNING backbone std_h {row['std_h']:.3f} < {CFG['monitor']['collapse_warn']} (ep {ep+1})")

        if (ep + 1) % CFG["monitor"]["knn_every"] == 0 or ep + 1 == tc["epochs"]:
            tk = time.time()
            evr = np.sort(np.r_[SSL_ROWS, Q_ROWS])
            E_cl = extract(online.backbone, evr, degraded=False)
            E_dg = extract(online.backbone, evr, degraded=True)
            pos = {r: i for i, r in enumerate(evr)}
            # Inner validation kNN (average of clean and degraded query kNN mAUROC on Fold 4)
            in_s = [pos[r] for r in INNER_TR_ROWS]; in_q = [pos[r] for r in INNER_VA_ROWS]
            knn_in_c, _, _ = knn_monitor(E_cl[in_s], Y_ALL[INNER_TR_ROWS], E_cl[in_q], Y_ALL[INNER_VA_ROWS], common_in, CFG["monitor"]["knn_k"])
            knn_in_d, _, _ = knn_monitor(E_cl[in_s], Y_ALL[INNER_TR_ROWS], E_dg[in_q], Y_ALL[INNER_VA_ROWS], common_in, CFG["monitor"]["knn_k"])
            row["inner_val_knn_clean"] = knn_in_c
            row["inner_val_knn_deg"] = knn_in_d
            row["inner_val_knn_mean"] = 0.5 * (knn_in_c + knn_in_d)
            # Dis
