# =============================================================================
# team_offirst — Retina encoder BASELINE (supervised fine-tune + frozen few-shot)
# -----------------------------------------------------------------------------
# Data: RFMiD Training split ONLY, via Kaggle dataset awsaf49/rfmid-train-dataset
#   (1,920 images, 28-class "Challenge" label CSV: ID, Disease_Risk, 27 diseases, OTHER).
#   Citation: RFMiD (Pachade et al., Data 6(2):14, 2021; IEEE DataPort,
#   doi:10.21227/s3g7-st65), CC BY 4.0. Training split (1,920 images) via Kaggle
#   mirror, verified byte-identical to the official RIADD release.
#   NOTE: the Kaggle page says CC0 — that is wrong; the official licence is CC BY 4.0.
# DATA RULES: no RFMiD Evaluation/Test sets, no IDRiD grading set, no EyePACS, no
#   other retina data. Only ImageNet-pretrained weights (timm). A guard below refuses
#   to run if forbidden data is mounted.
# Pipeline: 5-fold multilabel-stratified split -> fundus-circle crop + resize ->
#   fine-tune timm backbone (convnext_tiny / resnet50) with ASL -> per-epoch AUROC/mAP
#   -> best checkpoint -> L2-normalised pooled embeddings for all images ->
#   few-shot eval (prototypes / logistic regression / kNN) on frozen embeddings,
#   support from TRAIN folds only, query = VAL fold.
# Switch RUN_MODE below: "smoke" (tiny subset, 1 epoch) or "full".
# =============================================================================

RUN_MODE = "full"   # "smoke" | "full"   <-- only change this to launch the full run

CONFIG_YAML = r"""
seed: 42
data:
  mode: full_image          # full_image | crops   (crops = later, cropper dataset)
  labels_csv: auto          # auto = find RFMiD_Training_Labels.csv under input root
  images_dir: auto          # auto = folder containing the training PNGs
  crops_csv: null           # crops mode: CSV with columns crop_id,image_id,path (labels joined from image_id)
  circle_crop: auto         # auto = true for full_image, false for crops
  img_size: 512
  n_folds: 5
  fold: 0                   # validation fold index
  cache: true               # pre-crop/resize once into a uint8 memmap (fast epochs)
  num_workers: 4
aug:
  hflip: 0.5
  vflip: 0.5
  rotate_deg: 20
  brightness: 0.2
  contrast: 0.2
  saturation: 0.1
  hue_deg: 4
model:
  backbones: [convnext_tiny, resnet50]   # each runs in sequence
  pretrained: true                       # ImageNet init (timm)
train:
  epochs: 30
  batch_size: 16
  lr_head: 2.0e-4
  lr_backbone: 1.0e-4
  weight_decay: 0.05
  warmup_epochs: 2
  amp: true
  ema: 0.999
  loss: asl                 # asl | bce
  asl: {gamma_neg: 4, gamma_pos: 0, clip: 0.05}
  grad_clip: 5.0
eval:
  min_val_pos: 10           # labels with >= this many val positives enter mean AUROC
  batch_size: 32
fewshot:
  shots: [1, 5, 10]         # positives per label in the support set
  neg_per_pos: 1            # negatives sampled per positive (k pos + k*neg_per_pos neg)
  draws: 10
  min_label_pos: 20         # label must have >= this many positives in the train folds
  min_query_pos: 5          # and >= this many in the val fold
  knn_k: 5
  logreg_C: 1.0
  include_imagenet: true    # also run few-shot on frozen ImageNet-init embeddings (no fine-tune)
smoke:
  epochs: 1
  n_train: 128
  n_val: 64
  min_val_pos: 3
  fewshot_min_label_pos: 5
  fewshot_min_query_pos: 2
  fewshot_shots: [1, 5]
  fewshot_draws: 3
"""

import os, sys, json, time, math, random, copy, warnings
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
T0 = time.time()

def log(*a):
    msg = f"[{time.time()-T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    if LOGF is not None:
        LOGF.write(msg + "\n"); LOGF.flush()

LOGF = None
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
    CFG["eval"]["min_val_pos"] = s["min_val_pos"]
    CFG["fewshot"].update(min_label_pos=s["fewshot_min_label_pos"], min_query_pos=s["fewshot_min_query_pos"],
                          shots=s["fewshot_shots"], draws=s["fewshot_draws"])
# local-test overrides (env), ignored on Kaggle
for k_env, path in [("TOB_IMG_SIZE", ("data", "img_size")), ("TOB_PRETRAINED", ("model", "pretrained")),
                    ("TOB_WORKERS", ("data", "num_workers"))]:
    if k_env in os.environ:
        v = os.environ[k_env]; v = int(v) if v.isdigit() else (v.lower() == "true")
        CFG[path[0]][path[1]] = v
if "TOB_BACKBONES" in os.environ:
    CFG["model"]["backbones"] = os.environ["TOB_BACKBONES"].split(",")

INPUT = Path(os.environ.get("TOB_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("TOB_OUT", "/kaggle/working/out"))
TMP = Path(os.environ.get("TOB_TMP", "/kaggle/temp" if Path("/kaggle").exists() else "/tmp/tob"))
OUT.mkdir(parents=True, exist_ok=True); TMP.mkdir(parents=True, exist_ok=True)
LOGF = open(OUT / "run.log", "a")
log(f"RUN_MODE={RUN_MODE}  input={INPUT}  out={OUT}")

import torch, torch.nn as nn, torch.nn.functional as F
import cv2
cv2.setNumThreads(0)
try:
    import timm
except ImportError:
    os.system("pip install -q timm"); import timm
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log("device", DEV, torch.cuda.get_device_name(0) if DEV.type == "cuda" else "", "torch", torch.__version__, "timm", timm.__version__)

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(CFG["seed"])

# ----------------------------------------------------------------------------- data guard
def data_guard():
    bad_names = ["RFMiD_Testing_Labels.csv", "RFMiD_Validation_Labels.csv", "RFMiD_Evaluation_Labels.csv",
                 "B. Disease Grading", "IDRiD_001test.jpg", "trainLabels.csv"]  # trainLabels.csv = EyePACS
    for n in bad_names:
        if any(INPUT.rglob(n)):
            raise SystemExit(f"DATA GUARD: forbidden data mounted ({n}). Only RFMiD-train is allowed.")
    for d in ["Evaluation_Set", "Test_Set", "Validation", "Test"]:
        for p in INPUT.rglob(d):
            if p.is_dir() and any(p.glob("*.png")):
                raise SystemExit(f"DATA GUARD: RFMiD eval/test images mounted at {p}")
data_guard()

# ----------------------------------------------------------------------------- load labels / sources
def find_labels():
    if CFG["data"]["labels_csv"] != "auto":
        return Path(CFG["data"]["labels_csv"])
    c = sorted(INPUT.rglob("RFMiD_Training_Labels.csv"))
    if not c: raise SystemExit("RFMiD_Training_Labels.csv not found under input")
    return c[0]

def find_images_dir(lab_csv):
    if CFG["data"]["images_dir"] != "auto":
        return Path(CFG["data"]["images_dir"])
    for p in sorted(lab_csv.parent.rglob("1.png")):
        return p.parent
    for p in sorted(INPUT.rglob("1.png")):
        return p.parent
    raise SystemExit("training PNGs not found")

lab_csv = find_labels()
df = pd.read_csv(lab_csv)
LABELS = [c for c in df.columns if c != "ID"]
df["ID"] = df["ID"].astype(int)
log(f"labels csv {lab_csv}  rows={len(df)}  n_labels={len(LABELS)}: {LABELS}")
IMG_DIR = find_images_dir(lab_csv)
df["path"] = [str(IMG_DIR / f"{i}.png") for i in df["ID"]]
missing = [p for p in df["path"] if not os.path.exists(p)]
if missing:
    log(f"WARNING {len(missing)} images missing (e.g. {missing[:3]}); dropping them")
    df = df[[os.path.exists(p) for p in df["path"]]].reset_index(drop=True)
Y_IMG = df[LABELS].values.astype(np.float32)
log(f"images dir {IMG_DIR}  usable images={len(df)}")

# ----------------------------------------------------------------------------- multilabel stratified folds
def iterative_stratification(Y, n_folds, seed):
    """Sechidis et al. 2011 iterative stratification (own implementation, deterministic)."""
    rng = np.random.RandomState(seed)
    n, L = Y.shape
    folds = -np.ones(n, dtype=int)
    desired = np.full(n_folds, n / n_folds)
    desired_lab = np.tile(Y.sum(0) / n_folds, (n_folds, 1))
    remaining = np.ones(n, bool)
    while remaining.any():
        Yr = Y[remaining]
        counts = Yr.sum(0)
        if counts.sum() == 0:
            idx = np.where(remaining)[0]
            for i in rng.permutation(idx):
                f = np.argmax(desired + rng.rand(n_folds) * 1e-6)
                folds[i] = f; desired[f] -= 1
            break
        nz = np.where(counts > 0)[0]
        l = nz[np.argmin(counts[nz])]
        idx = np.where(remaining & (Y[:, l] == 1))[0]
        for i in rng.permutation(idx):
            m = desired_lab[:, l]
            cand = np.where(m == m.max())[0]
            if len(cand) > 1:
                d = desired[cand]; cand = cand[d == d.max()]
            f = rng.choice(cand)
            folds[i] = f; remaining[i] = False
            desired_lab[f] -= Y[i]; desired[f] -= 1
    return folds

def make_folds(Y, n_folds, seed):
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        folds = -np.ones(len(Y), int)
        for k, (_, va) in enumerate(MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(np.zeros(len(Y)), Y)):
            folds[va] = k
        return folds, "iterstrat"
    except ImportError:
        return iterative_stratification(Y, n_folds, seed), "own_iterative_stratification"

folds, split_impl = make_folds(Y_IMG, CFG["data"]["n_folds"], CFG["seed"])
df["fold"] = folds
FOLD = CFG["data"]["fold"]
df[["ID", "fold"] + LABELS].to_csv(OUT / "split_5fold_seed42.csv", index=False)
log(f"split via {split_impl}; fold sizes {np.bincount(folds).tolist()}; val fold={FOLD}")

# ----------------------------------------------------------------------------- samples table (full_image | crops)
MODE = CFG["data"]["mode"]
CIRCLE = CFG["data"]["circle_crop"]
CIRCLE = (MODE == "full_image") if CIRCLE == "auto" else bool(CIRCLE)
if MODE == "full_image":
    samples = df[["ID", "path", "fold"]].rename(columns={"ID": "image_id"}).copy()
    samples["sample_id"] = samples["image_id"].astype(str)
elif MODE == "crops":
    # STUB for the cropper dataset: crops.csv with columns crop_id,image_id,path.
    # Labels are inherited from the parent image; folds are by parent image (no leakage).
    if not CFG["data"]["crops_csv"]:
        raise SystemExit("data.mode=crops needs data.crops_csv (columns: crop_id,image_id,path)")
    cr = pd.read_csv(CFG["data"]["crops_csv"])
    cr["image_id"] = cr["image_id"].astype(int)
    samples = cr.merge(df[["ID", "fold"]], left_on="image_id", right_on="ID", how="inner").drop(columns="ID")
    samples["sample_id"] = samples["crop_id"].astype(str)
    log(f"crops mode: {len(samples)} crops from {samples.image_id.nunique()} images")
else:
    raise SystemExit(f"unknown data.mode {MODE}")
lab_map = dict(zip(df["ID"], range(len(df))))
samples["yrow"] = samples["image_id"].map(lab_map)
samples = samples.reset_index(drop=True)

if SMOKE:
    rng = np.random.RandomState(CFG["seed"])
    tr_ids = df.loc[df.fold != FOLD, "ID"].values; va_ids = df.loc[df.fold == FOLD, "ID"].values
    keep = set(rng.choice(tr_ids, min(CFG["smoke"]["n_train"], len(tr_ids)), replace=False)) | set(rng.choice(va_ids, min(CFG["smoke"]["n_val"], len(va_ids)), replace=False))
    samples = samples[samples.image_id.isin(keep)].reset_index(drop=True)
    log(f"SMOKE subset: {len(samples)} samples ({CFG['smoke']['n_train']} train / {CFG['smoke']['n_val']} val images)")

Y = Y_IMG[samples["yrow"].values]
IS_VAL = (samples["fold"].values == FOLD)

# ----------------------------------------------------------------------------- preprocessing + cache
S = CFG["data"]["img_size"]

def crop_fundus(img, thr=15):
    """Crop to the fundus circle by thresholding the black border, pad to square."""
    h, w = img.shape[:2]
    sc = 512.0 / max(h, w)
    small = cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
    g = small.max(axis=2)
    mask = (g > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    ys, xs = np.where(mask > 0)
    if len(xs) < 100:
        return img
    x0, x1 = int(xs.min() / sc), int((xs.max() + 1) / sc)
    y0, y1 = int(ys.min() / sc), int((ys.max() + 1) / sc)
    img = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    h, w = img.shape[:2]; m = max(h, w)
    out = np.zeros((m, m, 3), img.dtype)
    out[(m - h) // 2:(m - h) // 2 + h, (m - w) // 2:(m - w) // 2 + w] = img
    return out

def load_prep(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError(f"cannot read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if CIRCLE: img = crop_fundus(img)
    return cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)

CACHE = None
TIMINGS = {}
if CFG["data"]["cache"]:
    cpath = TMP / f"cache_{MODE}_{S}_{len(samples)}.u8"
    t = time.time()
    CACHE = np.lib.format.open_memmap(str(cpath) + ".npy", mode="w+", dtype=np.uint8, shape=(len(samples), S, S, 3))
    nw = max(1, CFG["data"]["num_workers"])
    with Pool(nw) as pool:
        for i, im in enumerate(pool.imap(load_prep, samples["path"].tolist(), chunksize=4)):
            CACHE[i] = im
            if (i + 1) % 200 == 0: log(f"  cached {i+1}/{len(samples)}")
    CACHE.flush()
    TIMINGS["cache_s"] = time.time() - t; TIMINGS["cache_s_per_img"] = TIMINGS["cache_s"] / len(samples)
    log(f"cache built: {len(samples)} imgs in {TIMINGS['cache_s']:.1f}s ({TIMINGS['cache_s_per_img']*1000:.0f} ms/img, {nw} procs)")
    cv2.imwrite(str(OUT / "preview_preprocessed.jpg"), cv2.cvtColor(np.concatenate(list(CACHE[:4]), 1), cv2.COLOR_RGB2BGR))

MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255

def augment(img, rng):
    a = CFG["aug"]
    if rng.rand() < a["hflip"]: img = img[:, ::-1]
    if rng.rand() < a["vflip"]: img = img[::-1]
    ang = rng.uniform(-a["rotate_deg"], a["rotate_deg"])
    M = cv2.getRotationMatrix2D((S / 2, S / 2), ang, 1.0)
    img = cv2.warpAffine(np.ascontiguousarray(img), M, (S, S), flags=cv2.INTER_LINEAR, borderValue=0)
    if a["saturation"] > 0 or a["hue_deg"] > 0:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-a["hue_deg"], a["hue_deg"]) / 2) % 180  # OpenCV hue 0..180
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(1 - a["saturation"], 1 + a["saturation"]), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    img = img.astype(np.float32)
    fg = img.max(2, keepdims=True) > 5  # keep black background black
    b = rng.uniform(-a["brightness"], a["brightness"]) * 255
    c = rng.uniform(1 - a["contrast"], 1 + a["contrast"])
    mu = img[fg[..., 0]].mean() if fg.any() else 0
    img = np.where(fg, (img - mu) * c + mu + b, img)
    return np.clip(img, 0, 255)

class DS(torch.utils.data.Dataset):
    def __init__(self, idx, train):
        self.idx = np.asarray(idx); self.train = train
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        img = CACHE[j] if CACHE is not None else load_prep(samples.at[j, "path"])
        if self.train:
            rng = np.random.RandomState((torch.initial_seed() + i * 7919) % 2**32)
            img = augment(img, rng)
        img = (np.asarray(img, np.float32) - MEAN) / STD
        return torch.from_numpy(img.transpose(2, 0, 1).copy()), torch.from_numpy(Y[j]), j

def loader(idx, train, bs):
    return torch.utils.data.DataLoader(DS(idx, train), batch_size=bs, shuffle=train, drop_last=train and len(idx) > bs,
                                       num_workers=CFG["data"]["num_workers"], pin_memory=DEV.type == "cuda",
                                       persistent_workers=False)

# ----------------------------------------------------------------------------- model / loss / ema
class Net(nn.Module):
    def __init__(self, name, n_out, pretrained):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0)  # pooled features
        self.head = nn.Linear(self.backbone.num_features, n_out)
    def embed(self, x): return self.backbone(x)
    def forward(self, x): return self.head(self.backbone(x))

class ASL(nn.Module):
    """Asymmetric loss (Ridnik et al. 2021), multi-label, on logits."""
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8):
        super().__init__(); self.gn, self.gp, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps
    def forward(self, logits, y):
        logits = logits.float()
        p = torch.sigmoid(logits); pn = 1 - p
        if self.clip and self.clip > 0: pn = (pn + self.clip).clamp(max=1)
        lp = y * torch.log(p.clamp(min=self.eps)); ln = (1 - y) * torch.log(pn.clamp(min=self.eps))
        loss = lp + ln
        pt = p * y + pn * (1 - y)
        gamma = self.gp * y + self.gn * (1 - y)
        loss = loss * torch.pow(1 - pt, gamma)
        return -loss.sum() / logits.shape[0]

class EMA:
    def __init__(self, model, decay):
        self.m = copy.deepcopy(model).eval(); self.decay = decay; self.n = 0
        for p in self.m.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))  # standard EMA warm-up
        msd = model.state_dict()
        for k, v in self.m.state_dict().items():
            if v.dtype.is_floating_point: v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else: v.copy_(msd[k])

def param_groups(model, tc):
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        part = "head" if n.startswith("head.") else "backbone"
        nowd = p.ndim <= 1 or n.endswith(".bias")
        key = (part, nowd)
        groups.setdefault(key, []).append(p)
    out = []
    for (part, nowd), ps in groups.items():
        out.append({"params": ps, "lr": tc["lr_head"] if part == "head" else tc["lr_backbone"],
                    "weight_decay": 0.0 if nowd else tc["weight_decay"], "name": f"{part}{'_nowd' if nowd else ''}"})
    return out

# ----------------------------------------------------------------------------- eval
def metrics(yt, ys, min_pos):
    res = {"per_label_auroc": {}, "eligible_labels": [], "n_val": int(len(yt))}
    pos = yt.sum(0)
    for k, name in enumerate(LABELS):
        if pos[k] >= min_pos and pos[k] < len(yt):
            res["per_label_auroc"][name] = float(roc_auc_score(yt[:, k], ys[:, k]))
            res["eligible_labels"].append(name)
    el = [LABELS.index(n) for n in res["eligible_labels"]]
    res["mean_auroc"] = float(np.mean([res["per_label_auroc"][n] for n in res["eligible_labels"]])) if el else float("nan")
    res["mAP"] = float(np.mean([average_precision_score(yt[:, k], ys[:, k]) for k in el])) if el else float("nan")
    anyp = [k for k in range(len(LABELS)) if 0 < pos[k] < len(yt)]
    res["mAP_all_labels_with_pos"] = float(np.mean([average_precision_score(yt[:, k], ys[:, k]) for k in anyp])) if anyp else float("nan")
    dr = LABELS.index("Disease_Risk") if "Disease_Risk" in LABELS else None
    res["disease_risk_auroc"] = float(roc_auc_score(yt[:, dr], ys[:, dr])) if dr is not None and 0 < pos[dr] < len(yt) else float("nan")
    res["val_pos_counts"] = {n: int(pos[k]) for k, n in enumerate(LABELS)}
    return res

@torch.no_grad()
def predict(model, dl, amp):
    model.eval(); outs, ys, js = [], [], []
    for x, y, j in dl:
        x = x.to(DEV, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(DEV.type, enabled=amp and DEV.type == "cuda"):
            outs.append(torch.sigmoid(model(x).float()).cpu().numpy())
        ys.append(y.numpy()); js.append(j.numpy())
    return np.concatenate(ys), np.concatenate(outs), np.concatenate(js)

def image_level(yt, ys, js):
    """crops mode: average predictions per parent image (identity in full_image mode)."""
    if MODE == "full_image": return yt, ys
    g = pd.DataFrame({"img": samples["image_id"].values[js]})
    order = g.groupby("img").indices
    return np.stack([yt[v[0]] for v in order.values()]), np.stack([ys[v].mean(0) for v in order.values()])

# ----------------------------------------------------------------------------- few-shot on frozen embeddings
def l2n(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def fewshot_eval(emb, img_ids, tag):
    fc = CFG["fewshot"]
    # image-level labels/folds for the embedding rows
    yrow = np.array([lab_map[i] for i in img_ids]); Ye = Y_IMG[yrow]; fe = folds[yrow]
    tr = np.where(fe != FOLD)[0]; va = np.where(fe == FOLD)[0]
    labs = [k for k in range(len(LABELS)) if Ye[tr, k].sum() >= fc["min_label_pos"] and Ye[va, k].sum() >= fc["min_query_pos"]
            and Ye[va, k].sum() < len(va)]
    rep = {"tag": tag, "labels": [LABELS[k] for k in labs], "n_support_pool": int(len(tr)), "n_query": int(len(va)),
           "train_pos": {LABELS[k]: int(Ye[tr, k].sum()) for k in labs}, "query_pos": {LABELS[k]: int(Ye[va, k].sum()) for k in labs},
           "results": {}}
    Q = emb[va]
    for kshot in fc["shots"]:
        res = {h: [] for h in ["prototype", "logreg", "knn"]}
        per_label = {h: {LABELS[k]: [] for k in labs} for h in res}
        for d in range(fc["draws"]):
            rng = np.random.RandomState(CFG["seed"] * 1000 + kshot * 100 + d)
            draw = {h: [] for h in res}
            for k in labs:
                pos = tr[Ye[tr, k] == 1]; neg = tr[Ye[tr, k] == 0]
                kp = min(kshot, len(pos)); kn = min(kshot * fc["neg_per_pos"], len(neg))
                sp = rng.choice(pos, kp, replace=False); sn = rng.choice(neg, kn, replace=False)
                yq = Ye[va, k]
                # prototypes: cosine to positive mean minus cosine to negative mean
                mp = l2n(emb[sp].mean(0, keepdims=True)); mn = l2n(emb[sn].mean(0, keepdims=True))
                s_proto = (Q @ mp.T - Q @ mn.T).ravel()
                # logistic regression (L2, balanced)
                Xs = np.concatenate([emb[sp], emb[sn]]); ys_ = np.r_[np.ones(kp), np.zeros(kn)]
                lr = LogisticRegression(C=fc["logreg_C"], class_weight="balanced", max_iter=2000)
                lr.fit(Xs, ys_); s_lr = lr.decision_function(Q)
                # kNN (cosine), score = similarity-weighted positive fraction of the k nearest
                kk = min(fc["knn_k"], len(Xs)); sims = Q @ Xs.T
                nn_idx = np.argsort(-sims, 1)[:, :kk]; w = np.take_along_axis(sims, nn_idx, 1) + 1.0
                s_knn = (w * ys_[nn_idx]).sum(1) / (w.sum(1) + 1e-12)
                for h, sc in [("prototype", s_proto), ("logreg", s_lr), ("knn", s_knn)]:
                    a = float(roc_auc_score(yq, sc)); draw[h].append(a); per_label[h][LABELS[k]].append(a)
            for h in res: res[h].append(float(np.mean(draw[h])))
        rep["results"][f"k={kshot}"] = {h: {"mean_auroc": float(np.mean(v)), "std_over_draws": float(np.std(v)),
                                            "per_label_mean": {n: float(np.mean(x)) for n, x in per_label[h].items()}}
                                        for h, v in res.items()}
        log(f"  fewshot[{tag}] k={kshot}: " + "  ".join(f"{h} {np.mean(v):.3f}±{np.std(v):.3f}" for h, v in res.items()))
    return rep

@torch.no_grad()
def extract_embeddings(model, amp):
    dl = loader(np.arange(len(samples)), False, CFG["eval"]["batch_size"])
    model.eval(); E, J = [], []
    for x, _, j in dl:
        x = x.to(DEV, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(DEV.type, enabled=amp and DEV.type == "cuda"):
            E.append(model.embed(x).float().cpu().numpy())
        J.append(j.numpy())
    E = np.concatenate(E); J = np.concatenate(J)
    E = E[np.argsort(J)]
    img_ids = samples["image_id"].values
    if MODE == "crops":  # mean-pool crop embeddings per parent image
        g = pd.DataFrame({"img": img_ids}).groupby("img").indices
        img_ids = np.array(list(g.keys())); E = np.stack([E[v].mean(0) for v in g.values()])
    return l2n(E).astype(np.float32), img_ids

# ----------------------------------------------------------------------------- train one backbone
def run_backbone(name):
    tc = CFG["train"]; amp = tc["amp"] and DEV.type == "cuda"
    od = OUT / name; od.mkdir(parents=True, exist_ok=True)
    seed_all(CFG["seed"])
    tr_idx = np.where(~IS_VAL)[0]; va_idx = np.where(IS_VAL)[0]
    log(f"=== {name}: train {len(tr_idx)} / val {len(va_idx)} samples, {len(LABELS)} outputs, loss={tc['loss']}")
    model = Net(name, len(LABELS), CFG["model"]["pretrained"]).to(DEV).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    rep = {"backbone": name, "n_params": n_params, "feat_dim": model.backbone.num_features}

    if CFG["fewshot"]["include_imagenet"]:
        t = time.time(); E0, ids0 = extract_embeddings(model, amp)
        np.save(od / "embeddings_imagenet_init.npy", E0)
        rep["fewshot_imagenet_init"] = fewshot_eval(E0, ids0, f"{name}/imagenet_init")
        TIMINGS[f"{name}_imagenet_embed_fewshot_s"] = time.time() - t

    crit = ASL(**tc["asl"]) if tc["loss"] == "asl" else nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(param_groups(model, tc), betas=(0.9, 0.999))
    dl_tr = loader(tr_idx, True, tc["batch_size"]); dl_va = loader(va_idx, False, CFG["eval"]["batch_size"])
    steps_ep = len(dl_tr); total = steps_ep * tc["epochs"]; warm = steps_ep * tc["warmup_epochs"]
    def lr_lambda(s):
        if s < warm: return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = EMA(model, tc["ema"]) if tc["ema"] else None
    hist, best, best_ep = [], -1.0, -1
    for ep in range(tc["epochs"]):
        model.train(); t = time.time(); tl, nb = 0.0, 0
        for x, y, _ in dl_tr:
            x = x.to(DEV, non_blocking=True).to(memory_format=torch.channels_last); y = y.to(DEV, non_blocking=True)
            with torch.autocast(DEV.type, enabled=amp):
                logits = model(x)
            loss = crit(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if tc["grad_clip"]:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt); scaler.update(); sched.step()
            if ema: ema.update(model)
            tl += loss.item(); nb += 1
        if DEV.type == "cuda": torch.cuda.synchronize()
        t_train = time.time() - t; t = time.time()
        row = {"epoch": ep + 1, "train_loss": tl / max(1, nb), "lr_backbone": next(g["lr"] for g in opt.param_groups if g["name"].startswith("backbone")), "train_s": t_train,
               "train_img_per_s": nb * tc["batch_size"] / max(t_train, 1e-9)}
        cands = [("raw", model)] + ([("ema", ema.m)] if ema else [])
        for tag, m in cands:
            yt, ys, js = predict(m, dl_va, amp); yt, ys = image_level(yt, ys, js)
            row[tag] = metrics(yt, ys, CFG["eval"]["min_val_pos"])
        row["eval_s"] = time.time() - t
        sel_tag = max(cands, key=lambda c: np.nan_to_num(row[c[0]]["mean_auroc"], nan=-1))[0]
        score = np.nan_to_num(row[sel_tag]["mean_auroc"], nan=-1)
        hist.append(row)
        log(f"  ep {ep+1}/{tc['epochs']} loss {row['train_loss']:.4f}  raw mAUROC {row['raw']['mean_auroc']:.4f} "
            + (f"ema mAUROC {row['ema']['mean_auroc']:.4f} " if ema else "")
            + f"DR-risk {row[sel_tag]['disease_risk_auroc']:.4f} mAP {row[sel_tag]['mAP']:.4f} "
            + f"[{len(row[sel_tag]['eligible_labels'])} labels]  train {t_train:.1f}s ({row['train_img_per_s']:.1f} img/s) eval {row['eval_s']:.1f}s")
        if score > best:
            best, best_ep = score, ep + 1
            torch.save({"model": (dict(cands)[sel_tag]).state_dict(), "weights": sel_tag, "epoch": ep + 1, "labels": LABELS,
                        "backbone": name, "config": CFG, "metrics": row[sel_tag]}, od / "best.pth")
        json.dump(hist, open(od / "history.json", "w"), indent=1)
    torch.save({"model": model.state_dict(), "ema": ema.m.state_dict() if ema else None, "epoch": tc["epochs"],
                "labels": LABELS, "backbone": name, "config": CFG}, od / "last.pth")
    rep.update(best_epoch=best_ep, best_mean_auroc=best, epochs=tc["epochs"], steps_per_epoch=steps_ep,
               eligible_labels=hist[-1]["raw"]["eligible_labels"])

    # embeddings + few-shot with best checkpoint
    t = time.time()
    ck = torch.load(od / "best.pth", map_location=DEV, weights_only=False); model.load_state_dict(ck["model"])
    E, ids = extract_embeddings(model, amp)
    np.save(od / "embeddings.npy", E)
    yrow = np.array([lab_map[i] for i in ids])
    pd.DataFrame({"ID": ids, "fold": folds[yrow], "is_val": folds[yrow] == FOLD}).to_csv(od / "embedding_ids.csv", index=False)
    rep["embedding_shape"] = list(E.shape); rep["best_weights"] = ck["weights"]
    rep["fewshot_finetuned"] = fewshot_eval(E, ids, f"{name}/finetuned_best")
    TIMINGS[f"{name}_embed_fewshot_s"] = time.time() - t
    TIMINGS[f"{name}_train_epoch_s"] = float(np.mean([h["train_s"] for h in hist]))
    TIMINGS[f"{name}_eval_epoch_s"] = float(np.mean([h["eval_s"] for h in hist]))
    TIMINGS[f"{name}_train_img_per_s"] = float(np.mean([h["train_img_per_s"] for h in hist]))
    json.dump({k: rep[k] for k in ["fewshot_finetuned"] + (["fewshot_imagenet_init"] if "fewshot_imagenet_init" in rep else [])},
              open(od / "fewshot_report.json", "w"), indent=1)
    del model, ema, opt; torch.cuda.empty_cache() if DEV.type == "cuda" else None
    return rep

summary = {"run_mode": RUN_MODE, "data_mode": MODE, "labels": LABELS, "n_samples": int(len(samples)),
           "n_train": int((~IS_VAL).sum()), "n_val": int(IS_VAL.sum()), "split_impl": split_impl, "fold": FOLD,
           "citation": "RFMiD (Pachade et al., Data 6(2):14, 2021; IEEE DataPort, doi:10.21227/s3g7-st65), CC BY 4.0",
           "config": CFG, "backbones": {}}
for bb in CFG["model"]["backbones"]:
    tb = time.time()
    summary["backbones"][bb] = run_backbone(bb)
    TIMINGS[f"{bb}_total_s"] = time.time() - tb
    summary["timings"] = TIMINGS
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=float)
TIMINGS["wall_s"] = time.time() - T0
summary["timings"] = TIMINGS
json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=float)
log("DONE", json.dumps(TIMINGS, default=float))
