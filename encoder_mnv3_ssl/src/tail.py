
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

# ----------------------------------------------------------------------------- rows: SSL / inner val / dev
IVF = CFG["data"]["inner_val_fold"]
assert IVF != EF, "inner_val_fold must differ from eval_fold"
SSL_EXCL = [EF] if CFG["data"]["ssl_include_inner_val"] else [EF, IVF]
SSL_ROWS = np.where(~np.isin(FOLDS, SSL_EXCL))[0]
N_FULL_SSL = int(len(SSL_ROWS))
EVAL_ROWS = np.arange(len(df))                       # every image gets embedded for eval (labels only used in probes)
if SMOKE:
    rng = np.random.RandomState(CFG["seed"])
    if CFG["smoke"]["n_train"] > 0: SSL_ROWS = np.sort(rng.choice(SSL_ROWS, CFG["smoke"]["n_train"], replace=False))
    if EVAL_SUB > 0:   # local CPU test only: subsample eval rows per fold
        EVAL_ROWS = np.sort(np.concatenate([rng.choice(np.where(FOLDS == f)[0], min(EVAL_SUB, int((FOLDS == f).sum())), replace=False)
                                            for f in range(CFG["data"]["n_folds"])]))
Q_ROWS = EVAL_ROWS[FOLDS[EVAL_ROWS] == EF]                              # dev query (fold 0)
DEV_S_ROWS = EVAL_ROWS[FOLDS[EVAL_ROWS] != EF]                          # dev support (folds 1-4), as parent
IV_ROWS = EVAL_ROWS[FOLDS[EVAL_ROWS] == IVF]                            # inner query (fold 4)
IN_S_ROWS = EVAL_ROWS[~np.isin(FOLDS[EVAL_ROWS], [EF, IVF])]            # inner support (folds 1-3)
INNER_ROWS = np.sort(np.r_[IN_S_ROWS, IV_ROWS])
assert not set(SSL_ROWS) & set(np.where(FOLDS == EF)[0]), "fold-0 leak into SSL"
if not CFG["data"]["ssl_include_inner_val"]: assert not set(SSL_ROWS) & set(np.where(FOLDS == IVF)[0]), "inner-val leak into SSL"
log(f"SSL images {len(SSL_ROWS)} from folds {sorted(set(FOLDS[SSL_ROWS].tolist()))} (full run would use {N_FULL_SSL}); "
    f"inner support {len(IN_S_ROWS)} / inner query fold {IVF} {len(IV_ROWS)}; dev support {len(DEV_S_ROWS)} / dev query fold {EF} {len(Q_ROWS)}")

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
    """camera-robust augs (all views): brightness gain, gaussian blur, JPEG re-encode. (v4 code unchanged, incl. its
    RGB array passed to cv2.imencode as if BGR; see README.)"""
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

def _colour(x, rng):
    """v4 weak colour jitter (first half of v4 _photometric, unchanged)."""
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
    return x

def _photometric(x, rng):
    return _camera(_colour(x, rng), rng, V["cam_bright"])

# >>> DEGRADE BLOCK (self-contained; test_degrade_equivalence.py execs this block)
# Visuscout degrader = degrade.py chain (CRF -> radial vignette -> defocus blur -> colour gains -> [leak] -> noise -> FOV mask
# -> JPEG), applied per VIEW after the geometric crop, RGB channel order. Fitted values copied verbatim from degrade.py
# DegradeParams defaults (fit 2026-11-06, 24 Visuscout frames). Do not edit FIT here; edit degrade.py + re-fit instead.
FIT = dict(crf_gamma=1.05, crf_toe=0.02, crf_shoulder=0.92, exposure=1.12, vignette_strength=0.16, vignette_power=2.0,
           blur_sigma=1.15, gain_r=0.85, gain_g=1.05, gain_b=1.15, leak_strength=0.0, noise_sigma=1.4, jpeg_quality=70)
assert FIT["leak_strength"] == 0.0, "light leak not implemented in the per-view degrader (fitted value is 0 = off)"
DG = CFG["degrade"]

def deg_params(a):
    """severity a in [0,1]: every fitted parameter linearly interpolated from identity (a=0) to the fit (a=1)."""
    f = FIT
    return {"gamma": 1 + a * (f["crf_gamma"] - 1), "toe": a * f["crf_toe"], "shoulder": 1 - a * (1 - f["crf_shoulder"]),
            "exposure": 1 + a * (f["exposure"] - 1), "vig": a * f["vignette_strength"], "vpow": f["vignette_power"],
            "blur": a * f["blur_sigma"], "noise": a * f["noise_sigma"], "q": int(round(100 - a * (100 - f["jpeg_quality"]))),
            "gains": np.array([1 + a * (f["gain_r"] - 1), 1 + a * (f["gain_g"] - 1), 1 + a * (f["gain_b"] - 1)], np.float32)}  # RGB

def _crf(x, p):   # degrade.crf with interpolated params (shoulder=1 -> no roll-off)
    x = np.clip(x * p["exposure"], 0, None)
    y = np.power(np.clip(x, 0, 1), p["gamma"]); s = p["shoulder"]
    if s < 1.0:
        hi = y > s; y[hi] = s + (1 - s) * (1 - np.exp(-(y[hi] - s) / (1 - s)))
    return p["toe"] + (1 - p["toe"]) * y

_GRID = {}
def _grid(n):
    if n not in _GRID:
        v, u = np.indices((n, n), dtype=np.float32); _GRID[n] = (u, v)
    return _GRID[n]

def visuscout(x, a, M, row, rng):
    """x: uint8 RGB view (n x n) obtained from the cache square (side CS) by affine M (cache -> view).
    Vignette uses the image's FOV centre/radius from GEOM mapped through M (so a local crop gets the rim falloff of
    its true position, not a ring around the crop). emulate_native: blur sigma and noise are converted from
    degrade.py's raw-image units (sigma = blur * raw_width/1024 raw px; noise per raw px) to view px."""
    p = deg_params(a); n = x.shape[0]
    w_raw, side_raw, fcx, fcy, fR = GEOM[row]
    m = cv2.morphologyEx((cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) > 15).astype(np.uint8), cv2.MORPH_OPEN,
                         np.ones((5, 5), np.uint8)).astype(np.float32)
    f = _crf(x.astype(np.float32) / 255, p)
    if p["vig"] > 0:
        Mi = cv2.invertAffineTransform(np.asarray(M, np.float64)); u, v = _grid(n)
        X = Mi[0, 0] * u + Mi[0, 1] * v + Mi[0, 2]; Y = Mi[1, 0] * u + Mi[1, 1] * v + Mi[1, 2]
        rad = np.sqrt((X - fcx * CS) ** 2 + (Y - fcy * CS) ** 2) / (fR * CS)
        f *= (1 - p["vig"] * np.power(np.clip(rad, 0, 1), p["vpow"]))[..., None]
    s_m = math.sqrt(abs(float(np.linalg.det(np.asarray(M, np.float64)[:, :2]))))   # view px per cache px
    if DG["emulate_native"]:
        raw_per_view = side_raw / CS / s_m                                             # raw px per view px
        sig = p["blur"] * (w_raw / 1024.0) / raw_per_view
        nz = p["noise"] / max(1.0, raw_per_view)                                       # iid noise averaged by downsampling
    else:
        sig = p["blur"] * n / 1024.0; nz = p["noise"]                                  # degrade.py applied as if the view were the image
    if sig > 0.05: f = cv2.GaussianBlur(f, (0, 0), sig)
    f *= p["gains"]
    if nz > 0: f += rng.normal(0, nz / 255, f.shape).astype(np.float32)
    f *= m[..., None]
    out = (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)
    if DG["jpeg"] and p["q"] < 100:
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, p["q"]])
        if ok: out = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return out

def resize_affine(S, n):
    """affine of cv2.resize(S x S -> n x n) (pixel-centre convention)."""
    k = n / S; return np.array([[k, 0, 0.5 * k - 0.5], [0, k, 0.5 * k - 0.5]], np.float64)

def degrade_eval(img, row, size):
    """deterministic degraded eval copy: full-image resize to size, then visuscout(eval_alpha) with a fixed per-image seed.
    No colour jitter / camera augs."""
    x = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    rng = np.random.RandomState((DG["eval_seed"] * 1000003 + int(row)) % 2**32)
    return visuscout(x, DG["eval_alpha"], resize_affine(img.shape[0], size), row, rng)
# <<< DEGRADE BLOCK

def _view(img, rng, row, out, center, wh, p_deg):
    """one view: geometry -> v4 colour jitter -> [Visuscout degrade, prob p_deg, alpha ~ U] -> v4 camera augs.
    Returns (uint8 view, alpha or -1 if clean)."""
    M = _affine(center, wh, out, rng.uniform(-V["rotate_deg"], V["rotate_deg"]), rng.rand() < V["hflip"], rng.rand() < V["vflip"])
    x = cv2.warpAffine(img, M, (out, out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    x = _colour(x, rng); a = -1.0
    if DG["enabled"] and p_deg > 0 and rng.rand() < p_deg:
        a = float(rng.uniform(*DG["alpha"])); x = visuscout(x, a, M, row, rng)
        if DG["camera_after"]: x = _camera(x, rng, V["cam_bright"])
    else:
        x = _camera(x, rng, V["cam_bright"])
    return x, a

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
    """-> globals (2,H,W,3), locals (nl,h,w,3), anchor types (nl,), alphas (2+nl,) (-1 = clean)."""
    S = img.shape[0]; gl, al = [], []
    for k, pg in enumerate([DG["p_global0"], DG["p_global1"]]):
        x, a = _view(img, rng, row, V["global_size"], *_rrc_box(rng, S, V["global_scale"]), pg); gl.append(x); al.append(a)
    if mode == "anatomy_v4":
        slots = ["disc"] * A2["n_disc"] + ["macula"] * A2["n_macula"] + ["random"] * A2["n_random"]
        assert len(slots) == V["n_local"], "anatomy_v4 slots must equal n_local"
        rng.shuffle(slots)
    else: slots = ["random"] * V["n_local"]
    lo, types = [], []
    for s in slots:
        if s == "random": c, wh, ty = _rrc_box(rng, S, V["local_scale"]) + (0,)
        else: c, wh, ty = anatomy_box(rng, row, S, s)
        x, a = _view(img, rng, row, V["local_size"], c, wh, DG["p_local"]); lo.append(x); al.append(a); types.append(ty)
    return np.stack(gl), np.stack(lo), np.array(types, np.int64), np.array(al, np.float32)

def _tile(v, T, text):
    v = cv2.resize(cv2.cvtColor(v, cv2.COLOR_RGB2BGR), (T, T)).copy()
    cv2.rectangle(v, (0, 0), (T, 18), (0, 0, 0), -1); cv2.putText(v, text, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return v

def crop_grid(path, n_img=12, seed=123):
    """PNG grid: per row one SSL image (disc/macula marks) + its 2 globals + 4 locals, labelled with anchor + severity."""
    rng0 = np.random.RandomState(seed)
    rows = rng0.choice(SSL_ROWS, min(n_img, len(SSL_ROWS)), replace=False); T = 200; out = []
    for r in rows:
        img = np.asarray(CACHE[r]); S = img.shape[0]
        ov = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR).copy(); dcx, dcy, dd, mcx, mcy = BOX_ARR[r] * S
        cv2.circle(ov, (int(dcx), int(dcy)), int(dd / 2), (0, 255, 0) if BOX_OK[r] else (0, 0, 255), 2)
        cv2.drawMarker(ov, (int(mcx), int(mcy)), (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
        ov = cv2.resize(ov, (T, T)); cv2.putText(ov, f"ID{df.ID.values[r]} c{BOX.confidence.values[r]:.2f}{'' if BOX_OK[r] else ' LOW'}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        gl, lo, ty, al = make_views(img, np.random.RandomState(seed + int(r)), r, "anatomy_v4")
        lab = lambda a: "clean" if a < 0 else f"deg a={a:.2f}"
        tiles = [ov] + [_tile(g, T, f"G{k} {lab(al[k])}") for k, g in enumerate(gl)]
        tiles += [_tile(v, T, f"{ANCHOR_NAMES[t][:6]} {lab(al[2 + k])}") for k, (v, t) in enumerate(zip(lo, ty))]
        out.append(np.concatenate(tiles, 1))
    cv2.imwrite(str(path), np.concatenate(out, 0)); log("crop grid ->", path)

def degrade_examples(path, n_img=4, seed=7):
    """PNG: per row clean eval image, degraded eval copy (eval_alpha, fixed seed), and a severity strip a=0.25/0.5/0.75."""
    rows = np.random.RandomState(seed).choice(EVAL_ROWS, n_img, replace=False); T = 256; out = []; n = CFG["eval"]["img_size"]
    for r in rows:
        img = np.asarray(CACHE[r]); cl = cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA); M = resize_affine(img.shape[0], n)
        tiles = [_tile(cl, T, f"ID{df.ID.values[r]} clean eval"), _tile(degrade_eval(img, r, n), T, f"eval deg a={DG['eval_alpha']}")]
        tiles += [_tile(visuscout(cl, a, M, r, np.random.RandomState(0)), T, f"a={a}") for a in (0.25, 0.5, 0.75)]
        out.append(np.concatenate(tiles, 1))
    cv2.imwrite(str(path), np.concatenate(out, 0)); log("degrade examples ->", path)

class SSLDS(torch.utils.data.Dataset):
    def __init__(self, rows, mode): self.rows = np.asarray(rows); self.mode = mode; self.epoch = 0
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        rng = np.random.RandomState((CFG["seed"] * 1000003 + self.epoch * 7919 + i * 104729 + torch.initial_seed()) % 2**32)
        g, l, a, al = make_views(np.asarray(CACHE[r]), rng, r, self.mode)
        return torch.from_numpy(g).permute(0, 3, 1, 2), torch.from_numpy(l).permute(0, 3, 1, 2), torch.from_numpy(a), torch.from_numpy(al)

class EvalDS(torch.utils.data.Dataset):
    def __init__(self, rows, size, degraded=False): self.rows = np.asarray(rows); self.size = size; self.degraded = degraded
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = int(self.rows[i]); img = np.asarray(CACHE[r])
        im = degrade_eval(img, r, self.size) if self.degraded else cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1), i

MR = CFG["eval"]["multi_region_probe"]
class RegionDS(torch.utils.data.Dataset):
    """eval-only: deterministic disc / macula square crop (MR[kind_dd] x DD, no jitter/rotation) resized to region_size."""
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

class GAPEncoder(nn.Module):
    """timm MobileNetV3: feature='gap' -> forward_features + global average pool (960-d; timm's conv_head 960->1280,
    head norm/act and classifier are removed so they hold no unused params). feature='timm_head' -> timm pre-logits."""
    def __init__(self, pretrained):
        super().__init__()
        mc = CFG["model"]; self.mode = mc["feature"]
        self.body = timm.create_model(mc["backbone"], pretrained=pretrained, num_classes=0)
        if self.mode == "gap":
            for n in ["conv_head", "norm_head", "act2", "flatten", "classifier", "global_pool"]:
                if hasattr(self.body, n): setattr(self.body, n, nn.Identity())
            self.num_features = int(self.body.feature_info[-1]["num_chs"])
        else:
            self.num_features = int(self.body.num_features if not hasattr(self.body, "head_hidden_size") else self.body.head_hidden_size)
    def set_grad_checkpointing(self, enable=True): self.body.set_grad_checkpointing(enable)
    def forward(self, x):
        if self.mode == "gap": return self.body.forward_features(x).mean((2, 3))
        return self.body(x)

class BYOLNet(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        mc = CFG["model"]
        self.backbone = GAPEncoder(pretrained)
        d = self.backbone.num_features
        self.neck = nn.Linear(d, mc["neck_dim"]) if mc["neck_dim"] else nn.Identity()
        d = mc["neck_dim"] or d; self.emb_dim = d
        self.projector = mlp(d, mc["proj_hidden"], mc["proj_out"])
        self.predictor = mlp(mc["proj_out"], mc["pred_hidden"], mc["proj_out"])
    def project(self, x):
        h = self.neck(self.backbone(x)); return h, self.projector(h)

class Target(nn.Module):
    def __init__(self, online):
        super().__init__()
        self.backbone = copy.deepcopy(online.backbone); self.neck = copy.deepcopy(online.neck); self.projector = copy.deepcopy(online.projector)
        for p in self.parameters(): p.requires_grad_(False)
    def forward(self, x): return self.projector(self.neck(self.backbone(x)))

def encoder(net):
    """the frozen-eval encoder = backbone (+ neck); shares modules with net."""
    return nn.Sequential(net.backbone, net.neck)

@torch.no_grad()
def ema_update(target, online, tau):
    for tm, om in [(target.backbone, online.backbone), (target.neck, online.neck), (target.projector, online.projector)]:
        for tp, op in zip(tm.parameters(), om.parameters()): tp.mul_(tau).add_(op.detach(), alpha=1 - tau)
        for tb, ob in zip(tm.buffers(), om.buffers()): tb.copy_(ob)

VC = CFG["train"]["vicreg"]
def vicreg_var_cov(z, gamma=1.0, eps=1e-4):
    """VICReg variance + covariance terms (kept from v4; disabled by config)."""
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
        part = "backbone" if n.startswith("backbone.") else "heads"   # neck (new layer) -> heads LR
        nowd = p.ndim <= 1 or n.endswith(".bias")
        groups.setdefault((part, nowd), []).append(p)
    return [{"params": ps, "lr": tc["lr_backbone"] if part == "backbone" else tc["lr_heads"], "base_lr": tc["lr_backbone"] if part == "backbone" else tc["lr_heads"],
             "weight_decay": 0.0 if nowd else tc["weight_decay"], "name": f"{part}{'_nowd' if nowd else ''}"} for (part, nowd), ps in groups.items()]

def train_step(online, target, opt, scaler, G, L, tau, amp):
    """G: (B,2,3,H,W) uint8, L: (B,nl,3,h,w) uint8. (v4 code; VICReg branch inactive.)"""
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
    if VC["enabled"]:
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
def _loader(ds):
    return torch.utils.data.DataLoader(ds, batch_size=CFG["eval"]["batch_size"], shuffle=False,
                                       num_workers=CFG["data"]["num_workers"], pin_memory=DEV.type == "cuda")

@torch.no_grad()
def extract(enc, rows, size=None, degraded=False):
    size = size or CFG["eval"]["img_size"]
    was = enc.training; enc.eval(); E = []
    for x, _ in _loader(EvalDS(rows, size, degraded)):
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
            E.append(enc(norm_gpu(x)).float().cpu().numpy())
    enc.train(was)
    E = np.concatenate(E); return (E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)).astype(np.float32)

def extract_pair(enc, rows):
    """clean + fixed-seed degraded embeddings of the same rows."""
    return extract(enc, rows), extract(enc, rows, degraded=True)

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

def _lp_summary(out):
    dz = [l for l in out if l != "Disease_Risk"]
    return {"per_label": out, "mean_auroc_common": float(np.mean([out[l]["auroc"] for l in dz])) if dz else float("nan"),
            "disease_risk_auroc": out.get("Disease_Risk", {}).get("auroc", float("nan"))}

def linear_probe(Es, ys, Eq, yq, labs, extra=None):
    """v4 probe (C by 3-fold CV on support). extra = {name: (Eq2, yq2)} scored with the SAME fitted probes."""
    sc = StandardScaler().fit(Es); Xs = sc.transform(Es); Xq = sc.transform(Eq)
    Xe = {k: (sc.transform(e), y) for k, (e, y) in (extra or {}).items()}
    out, oute = {}, {k: {} for k in Xe}
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
        for k, (X2, y2) in Xe.items(): oute[k][LABELS[j]] = {"auroc": float(roc_auc_score(y2[:, j], m.decision_function(X2))), "C": bestC}
    res = _lp_summary(out)
    if Xe: res["query_variants"] = {k: _lp_summary(v) for k, v in oute.items()}
    return res

def fewshot(Es, ys, Eq, yq, labs, headline, extra=None):
    """v4 few-shot (same draws for every encoder). extra = {name: (Eq2, yq2)} scored with the same support/fits.
    Returns (report on Eq, {name: report on Eq2})."""
    ec = CFG["eval"]; Q = [("clean", Eq, yq)] + [(k, e, y) for k, (e, y) in (extra or {}).items()]
    reps = {n: {} for n, _, _ in Q}
    for kshot in ec["shots"]:
        per = {n: {h: {LABELS[j]: [] for j in labs} for h in ["prototype", "logreg", "knn"]} for n, _, _ in Q}
        for d in range(ec["draws"]):
            rng = np.random.RandomState(CFG["seed"] * 1000 + kshot * 100 + d)   # same draws for every encoder
            for j in labs:
                pos = np.where(ys[:, j] == 1)[0]; neg = np.where(ys[:, j] == 0)[0]
                sp = rng.choice(pos, min(kshot, len(pos)), replace=False); sn = rng.choice(neg, min(kshot * ec["neg_per_pos"], len(neg)), replace=False)
                Xs = np.concatenate([Es[sp], Es[sn]]); y_ = np.r_[np.ones(len(sp)), np.zeros(len(sn))]
                mp = Es[sp].mean(0); mn = Es[sn].mean(0)
                mp /= np.linalg.norm(mp) + 1e-12; mn /= np.linalg.norm(mn) + 1e-12
                lr = LogisticRegression(C=ec["logreg_C"], class_weight="balanced", max_iter=2000).fit(Xs, y_)
                for n, E_, yq_ in Q:
                    s_p = E_ @ mp - E_ @ mn
                    s_l = lr.decision_function(E_)
                    kk = min(ec["knn_k"], len(Xs)); sims = E_ @ Xs.T; ix = np.argsort(-sims, 1)[:, :kk]
                    w = np.take_along_axis(sims, ix, 1) + 1.0; s_k = (w * y_[ix]).sum(1) / (w.sum(1) + 1e-12)
                    for h, s in [("prototype", s_p), ("logreg", s_l), ("knn", s_k)]:
                        per[n][h][LABELS[j]].append(float(roc_auc_score(yq_[:, j], s)))
        for n, _, _ in Q:
            res = {}
            for h, pl in per[n].items():
                if not pl: res[h] = {"headline_labels": [], "headline_mean": float("nan"), "headline_std_over_draws": float("nan"),
                                     "all_rare_mean": float("nan"), "all_rare_std_over_draws": float("nan"), "per_label": {}}; continue
                hl = [l for l in headline if l in pl] or list(pl)
                draws_mean_all = np.mean([pl[l] for l in pl], 0); draws_mean_hl = np.mean([pl[l] for l in hl], 0)
                res[h] = {"headline_labels": hl, "headline_mean": float(draws_mean_hl.mean()), "headline_std_over_draws": float(draws_mean_hl.std()),
                          "all_rare_mean": float(draws_mean_all.mean()), "all_rare_std_over_draws": float(draws_mean_all.std()),
                          "per_label": {l: [float(np.mean(v)), float(np.std(v))] for l, v in pl.items()}}
            reps[n][f"k={kshot}"] = res
    return reps["clean"], {n: reps[n] for n, _, _ in Q[1:]}

def erank(Z):
    """effective rank (Roy & Vetterli 2007) of centred Z: exp(entropy of normalised singular values)."""
    s = np.linalg.svd((Z - Z.mean(0, keepdims=True)).astype(np.float64), compute_uv=False)
    p = s / max(s.sum(), 1e-12); p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))

def _pair_stats(A, B):
    S = A @ B.T; n = len(A); paired = float(np.diag(S).mean()); rand = float((S.sum() - np.trace(S)) / max(1, n * (n - 1)))
    return paired, rand, float((S.argmax(1) == np.arange(n)).mean())

def diagnostics(Ec, Ed=None):
    """collapse / invariance on L2-normed query embeddings. paired = mean cos(clean_i, deg_i); random = mean cos(clean_i, deg_j), i!=j.
    *_centered: same after subtracting the clean mean (GAP features are mostly positive, so raw cosines sit high)."""
    d = {"n": int(len(Ec)), "dim": int(Ec.shape[1]), "erank_clean": erank(Ec), "std_metric_clean": std_metric(torch.from_numpy(Ec))}
    if Ed is not None:
        d["erank_degraded"] = erank(Ed); d["std_metric_degraded"] = std_metric(torch.from_numpy(Ed))
        p, r, t1 = _pair_stats(Ec, Ed); d.update({"paired_cos": p, "random_pair_cos": r, "paired_minus_random": p - r, "clean_to_deg_top1": t1})
        mu = Ec.mean(0, keepdims=True); nz = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        p, r, _ = _pair_stats(nz(Ec - mu), nz(Ed - mu)); d.update({"paired_cos_centered": p, "random_pair_cos_centered": r, "paired_minus_random_centered": p - r})
    return d

def evaluate(Eall_rows, E, tag, E_deg=None, split="dev"):
    """E (and E_deg, same rows/order): L2-normed embeddings aligned with Eall_rows.
    split 'dev': support folds != EF, query EF (parent protocol). split 'inner': support folds not in (EF, IVF), query IVF."""
    rows = np.asarray(Eall_rows); pos = {r: i for i, r in enumerate(rows)}
    if split == "dev": s_rows = [r for r in rows if FOLDS[r] != EF]; q_rows = [r for r in rows if FOLDS[r] == EF]
    else:              s_rows = [r for r in rows if FOLDS[r] not in (EF, IVF)]; q_rows = [r for r in rows if FOLDS[r] == IVF]
    si = [pos[r] for r in s_rows]; qi = [pos[r] for r in q_rows]
    Es = E[si]; Eq = E[qi]; ys = Y_ALL[s_rows]; yq = Y_ALL[q_rows]
    common, rare = label_sets(q_rows, s_rows)
    t = time.time()
    ext = {"degraded": (E_deg[qi], yq)} if E_deg is not None else None
    lp = linear_probe(Es, ys, Eq, yq, common, ext)
    fs, fs_x = fewshot(Es, ys, Eq, yq, rare, CFG["eval"]["rare_headline"], ext)
    rep = {"tag": tag, "split": split, "n_support": len(s_rows), "n_query": len(q_rows),
           "common_labels": [LABELS[j] for j in common], "rare_labels": [LABELS[j] for j in rare],
           "rare_counts": {LABELS[j]: {"support_pos": int(ys[:, j].sum()), "query_pos": int(yq[:, j].sum())} for j in rare},
           "linear_probe": lp, "fewshot_rare": fs}
    kc = knn_monitor(Es, ys, Eq, yq, common, CFG["monitor"]["knn_k"]); rep["knn"] = {"clean_mauroc": kc[0], "clean_dr_auroc": kc[2]}
    if E_deg is not None:
        rep["linear_probe_clean_to_deg"] = lp["query_variants"]["degraded"]
        rep["fewshot_rare_clean_to_deg"] = fs_x["degraded"]
        kd = knn_monitor(Es, ys, E_deg[qi], yq, common, CFG["monitor"]["knn_k"]); rep["knn"].update({"deg_query_mauroc": kd[0], "deg_query_dr_auroc": kd[2]})
        if CFG["eval"]["degraded_support_probe"]:
            rep["linear_probe_deg_to_deg"] = linear_probe(E_deg[si], ys, E_deg[qi], yq, common)
    rep["diagnostics"] = diagnostics(Eq, E_deg[qi] if E_deg is not None else None)
    rep["eval_s"] = time.time() - t
    fsl = lambda f: "  ".join(f"{k} {v['logreg']['headline_mean']:.3f}" for k, v in f.items())
    msg = (f"  EVAL[{tag}|{split}] LP clean {lp['mean_auroc_common']:.3f} DR {lp['disease_risk_auroc']:.3f}"
           + (f" | LP c->deg {rep['linear_probe_clean_to_deg']['mean_auroc_common']:.3f}" if E_deg is not None else "")
           + (f" deg->deg {rep['linear_probe_deg_to_deg']['mean_auroc_common']:.3f}" if "linear_probe_deg_to_deg" in rep else "")
           + f" | fewshot(logreg {'/'.join(CFG['eval']['rare_headline'])}) {fsl(fs)}"
           + (f" | c->deg {fsl(rep['fewshot_rare_clean_to_deg'])}" if E_deg is not None else "")
           + f" | kNN {kc[0]:.3f}" + (f"/{rep['knn']['deg_query_mauroc']:.3f}" if E_deg is not None else "")
           + f" | erank {rep['diagnostics']['erank_clean']:.1f}"
           + (f"/{rep['diagnostics']['erank_degraded']:.1f} pair-rand {rep['diagnostics']['paired_minus_random']:.3f}"
              f" (centred {rep['diagnostics']['paired_minus_random_centered']:.3f})" if E_deg is not None else "")
           + f"  ({rep['eval_s']:.0f}s)")
    log(msg)
    return rep

def select_score(rep):
    m = CFG["monitor"]["select_metric"]
    if m == "lp_clean": return rep["linear_probe"]["mean_auroc_common"]
    if m == "knn_clean": return rep["knn"]["clean_mauroc"]
    if m == "lp_mean_clean_deg": return 0.5 * (rep["linear_probe"]["mean_auroc_common"] + rep["linear_probe_clean_to_deg"]["mean_auroc_common"])
    raise ValueError(m)

@torch.no_grad()
def extract_region(enc, rows, kind):
    was = enc.training; enc.eval(); E = []
    for x, _ in _loader(RegionDS(rows, kind)):
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
            E.append(enc(norm_gpu(x)).float().cpu().numpy())
    enc.train(was); E = np.concatenate(E)
    return (E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)).astype(np.float32)

def multi_region_eval(enc, rows, E_glob, tag, od=None):
    """eval-only readout (disabled by default): concat [global, disc, macula] L2-normed embeddings."""
    E = np.concatenate([E_glob, extract_region(enc, rows, "disc"), extract_region(enc, rows, "macula")], 1)
    if MR["power_norm"]: E = np.sign(E) * np.sqrt(np.abs(E))
    E = (E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
    if od is not None: np.save(od / f"embeddings_{tag.split('/')[-1]}.npy", E)
    r = evaluate(rows, E, tag); r["low_conf_frac"] = float((~BOX_OK[np.asarray(rows)]).mean()); return r

def baseline_embeddings():
    """supervised ConvNeXt-T (best.pth) embeddings from the baseline kernel output, if mounted (clean only)."""
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
    if DEV.type != "cuda": return int(os.environ.get("TOB_BS", 4)), False, None
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

def ckpt_dict(online, target, ep, mode, **extra):
    d = {"online": online.state_dict(), "target_backbone": target.backbone.state_dict(), "target_neck": target.neck.state_dict(),
         "target_projector": target.projector.state_dict(), "epoch": ep, "crop_mode": mode, "config": CFG,
         "arch": {"backbone": CFG["model"]["backbone"], "feature": CFG["model"]["feature"], "neck_dim": CFG["model"]["neck_dim"]}}
    d.update(extra); return d

def load_ckpt(path):
    """-> (online BYOLNet, Target) from a checkpoint written by this script."""
    ck = torch.load(path, map_location=DEV, weights_only=False)
    net = BYOLNet(False).to(DEV).to(memory_format=torch.channels_last); net.load_state_dict(ck["online"])
    tg = Target(net).to(DEV); tg.backbone.load_state_dict(ck["target_backbone"]); tg.neck.load_state_dict(ck["target_neck"])
    tg.projector.load_state_dict(ck["target_projector"]); return net, tg, ck

def run_mode(mode, bs, ck):
    tc = CFG["train"]; mon = CFG["monitor"]; amp = tc["amp"] and DEV.type == "cuda"
    od = OUT / f"byol_mnv3_{mode}"; od.mkdir(parents=True, exist_ok=True)   # v4 wrote byol_{mode}; renamed so outputs never mix with v4
    online, target, opt = build(bs, ck); scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ds = SSLDS(SSL_ROWS, mode)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=CFG["data"]["num_workers"],
                                     pin_memory=DEV.type == "cuda", persistent_workers=False)
    steps_ep = len(dl); total = steps_ep * tc["epochs"]; warm = steps_ep * tc["warmup_epochs"]
    log(f"=== BYOL {CFG['model']['backbone']} crop_mode={mode}: {len(SSL_ROWS)} imgs, bs {bs}, grad_ckpt {ck}, {steps_ep} steps/ep, {tc['epochs']} ep, "
        f"emb dim {online.emb_dim}, params {sum(p.numel() for p in online.parameters())/1e6:.2f}M (backbone {sum(p.numel() for p in online.backbone.parameters())/1e6:.2f}M)")
    if DEV.type == "cuda": torch.cuda.reset_peak_memory_stats()
    common_in, _ = label_sets(IV_ROWS, IN_S_ROWS)     # kNN monitor: support folds 1-3, query fold 4 (labels only for monitoring)
    sel_net = lambda: encoder(target if mon["select_encoder"] == "target" else online)
    oth_net = lambda: encoder(online if mon["select_encoder"] == "target" else target)
    oth_name = "online" if mon["select_encoder"] == "target" else "target"
    hist, step = [], 0
    best_score, best_ep = -1.0, 0; epoch_evals = {}; sel_scores = {}
    for ep in range(tc["epochs"]):
        ds.epoch = ep; online.train(); target.train()
        t = time.time(); tl, nb, zs, hs, an, tstep, als = 0.0, 0, [], [], np.zeros(N_ANCH), [], []
        for G, L, a, al in dl:
            ts = time.time()
            for gp in opt.param_groups: gp["lr"] = gp["base_lr"] * lr_factor(step, total, warm)
            tau = 1 - (1 - tc["tau_base"]) * (math.cos(math.pi * step / max(1, total)) + 1) / 2
            loss, z, h, gn, vc = train_step(online, target, opt, scaler, G, L, tau, amp)
            tl += float(loss); nb += 1; step += 1; an += np.bincount(a.numpy().ravel(), minlength=N_ANCH); als.append(al.numpy())
            zs.append(z); hs.append(h)
            if DEV.type == "cuda": torch.cuda.synchronize()
            tstep.append(time.time() - ts)
        t_tr = time.time() - t
        zc = torch.cat(zs); hc = torch.cat(hs); als = np.concatenate(als)
        row = {"epoch": ep + 1, "loss": tl / max(1, nb), "std_z": std_metric(zc), "std_h": std_metric(hc), "tau_end": tau,
               "lr_bb": opt.param_groups[0]["lr"], "train_s": t_tr, "img_per_s": nb * bs / max(t_tr, 1e-9),
               "gpu_img_per_s_median_step": bs / float(np.median(tstep)) if tstep else float("nan"),
               "anchor_frac": {ANCHOR_NAMES[k]: float(an[k] / max(1, an.sum())) for k in range(N_ANCH)},
               "anatomy_views_frac": float(an[1:3].sum() / max(1, an.sum())),
               "deg_frac_g0": float((als[:, 0] >= 0).mean()), "deg_frac_g1": float((als[:, 1] >= 0).mean()),
               "deg_frac_local": float((als[:, 2:] >= 0).mean()), "deg_alpha_mean": float(als[als >= 0].mean()) if (als >= 0).any() else float("nan")}
        if not math.isfinite(row["loss"]):
            json.dump(hist + [row], open(od / "history.json", "w"), indent=1, default=float)
            raise SystemExit(f"non-finite loss at epoch {ep+1}")
        if row["std_z"] < mon["collapse_warn"]: log(f"  WARNING possible collapse: std_z {row['std_z']:.3f}")
        if row["std_h"] < mon["collapse_warn"]: log(f"  WARNING backbone std_h {row['std_h']:.3f} < {mon['collapse_warn']} (ep {ep+1})")
        if (ep + 1) % mon["knn_every"] == 0 or ep + 1 == tc["epochs"]:
            tk = time.time(); enc = sel_net()
            Eb = extract(enc, IN_S_ROWS); Eq = extract(enc, IV_ROWS); Eqd = extract(enc, IV_ROWS, degraded=True)
            row["knn_mauroc"], row["knn_dr_acc"], row["knn_dr_auroc"] = knn_monitor(Eb, Y_ALL[IN_S_ROWS], Eq, Y_ALL[IV_ROWS], common_in, mon["knn_k"])
            row["knn_mauroc_deg_query"] = knn_monitor(Eb, Y_ALL[IN_S_ROWS], Eqd, Y_ALL[IV_ROWS], common_in, mon["knn_k"])[0]
            row["knn_split"] = f"inner: support folds 1-3, query fold {IVF}, {mon['select_encoder']} encoder"
            dg = diagnostics(Eq, Eqd); row.update({"erank_q": dg["erank_clean"], "pair_minus_rand_q": dg["paired_minus_random"],
                                                   "pair_minus_rand_centered_q": dg["paired_minus_random_centered"]})
            row["knn_s"] = time.time() - tk
        hist.append(row)
        log(f"  [{mode}] ep {ep+1}/{tc['epochs']} loss {row['loss']:.4f} std_z {row['std_z']:.3f} std_h {row['std_h']:.3f} tau {tau:.4f} "
            f"lr_bb {row['lr_bb']:.2e} {row['img_per_s']:.1f} img/s ({t_tr:.0f}s)"
            + (f" | inner kNN {row['knn_mauroc']:.3f} deg-q {row['knn_mauroc_deg_query']:.3f} erank {row['erank_q']:.1f} "
               f"pair-rand {row['pair_minus_rand_q']:.3f}/{row['pair_minus_rand_centered_q']:.3f}c ({row['knn_s']:.0f}s)" if "knn_mauroc" in row else "")
            + f" | deg g0 {row['deg_frac_g0']:.2f} g1 {row['deg_frac_g1']:.2f} loc {row['deg_frac_local']:.2f} a {row['deg_alpha_mean']:.2f}"
            + f" | anat {row['anatomy_views_frac']:.2f} " + " ".join(f"{k[:6]}={v:.2f}" for k, v in row["anchor_frac"].items()))
        json.dump(hist, open(od / "history.json", "w"), indent=1, default=float)
        last = ep + 1 == tc["epochs"]
        if (ep + 1) % tc["ckpt_every"] == 0 or last or ep + 1 in mon["eval_epochs"]:
            torch.save(ckpt_dict(online, target, ep + 1, mode), od / ("last.pth" if last else f"ep{ep+1:03d}.pth"))
        if ep + 1 in mon["eval_epochs"] or (last and not epoch_evals):   # inner-val eval -> checkpoint selection (fold 0 untouched)
            te = time.time(); E, Ed = extract_pair(sel_net(), INNER_ROWS)
            ee = {mon["select_encoder"]: evaluate(INNER_ROWS, E, f"byol_mnv3_{mode}/ep{ep+1}_{mon['select_encoder']}", Ed, "inner")}
            if mon["eval_both_encoders"]:
                E2, Ed2 = extract_pair(oth_net(), INNER_ROWS); ee[oth_name] = evaluate(INNER_ROWS, E2, f"byol_mnv3_{mode}/ep{ep+1}_{oth_name}", Ed2, "inner")
            sc = select_score(ee[mon["select_encoder"]]); sel_scores[ep + 1] = sc
            ee["select_score"] = sc; ee["eval_s"] = time.time() - te; epoch_evals[f"ep{ep+1}"] = ee
            if sc > best_score:
                best_score, best_ep = sc, ep + 1
                torch.save(ckpt_dict(online, target, ep + 1, mode, select_metric=mon["select_metric"], select_score=sc,
                                     select_encoder=mon["select_encoder"]), od / "best.pth")
                log(f"  inner-val select ({mon['select_metric']}, {mon['select_encoder']}) {sc:.4f} at ep {ep+1} -> best.pth")
            json.dump(epoch_evals, open(od / "epoch_evals.json", "w"), indent=1, default=float)
    peak = torch.cuda.max_memory_allocated() / 2**30 if DEV.type == "cuda" else float("nan")
    rep = {"crop_mode": mode, "batch_size": bs, "grad_ckpt": ck, "steps_per_epoch": steps_ep, "history": hist, "peak_gpu_gib": peak,
           "selection": {"metric": mon["select_metric"], "encoder": mon["select_encoder"], "split": f"inner (folds 1-3 -> fold {IVF})",
                         "scores_by_epoch": sel_scores, "selected_epoch": best_ep, "selected_score": best_score}}
    # dev split (fold 0, disclosed development split): last online + last target + inner-selected checkpoint
    t = time.time()
    E_on, Ed_on = extract_pair(encoder(online), EVAL_ROWS); np.save(od / "embeddings_online.npy", E_on); np.save(od / "embeddings_online_degraded.npy", Ed_on)
    E_tg, Ed_tg = extract_pair(encoder(target), EVAL_ROWS); np.save(od / "embeddings_target.npy", E_tg); np.save(od / "embeddings_target_degraded.npy", Ed_tg)
    pd.DataFrame({"ID": df.ID.values[EVAL_ROWS], "fold": FOLDS[EVAL_ROWS]}).to_csv(od / "embedding_ids.csv", index=False)
    rep["eval_online"] = evaluate(EVAL_ROWS, E_on, f"byol_mnv3_{mode}/online", Ed_on, "dev")
    rep["eval_target"] = evaluate(EVAL_ROWS, E_tg, f"byol_mnv3_{mode}/target", Ed_tg, "dev")
    if MR["enabled"]: rep["eval_multi_region_online"] = multi_region_eval(encoder(online), EVAL_ROWS, E_on, f"byol_mnv3_{mode}/multiregion_online", od)
    rep["epoch_evals"] = epoch_evals
    rep["best_knn"] = {"note": "v4 key kept; now = inner-val selection (not fold-0 kNN)", "epoch": best_ep, "score": best_score}
    if best_ep and best_ep != tc["epochs"]:
        net, tg, _ = load_ckpt(od / "best.pth"); enc = encoder(tg if mon["select_encoder"] == "target" else net)
        E_b, Ed_b = extract_pair(enc, EVAL_ROWS); np.save(od / f"embeddings_selected_{mon['select_encoder']}.npy", E_b)
        np.save(od / f"embeddings_selected_{mon['select_encoder']}_degraded.npy", Ed_b)
        rep["eval_selected"] = evaluate(EVAL_ROWS, E_b, f"byol_mnv3_{mode}/selected_ep{best_ep}_{mon['select_encoder']}", Ed_b, "dev")
        del net, tg
    else:
        rep["eval_selected"] = {"same_as": f"eval_{mon['select_encoder']}", "epoch": best_ep}
    rep["final_eval_s"] = time.time() - t
    json.dump(rep, open(od / "report.json", "w"), indent=1, default=float)
    del online, target, opt; torch.cuda.empty_cache() if DEV.type == "cuda" else None
    return rep

# ----------------------------------------------------------------------------- main
summary = {"run_mode": RUN_MODE, "labels": LABELS, "split_source": split_src, "eval_fold": EF, "inner_val_fold": IVF,
           "ssl_folds": sorted(set(FOLDS[SSL_ROWS].tolist())), "n_ssl": int(len(SSL_ROWS)), "n_ssl_full": N_FULL_SSL,
           "n_query": int(len(Q_ROWS)), "n_inner_query": int(len(IV_ROWS)), "n_eval_rows": int(len(EVAL_ROWS)),
           "locator": LOC_STATS, "degrade_fit": FIT, "geom": GEOM_STATS, "config": CFG, "modes": {}, "references": {},
           "citation": "RFMiD (Pachade et al., Data 6(2):14, 2021; IEEE DataPort, doi:10.21227/s3g7-st65), CC BY 4.0"}
def dump(): summary["timings"] = TIM; json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=float)

degrade_examples(OUT / "degrade_eval_examples.png")
# (c) references: supervised baseline embeddings (clean only) + ImageNet-init MobileNetV3 (clean + degraded), dev + inner
be = baseline_embeddings() if not SMOKE else None
if be is not None:
    brows, bE, bsrc = be; keep = np.isin(brows, EVAL_ROWS)
    summary["references"]["supervised_baseline"] = evaluate(brows[keep], bE[keep], "supervised_convnext_tiny_best")
    summary["references"]["supervised_baseline"]["source"] = bsrc
    summary["references"]["supervised_baseline"]["note"] = "ConvNeXt-T embeddings at 512 px, clean only; TRAINED on all 28 labels of folds 1-4"
else:
    log("WARNING: supervised baseline embeddings not mounted (or smoke); skipping reference (c)")
t = time.time()
if not SMOKE or os.environ.get("TOB_REFS_IN_SMOKE") == "1":
    _m = GAPEncoder(CFG["model"]["pretrained"]).to(DEV).to(memory_format=torch.channels_last)
    _E, _Ed = extract_pair(_m, EVAL_ROWS)
    summary["references"]["imagenet_init"] = evaluate(EVAL_ROWS, _E, f"imagenet_init_{CFG['eval']['img_size']}", _Ed, "dev")
    summary["references"]["imagenet_init_inner"] = evaluate(EVAL_ROWS, _E, f"imagenet_init_{CFG['eval']['img_size']}", _Ed, "inner")
    del _m
TIM["ref_eval_s"] = time.time() - t; dump()
if os.environ.get("TOB_STOP_AFTER_REFS") == "1": log("STOP_AFTER_REFS (local test)"); sys.exit(0)

if RUN_MODE == "eval":
    # standalone: evaluate a saved checkpoint (TOB_CKPT=path to last.pth / epNNN.pth / best.pth), dev + inner, online + target
    net, tg, ckd = load_ckpt(os.environ["TOB_CKPT"])
    for nm, m_ in [("online", net), ("target", tg)]:
        E, Ed = extract_pair(encoder(m_), EVAL_ROWS)
        summary["modes"][f"eval_ckpt_{nm}"] = evaluate(EVAL_ROWS, E, f"ckpt:{os.environ['TOB_CKPT']}:{nm}", Ed, "dev")
        summary["modes"][f"eval_ckpt_{nm}_inner"] = evaluate(EVAL_ROWS, E, f"ckpt:{os.environ['TOB_CKPT']}:{nm}", Ed, "inner")
    dump(); log("DONE eval"); sys.exit(0)

amp = CFG["train"]["amp"] and DEV.type == "cuda"
bs, ck, probe_peak = probe_batch(amp)
summary["batch"] = {"batch_size": bs, "grad_ckpt": ck, "probe_peak_gib": probe_peak}
dump()
for mode in CFG["train"]["crop_modes"]:
    tm = time.time()
    crop_grid(OUT / "local_views_grid.png")
    summary["modes"][mode] = run_mode(mode, bs, ck)
    TIM[f"{mode}_total_s"] = time.time() - tm; dump()

# full-run time estimate (from measured throughput of THIS run; on CPU this is not representative of a T4)
est = {}
for mode, r in summary["modes"].items():
    h = r["history"]
    ips = max(x["img_per_s"] for x in h)            # wall img/s incl. dataloader (best epoch = warm)
    knn_s = np.mean([x["knn_s"] for x in h if "knn_s" in x]) if any("knn_s" in x for x in h) else 0
    ee_s = np.mean([v["eval_s"] for v in r["epoch_evals"].values()]) if r["epoch_evals"] else 0
    scale_eval = len(df) / max(1, len(EVAL_ROWS))
    ep_s = N_FULL_SSL / ips; E_ = CFG_FULL_EPOCHS
    n_knn = E_ // 5; n_ev = 3
    est[mode] = {"device": str(DEV), "wall_img_per_s": ips, "sec_per_epoch_full_data": ep_s, "knn_s_full": knn_s * scale_eval,
                 "inner_eval_s_full": ee_s * scale_eval, "final_eval_s_full": r["final_eval_s"] * scale_eval,
                 "est_full_hours": (E_ * ep_s + n_knn * knn_s * scale_eval + n_ev * ee_s * scale_eval + r["final_eval_s"] * scale_eval) / 3600,
                 "note": "assumes 40 ep, kNN every 5, 3 inner evals; extrapolated from this run's measured times"}
summary["estimate_full"] = est; TIM["wall_s"] = time.time() - T0; dump()
log("ESTIMATE", json.dumps(est, default=float))
log("DONE", json.dumps(TIM, default=float))
