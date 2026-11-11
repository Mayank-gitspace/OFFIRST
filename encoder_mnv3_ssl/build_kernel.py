"""OFFIRST direct small-model SSL (BYOL on MobileNetV3-Large) kernel builder.
parent_team_offirst_encoder_byol_anatomy_v4.py (byte copy of v4 kernels/full) --(this file)--> kernels/{smoke,full}/team_offirst_encoder_byol_mnv3.py
Kept verbatim from the parent: imports, data guard, labels/split, crop_fundus, box table (BOX_B64), locator stats.
Replaced: header + CONFIG (src/header_config.py), cache builder (+ per-image FOV/raw geometry for the degrader),
everything from '# --- torch' on (src/tail.py: views + degrader, model, eval, train loop, main).
Run: python3 build_kernel.py
"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
PARENT = HERE / "parent_team_offirst_encoder_byol_anatomy_v4.py"
src = PARENT.read_text()
def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (old[:80], src.count(old)); src = src.replace(old, new)

# ---------------------------------------------------------------- header + config
i0 = src.index("import os, sys, json, time, math, random, copy, warnings")
src = (HERE / "src" / "header_config.py").read_text() + src[i0:]

# ---------------------------------------------------------------- local-test overrides (env; never set on Kaggle)
rep('LOCATE_ONLY = os.environ.get("TOB_LOCATE_ONLY") == "1"   # local CPU: heuristic + contact sheet only',
'''LOCATE_ONLY = os.environ.get("TOB_LOCATE_ONLY") == "1"   # local CPU: heuristic + contact sheet only
for _e, (_a, _b) in {"TOB_GLOBAL_SIZE": ("views", "global_size"), "TOB_LOCAL_SIZE": ("views", "local_size"),
                     "TOB_CACHE_SIZE": ("data", "cache_size"), "TOB_EVAL_SIZE": ("eval", "img_size")}.items():
    if _e in os.environ: CFG[_a][_b] = int(os.environ[_e])          # local CPU test only
if SMOKE and "TOB_EVAL_EPOCHS" in os.environ: CFG["monitor"]["eval_epochs"] = [int(x) for x in os.environ["TOB_EVAL_EPOCHS"].split(",")]
if SMOKE and "TOB_KNN_EVERY" in os.environ: CFG["monitor"]["knn_every"] = int(os.environ["TOB_KNN_EVERY"])
EVAL_SUB = int(os.environ.get("TOB_EVAL_SUB", "0")) if SMOKE else 0   # local CPU test only: eval rows per fold''')
rep('log(f"RUN_MODE={RUN_MODE}  input={INPUT}  out={OUT}")',
    'log(f"RUN_MODE={RUN_MODE}  backbone={CFG[\'model\'][\'backbone\']}  global={CFG[\'views\'][\'global_size\']}  local={CFG[\'views\'][\'local_size\']}  cache={CFG[\'data\'][\'cache_size\']}  input={INPUT}  out={OUT}")')

# ---------------------------------------------------------------- cache: also store per-image geometry for the degrader
rep('''def load_prep(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError(f"cannot read {path}")
    return cv2.resize(crop_fundus(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), (CS, CS), interpolation=cv2.INTER_AREA)
''', '''def fov_geom(im, w_raw, side_raw):
    """[raw width, raw side of the fundus square, FOV centre x, centre y, radius] (last 3 as fractions of the square side).
    FOV = degrade.py _fov(): gray > 15 + 5x5 opening; R = sqrt(area / pi)."""
    m = cv2.morphologyEx((cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) > 15).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    ys, xs = np.nonzero(m); S = im.shape[0]
    if len(xs) < 100: return [float(w_raw), float(side_raw), 0.5, 0.5, 0.5]
    return [float(w_raw), float(side_raw), float(xs.mean() / S), float(ys.mean() / S), float(math.sqrt(len(xs) / math.pi) / S)]

def load_prep(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError(f"cannot read {path}")
    sq = crop_fundus(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    out = cv2.resize(sq, (CS, CS), interpolation=cv2.INTER_AREA)
    return out, fov_geom(out, img.shape[1], sq.shape[1])
''')
rep('''cpath = TMP / f"cache_{CS}_{len(df)}.npy"
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
TIM["cache_s"] = time.time() - t''', '''cpath = TMP / f"cache_{CS}_{len(df)}.npy"; gpath = TMP / f"cache_{CS}_{len(df)}_geom.npy"
t = time.time()
if cpath.exists() and gpath.exists():
    CACHE = np.load(cpath, mmap_mode="r"); GEOM = np.load(gpath)
else:
    CACHE = np.lib.format.open_memmap(str(cpath) + ".part.npy", mode="w+", dtype=np.uint8, shape=(len(df), CS, CS, 3))
    GEOM = np.zeros((len(df), 5), np.float64)
    with Pool(max(1, CFG["data"]["num_workers"])) as pool:
        for i, (im, g) in enumerate(pool.imap(load_prep, df["path"].tolist(), chunksize=4)):
            CACHE[i] = im; GEOM[i] = g
            if (i + 1) % 400 == 0: log(f"  cached {i+1}/{len(df)}")
    CACHE.flush(); del CACHE
    np.save(gpath, GEOM); os.replace(str(cpath) + ".part.npy", cpath); CACHE = np.load(cpath, mmap_mode="r")
assert GEOM.shape == (len(df), 5) and np.isfinite(GEOM).all()
GEOM_STATS = {"raw_width_over_side_median": float(np.median(GEOM[:, 0] / GEOM[:, 1])), "raw_side_median": float(np.median(GEOM[:, 1])),
              "fov_radius_frac_median": float(np.median(GEOM[:, 4])), "fov_centre_offset_frac_max": float(np.abs(GEOM[:, 2:4] - 0.5).max())}
log("degrader geometry:", json.dumps(GEOM_STATS))
TIM["cache_s"] = time.time() - t''')

# ---------------------------------------------------------------- locator stat: SSL folds now exclude the inner-val fold by default
rep('"low_conf_frac_ssl": float(BOX.low_conf.values[FOLDS != EF].mean())}',
    '"low_conf_frac_ssl": float(BOX.low_conf.values[~np.isin(FOLDS, [EF] if CFG["data"]["ssl_include_inner_val"] else [EF, CFG["data"]["inner_val_fold"]])].mean())}')

# ---------------------------------------------------------------- tail
i0 = src.index("# ----------------------------------------------------------------------------- torch")
src = src[:i0].rstrip("\n") + "\n" + (HERE / "src" / "tail.py").read_text()

for mode in ["smoke", "full"]:
    d = HERE / "kernels" / mode; d.mkdir(parents=True, exist_ok=True)
    (d / "team_offirst_encoder_byol_mnv3.py").write_text(src.replace('RUN_MODE = "__RUN_MODE__"', f'RUN_MODE = "{mode}"'))
    kid = "teamoffirst/team-offirst-encoder-byol-mnv3" + ("-smoke" if mode == "smoke" else "")
    meta = {"id": kid, "title": kid.split("/")[1], "code_file": "team_offirst_encoder_byol_mnv3.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": ["awsaf49/rfmid-train-dataset"], "competition_sources": [],
            "kernel_sources": [], "machine_shape": "NvidiaTeslaT4"}
    (d / "kernel-metadata.json").write_text(json.dumps(meta))
print("built", len(src.splitlines()), "lines")
