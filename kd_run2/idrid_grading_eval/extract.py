"""IDRiD Disease Grading cross-domain eval, step 1: frozen-encoder embeddings on the box CPU.
Preprocessing is the KD Run 2 kernel code itself (crop_fundus, fov_geom, DEGRADE BLOCK, config) sliced from the kernel source
and exec'd, so nothing is re-implemented: raw JPG -> BGR2RGB -> fundus crop/pad square -> INTER_AREA 640 cache (+GEOM) ->
INTER_AREA 512 (MNV3 student / SSL) or 384 (teacher) -> ImageNet mean/std. Degraded copy = degrade_eval(alpha=1, eval_seed 2026,
per-image seed from the row index in THIS dataset). Only deviation: CPU fp32 instead of Kaggle CUDA fp16 autocast
(checked below on 16 RFMiD images against the kernel's own saved embeddings).
Models: KD768 = kd_final.pth (post BN-recal, with the 960->768 head); SSL960 = student_init_mnv3_ssl_ep40_target.pth GAP;
SSL768p = SSL960 GAP @ procrustes_W.T (the kernel's fixed W); teacher = BYOL v4 ConvNeXt-T ep20 target at 384."""
import os, sys, math, json, time, re
from pathlib import Path
import numpy as np, pandas as pd, cv2, yaml, torch, timm, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(8); cv2.setNumThreads(4)
HERE = Path(__file__).resolve().parent; K = HERE.parent
KSRC = (K / "kernels/full/team_offirst_kd_run2.py").read_text()
OUTK = K / "full_out/out/kd_run2"; WDS = K / "weights_ds"
ROOT = Path("/workspace/idrid_grading/B.%20Disease%20Grading/B. Disease Grading")
def grab_def(name):
    i = KSRC.index(f"def {name}("); j = KSRC.index("\ndef ", i + 1); return KSRC[i:j]
CFG = yaml.safe_load(KSRC[KSRC.index('CONFIG_YAML = r"""') + 18:KSRC.index('"""', KSRC.index('CONFIG_YAML = r"""') + 18)])
CS = CFG["data"]["cache_size"]; SS = CFG["kd"]["student_size"]; TS = CFG["kd"]["teacher_size"]
assert (CS, SS, TS) == (640, 512, 384)
NS = {"np": np, "cv2": cv2, "math": math, "CS": CS, "CFG": CFG}
exec(grab_def("crop_fundus"), NS); exec(grab_def("fov_geom"), NS)
exec(KSRC[KSRC.index("# >>> DEGRADE BLOCK"):KSRC.index("# <<< DEGRADE BLOCK")], NS)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255; STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255
def load_prep(path):   # = kernel load_prep (uses kernel crop_fundus / fov_geom)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR); sq = NS["crop_fundus"](cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    out = cv2.resize(sq, (CS, CS), interpolation=cv2.INTER_AREA); return out, NS["fov_geom"](out, img.shape[1], sq.shape[1]), img.shape

class GAPEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.body = timm.create_model("mobilenetv3_large_100", pretrained=False, num_classes=0)
        for n in ["conv_head", "norm_head", "act2", "flatten", "classifier", "global_pool"]:
            if hasattr(self.body, n): setattr(self.body, n, nn.Identity())
    def forward(self, x): return self.body.forward_features(x).mean((2, 3))
class KDStudent(nn.Module):
    def __init__(self):
        super().__init__(); self.backbone = GAPEncoder(); self.head = nn.Linear(960, 768, bias=False)
    def forward(self, x): return self.head(self.backbone(x))
def build():
    kd = KDStudent(); ck = torch.load(OUTK / "kd_final.pth", map_location="cpu", weights_only=False)
    assert ck["bn_recal"] and ck["epoch"] == 40; kd.load_state_dict(ck["student"], strict=True)
    ssl = GAPEncoder(); ssl.load_state_dict(torch.load(WDS / "student_init_mnv3_ssl_ep40_target.pth", map_location="cpu", weights_only=False)["state_dict"], strict=True)
    t = timm.create_model("convnext_tiny", pretrained=False, num_classes=0)
    t.load_state_dict(torch.load(WDS / "teacher_byol_v4_ep20_target_convnext_tiny.pth", map_location="cpu", weights_only=False)["state_dict"], strict=True)
    return kd.eval(), ssl.eval(), t.eval()
W = np.load(OUTK / "procrustes_W.npy")   # (768, 960)
l2n = lambda E: (E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)).astype(np.float32)

@torch.no_grad()
def embed(models, cache, geom, rows_seed, degraded, bs=16):
    """returns dict name -> (N, d) L2-normalised. rows_seed = per-image integer used as the degrader row seed."""
    kd, ssl, t = models; NS["GEOM"] = geom; out = {k: [] for k in ["kd768", "ssl960", "ssl768p", "teacher"]}
    for i0 in range(0, len(cache), bs):
        xs, xt = [], []
        for i in range(i0, min(len(cache), i0 + bs)):
            img = np.asarray(cache[i])
            if degraded:
                xs.append(NS["degrade_eval"](img, i, SS)); xt.append(NS["degrade_eval"](img, i, TS))
            else:
                xs.append(cv2.resize(img, (SS, SS), interpolation=cv2.INTER_AREA)); xt.append(cv2.resize(img, (TS, TS), interpolation=cv2.INTER_AREA))
        f = lambda L: (torch.from_numpy(np.stack(L)).permute(0, 3, 1, 2).float() - MEAN) / STD
        s, tt = f(xs), f(xt)
        out["kd768"].append(kd(s).numpy()); g = ssl(s).numpy(); out["ssl960"].append(g); out["ssl768p"].append(l2n(g) @ W.T)
        out["teacher"].append(t(tt).numpy())
    return {k: l2n(np.concatenate(v)) for k, v in out.items()}

def sanity_rfmid(models, n=16):
    """CPU fp32 box path vs the kernel's saved Kaggle (fp16) embeddings on n RFMiD rows (clean + degraded)."""
    ids = pd.read_csv(OUTK / "embedding_ids.csv"); rows = np.linspace(0, len(ids) - 1, n).astype(int)
    d = Path("/workspace/rfmid_train_dl/Training_Set/Training")
    cache, geom = [], []
    for r in rows: c, g, _ = load_prep(d / f"{ids.ID[r]}.png"); cache.append(c); geom.append(g)
    res = {}
    for deg in (False, True):
        # degrader seed in the kernel = kernel row index; GEOM indexed by that row -> build a sparse geom table
        G = np.zeros((len(ids), 5)); G[rows] = geom; NS["GEOM"] = G
        Ebox = {k: [] for k in ["kd768", "ssl960", "ssl768p", "teacher"]}
        for r, c in zip(rows, cache):
            e = embed_one(models, c, int(r), G, deg)
            for k in Ebox: Ebox[k].append(e[k])
        for k, fn in [("kd768", "embeddings_kd768"), ("ssl960", "embeddings_ssl_ep40_960"), ("ssl768p", "embeddings_ssl_ep40_768procrustes"), ("teacher", "embeddings_teacher")]:
            Ek = np.load(OUTK / f"{fn}{'_degraded' if deg else ''}.npy")[rows]; cs = (np.stack(Ebox[k]) * Ek).sum(1)
            res[f"{k}_{'deg' if deg else 'clean'}"] = {"cos_min": float(cs.min()), "cos_mean": float(cs.mean())}
    return res

@torch.no_grad()
def embed_one(models, img, seed_row, geom, deg):
    kd, ssl, t = models; NS["GEOM"] = geom
    xs = NS["degrade_eval"](img, seed_row, SS) if deg else cv2.resize(img, (SS, SS), interpolation=cv2.INTER_AREA)
    xt = NS["degrade_eval"](img, seed_row, TS) if deg else cv2.resize(img, (TS, TS), interpolation=cv2.INTER_AREA)
    f = lambda x: (torch.from_numpy(x[None]).permute(0, 3, 1, 2).float() - MEAN) / STD
    g = ssl(f(xs)).numpy()
    return {"kd768": l2n(kd(f(xs)).numpy())[0], "ssl960": l2n(g)[0], "ssl768p": l2n(l2n(g) @ W.T)[0], "teacher": l2n(t(f(xt)).numpy())[0]}

if __name__ == "__main__":
    T0 = time.time(); models = build(); EMB = HERE / "embeddings"; EMB.mkdir(exist_ok=True)
    san = sanity_rfmid(models); print("sanity vs kernel embeddings:", json.dumps(san, indent=1), flush=True)
    rows = []
    for split, sub, lab in [("train", "a. Training Set", "a. IDRiD_Disease Grading_Training Labels.csv"), ("test", "b. Testing Set", "b. IDRiD_Disease Grading_Testing Labels.csv")]:
        df = pd.read_csv(ROOT / "2. Groundtruths" / lab).iloc[:, :3].dropna(); df.columns = ["name", "dr", "dme"]
        for _, r in df.iterrows(): rows.append({"split": split, "name": r["name"], "dr": int(r.dr), "dme": int(r.dme), "path": str(ROOT / "1. Original Images" / sub / f"{r['name']}.jpg")})
    meta = pd.DataFrame(rows); assert meta.path.map(os.path.exists).all()
    cache = np.lib.format.open_memmap("/tmp/idrid_cache_640.npy", mode="w+", dtype=np.uint8, shape=(len(meta), CS, CS, 3)); geom = np.zeros((len(meta), 5)); shapes = []
    for i, p in enumerate(meta.path):
        c, g, sh = load_prep(p); cache[i] = c; geom[i] = g; shapes.append(sh[:2])
    meta["raw_h"] = [s[0] for s in shapes]; meta["raw_w"] = [s[1] for s in shapes]
    meta["fov_radius_frac"] = geom[:, 4]; meta.drop(columns="path").to_csv(EMB / "ids.csv", index=False); np.save(EMB / "geom.npy", geom)
    print(f"cached {len(meta)} in {time.time()-T0:.0f}s; raw sizes {meta.groupby(['raw_w','raw_h']).size().to_dict()}", flush=True)
    for deg in (False, True):
        E = embed(models, cache, geom, None, deg)
        for k, v in E.items(): np.save(EMB / f"{k}{'_degraded' if deg else ''}.npy", v)
        print(f"embedded degraded={deg} {time.time()-T0:.0f}s", flush=True)
    for i in [0, 450]:   # example images for the report
        img = np.asarray(cache[i]); NS["GEOM"] = geom
        cv2.imwrite(str(HERE / f"example_{meta.split[i]}_{meta.name[i]}.jpg"), cv2.cvtColor(np.concatenate([cv2.resize(img, (SS, SS), interpolation=cv2.INTER_AREA), NS["degrade_eval"](img, i, SS)], 1), cv2.COLOR_RGB2BGR))
    json.dump({"sanity_vs_kernel_rfmid16": san, "n": len(meta), "secs": time.time() - T0, "geom_fov_radius_median": float(np.median(geom[:, 4]))}, open(EMB / "extract_meta.json", "w"), indent=1)
    print("DONE", time.time() - T0)
