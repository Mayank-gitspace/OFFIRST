"""ODIR-5K step 3: frozen embeddings -> LP / kNN / few-shot -> paired bootstrap (box CPU).
LP: per-label binary (one-vs-rest, multi-label) on L2-normalised embeddings: StandardScaler(train) -> LogisticRegression(balanced,
    max_iter 3000); C from [1e-4,1e-3,1e-2,0.1,1] by 3-fold StratifiedGroupKFold (groups = patient) on train only, metric AUROC;
    refit on all train. Conditions: clean->clean, clean probe->degraded test, degraded train->degraded test. Macro = mean of 8 labels.
kNN (A, G, M): bank = clean train, k=20 cosine; 'w' = exp(sim/0.07)-weighted vote (paired_bootstrap.py convention), 'maj' = plain
    fraction of positive neighbours; clean and degraded queries.
Few-shot (A, G, M): RFMiD rare protocol mirrored: k in {1,5,10} positives + k negatives (neg_per_pos 1) drawn from TRAIN eyes,
    20 draws, rng seed 42*1000+k*100+d consumed over labels in order A,G,M; LogisticRegression(C=1, balanced) on L2-normalised
    embeddings (no scaler, as evaluate()); query = all test eyes; AUROC averaged over draws. Clean only.
Bootstrap: 1000 resamples of TEST PATIENTS (cluster bootstrap: both eyes move together; seed 12345), shared across models/metrics."""
import json, time, numpy as np, pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
HERE = Path(__file__).resolve().parent; EMB = HERE / "embeddings"
GRID = [1e-4, 1e-3, 1e-2, 0.1, 1.0]; L8 = list("NDGCAHMO"); RARE = ["A", "G", "M"]; NB = 1000; BSEED = 12345
MODELS = ["kd768", "ssl960", "ssl768p", "teacher"]
M = pd.read_csv(EMB / "ids.csv"); tr = np.where(M.split == "train")[0]; te = np.where(M.split == "test")[0]
Y = M[L8].values.astype(int); grp = M.patient.values
nz = lambda E: (E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)).astype(np.float32)

def auc(y, s):
    npos = y.sum(); nneg = len(y) - npos
    if npos == 0 or nneg == 0: return np.nan
    r = rankdata(s); return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)

def probe_one(Xtr, ytr, g, Xqs):
    sc = StandardScaler().fit(Xtr); Xs = sc.transform(Xtr); cv = {}
    for C in GRID:
        a = []
        for i, j in StratifiedGroupKFold(3, shuffle=True, random_state=0).split(Xs, ytr, g):
            m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(Xs[i], ytr[i]); a.append(auc(ytr[j], m.decision_function(Xs[j])))
        cv[C] = float(np.nanmean(a))
    C = max(cv, key=cv.get); m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(Xs, ytr)
    return C, cv, [m.decision_function(sc.transform(x)) for x in Xqs]

def knn(Eb, yb, Eq, k=20):
    S = Eq @ Eb.T; idx = np.argsort(-S, 1)[:, :k]; w = np.exp(np.take_along_axis(S, idx, 1) / 0.07)
    return (w * yb[idx]).sum(1) / w.sum(1), yb[idx].mean(1)

def fewshot(E):
    out = {}
    for k in [1, 5, 10]:
        for l in RARE: out[(k, l)] = []
        for d in range(20):
            rng = np.random.RandomState(42 * 1000 + k * 100 + d)
            for l in RARE:
                j = L8.index(l); pos = tr[Y[tr, j] == 1]; neg = tr[Y[tr, j] == 0]
                sp = rng.choice(pos, k, replace=False); sn = rng.choice(neg, k, replace=False)
                m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(np.r_[E[sp], E[sn]], np.r_[np.ones(k), np.zeros(k)])
                out[(k, l)].append(m.decision_function(E[te]))
        for l in RARE: out[(k, l)] = np.stack(out[(k, l)])
    return out

T0 = time.time(); SC = {}; CSEL = {}
for mn in MODELS:
    E = nz(np.load(EMB / f"{mn}.npy")); Ed = nz(np.load(EMB / f"{mn}_degraded.npy"))
    jobs = [(l, "clean") for l in L8] + [(l, "degtrain") for l in L8]
    res = Parallel(n_jobs=8)(delayed(probe_one)((E if c == "clean" else Ed)[tr], Y[tr, L8.index(l)], grp[tr], [E[te], Ed[te]] if c == "clean" else [Ed[te]]) for l, c in jobs)
    s = {"clean": {}, "clean_to_deg": {}, "deg_to_deg": {}}
    for (l, c), (C, cv, outs) in zip(jobs, res):
        CSEL[f"{mn}/{l}/{c}"] = {"C": C, "cv": cv}
        if c == "clean": s["clean"][f"LP_{l}"] = outs[0][None]; s["clean_to_deg"][f"LP_{l}"] = outs[1][None]
        else: s["deg_to_deg"][f"LP_{l}"] = outs[0][None]
    for l in RARE:
        j = L8.index(l)
        for cond, Q in [("clean", E[te]), ("clean_to_deg", Ed[te])]:
            w, mj = knn(E[tr], Y[tr, j], Q); s[cond][f"kNNw_{l}"] = w[None]; s[cond][f"kNNmaj_{l}"] = mj[None]
    fs = fewshot(E)
    for (k, l), v in fs.items(): s["clean"][f"FS{k}_{l}"] = v
    SC[mn] = s; print(f"{mn} scored {time.time()-T0:.0f}s", flush=True)

def lab_of(name): return L8.index(name.split("_")[1])
GROUPS = {"LP_macro": [f"LP_{l}" for l in L8], "kNNw_mean": [f"kNNw_{l}" for l in RARE], "kNNmaj_mean": [f"kNNmaj_{l}" for l in RARE],
          **{f"FS{k}_mean": [f"FS{k}_{l}" for l in RARE] for k in [1, 5, 10]}}
def metrics(s, idx):
    o = {}
    for nm, v in s.items():
        y = Y[te][idx, lab_of(nm)]; o[nm] = float(np.nanmean([auc(y, row[idx]) for row in v]))
    for g, mem in GROUPS.items():
        if all(m in o for m in mem): o[g] = float(np.mean([o[m] for m in mem]))
    return o
ALL = np.arange(len(te)); point = {m: {c: metrics(SC[m][c], ALL) for c in SC[m]} for m in MODELS}
# cluster bootstrap over test patients
tp = M.patient.values[te]; up = np.unique(tp); pos_of = {p: np.where(tp == p)[0] for p in up}
rng = np.random.default_rng(BSEED); IDX = [np.concatenate([pos_of[p] for p in rng.choice(up, len(up))]) for _ in range(NB)]
def boot_model(m): return {c: [metrics(SC[m][c], I) for I in IDX] for c in SC[m]}
boot = dict(zip(MODELS, Parallel(n_jobs=4)(delayed(boot_model)(m) for m in MODELS)))
print(f"bootstrap done {time.time()-T0:.0f}s", flush=True)
def paired(b, a):
    out = {}
    for c in point[b]:
        out[c] = {}
        for k in point[b][c]:
            d = np.array([x[k] for x in boot[b][c]]) - np.array([x[k] for x in boot[a][c]]); d = d[np.isfinite(d)]; lo, hi = np.percentile(d, [2.5, 97.5])
            out[c][k] = {"A": point[a][c][k], "B": point[b][c][k], "delta": point[b][c][k] - point[a][c][k], "ci95": [float(lo), float(hi)], "p_delta_gt_0": float((d > 0).mean())}
    return out
ci = {m: {c: {k: [float(x) for x in np.nanpercentile([r[k] for r in boot[m][c]], [2.5, 97.5])] for k in point[m][c]} for c in point[m]} for m in MODELS}
cnt = lambda sp: {l: {"train": int(Y[tr, L8.index(l)].sum()), "test": int(Y[te, L8.index(l)].sum())} for l in L8}
R = {"counts": {"train_eyes": int(len(tr)), "test_eyes": int(len(te)), "train_patients": int(M.patient[tr].nunique()), "test_patients": int(len(up)), "positives": cnt(None)},
     "point": point, "ci95_single_model": ci, "C_selected": CSEL,
     "paired_KD768_minus_SSL960": paired("kd768", "ssl960"), "paired_KD768_minus_SSL768p": paired("kd768", "ssl768p"),
     "paired_Teacher_minus_KD768_reference": paired("teacher", "kd768"), "n_boot": NB, "boot_seed": BSEED, "boot_unit": "test patient (cluster)",
     "extract_meta": json.load(open(EMB / "extract_meta.json")), "secs": time.time() - T0}
json.dump(R, open(HERE / "results.json", "w"), indent=1, default=float)
for name in ["paired_KD768_minus_SSL960", "paired_KD768_minus_SSL768p", "paired_Teacher_minus_KD768_reference"]:
    print("\n==", name)
    for c, d in R[name].items():
        for k, v in d.items(): print(f"  {c:<13} {k:<12} A {v['A']:.3f}  B {v['B']:.3f}  d {v['delta']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]  P {v['p_delta_gt_0']:.3f}")
print("counts", json.dumps(R["counts"])); print("C:", json.dumps({k: v["C"] for k, v in CSEL.items()})); print("DONE", time.time() - T0)
