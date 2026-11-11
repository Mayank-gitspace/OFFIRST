"""IDRiD Disease Grading cross-domain eval, step 2: frozen embeddings -> linear probes / kNN -> paired bootstrap.
Probe (OFFIRST convention): L2-normalised embedding -> StandardScaler (fit on the 413 official train) -> LogisticRegression
(class_weight='balanced', lbfgs multinomial, max_iter 3000), C from the protocol grid [0.01, 0.1, 1.0] by 5-fold stratified CV
on train only (selection metric = macro OvR AUROC for DR5/DME3, AUROC for RDR); refit on all 413; score the 103 official test.
Tasks: DR grade 0-4 (macro OvR AUROC, QWK on argmax), DME risk 0-2 (macro OvR AUROC, accuracy), referable DR = grade>=2 (AUROC).
Conditions: clean->clean, clean-trained probe -> degraded test, degraded-train -> degraded-test.
kNN (RDR): k=20 cosine, exp(sim/0.07)-weighted vote (same scoring as paired_bootstrap.knn_scores), bank = clean train, clean and degraded queries.
Paired bootstrap: 1000 resamples of the 103 test images (seed 12345), one index vector shared by all models/metrics; classes absent
from a resample are dropped from the macro AUROC of that resample (both models identically)."""
import json, time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, cohen_kappa_score, accuracy_score
HERE = Path(__file__).resolve().parent; EMB = HERE / "embeddings"
GRID = [0.01, 0.1, 1.0]; NB = 1000; SEED = 12345
meta = pd.read_csv(EMB / "ids.csv"); tr = np.where(meta.split == "train")[0]; te = np.where(meta.split == "test")[0]
assert len(tr) == 413 and len(te) == 103
Y = {"DR5": meta.dr.values, "DME3": meta.dme.values, "RDR": (meta.dr.values >= 2).astype(int)}
MODELS = ["kd768", "ssl960", "ssl768p", "teacher"]
nz = lambda E: (E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)).astype(np.float32)

def auc_task(y, P, idx=None):
    if idx is not None: y, P = y[idx], P[idx]
    if P.ndim == 1: return roc_auc_score(y, P) if len(np.unique(y)) == 2 else np.nan
    a = [roc_auc_score(y == c, P[:, c]) for c in range(P.shape[1]) if 0 < (y == c).sum() < len(y)]
    return float(np.mean(a))

def fit_probe(X, y):
    sc = StandardScaler().fit(X); Xs = sc.transform(X); cv = {}
    for C in GRID:
        s = []
        for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, y):
            m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(Xs[a], y[a])
            p = m.predict_proba(Xs[b]); s.append(auc_task(y[b], p[:, 1] if p.shape[1] == 2 else p))
        cv[C] = float(np.nanmean(s))
    C = max(cv, key=cv.get); m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(Xs, y)
    return (lambda Xq: m.predict_proba(sc.transform(Xq))), C, cv

def knn(Eb, yb, Eq, k=20):
    S = Eq @ Eb.T; idx = np.argsort(-S, 1)[:, :k]; w = np.exp(np.take_along_axis(S, idx, 1) / 0.07)
    return (w * yb[idx]).sum(1) / w.sum(1)

def metrics(pred, idx=None):
    """pred: dict of test-set outputs for one model+condition -> metric dict (on resample idx)."""
    yD, yM, yR = Y["DR5"][te], Y["DME3"][te], Y["RDR"][te]
    I = np.arange(len(te)) if idx is None else idx
    o = {}
    if "DR5" in pred:
        P = pred["DR5"]; o["DR5_macroAUROC"] = auc_task(yD, P, I); o["DR5_QWK"] = cohen_kappa_score(yD[I], P[I].argmax(1), weights="quadratic")
    if "DME3" in pred:
        P = pred["DME3"]; o["DME3_macroAUROC"] = auc_task(yM, P, I); o["DME3_acc"] = accuracy_score(yM[I], P[I].argmax(1))
    if "RDR" in pred: o["RDR_AUROC"] = auc_task(yR, pred["RDR"], I)
    if "kNN_RDR" in pred: o["kNN_RDR_AUROC"] = auc_task(yR, pred["kNN_RDR"], I)
    return o

T0 = time.time(); PRED = {}; CSEL = {}
for mname in MODELS:
    E = nz(np.load(EMB / f"{mname}.npy")); Ed = nz(np.load(EMB / f"{mname}_degraded.npy"))
    pc, pcd, pdd = {}, {}, {}
    for task in ["DR5", "DME3", "RDR"]:
        y = Y[task]; f, C, cv = fit_probe(E[tr], y[tr]); fd, Cd, cvd = fit_probe(Ed[tr], y[tr])
        CSEL[f"{mname}/{task}"] = {"clean_C": C, "clean_cv": cv, "degtrain_C": Cd, "degtrain_cv": cvd}
        g = (lambda P: P[:, 1]) if task == "RDR" else (lambda P: P)
        pc[task] = g(f(E[te])); pcd[task] = g(f(Ed[te])); pdd[task] = g(fd(Ed[te]))
    pc["kNN_RDR"] = knn(E[tr], Y["RDR"][tr], E[te]); pcd["kNN_RDR"] = knn(E[tr], Y["RDR"][tr], Ed[te])
    PRED[mname] = {"clean": pc, "clean_to_deg": pcd, "deg_to_deg": pdd}
    print(f"{mname} probes done {time.time()-T0:.0f}s", flush=True)

point = {m: {c: metrics(PRED[m][c]) for c in PRED[m]} for m in MODELS}
rng = np.random.default_rng(SEED); IDX = [rng.integers(0, len(te), len(te)) for _ in range(NB)]
boot = {m: {c: [metrics(PRED[m][c], I) for I in IDX] for c in PRED[m]} for m in MODELS}
def paired(b, a):
    out = {}
    for c in point[b]:
        out[c] = {}
        for k in point[b][c]:
            d = np.array([x[k] for x in boot[b][c]]) - np.array([x[k] for x in boot[a][c]]); d = d[np.isfinite(d)]
            lo, hi = np.percentile(d, [2.5, 97.5])
            out[c][k] = {"A": point[a][c][k], "B": point[b][c][k], "delta": point[b][c][k] - point[a][c][k], "ci95": [float(lo), float(hi)],
                         "p_delta_gt_0": float((d > 0).mean()), "n_boot_valid": int(len(d))}
    return out
def ci(m):
    return {c: {k: [float(x) for x in np.percentile([r[k] for r in boot[m][c]], [2.5, 97.5])] for k in point[m][c]} for c in point[m]}
res = {"data": {"source": "kaggle aaryapatel98/indian-diabetic-retinopathy-image-dataset (mirror of the official IDRiD release, B. Disease Grading)",
                "n_train": int(len(tr)), "n_test": int(len(te)),
                "train_DR": meta.dr[tr].value_counts().sort_index().to_dict(), "test_DR": meta.dr[te].value_counts().sort_index().to_dict(),
                "train_DME": meta.dme[tr].value_counts().sort_index().to_dict(), "test_DME": meta.dme[te].value_counts().sort_index().to_dict(),
                "test_RDR_pos": int(Y["RDR"][te].sum()), "train_RDR_pos": int(Y["RDR"][tr].sum())},
       "point": point, "ci95_single_model": {m: ci(m) for m in MODELS}, "C_selected": CSEL,
       "paired_KD768_minus_SSL960": paired("kd768", "ssl960"), "paired_KD768_minus_SSL768p": paired("kd768", "ssl768p"),
       "paired_Teacher_minus_KD768_reference": paired("teacher", "kd768"),
       "extract_meta": json.load(open(EMB / "extract_meta.json")), "n_boot": NB, "boot_seed": SEED, "secs": time.time() - T0}
json.dump(res, open(HERE / "results.json", "w"), indent=1, default=lambda x: int(x) if isinstance(x, np.integer) else float(x))
for name in ["paired_KD768_minus_SSL960", "paired_KD768_minus_SSL768p", "paired_Teacher_minus_KD768_reference"]:
    print("\n==", name, "(A -> B)")
    for c, d in res[name].items():
        for k, v in d.items(): print(f"  {c:<13} {k:<16} A {v['A']:.3f}  B {v['B']:.3f}  d {v['delta']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]  P(d>0) {v['p_delta_gt_0']:.3f}")
print("C selected:", json.dumps({k: (v["clean_C"], v["degtrain_C"]) for k, v in CSEL.items()}))
print("DONE", time.time() - T0)
