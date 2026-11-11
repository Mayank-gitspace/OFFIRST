#!/usr/bin/env python3
"""Per-sample ODIR-5K test outputs: linear-probe probabilities + top-3 train neighbours.

Reuses evaluate.py's exact patient split (embeddings/ids.csv) and the C values
already selected in results.json['C_selected'] under the clean condition.
Encoders stay frozen; only StandardScaler + LogisticRegression are fit on train.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path("kd_run2/odir5k_eval")
EMB = HERE / "embeddings"
OUT = HERE / "samples"
KSRC_PATH = Path("kd_run2/kernels/full/team_offirst_kd_run2.py")
CACHE_PATH = Path("/workspace/odir5k/cache_640.npy")
L8 = list("NDGCAHMO")
LNAMES = {
    "N": "Normal", "D": "DR", "G": "Glaucoma", "C": "Cataract",
    "A": "AMD", "H": "Hypertension", "M": "Myopia", "O": "Other",
}
MODELS = ["kd768", "ssl960", "teacher"]
MNAME = {"kd768": "KD 768", "ssl960": "SSL 960", "teacher": "Teacher"}
SEED = 7


def l2n(E):
    return (E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def pos_letters(row):
    return "".join(l for l in L8 if int(row[l]) == 1) or "—"


def pos_pretty(row):
    letters = [l for l in L8 if int(row[l]) == 1]
    if not letters:
        return "—"
    return "+".join(letters) + " (" + ", ".join(LNAMES[l] for l in letters) + ")"


def kw_text(s):
    if pd.isna(s) or str(s).strip() == "":
        return "—"
    return str(s).replace("|", "; ")


def load_degrader(geom):
    KSRC = KSRC_PATH.read_text()
    CFG = yaml.safe_load(KSRC[KSRC.index('CONFIG_YAML = r"""') + 18:KSRC.index('"""', KSRC.index('CONFIG_YAML = r"""') + 18)])
    CS = CFG["data"]["cache_size"]
    NS = {"np": np, "cv2": cv2, "math": math, "CS": CS, "CFG": CFG, "GEOM": geom}
    i = KSRC.index("# >>> DEGRADE BLOCK")
    j = KSRC.index("# <<< DEGRADE BLOCK")
    exec(KSRC[i:j], NS)
    return NS


def select_samples(M):
    te = np.where(M.split == "test")[0]
    disease = te[(M.loc[te, list("AGMD")].values.astype(int).sum(1) > 0)]
    # Record what a purely random draw would have produced.
    rng_pure = np.random.RandomState(SEED)
    pure = rng_pure.choice(te, 5, replace=False)
    n_pure = int((M.loc[pure, list("AGMD")].values.astype(int).sum(1) > 0).sum())
    if n_pure >= 2:
        return pure, "purely random", (
            f"Purely random draw of 5 test eyes with seed {SEED} already included "
            f"{n_pure} eyes with a disease label among A/G/M/D, so that draw was used."
        )
    # Preferred: 2 from A/G/M/D-positive test eyes, 3 from the remaining test eyes.
    rng = np.random.RandomState(SEED)
    d2 = rng.choice(disease, 2, replace=False)
    rest = np.setdiff1d(te, d2)
    r3 = rng.choice(rest, 3, replace=False)
    chosen = np.concatenate([d2, r3])
    n_pref = int((M.loc[chosen, list("AGMD")].values.astype(int).sum(1) > 0).sum())
    pure_files = ", ".join(
        f"{M.file.iloc[i]} ({pos_letters(M.iloc[i])})" for i in pure
    )
    note = (
        f"Not purely random. A purely random draw of 5 test eyes with seed {SEED} "
        f"included only {n_pure} eye(s) with a disease label among A/G/M/D "
        f"[{pure_files}]. With the same seed, 2 eyes were therefore drawn from the "
        f"A/G/M/D-positive test subset and 3 from the remaining test eyes "
        f"(final count with A/G/M/D: {n_pref})."
    )
    return chosen, "preferred (≥2 with A/G/M/D)", note


def fit_probes(E, Y, tr, Csel, model):
    """Return dict label -> (scaler, clf) trained on clean train embeddings."""
    probes = {}
    Xtr = E[tr]
    for l in L8:
        C = float(Csel[f"{model}/{l}/clean"]["C"])
        y = Y[tr, L8.index(l)]
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
        clf.fit(sc.transform(Xtr), y)
        probes[l] = (sc, clf)
    return probes


def predict_probs(probes, X):
    """(n, 8) probabilities, column order L8."""
    P = np.zeros((len(X), 8), dtype=np.float64)
    for j, l in enumerate(L8):
        sc, clf = probes[l]
        P[:, j] = clf.predict_proba(sc.transform(X))[:, 1]
    return P


def topk_neighbors(Eq, Etr, tr_idx, k=3):
    """Cosine on already L2-normalised embeddings. Returns (n, k) indices into full table, and sims."""
    S = Eq @ Etr.T  # (n, n_train)
    part = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
    # sort those k by similarity desc
    row = np.arange(len(Eq))[:, None]
    order = np.argsort(-S[row, part], axis=1)
    nn_rel = np.take_along_axis(part, order, axis=1)
    sims = np.take_along_axis(S, nn_rel, axis=1)
    nn_abs = tr_idx[nn_rel]
    return nn_abs, sims


def thumb(img, size=512):
    return cv2.resize(np.asarray(img), (size, size), interpolation=cv2.INTER_AREA)


def caption_wrap(s, width=28):
    words = s.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if len(trial) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def make_montage(clean, degraded, nn_imgs, nn_caps, title, path):
    fig = plt.figure(figsize=(16.5, 4.6), dpi=130)
    gs = GridSpec(1, 5, figure=fig, wspace=0.08, left=0.02, right=0.98, top=0.78, bottom=0.18)
    panels = [("Clean query", clean), ("Degraded query", degraded)]
    for img, cap in zip(nn_imgs, nn_caps):
        panels.append((cap, img))
    for i, (cap, img) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
        ax.set_xlabel(caption_wrap(cap, 32), fontsize=8, labelpad=6)
    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fmt_p(p):
    return f"{p:.3f}"


def md_escape(s):
    return str(s).replace("|", "\\|")


def main():
    OUT.mkdir(exist_ok=True)
    results = json.loads((HERE / "results.json").read_text())
    Csel = results["C_selected"]
    M = pd.read_csv(EMB / "ids.csv")
    Y = M[L8].values.astype(int)
    tr = np.where(M.split == "train")[0]
    chosen, mode, note = select_samples(M)
    n_disease = int((M.loc[chosen, list("AGMD")].values.astype(int).sum(1) > 0).sum())
    print("selection mode:", mode)
    print("indices:", chosen.tolist())
    print(M.iloc[chosen][["patient", "eye", "file", "keywords", *L8]].to_string())

    # Load embeddings (L2-normalise, matching evaluate.py)
    E = {mn: l2n(np.load(EMB / f"{mn}.npy")) for mn in MODELS}
    Ed = {mn: l2n(np.load(EMB / f"{mn}_degraded.npy")) for mn in ["kd768", "ssl960"]}

    probes = {}
    for mn in MODELS:
        print(f"fitting probes {mn} ...", flush=True)
        probes[mn] = fit_probes(E[mn], Y, tr, Csel, mn)

    # Probabilities for the 5 clean queries; degraded for KD and SSL
    P_clean = {mn: predict_probs(probes[mn], E[mn][chosen]) for mn in MODELS}
    P_deg = {mn: predict_probs(probes[mn], Ed[mn][chosen]) for mn in ["kd768", "ssl960"]}

    NN = {}
    SIM = {}
    for mn in MODELS:
        nn, sm = topk_neighbors(E[mn][chosen], E[mn][tr], tr, k=3)
        NN[f"{mn}_clean"] = nn
        SIM[f"{mn}_clean"] = sm
    for mn in ["kd768", "ssl960"]:
        nn, sm = topk_neighbors(Ed[mn][chosen], E[mn][tr], tr, k=3)
        NN[f"{mn}_deg"] = nn
        SIM[f"{mn}_deg"] = sm

    # Images + degrader (same kernel block / row seed as extract.py)
    geom = np.load(EMB / "geom.npy")
    NS = load_degrader(geom)
    cache = np.load(CACHE_PATH, mmap_mode="r")
    SS = 512

    png_paths = []
    # Gather all unique rows we need (queries + KD clean neighbours)
    need = set(int(i) for i in chosen)
    for r in range(len(chosen)):
        for j in range(3):
            need.add(int(NN["kd768_clean"][r, j]))
    crops = {i: np.asarray(cache[i]) for i in need}

    for s, idx in enumerate(chosen):
        idx = int(idx)
        row = M.iloc[idx]
        clean = thumb(crops[idx], SS)
        degraded = NS["degrade_eval"](crops[idx], idx, SS)
        nn_imgs, nn_caps = [], []
        for j in range(3):
            ni = int(NN["kd768_clean"][s, j])
            nr = M.iloc[ni]
            nn_imgs.append(thumb(crops[ni], SS))
            nn_caps.append(
                f"KD NN{j+1}  cos={SIM['kd768_clean'][s, j]:.3f}  {pos_letters(nr)}\n"
                f"{nr.file}  {kw_text(nr.keywords)}"
            )
        title = (
            f"{row.file}   patient {int(row.patient)} {row.eye}   "
            f"true {pos_pretty(row)}   keywords: {kw_text(row.keywords)}"
        )
        png = OUT / f"sample_{s+1:02d}_{Path(str(row.file)).stem}.png"
        make_montage(clean, degraded, nn_imgs, nn_caps, title, png)
        png_paths.append(png)
        print("wrote", png, flush=True)

    # ---- markdown report ----
    lines = []
    lines.append("# ODIR-5K per-sample probe outputs")
    lines.append("")
    lines.append(f"**Selection:** {mode} (seed {SEED}). {note}")
    lines.append("")
    lines.append(
        "Protocol matches `evaluate.py`: frozen encoders, L2-normalised embeddings, "
        "patient-level 70/30 split from `embeddings/ids.csv`, `StandardScaler` fit on the "
        "train split, one-vs-rest `LogisticRegression(class_weight='balanced', max_iter=3000)` "
        "with the **clean-condition** `C` values already selected in `results.json['C_selected']`. "
        "Predicted values are class probabilities (`predict_proba`). Neighbours are the top-3 "
        "train eyes by cosine similarity on L2-normalised embeddings (clean train bank, including "
        "when the query is degraded). Encoders were not retrained. Degraded thumbnails use the "
        "kernel `degrade_eval` from `extract.py` (row index = `ids.csv` row, size 512)."
    )
    lines.append("")
    lines.append("C (clean) used:")
    lines.append("")
    lines.append("| Model | " + " | ".join(L8) + " |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for mn in MODELS:
        cs = [str(Csel[f"{mn}/{l}/clean"]["C"]) for l in L8]
        lines.append(f"| {MNAME[mn]} | " + " | ".join(cs) + " |")
    lines.append("")

    def prob_table(models, Pmap):
        hdr = "| Label | True | " + " | ".join(MNAME[m] for m in models) + " |"
        sep = "|---|---:|" + "|".join(["---:"] * len(models)) + "|"
        out = [hdr, sep]
        true = Y[idx]
        for j, l in enumerate(L8):
            cells = [f"{l} ({LNAMES[l]})", str(int(true[j]))]
            for m in models:
                p = Pmap[m][s, j]
                cell = fmt_p(p)
                if p >= 0.5:
                    cell = f"**{cell}**"
                cells.append(cell)
            out.append("| " + " | ".join(cells) + " |")
        return "\n".join(out)

    def nn_block(mn, cond):
        key = f"{mn}_{cond}"
        out = [f"**{MNAME[mn]}**", "", "| Rank | Train neighbour | Cosine | Labels | Keywords |",
               "|---:|---|---:|---|---|"]
        for j in range(3):
            ni = int(NN[key][s, j])
            nr = M.iloc[ni]
            out.append(
                f"| {j+1} | {nr.file} (patient {int(nr.patient)} {nr.eye}) | "
                f"{SIM[key][s, j]:.3f} | {pos_letters(nr)} | {md_escape(kw_text(nr.keywords))} |"
            )
        return "\n".join(out)

    for s, idx in enumerate(chosen):
        idx = int(idx)
        row = M.iloc[idx]
        true_vec = ", ".join(f"{l}={int(row[l])}" for l in L8)
        lines.append(f"## Sample {s+1}: {row.file}")
        lines.append("")
        lines.append(f"- Patient {int(row.patient)}, {row.eye} eye, age {row.age}, {row.sex}, test split")
        lines.append(f"- True labels: `{true_vec}`  →  **{pos_pretty(row)}**")
        lines.append(f"- Original ODIR diagnostic keywords: {kw_text(row.keywords)}")
        lines.append(f"- Montage: `{png_paths[s]}`")
        lines.append("")
        lines.append("### Clean query — predicted probabilities")
        lines.append("")
        lines.append(prob_table(MODELS, P_clean))
        lines.append("")
        lines.append("### Clean query — top-3 train neighbours (cosine)")
        lines.append("")
        for mn in MODELS:
            lines.append(nn_block(mn, "clean"))
            lines.append("")
        lines.append("### Degraded query — predicted probabilities (clean-trained probe)")
        lines.append("")
        lines.append(prob_table(["kd768", "ssl960"], P_deg))
        lines.append("")
        lines.append("### Degraded query — top-3 train neighbours (clean train bank)")
        lines.append("")
        for mn in ["kd768", "ssl960"]:
            lines.append(nn_block(mn, "deg"))
            lines.append("")

    lines.append("## PNG paths")
    lines.append("")
    for p in png_paths:
        lines.append(f"- `{p}`")
    lines.append("")

    md_path = OUT / "samples.md"
    md_path.write_text("\n".join(lines))
    print("wrote", md_path)
    # compact stdout summary for the parent
    print("\n========== COMPACT TABLES ==========\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
