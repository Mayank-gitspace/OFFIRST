"""Before/after grid + stats table on 6 diverse RFMiD images (by brightness spread)."""
import cv2, numpy as np, glob, json, os
from degrade import degrade, DegradeParams
from stats import compute, SCALARS
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo")
paths = sorted(glob.glob("data/rfmid/**/*.png", recursive=True))
rng = np.random.default_rng(0); cand = list(rng.choice(paths, 40, replace=False))
br = [cv2.imread(p, 0)[::8, ::8].mean() for p in cand]; order = np.argsort(br)
pick = [cand[i] for i in order[np.linspace(0, len(cand) - 1, 6).astype(int)]]
rows, table = [], []
for i, p in enumerate(pick):
    im = cv2.imread(p); prm = DegradeParams(seed=i, leak_strength=0.6 if i in (2, 5) else 0.0)
    d = degrade(im, prm); sc, sd = compute(im), compute(d)
    t = 384; a = cv2.resize(im, (int(im.shape[1] * t / im.shape[0]), t)); b = cv2.resize(d, a.shape[1::-1])
    for img, lab in ((a, "clean"), (b, "degraded" + (" +leak" if prm.leak_strength else ""))): cv2.putText(img, f"{os.path.basename(p)} {lab}", (8, 24), 0, 0.6, (255, 255, 255), 2)
    rows.append(np.hstack([a, b])); table.append({"image": p, "leak": prm.leak_strength, "clean": {k: sc[k] for k in SCALARS}, "degraded": {k: sd[k] for k in SCALARS}})
W = max(r.shape[1] for r in rows); rows = [cv2.copyMakeBorder(r, 0, 0, 0, W - r.shape[1], 0) for r in rows]
cv2.imwrite(f"{OUT}/grid_clean_vs_degraded.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
json.dump(table, open(f"{OUT}/stats_clean_vs_degraded.json", "w"), indent=1)
with open(f"{OUT}/stats_table.md", "w") as f:
    f.write("Defaults are UNCALIBRATED. Values: clean -> degraded.\n\n| image | " + " | ".join(SCALARS) + " |\n|" + "---|" * (len(SCALARS) + 1) + "\n")
    for r in table: f.write(f"| {os.path.basename(r['image'])}{' (leak)' if r['leak'] else ''} | " + " | ".join(f"{r['clean'][k]:.3g}→{r['degraded'][k]:.3g}" for k in SCALARS) + " |\n")
print(open(f"{OUT}/stats_table.md").read())
