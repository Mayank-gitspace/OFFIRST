"""Sort EyePACS images into ImageFolder layout using EyeQ labels.
EyeQ repo: https://github.com/HzFu/EyeQ (label CSVs under data/). Check the CSV header first;
this assumes an image-name column and a quality column with 0=Good, 1=Usable, 2=Reject.
Usage: python prepare_eyeq.py LABELS.csv IMG_DIR OUT_DIR/train  [--binary]   (repeat for the test CSV -> OUT_DIR/val)
--binary merges Usable into Good, for a simpler Upload/Retake model.
"""
import argparse, csv, shutil
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("imgs"); ap.add_argument("out")
ap.add_argument("--binary", action="store_true"); a = ap.parse_args()
names = {0: "good", 1: "good" if a.binary else "usable", 2: "reject"}
rows = list(csv.DictReader(open(a.csv)))
hdr = rows[0].keys(); img_col = next(c for c in hdr if "image" in c.lower()); q_col = next(c for c in hdr if "quality" in c.lower())
have = {p.name: p for p in Path(a.imgs).rglob("*.jpeg")} | {p.name: p for p in Path(a.imgs).rglob("*.jpg")}
n = miss = 0
for r in rows:
    src = have.get(r[img_col]) or have.get(Path(r[img_col]).stem + ".jpeg")
    if not src: miss += 1; continue
    d = Path(a.out) / names[int(float(r[q_col]))]; d.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, d / src.name); n += 1
print(f"copied {n}, missing {miss} (fine if you only downloaded some EyePACS parts)")
