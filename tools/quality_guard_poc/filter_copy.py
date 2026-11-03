"""Copy only the Day-1 subset images from an EyePACS dump into ImageFolder layout.
Usage:
  python filter_copy.py /path/to/eyepacs_images data_ready --binary
Looks up names in data_subset/needed_images.txt and labels in data_subset/subset_{train,val}.csv
"""
import argparse, csv, shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("imgs"); ap.add_argument("out"); ap.add_argument("--binary", action="store_true")
a = ap.parse_args()
names = {0: "good", 1: "good" if a.binary else "usable", 2: "reject"}
needed = set(Path("data_subset/needed_images.txt").read_text().split())
have = {}
for p in Path(a.imgs).rglob("*"):
    if p.suffix.lower() in {".jpeg", ".jpg"}:
        have[p.name] = p
n = miss = 0
for split, csv_name in [("train", "subset_train.csv"), ("val", "subset_val.csv")]:
    for r in csv.DictReader(open(f"data_subset/{csv_name}")):
        if r["image"] not in needed: continue
        src = have.get(r["image"])
        if not src: miss += 1; continue
        d = Path(a.out) / split / names[int(r["quality"])]
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, d / src.name); n += 1
print(f"copied {n}, missing {miss}")
