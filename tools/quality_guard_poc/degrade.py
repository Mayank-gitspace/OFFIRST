"""Fallback data plan: turn good fundus photos (e.g. APTOS 2019) into labelled bad ones.
Each defect maps to a real capture error, which is the optics story:
  blur      -> patient moved / camera out of focus (motion or defocus PSF)
  dark      -> small undilated pupil, under-exposure
  glare     -> flash reflection off cornea / lens
  crop      -> camera not aligned, optic disc out of field
Usage: python degrade.py SRC_DIR OUT_DIR [--n-per-class 1500]
Writes OUT_DIR/{good,reject}/... plus labels.csv (file,label,defect).
"""
import argparse, csv, random
from pathlib import Path
import cv2
import numpy as np


def motion_blur(img, k=None):
    k = k or random.choice([9, 13, 17, 21])
    ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), random.uniform(0, 180), 1)
    ker = cv2.warpAffine(ker, M, (k, k)); ker /= ker.sum()
    return cv2.filter2D(img, -1, ker)

def defocus(img):
    return cv2.GaussianBlur(img, (0, 0), random.uniform(3, 7))

def dark(img):
    g = random.uniform(1.8, 3.0)  # gamma > 1 darkens
    lut = np.array([((i / 255.0) ** g) * 255 * random.uniform(0.5, 0.8) for i in range(256)], np.uint8)
    return cv2.LUT(img, lut)

def glare(img):
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    cx, cy = random.randint(w // 4, 3 * w // 4), random.randint(h // 4, 3 * h // 4)
    yy, xx = np.mgrid[:h, :w]
    spot = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (random.uniform(0.08, 0.18) * w) ** 2))
    out += spot[..., None] * 255 * random.uniform(0.8, 1.2)
    return np.clip(out, 0, 255).astype(np.uint8)

def crop_shift(img):
    h, w = img.shape[:2]
    dx = int(random.choice([-1, 1]) * random.uniform(0.3, 0.45) * w)
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h))

DEFECTS = {"motion_blur": motion_blur, "defocus": defocus, "dark": dark, "glare": glare, "crop": crop_shift}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--n-per-class", type=int, default=1500)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); random.seed(a.seed); np.random.seed(a.seed)
    files = sorted(p for p in Path(a.src).rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    random.shuffle(files)
    out = Path(a.out); (out / "good").mkdir(parents=True, exist_ok=True); (out / "reject").mkdir(exist_ok=True)
    rows = []
    for i, p in enumerate(files[: 2 * a.n_per_class]):
        img = cv2.resize(cv2.imread(str(p)), (a.size, a.size), interpolation=cv2.INTER_AREA)
        if i % 2 == 0:
            name = f"good/{p.stem}.png"; defect = "none"
        else:
            defect = random.choice(list(DEFECTS)); img = DEFECTS[defect](img)
            name = f"reject/{p.stem}_{defect}.png"
        cv2.imwrite(str(out / name), img); rows.append((name, name.split("/")[0], defect))
    with open(out / "labels.csv", "w", newline="") as f:
        csv.writer(f).writerows([("file", "label", "defect"), *rows])
    print(f"wrote {len(rows)} images to {out}")

if __name__ == "__main__":
    main()
