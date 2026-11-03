"""Tune per-class inference.min_score for a (v5) checkpoint WITHOUT touching evaluation-only data.

Data used (and nothing else):
  * healthy : RFMiD TRAINING-split normals listed in data_splits/rfmid_thresh_tune.txt
              (Disease_Risk=0, disjoint from the 50 train negatives and 10 FP-monitor negatives)
  * DR      : IDRiD SEGMENTATION subset, official test images (IDRiD_55..81) with pixel GT
Never used: IDRiD Disease-Grading set (IDRiD_001..516[test]), RFMiD validation/test sets.
The script refuses such paths / ids.

Rule: for each class, min_score = smallest grid value where the fraction of healthy images
whose top blob of that class survives is <= --max_healthy_rate. DR pick rate / hit rate are
reported at that value (hit = a GT pixel of the same class inside the crop around the pick).

Usage:
  python tune_thresholds.py --ckpt ckpt/best.pth --rfmid_dir <RFMiD Training_Set/Training> \
      --idrid_root <IDRiD seg root> --out thresholds_v5.json
"""
import argparse, json, re, csv
from pathlib import Path
import cv2, numpy as np, torch
from tqdm import tqdm
from common import load_config, preprocess_from_cfg, normalize, sliding_window_predict, preprocess_mask
from inference import load_model, sync_preprocessing, find_candidates
from fov_mask import apply_fov_mask, filter_candidates
from train import list_idrid, read_binary_mask

HERE = Path(__file__).resolve().parent
SPLITS = HERE.parent / "data_splits"
FORBIDDEN_PATH = re.compile(r"(Disease Grading|idrid_grade|Imagenes|Test_Set|Testing|Validation_Set|Evaluation|RFMiD_Testing|RFMiD_Validation)", re.I)
GRADING_ID = re.compile(r"^IDRiD_\d{3}")          # grading set ids have 3 digits (IDRiD_001, IDRiD_001test)


def guard_path(p):
    if FORBIDDEN_PATH.search(str(p)):
        raise SystemExit(f"Refusing evaluation-only / grading path: {p}")


def healthy_items(rfmid_dir, ids, labels_csv=None):
    guard_path(rfmid_dir)
    if labels_csv:
        guard_path(labels_csv)
        with open(labels_csv, encoding="utf-8-sig") as f:
            rows = {r["ID"]: r for r in csv.DictReader(f)}
        bad = [i for i in ids if i not in rows or rows[i]["Disease_Risk"] != "0"]
        if bad:
            raise SystemExit(f"Not RFMiD-train normals: {bad[:10]}")
    out = []
    for i in ids:
        if GRADING_ID.match(i) or i.startswith("IDRiD"):
            raise SystemExit(f"Refusing IDRiD id in healthy list: {i}")
        p = Path(rfmid_dir) / f"{i}.png"
        if not p.exists():
            raise SystemExit(f"Missing {p}")
        out.append({"id": f"rfmid_{i}", "image": p, "masks": None})
    return out


def top_per_class(model, item, classes, cfg, dev, amp, d, inf):
    rgb = cv2.cvtColor(cv2.imread(str(item["image"])), cv2.COLOR_BGR2RGB)
    img_r, meta = preprocess_from_cfg(rgb, d)
    gt = None
    if item["masks"] is not None:
        m = np.stack([read_binary_mask(item["masks"].get(c), rgb.shape[:2]) for c in classes], -1)
        gt = preprocess_mask(m, meta, img_r.shape[:2])
    probs = sliding_window_predict(model, normalize(img_r), dev, inf["tile"], inf["overlap"], amp)
    if inf.get("use_fov_mask", True):
        probs, fov = apply_fov_mask(probs, img_r, inf)
    cands = find_candidates(probs, classes, cfg)
    if inf.get("use_fov_mask", True):
        cands = filter_candidates(cands, fov["safe"])
    H, W = img_r.shape[:2]; s = inf["crop_size"] // 2
    res = {}
    for ci, c in enumerate(classes):
        xs = [x for x in cands if x["lesion_type"] == c]
        if not xs:
            res[c] = None; continue
        b = max(xs, key=lambda x: x["confidence"])
        r = {"score": b["confidence"]}
        if gt is not None:
            x0, y0 = int(max(0, b["cx"] - s)), int(max(0, b["cy"] - s))
            r["hit"] = bool(gt[y0:y0 + 2 * s, x0:x0 + 2 * s, ci].any())
        res[c] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rfmid_dir", required=True, help="RFMiD TRAINING images folder (<id>.png)")
    ap.add_argument("--rfmid_labels", default=None, help="RFMiD_Training_Labels.csv (verifies Disease_Risk=0)")
    ap.add_argument("--tune_list", default=str(SPLITS / "rfmid_thresh_tune.txt"))
    ap.add_argument("--idrid_root", required=True, help="IDRiD SEGMENTATION subset root")
    ap.add_argument("--max_healthy_rate", type=float, default=0.10)
    ap.add_argument("--limit_healthy", type=int, default=None)
    ap.add_argument("--cache", default=None, help="json cache of per-image top scores")
    ap.add_argument("--out", default="thresholds_v5.json")
    a = ap.parse_args()
    for p in (a.idrid_root, a.rfmid_dir, a.tune_list) + ((a.rfmid_labels,) if a.rfmid_labels else ()):
        guard_path(p)

    cfg = load_config(a.config)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"
    sync_preprocessing(cfg, a.ckpt, dev)
    model, classes = load_model(a.ckpt, cfg, dev)
    d, inf = cfg["data"], cfg["inference"]

    ids = [l.strip() for l in open(a.tune_list) if l.strip()]
    for f in ("rfmid_neg_train.txt", "rfmid_neg_valfp.txt"):
        if (SPLITS / f).exists():
            used = {l.strip() for l in open(SPLITS / f) if l.strip()}
            assert not used & set(ids), f"tune list overlaps {f}"
    if a.limit_healthy:
        ids = ids[:a.limit_healthy]
    healthy = healthy_items(a.rfmid_dir, ids, a.rfmid_labels)
    dr = list_idrid(a.idrid_root, "test", classes)
    for it in dr:
        if GRADING_ID.match(it["id"]):
            raise SystemExit(f"Grading id in DR list: {it['id']}")

    cache = json.load(open(a.cache)) if a.cache and Path(a.cache).exists() else {}
    for group, items in (("healthy", healthy), ("dr", dr)):
        for it in tqdm(items, desc=group):
            k = f"{group}/{it['id']}"
            if k not in cache:
                cache[k] = top_per_class(model, it, classes, cfg, dev, amp, d, inf)
                if a.cache:
                    json.dump(cache, open(a.cache, "w"))

    grid = np.round(np.arange(0.50, 0.9951, 0.005), 3)
    H = [v for k, v in cache.items() if k.startswith("healthy/")]
    D = [v for k, v in cache.items() if k.startswith("dr/")]
    out = {"ckpt": str(a.ckpt), "n_healthy": len(H), "n_dr": len(D),
           "max_healthy_rate": a.max_healthy_rate, "min_score": {}, "report": {}}
    for c in classes:
        hs = np.array([v[c]["score"] if v[c] else 0.0 for v in H])
        chosen = next((float(t) for t in grid if (hs >= t).mean() <= a.max_healthy_rate), float(grid[-1]))
        picks = [v[c] for v in D if v[c] and v[c]["score"] >= chosen]
        out["min_score"][c] = chosen
        out["report"][c] = {"healthy_pick_rate": round(float((hs >= chosen).mean()), 3),
                            "dr_pick_rate": round(len(picks) / max(1, len(D)), 3),
                            "dr_hit_rate": round(float(np.mean([p["hit"] for p in picks])), 3) if picks else None}
        print(c, chosen, out["report"][c])
    json.dump(out, open(a.out, "w"), indent=2)
    print("min_score for config.yaml inference:", json.dumps(out["min_score"]))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
