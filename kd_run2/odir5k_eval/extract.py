"""ODIR-5K step 2: patient-level stratified 70/30 split + frozen-encoder embeddings (clean + degraded) on the box CPU.
Re-uses the IDRiD extractor (which exec's the KD Run 2 kernel's crop_fundus / fov_geom / DEGRADE BLOCK / config) unchanged."""
import sys, json, time, numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "idrid_grading_eval"))
import extract as X   # IDRiD extractor: load_prep, build, embed, sanity_rfmid, l2n
IMG = Path("/workspace/odir5k/ODIR-5K/ODIR-5K/Training Images"); L8 = list("NDGCAHMO"); SEED = 42
def prep(f): c, g, sh = X.load_prep(IMG / f); return c, g, sh[:2]
if __name__ == "__main__":
    T0 = time.time(); EMB = HERE / "embeddings"; EMB.mkdir(exist_ok=True)
    E = pd.read_csv(HERE / "eye_labels_all.csv"); n_all = len(E)
    E = E[~E.exclude_quality].reset_index(drop=True)
    # patient-level multilabel-stratified 70/30 split on the kept eyes (patient label = OR over kept eyes)
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    P = E.groupby("patient")[L8].max().reset_index()
    tr_i, te_i = next(MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED).split(P[["patient"]], P[L8].values))
    test_p = set(P.patient.values[te_i]); E["split"] = np.where(E.patient.isin(test_p), "test", "train")
    assert E.groupby("patient").split.nunique().max() == 1
    E.to_csv(EMB / "ids.csv", index=False)
    print(f"eyes kept {len(E)}/{n_all}; patients {len(P)}; train eyes {int((E.split=='train').sum())} test eyes {int((E.split=='test').sum())}", flush=True)
    models = X.build()
    cache = np.lib.format.open_memmap("/workspace/odir5k/cache_640.npy", mode="w+", dtype=np.uint8, shape=(len(E), X.CS, X.CS, 3))
    geom = np.zeros((len(E), 5)); shp = np.zeros((len(E), 2), int)
    with Pool(6) as pool:
        for i, (c, g, s) in enumerate(pool.imap(prep, E.file.tolist(), chunksize=8)):
            cache[i] = c; geom[i] = g; shp[i] = s
            if (i + 1) % 1000 == 0: print(f"  cached {i+1} {time.time()-T0:.0f}s", flush=True)
    cache.flush(); np.save(EMB / "geom.npy", geom); np.save(EMB / "raw_hw.npy", shp)
    print(f"cache done {time.time()-T0:.0f}s", flush=True)
    for deg in (False, True):
        Z = X.embed(models, cache, geom, None, deg, bs=32)
        for k, v in Z.items(): np.save(EMB / f"{k}{'_degraded' if deg else ''}.npy", v)
        print(f"embedded degraded={deg} {time.time()-T0:.0f}s", flush=True)
    # near-duplicate check vs every RFMiD training image the encoders could have seen (kernel teacher + SSL embeddings)
    dup = {}
    for k, fn in [("teacher", "embeddings_teacher"), ("ssl960", "embeddings_ssl_ep40_960")]:
        R = np.load(X.OUTK / f"{fn}.npy"); Z = np.load(EMB / f"{k}.npy"); m = (Z @ R.T).max(1)
        dup[k] = {"max_cos_to_rfmid_max": float(m.max()), "p99": float(np.percentile(m, 99)), "n_ge_0.98": int((m >= 0.98).sum()), "n_ge_0.95": int((m >= 0.95).sum())}
    for i in [0, int(np.where(E.A == 1)[0][0])]:
        X.NS["GEOM"] = geom; img = np.asarray(cache[i])
        X.cv2.imwrite(str(HERE / f"example_{E.file[i]}"), X.cv2.cvtColor(np.concatenate([X.cv2.resize(img, (512, 512), interpolation=X.cv2.INTER_AREA), X.NS["degrade_eval"](img, i, 512)], 1), X.cv2.COLOR_RGB2BGR))
    json.dump({"n_eyes_all": n_all, "n_kept": len(E), "near_dup_vs_rfmid": dup, "secs": time.time() - T0,
               "raw_sizes_top": pd.Series([f"{w}x{h}" for h, w in shp]).value_counts().head(15).to_dict(), "n_distinct_sizes": int(len(set(map(tuple, shp))))},
              open(EMB / "extract_meta.json", "w"), indent=1)
    print("near-dup:", json.dumps(dup)); print("DONE", time.time() - T0)
