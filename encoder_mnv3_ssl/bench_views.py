"""CPU cost per image of the SSL view pipeline (2x512 + 4x224) with / without the Visuscout degrader, and of a degraded
eval copy. Single process, single thread (cv2.setNumThreads(0) as in the kernel). Box CPU, NOT Kaggle's CPU.
Run with the same TOB_* env as the local smoke (needs the 640 cache in TOB_TMP)."""
import os, time, json, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
src = (HERE / "kernels/smoke/team_offirst_encoder_byol_mnv3.py").read_text()
src = src[:src.index("# ----------------------------------------------------------------------------- model")]
g = {"__name__": "bench"}; exec(compile(src, "kernel", "exec"), g)
rows = g["SSL_ROWS"][:40]; out = {}
for name, en in [("views_with_degrade(default p)", True), ("views_no_degrade(v4-equivalent)", False)]:
    g["DG"]["enabled"] = en; t = time.time(); n = 0
    for r in rows:
        g["make_views"](np.asarray(g["CACHE"][r]), np.random.RandomState(int(r)), r, "anatomy_v4"); n += 1
    out[name] = (time.time() - t) / n * 1000
g["DG"]["enabled"] = True
for name, p in [("views_all_degraded(p=1 on g1+locals)", 1.0)]:
    g["DG"]["p_global1"] = p; g["DG"]["p_local"] = p; t = time.time()
    for r in rows: g["make_views"](np.asarray(g["CACHE"][r]), np.random.RandomState(int(r)), r, "anatomy_v4")
    out[name] = (time.time() - t) / len(rows) * 1000
t = time.time()
for r in rows: g["degrade_eval"](np.asarray(g["CACHE"][r]), r, 512)
out["degrade_eval_512"] = (time.time() - t) / len(rows) * 1000
import cv2
t = time.time()
for r in rows: cv2.resize(np.asarray(g["CACHE"][r]), (512, 512), interpolation=cv2.INTER_AREA)
out["clean_eval_resize_512"] = (time.time() - t) / len(rows) * 1000
out = {k + "_ms_per_image": round(v, 1) for k, v in out.items()}; out["n_images"] = len(rows); out["cpu"] = os.cpu_count()
print(json.dumps(out, indent=1)); json.dump(out, open(HERE / "localtest/bench_views.json", "w"), indent=1)
