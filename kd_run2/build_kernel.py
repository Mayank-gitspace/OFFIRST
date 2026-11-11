"""OFFIRST KD Run 2 kernel builder.
Prefix (imports, data guard, labels, split, 640 cache + FOV geometry, box table) = the MNV3 SSL kernel verbatim,
with our header/CONFIG (src/header_config.py). Reused from the SSL arm's src/tail.py: torch setup, row split,
views + Visuscout DEGRADE BLOCK, EvalDS, norm_gpu, GAPEncoder, std_metric, the whole eval section
(extract / probes / few-shot / kNN / diagnostics / evaluate). New: src/kd_tail.py.
Run: python3 build_kernel.py
"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
SSL = HERE.parent / "encoder_mnv3_ssl"
parent = (SSL / "kernels" / "full" / "team_offirst_encoder_byol_mnv3.py").read_text()
i0 = parent.index("import os, sys, json, time, math, random, copy, warnings")
i1 = parent.index("# ----------------------------------------------------------------------------- torch")
prefix = (HERE / "src" / "header_config.py").read_text() + parent[i0:i1]
prefix = prefix.replace('log(f"RUN_MODE={RUN_MODE}  backbone=', 'log(f"KD RUN 2  RUN_MODE={RUN_MODE}  backbone=')
tail = (SSL / "src" / "tail.py").read_text()
def cut(a, b):
    i = tail.index(a); j = tail.index(b, i); return tail[i:j]
segA = cut("# ----------------------------------------------------------------------------- torch", "# ----------------------------------------------------------------------------- model")
gap = cut("class GAPEncoder(nn.Module):", "class BYOLNet(nn.Module):")
stdm = cut("def std_metric(z):", "# ----------------------------------------------------------------------------- embeddings + eval")
evals = cut("# ----------------------------------------------------------------------------- embeddings + eval", "@torch.no_grad()\ndef extract_region")
evals += cut("def select_score(rep):", "@torch.no_grad()\ndef extract_region")[:0]
src = prefix + segA + "# ----------------------------------------------------------------------------- model (SSL arm)\n" + gap + stdm + evals + (HERE / "src" / "kd_tail.py").read_text()
for mode in ["smoke", "full"]:
    d = HERE / "kernels" / mode; d.mkdir(parents=True, exist_ok=True)
    (d / "team_offirst_kd_run2.py").write_text(src.replace('RUN_MODE = "__RUN_MODE__"', f'RUN_MODE = "{mode}"'))
    kid = "teamoffirst/team-offirst-kd-run2" + ("-smoke" if mode == "smoke" else "")
    meta = {"id": kid, "title": kid.split("/")[1], "code_file": "team_offirst_kd_run2.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": ["awsaf49/rfmid-train-dataset", "teamoffirst/offirst-kd-run2-weights"], "competition_sources": [],
            "kernel_sources": [], "machine_shape": "NvidiaTeslaT4"}
    (d / "kernel-metadata.json").write_text(json.dumps(meta))
print("built", len(src.splitlines()), "lines")
