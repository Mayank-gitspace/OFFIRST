"""ODIR-5K per-eye labels from the per-eye diagnostic keywords (data.xlsx, official training set, 3,500 patients / 7,000 eyes).
Mapping (validated below against the official PATIENT-level N..O columns: patient label = OR over eyes, N = both eyes normal):"""
import re, json, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path("/workspace/odir5k/ODIR-5K/ODIR-5K"); OUT = Path(__file__).resolve().parent
QUALITY = {"lens dust", "low image quality", "no fundus image", "image offset", "anterior segment image", "optic disk photographically invisible"}
def kw_label(k):
    if k == "normal fundus": return "N"
    if "hypertensive" in k: return "H"
    if "glaucoma" in k: return "G"                       # glaucoma, suspected glaucoma
    if "cataract" in k: return "C"                       # cataract, suspected cataract
    if "age-related macular degeneration" in k: return "A"   # dry / wet / unspecified AMD (drusen -> O, see validation)
    if k in ("pathological myopia", "myopia retinopathy", "myopic retinopathy", "myopic maculopathy"): return "M"
    if "proliferative retinopathy" in k or "diabetic retinopathy" in k: return "D"   # mild/moderate/severe NPDR, PDR, DR, suspected DR
    return "O"
L8 = list("NDGCAHMO")
d = pd.read_excel(ROOT / "data.xlsx"); rows = []; recon = []
for _, r in d.iterrows():
    eyes = {}
    for side in ("Left", "Right"):
        kws = [k.strip() for k in re.split("[,，]", str(r[f"{side}-Diagnostic Keywords"])) if k.strip()]
        q = [k for k in kws if k in QUALITY]; dx = [k for k in kws if k not in QUALITY]
        lab = {l: 0 for l in L8}
        for k in dx: lab[kw_label(k)] = 1
        if lab["N"] and sum(lab.values()) > 1: lab["N"] = 0          # never happens in practice; safety
        eyes[side] = lab
        rows.append({"patient": int(r.ID), "eye": side.lower(), "file": r[f"{side}-Fundus"], "keywords": "|".join(kws), "quality_kw": "|".join(q),
                     "n_dx_kw": len(dx), "age": r["Patient Age"], "sex": r["Patient Sex"], **lab})
    pat = {l: int(eyes["Left"][l] or eyes["Right"][l]) for l in L8 if l != "N"}
    pat["N"] = int(eyes["Left"]["N"] and eyes["Right"]["N"]) if not any(pat.values()) else 0
    recon.append({l: (pat[l], int(r[l])) for l in L8})
agree = {l: float(np.mean([x[l][0] == x[l][1] for x in recon])) for l in L8}
mism = {l: int(sum(x[l][0] != x[l][1] for x in recon)) for l in L8}
E = pd.DataFrame(rows)
E["exclude_quality"] = E.quality_kw != ""
E["exclude_no_dx"] = E.n_dx_kw == 0
print("patient-level reconstruction agreement:", json.dumps(agree)); print("mismatches:", mism)
print("eyes", len(E), "quality-keyword eyes", int(E.exclude_quality.sum()), "of which with no other dx kw", int((E.exclude_quality & E.exclude_no_dx).sum()))
E.to_csv(OUT / "eye_labels_all.csv", index=False)
json.dump({"quality_keywords": sorted(QUALITY), "patient_recon_agreement": agree, "patient_recon_mismatches": mism}, open(OUT / "label_build_meta.json", "w"), indent=1)
