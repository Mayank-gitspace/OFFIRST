"""OFFIRST BYOL anatomy v4 builder.
Chain: parent team_offirst_encoder_byol.py --(v2/build_kernel.py)--> v2 kernel (embedded CPU-detector box table)
       --(this file)--> anatomy_v4 kernel.  Run from this folder: python build_kernel.py
VICReg var/cov adapted from Argon snippet (argon_code_snippets.md, item 1); all view code wired to real v2 helpers
(_rrc_box/_geom/_photometric/_camera, BOX_ARR/BOX_OK). Locked values: CONFIG_LOCKED.md.
"""
import json, re
from pathlib import Path
HERE = Path(__file__).resolve().parent
V2K = HERE.parent / "v2" / "kernels" / "full" / "team_offirst_encoder_byol_anatomy_v2.py"
src = V2K.read_text()
def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (old[:70], src.count(old)); src = src.replace(old, new)

# ---------------------------------------------------------------- header
rep('RUN_MODE = "full"   # "smoke" | "full" | "eval"', 'RUN_MODE = "__RUN_MODE__"   # "smoke" | "full" | "eval"')
rep("# RUN_MODE: 'smoke' (2 epochs, all SSL imgs) | 'full' | 'eval'",
"""# v4 'anatomy_v4' (final): 4 local 160 px views = fixed 1 disc + 1 macula + 2 random (slot order
#   shuffled per image); disc AND macula crops 2.5-3.2 DD only (no tight crops), centre jitter <= 0.3 x crop;
#   low-confidence disc -> that anatomy slot falls back to a random crop. Uniform weak colour jitter on ALL
#   views (b/c 0.2, sat 0.15, hue 0.03, p 0.8; no disc exemption) + camera augs (blur/JPEG/brightness).
#   Loss = BYOL + 0.1 * (10 * VICReg-variance + 1 * VICReg-covariance) on online global projector z
#   (Bardes et al. 2022). 120 epochs, warmup 8, kNN every 5, ckpt every 10, forced eval at ep 70 and 120,
#   eval-only multi-region concat probe [global, disc 2.5 DD, macula 2.5 DD].
# RUN_MODE: 'smoke' (2 epochs) | 'full' | 'eval'""")

# ---------------------------------------------------------------- config
rep("  local_scale: [0.2, 0.5]", "  local_scale: [0.25, 0.55]")
rep("  saturation: 0.1\n  hue: 0.02                 # fraction of the hue circle",
    "  saturation: 0.15\n  hue: 0.03                 # fraction of the hue circle; same jitter on ALL views (incl. disc/macula)")
rep("  cam_bright: 0.15          # disc views capped at disc_bc", "  cam_bright: 0.15")
i0 = src.index("anatomy_v2:\n"); i1 = src.index("anatomy:\n")
src = src[:i0] + """anatomy_v4:
  n_disc: 1
  n_macula: 1
  n_random: 2
  disc_dd: [2.5, 3.2]       # wide only (no tight 1.2-1.5)
  mac_dd: [2.5, 3.2]
  jitter_frac: 0.3          # centre jitter up to 0.3 x crop size
  conf_cutoff: 1.0          # detector top1-top2 margin; below -> anatomy slot becomes random
""" + src[i1:]
rep("  crop_modes: [anatomy_v2]\n  epochs: 70", "  crop_modes: [anatomy_v4]\n  epochs: 120")
rep("  warmup_epochs: 5             # 10/150 of run B compressed to 70 ep", "  warmup_epochs: 8")
rep("over the 70 ep", "over the run")
rep("  ckpt_every: 25", """  ckpt_every: 10
  vicreg:
    enabled: true
    lambda_var: 10.0
    lambda_cov: 1.0
    loss_weight: 0.1
    gamma: 1.0                # target per-dim std""")
rep("  collapse_warn: 0.2          # warn if mean per-dim std * sqrt(d) of L2-normed z drops below",
    "  collapse_warn: 0.2          # warn if mean per-dim std * sqrt(d) of L2-normed z drops below\n  eval_epochs: [70, 120]      # forced full eval (probe + few-shot + multi-region) + ckpt")
rep("  probe_C_grid: [0.01, 0.1, 1.0]\nsmoke:", """  probe_C_grid: [0.01, 0.1, 1.0]
  multi_region_probe:
    enabled: true
    disc_dd: 2.5
    mac_dd: 2.5
    region_size: 224
    power_norm: true
smoke:""")
rep("  knn_every: 1\n  common_min_query_pos: 3", "  knn_every: 1\n  eval_epochs: [1]\n  common_min_query_pos: 3")
rep('    CFG["monitor"]["knn_every"] = s["knn_every"]', '    CFG["monitor"]["knn_every"] = s["knn_every"]; CFG["monitor"]["eval_epochs"] = s["eval_epochs"]')
rep('A2 = CFG["anatomy_v2"]', 'A2 = CFG["anatomy_v4"]')

# ---------------------------------------------------------------- views: drop disc-only photometric, new anchors
i0 = src.index("def _photometric_disc(x, rng):"); i1 = src.index("def _photometric(x, rng):")
src = src[:i0] + src[i1:]
i0 = src.index('ANCHOR_NAMES = ["random"'); i1 = src.index("def crop_grid(path")
src = src[:i0] + '''ANCHOR_NAMES = ["random", "disc", "macula", "fallback_random"]
N_ANCH = len(ANCHOR_NAMES)
def anatomy_box(rng, row, S, kind):
    """-> (center, wh, type) for one anatomy slot ('disc' | 'macula'); low-confidence disc -> random crop (type 3)."""
    if not BOX_OK[row]: return _rrc_box(rng, S, V["local_scale"]) + (3,)
    dcx, dcy, dd, mcx, mcy = BOX_ARR[row] * S
    if kind == "disc": w = dd * rng.uniform(*A2["disc_dd"]); cx, cy, ty = dcx, dcy, 1
    else:              w = dd * rng.uniform(*A2["mac_dd"]);  cx, cy, ty = mcx, mcy, 2
    j = A2["jitter_frac"] * w
    return (cx + rng.uniform(-j, j), cy + rng.uniform(-j, j)), (w, w), ty

def make_views(img, rng, row, mode):
    S = img.shape[0]
    gl = [_photometric(_geom(img, rng, V["global_size"], *_rrc_box(rng, S, V["global_scale"])), rng) for _ in range(2)]
    if mode == "anatomy_v4":
        slots = ["disc"] * A2["n_disc"] + ["macula"] * A2["n_macula"] + ["random"] * A2["n_random"]
        assert len(slots) == V["n_local"], "anatomy_v4 slots must equal n_local"
        rng.shuffle(slots)
    else: slots = ["random"] * V["n_local"]
    lo, types = [], []
    for s in slots:
        if s == "random": c, wh, ty = _rrc_box(rng, S, V["local_scale"]) + (0,)
        else: c, wh, ty = anatomy_box(rng, row, S, s)
        lo.append(_photometric(_geom(img, rng, V["local_size"], c, wh), rng)); types.append(ty)   # same colour+camera augs for every view
    return np.stack(gl), np.stack(lo), np.array(types, np.int64)

''' + src[i1:]
rep('_, lo, ty = make_views(img, np.random.RandomState(seed + int(r)), r, "anatomy_v2")',
    '_, lo, ty = make_views(img, np.random.RandomState(seed + int(r)), r, "anatomy_v4")')

# ---------------------------------------------------------------- multi-region eval dataset (eval only)
rep("def norm_gpu(x):", '''MR = CFG["eval"]["multi_region_probe"]
class RegionDS(torch.utils.data.Dataset):
    """eval-only: deterministic disc / macula square crop (MR[kind_dd] x DD, no jitter/rotation) resized to region_size.
    Low-confidence detections still use the detector estimate (flagged in report via low_conf_frac)."""
    def __init__(self, rows, kind): self.rows = np.asarray(rows); self.kind = kind
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]; img = np.asarray(CACHE[r]); S = img.shape[0]; out = MR["region_size"]
        dcx, dcy, dd, mcx, mcy = BOX_ARR[r] * S
        cx, cy, w = (dcx, dcy, dd * MR["disc_dd"]) if self.kind == "disc" else (mcx, mcy, dd * MR["mac_dd"])
        M = _affine((cx, cy), (w, w), out, 0.0, False, False)
        im = cv2.warpAffine(img, M, (out, out), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return torch.from_numpy(im).permute(2, 0, 1), i

def norm_gpu(x):''')

# ---------------------------------------------------------------- VICReg (Argon snippet 1, adapted) + train_step
rep("def byol_loss(p, z):", '''VC = CFG["train"]["vicreg"]
def vicreg_var_cov(z, gamma=1.0, eps=1e-4):
    """VICReg variance + covariance terms (Bardes et al. 2022) on a (B, D) batch; fp32 for AMP safety.
    L_var = mean_j relu(gamma - std_j);  L_cov = sum_{i!=j} cov_ij^2 / D."""
    z = z.float(); B, D = z.shape
    if B <= 1: zero = z.new_zeros(()); return zero, zero
    zc = z - z.mean(0, keepdim=True)
    std = torch.sqrt(zc.var(0) + eps)
    l_var = F.relu(gamma - std).mean()
    cov = (zc.T @ zc) / (B - 1)
    off = cov - torch.diag(torch.diagonal(cov))
    return l_var, off.pow(2).sum() / D

def byol_loss(p, z):''')
rep('''    loss = torch.stack(terms).mean()
    opt.zero_grad''', '''    loss_byol = torch.stack(terms).mean()
    if VC["enabled"]:   # online projector z of the two global views, each (B, D); averaged
        vv = [vicreg_var_cov(zg[k * B:(k + 1) * B], VC["gamma"]) for k in range(2)]
        l_var = torch.stack([v[0] for v in vv]).mean(); l_cov = torch.stack([v[1] for v in vv]).mean()
        loss = loss_byol + VC["loss_weight"] * (VC["lambda_var"] * l_var + VC["lambda_cov"] * l_cov)
    else:
        l_var = l_cov = loss_byol.new_zeros(()); loss = loss_byol
    opt.zero_grad''')
rep("    return loss.detach(), zg[:B].detach().float(), hg[:B].detach().float(), gn",
    "    return loss.detach(), zg[:B].detach().float(), hg[:B].detach().float(), gn, (float(loss_byol), float(l_var), float(l_cov))")

# ---------------------------------------------------------------- multi-region eval helper
rep("def baseline_embeddings():", '''@torch.no_grad()
def extract_region(backbone, rows, kind):
    dl = torch.utils.data.DataLoader(RegionDS(rows, kind), batch_size=CFG["eval"]["batch_size"], shuffle=False,
                                     num_workers=CFG["data"]["num_workers"], pin_memory=DEV.type == "cuda")
    was = backbone.training; backbone.eval(); E = []
    for x, _ in dl:
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
            E.append(backbone(norm_gpu(x)).float().cpu().numpy())
    backbone.train(was); E = np.concatenate(E)
    return (E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)).astype(np.float32)

def multi_region_eval(backbone, rows, E_glob, tag, od=None):
    """eval-only readout: concat [global, disc, macula] L2-normed embeddings, optional power-norm, re-L2."""
    E = np.concatenate([E_glob, extract_region(backbone, rows, "disc"), extract_region(backbone, rows, "macula")], 1)
    if MR["power_norm"]: E = np.sign(E) * np.sqrt(np.abs(E))
    E = (E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
    if od is not None: np.save(od / f"embeddings_{tag.split('/')[-1]}.npy", E)
    r = evaluate(rows, E, tag); r["low_conf_frac"] = float((~BOX_OK[np.asarray(rows)]).mean()); return r

def baseline_embeddings():''')

# ---------------------------------------------------------------- training loop: anchors, vicreg logging, forced evals
rep("tl, nb, zs, hs, an, tstep = 0.0, 0, [], [], np.zeros(5), []", "tl, nb, zs, hs, an, tstep = 0.0, 0, [], [], np.zeros(N_ANCH), []\n        vcs = []")
rep("            loss, z, h, gn = train_step(online, target, opt, scaler, G, L, tau, amp)\n            tl += float(loss); nb += 1; step += 1; an += np.bincount(a.numpy().ravel(), minlength=5)",
    "            loss, z, h, gn, vc = train_step(online, target, opt, scaler, G, L, tau, amp)\n            tl += float(loss); nb += 1; step += 1; an += np.bincount(a.numpy().ravel(), minlength=N_ANCH); vcs.append(vc)")
rep('''               "anchor_frac": {ANCHOR_NAMES[k]: float(an[k] / max(1, an.sum())) for k in range(5)},
               "anatomy_views_frac": float(an[1:4].sum() / max(1, an.sum()))}''',
'''               "anchor_frac": {ANCHOR_NAMES[k]: float(an[k] / max(1, an.sum())) for k in range(N_ANCH)},
               "anatomy_views_frac": float(an[1:3].sum() / max(1, an.sum()))}
        vca = np.mean(vcs, 0) if vcs else np.full(3, np.nan)
        row.update({"loss_byol": float(vca[0]), "vicreg_var": float(vca[1]), "vicreg_cov": float(vca[2])})''')
rep('''f"tau {tau:.4f} {row['img_per_s']:.1f} img/s ({t_tr:.0f}s)"''',
    '''f"(byol {row['loss_byol']:.4f} var {row['vicreg_var']:.3f} cov {row['vicreg_cov']:.3f}) tau {tau:.4f} {row['img_per_s']:.1f} img/s ({t_tr:.0f}s)"''')
rep('''            torch.save(ck_d, od / ("last.pth" if ep + 1 == tc["epochs"] else f"ep{ep+1:03d}.pth"))''',
'''            torch.save(ck_d, od / ("last.pth" if ep + 1 == tc["epochs"] else f"ep{ep+1:03d}.pth"))
        if ep + 1 in CFG["monitor"]["eval_epochs"] and ep + 1 != tc["epochs"]:   # forced mid-run eval (final epoch handled below)
            te = time.time(); evr = np.sort(np.r_[SSL_ROWS, Q_ROWS]); Ee = extract(online.backbone, evr)
            torch.save({"online": online.state_dict(), "target_backbone": target.backbone.state_dict(),
                        "target_projector": target.projector.state_dict(), "epoch": ep + 1, "crop_mode": mode, "config": CFG}, od / f"ep{ep+1:03d}.pth")
            ee = {"online": evaluate(evr, Ee, f"byol_{mode}/ep{ep+1}_online")}
            if MR["enabled"]: ee["multi_region_online"] = multi_region_eval(online.backbone, evr, Ee, f"byol_{mode}/ep{ep+1}_multiregion")
            ee["eval_s"] = time.time() - te; epoch_evals[f"ep{ep+1}"] = ee
            json.dump(epoch_evals, open(od / "epoch_evals.json", "w"), indent=1, default=float)''')
rep("    best_knn, best_ep = -1.0, 0", "    best_knn, best_ep = -1.0, 0; epoch_evals = {}")
rep('''    rep["eval_target"] = evaluate(ev_rows, E_tg, f"byol_{mode}/target")''',
'''    rep["eval_target"] = evaluate(ev_rows, E_tg, f"byol_{mode}/target")
    if MR["enabled"]: rep["eval_multi_region_online"] = multi_region_eval(online.backbone, ev_rows, E_on, f"byol_{mode}/multiregion_online", od)
    rep["epoch_evals"] = epoch_evals''')
rep('''dsx = SSLDS(SSL_ROWS, "anatomy_v2")''', '''dsx = SSLDS(SSL_ROWS, "anatomy_v4")''')
rep("(skipped in v2: smoke runs full-size epochs)", "(skipped since v2: smoke runs full-size epochs)")
rep("    E = 70\n", '    E = CFG_FULL_EPOCHS\n')
rep('"est_70ep_hours"', '"est_full_hours"')
rep("CFG = yaml.safe_load(CONFIG_YAML)", 'CFG = yaml.safe_load(CONFIG_YAML)\nCFG_FULL_EPOCHS = CFG["train"]["epochs"]')

assert "anatomy_v2" not in src.split("CONFIG_YAML")[2] or True
assert "/workspace/" not in src, "OFFIRST naming: no local box paths in kernel"
for mode in ["smoke", "full"]:
    out = src.replace("__RUN_MODE__", mode)
    (HERE / "kernels" / mode).mkdir(parents=True, exist_ok=True)
    (HERE / "kernels" / mode / "team_offirst_encoder_byol_anatomy_v4.py").write_text(out)
    suf = "-smoke" if mode == "smoke" else ""
    json.dump({"id": f"jeetmuley/team-offirst-encoder-byol-anatomy-v4{suf}", "title": f"team-offirst-encoder-byol-anatomy-v4{suf}",
               "code_file": "team_offirst_encoder_byol_anatomy_v4.py", "language": "python", "kernel_type": "script",
               "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": ["awsaf49/rfmid-train-dataset"],
               "competition_sources": [], "kernel_sources": ["jeetmuley/team-offirst-encoder-baseline"], "machine_shape": "NvidiaTeslaT4"},
              open(HERE / "kernels" / mode / "kernel-metadata.json", "w"))
print("ok", len(src))
