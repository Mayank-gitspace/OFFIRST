
# ============================================================================= KD RUN 2 (adapted from Argon's train_kd.py)
# Kept from Argon: two-stage schedule (head-only -> joint), Procrustes init (svd(Hs^T Zt) -> (U Vt)^T), fp32 loss
#   (1-cos) + lambda * tau_s^2 * KL(p_t || p_s) vs a static clean-teacher bank, stage-2 LR schedule, BN recal with
#   momentum=None (cumulative average), teacher 384 view = resize of the SAME clean 512 view.
# Changed vs Argon (see kd_run2/WORKFILE.md "Argon code review"): our cached fundus-square pipeline instead of raw PNGs
#   (crop_fundus = teacher's training preprocessing), real FOV-aware Visuscout degrader, strict checkpoint loads,
#   GPU antialiased 512->384 resize, no wd on norm/bias, grad clip, batch ladder, fold guards, logging + eval.
KD = CFG["kd"]
if SMOKE:
    KD["head_epochs"] = CFG["smoke"]["kd_head_epochs"]; KD["bb_warmup_epochs"] = CFG["smoke"]["kd_bb_warmup_epochs"]
    KD["inner_eval_epochs"] = CFG["smoke"]["kd_inner_eval_epochs"]
if "TOB_KD_HEAD_EPOCHS" in os.environ: KD["head_epochs"] = int(os.environ["TOB_KD_HEAD_EPOCHS"])      # local test only
if "TOB_KD_SSIZE" in os.environ: KD["student_size"] = int(os.environ["TOB_KD_SSIZE"]); KD["teacher_size"] = int(os.environ["TOB_KD_TSIZE"])
REFS = CFG["smoke"]["refs"] if SMOKE else ["ssl960", "ssl768p", "teacher"]
if "TOB_KD_REFS" in os.environ: REFS = [r for r in os.environ["TOB_KD_REFS"].split(",") if r]
SS, TS = KD["student_size"], KD["teacher_size"]
CFG["eval"]["img_size"] = SS
TRAIN_ROWS = SSL_ROWS                                     # folds 1-3 (prefix: SSL_EXCL = [EF, IVF])
assert not np.isin(FOLDS[TRAIN_ROWS], [EF]).any(), "fold 0 in KD training rows"
assert not np.isin(FOLDS[TRAIN_ROWS], [IVF]).any(), "fold 4 in KD training rows"
log(f"KD train rows {len(TRAIN_ROWS)} folds {sorted(set(FOLDS[TRAIN_ROWS].tolist()))}; student {SS}px teacher {TS}px")

def find_input(name):
    if f"TOB_{name.split('_')[0].upper()}_CKPT" in os.environ: return Path(os.environ[f"TOB_{name.split('_')[0].upper()}_CKPT"])
    c = sorted(INPUT.rglob(name))
    if not c: raise SystemExit(f"{name} not found under {INPUT} (attach dataset teamoffirst/offirst-kd-run2-weights)")
    return c[0]

# ----------------------------------------------------------------------------- shared-view KD dataset
def kd_box(rng, S):
    """random resized crop box of the S x S fundus square: scale ~ U(rrc_scale) of the area, log-uniform ratio."""
    for _ in range(10):
        area = S * S * rng.uniform(*KD["rrc_scale"]); r = math.exp(rng.uniform(math.log(KD["rrc_ratio"][0]), math.log(KD["rrc_ratio"][1])))
        w = math.sqrt(area * r); h = math.sqrt(area / r)
        if w <= S and h <= S: return (rng.uniform(w / 2, S - w / 2), rng.uniform(h / 2, S - h / 2)), (w, h)
    return (S / 2, S / 2), (S, S)

def kd_view(img, rng, row, out, p_deg, force=None):
    """-> (clean uint8 out x out, student uint8 out x out, alpha (-1 = clean), M, geometry). One affine (crop + rot +
    flips) from the cache square; the student copy is the SAME pixels, degraded with prob p_deg (teacher never)."""
    S = img.shape[0]; c, wh = kd_box(rng, S)
    ang = rng.uniform(-KD["rotate_deg"], KD["rotate_deg"]); hf = rng.rand() < KD["hflip"]; vf = rng.rand() < KD["vflip"]
    M = _affine(c, wh, out, ang, hf, vf)
    x = cv2.warpAffine(img, M, (out, out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if KD["colour_jitter"]: x = _colour(x, rng)
    clean = x; a = -1.0
    deg = (rng.rand() < p_deg) if force is None else force
    if deg:
        a = float(rng.uniform(*DG["alpha"])); x = visuscout(clean, a, M, row, rng)
    if KD["camera_augs"]: x = _camera(x, rng, V["cam_bright"])
    return clean, x, a, M, (c, wh, ang, hf, vf)

class KDDS(torch.utils.data.Dataset):
    """mode 'train': degrade each image independently with p_deg. mode 'recal': exactly every other sample degraded."""
    def __init__(self, rows, mode="train"): self.rows = np.asarray(rows); self.mode = mode; self.epoch = 0
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = int(self.rows[i])
        rng = np.random.RandomState((CFG["seed"] * 1000003 + self.epoch * 7919 + i * 104729 + torch.initial_seed()) % 2**32)
        force = (i % 2 == 0) if self.mode == "recal" else None
        cl, st, a, _, _ = kd_view(np.asarray(CACHE[r]), rng, r, SS, KD["p_deg"], force)
        return (torch.from_numpy(np.ascontiguousarray(cl)).permute(2, 0, 1), torch.from_numpy(np.ascontiguousarray(st)).permute(2, 0, 1),
                torch.tensor(a, dtype=torch.float32), torch.tensor(r, dtype=torch.int64))

def teacher_input(xc_norm):
    """teacher view = the student's clean normalised view resized SS -> TS (same field of view by construction)."""
    if xc_norm.shape[-1] == TS: return xc_norm
    return F.interpolate(xc_norm, size=(TS, TS), mode="bilinear", antialias=True, align_corners=False).contiguous(memory_format=torch.channels_last)

# ----------------------------------------------------------------------------- models
class Teacher(nn.Module):
    """frozen BYOL v4 ConvNeXt-T target backbone (timm num_classes=0 -> head.norm -> 768-d), L2-normalised output."""
    def __init__(self, path):
        super().__init__()
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.body = timm.create_model(KD["teacher_arch"], pretrained=False, num_classes=0)
        r = self.body.load_state_dict(ck["state_dict"], strict=True)
        self.load_report = {"path": str(path), "missing": list(r.missing_keys), "unexpected": list(r.unexpected_keys),
                            "n_tensors": len(ck["state_dict"]), "epoch": ck.get("epoch"), "source": ck.get("source")}
        self.eval(); self.requires_grad_(False)
    def train(self, mode=True): return super().train(False)          # always eval (no BN anyway; LayerNorm)
    @torch.no_grad()
    def forward(self, x):
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
            z = self.body(x)
        return F.normalize(z.float(), dim=-1)

class KDStudent(nn.Module):
    """MobileNetV3-L GAP (960) -> Linear(960, 768, bias=False) -> L2. The head is part of the exported encoder."""
    def __init__(self, path):
        super().__init__()
        self.backbone = GAPEncoder(False)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        r = self.backbone.load_state_dict(ck["state_dict"], strict=True)
        self.load_report = {"path": str(path), "missing": list(r.missing_keys), "unexpected": list(r.unexpected_keys),
                            "n_tensors": len(ck["state_dict"]), "epoch": ck.get("epoch"), "source": ck.get("source")}
        self.head = nn.Linear(self.backbone.num_features, KD["emb_dim"], bias=False)
        self.emb_dim = KD["emb_dim"]
    def forward(self, x): return F.normalize(self.head(self.backbone(x)).float(), dim=-1)

class Proj(nn.Module):
    """eval-only: a backbone + fixed linear map (SSL ep40 Procrustes-768 reference)."""
    def __init__(self, backbone, W):
        super().__init__(); self.backbone = backbone; self.register_buffer("W", torch.as_tensor(W, dtype=torch.float32))
    def forward(self, x): return F.normalize(self.backbone(x).float() @ self.W.T, dim=-1)

@torch.no_grad()
def extract_t(teacher, rows, degraded=False):
    """teacher embeddings of full clean (or degraded-eval) images at TS: cache -> INTER_AREA resize (= v4 eval path)."""
    E = []
    for x, _ in _loader(EvalDS(rows, TS, degraded)): E.append(teacher(norm_gpu(x)).cpu().numpy())
    return np.concatenate(E).astype(np.float32)

@torch.no_grad()
def extract_raw(net, rows, size):
    """un-normalised outputs (for Procrustes on L2-normalised GAP features, done by the caller)."""
    was = net.training; net.eval(); E = []
    for x, _ in _loader(EvalDS(rows, size)):
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
            E.append(net(norm_gpu(x)).float().cpu().numpy())
    net.train(was); return np.concatenate(E)

def l2n(E): return (E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)).astype(np.float32)

def procrustes(Hs, Zt):
    """orthogonal Procrustes: W = (U Vt)^T, U,_,Vt = svd(Hs^T Zt); Hs (N,960), Zt (N,768) both L2-normalised. W: (768,960)."""
    U, _, Vt = np.linalg.svd(Hs.astype(np.float64).T @ Zt.astype(np.float64), full_matrices=False)
    return (U @ Vt).T.astype(np.float32)

# ----------------------------------------------------------------------------- loss
def kd_loss(zs, zt, bank, self_idx=None):
    """zs, zt: (B,768) L2-normed fp32; bank: (N,768) clean teacher embeddings of the training images.
    -> total, cos, kl (scaled by tau_s^2), stats."""
    zs = zs.float(); zt = zt.float()
    l_cos = (1.0 - (zs * zt).sum(-1)).mean()
    lt = (zt @ bank.T) / KD["tau_t"]; ls = (zs @ bank.T) / KD["tau_s"]
    if KD["seed_mask_self"] and self_idx is not None:
        m = torch.zeros_like(lt, dtype=torch.bool); m[torch.arange(len(zt)), self_idx] = True
        lt = lt.masked_fill(m, float("-inf")); ls = ls.masked_fill(m, float("-inf"))
    with torch.no_grad(): p_t = F.softmax(lt, -1)
    log_p_s = F.log_softmax(ls, -1)
    kl = (p_t * (torch.log(p_t.clamp_min(1e-30)) - log_p_s)).masked_fill(p_t == 0, 0).sum(-1).mean() * KD["tau_s"] ** 2
    st = {}
    with torch.no_grad():
        if self_idx is not None and not KD["seed_mask_self"]:
            st["p_t_self"] = float(p_t[torch.arange(len(zt)), self_idx].mean())
            st["p_s_self"] = float(log_p_s.exp()[torch.arange(len(zt)), self_idx].mean())
        st["top1_agree"] = float((lt.argmax(-1) == ls.argmax(-1)).float().mean())
        st["p_t_max"] = float(p_t.max(-1).values.mean())
        st["p_t_top5_mass"] = float(p_t.topk(min(5, p_t.shape[1]), -1).values.sum(-1).mean())
        st["self_masked"] = float(bool(KD["seed_mask_self"] and self_idx is not None))
        st["p_t_entropy"] = float(-(p_t * torch.log(p_t.clamp_min(1e-30))).sum(-1).mean())
    return l_cos + KD["lambda_seed"] * kl, l_cos, kl, st

# ----------------------------------------------------------------------------- train step + schedule
def bn_layers(m): return [x for x in m.modules() if isinstance(x, nn.modules.batchnorm._BatchNorm)]

def kd_step(student, teacher, opt, scaler, C, Sx, rows, bank, bank_pos, amp, stage):
    xc = norm_gpu(C); xs = norm_gpu(Sx)
    zt = teacher(teacher_input(xc))
    with torch.autocast(DEV.type, dtype=torch.float16, enabled=amp):
        if stage == 1:
            with torch.no_grad(): h = student.backbone(xs)
            zs = F.normalize(student.head(h.float()), dim=-1)
        else:
            zs = student(xs)
    sidx = torch.as_tensor(bank_pos[rows.numpy()], device=zs.device)
    loss, lc, lk, st = kd_loss(zs.float(), zt, bank, sidx)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    gn = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], KD["grad_clip"])
    scaler.step(opt); scaler.update()
    with torch.no_grad(): cos_i = (zs.float() * zt).sum(-1)
    return float(loss), float(lc), float(lk), float(gn), cos_i.detach().cpu().numpy(), zs.detach().float(), st

def make_opt(student, stage):
    if stage == 1:
        return torch.optim.AdamW([{"params": list(student.head.parameters()), "lr": KD["head_lr"], "weight_decay": KD["head_wd"], "name": "head"}])
    g = {"bb": [], "bb_nowd": [], "head": []}
    for n, p in student.named_parameters():
        if n.startswith("head."): g["head"].append(p)
        elif p.ndim <= 1 or n.endswith(".bias"): g["bb_nowd"].append(p)
        else: g["bb"].append(p)
    return torch.optim.AdamW([{"params": g["bb"], "lr": KD["lr_backbone_start"], "weight_decay": KD["weight_decay"], "name": "bb"},
                              {"params": g["bb_nowd"], "lr": KD["lr_backbone_start"], "weight_decay": 0.0, "name": "bb_nowd"},
                              {"params": g["head"], "lr": KD["lr_head"], "weight_decay": KD["weight_decay"], "name": "head"}])

def stage2_lrs(s, warm, total):
    """Argon's schedule: backbone linear warmup start->peak over `warm` steps then cosine -> lr_min; head cosine
    lr_head -> lr_min over the whole stage."""
    if s < warm: bb = KD["lr_backbone_start"] + (s + 1) / max(1, warm) * (KD["lr_backbone"] - KD["lr_backbone_start"])
    else: bb = KD["lr_min"] + (KD["lr_backbone"] - KD["lr_min"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm))))
    hd = KD["lr_min"] + (KD["lr_head"] - KD["lr_min"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total))))
    return bb, hd

@torch.no_grad()
def bn_recalibrate(student, rows, bs):
    """reset BN running stats; one train-mode no-grad pass (momentum=None = exact cumulative average) with exactly
    50/50 clean/degraded student views (same spatial augs as training)."""
    L = bn_layers(student); before = [(m.running_mean.clone(), m.running_var.clone()) for m in L]; mom = [m.momentum for m in L]
    for m in L: m.reset_running_stats(); m.momentum = None
    ds = KDDS(rows, "recal"); ds.epoch = 10_000
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, drop_last=False, num_workers=CFG["data"]["num_workers"], pin_memory=DEV.type == "cuda")
    student.train(); n = 0; nd = 0
    for C, Sx, a, r in dl:
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"): student(norm_gpu(Sx))
        n += len(a); nd += int((a >= 0).sum())
    for m, mo in zip(L, mom): m.momentum = mo
    student.eval()
    dmean = float(np.mean([float((m.running_mean - b[0]).abs().mean()) for m, b in zip(L, before)]))
    dvar = float(np.mean([float(((m.running_var - b[1]).abs() / (b[1].abs() + 1e-6)).mean()) for m, b in zip(L, before)]))
    rep = {"n_bn": len(L), "n_images": n, "deg_frac": nd / max(1, n), "mean_abs_delta_running_mean": dmean,
           "mean_rel_delta_running_var": dvar, "num_batches_tracked_first": int(L[0].num_batches_tracked) if L else 0}
    log("BN recal:", json.dumps(rep)); return rep

def kd_probe_batch(student, teacher, bank, bank_pos, amp):
    if DEV.type != "cuda": return int(os.environ.get("TOB_BS", 4)), False, None
    sd = copy.deepcopy(student.state_dict())
    for bs, ck in KD["batch_ladder"]:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            student.backbone.set_grad_checkpointing(ck); student.train()
            opt = make_opt(student, 2); scaler = torch.amp.GradScaler("cuda", enabled=amp)
            C = torch.randint(0, 255, (bs, 3, SS, SS), dtype=torch.uint8); r = torch.as_tensor(TRAIN_ROWS[:bs])
            for _ in range(2): kd_step(student, teacher, opt, scaler, C, C, r, bank, bank_pos, amp, 2)
            torch.cuda.synchronize(); peak = torch.cuda.max_memory_allocated() / 2**30
            log(f"KD batch probe: bs={bs} grad_ckpt={ck} fits, peak {peak:.2f} GiB (joint stage, teacher live)")
            del opt; student.load_state_dict(sd); torch.cuda.empty_cache(); return bs, ck, peak
        except torch.cuda.OutOfMemoryError:
            log(f"KD batch probe: bs={bs} grad_ckpt={ck} OOM"); opt = None; torch.cuda.empty_cache()
    raise SystemExit("KD: even the smallest batch OOMs")

# ----------------------------------------------------------------------------- validation (fold 4, logged only)
@torch.no_grad()
def kd_val(student, Zt_val, bank):
    E = extract(student, IV_ROWS); Ed = extract(student, IV_ROWS, degraded=True)
    zt = torch.from_numpy(Zt_val).to(DEV)
    out = {}
    for nm, X in [("clean", E), ("deg", Ed)]:
        zs = torch.from_numpy(X).to(DEV); _, lc, lk, _ = kd_loss(zs, zt, bank)
        out[f"val_cos_{nm}"] = float((zs * zt).sum(-1).mean()); out[f"val_kl_{nm}"] = float(lk)
    return out

# ----------------------------------------------------------------------------- main
summary = {"run_mode": RUN_MODE, "labels": LABELS, "split_source": split_src, "eval_fold": EF, "inner_val_fold": IVF,
           "train_folds": sorted(set(FOLDS[TRAIN_ROWS].tolist())), "n_train": int(len(TRAIN_ROWS)), "n_eval_rows": int(len(EVAL_ROWS)),
           "degrade_fit": FIT, "geom": GEOM_STATS, "config": CFG, "checks": {}, "references": {}, "kd": {},
           "citation": "RFMiD (Pachade et al., Data 6(2):14, 2021; IEEE DataPort, doi:10.21227/s3g7-st65), CC BY 4.0"}
def dump(): summary["timings"] = TIM; json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=float)
od = OUT / "kd_run2"; od.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"ID": df.ID.values[EVAL_ROWS], "fold": FOLDS[EVAL_ROWS]}).to_csv(od / "embedding_ids.csv", index=False)
degrade_examples(OUT / "degrade_eval_examples.png")
amp = CFG["train"]["amp"] and DEV.type == "cuda"

t = time.time()
teacher = Teacher(find_input(KD["teacher_ckpt"])).to(DEV).to(memory_format=torch.channels_last)
student = KDStudent(find_input(KD["student_ckpt"])).to(DEV).to(memory_format=torch.channels_last)
for nm, m in [("teacher", teacher), ("student_init", student)]:
    lr_ = m.load_report; ok = not lr_["missing"] and not lr_["unexpected"]
    summary["checks"][f"{nm}_load_strict"] = {"pass": ok, **lr_}; log(f"CHECK {nm} strict load: {'PASS' if ok else 'FAIL'} {json.dumps(lr_)}")

# teacher embeddings: all eval rows clean (bank, Procrustes, val, reference) [+ degraded for the teacher reference]
T_ROWS = np.union1d(EVAL_ROWS, TRAIN_ROWS)            # = all 1,920 rows on Kaggle (local tests subsample)
Zt_all = extract_t(teacher, T_ROWS); pos_all = {int(r): i for i, r in enumerate(T_ROWS)}
Zt_eval = Zt_all[[pos_all[int(r)] for r in EVAL_ROWS]]
np.save(od / "embeddings_teacher.npy", Zt_eval)
bank_rows = TRAIN_ROWS
bank_np = Zt_all[[pos_all[int(r)] for r in bank_rows]]
f0_in_bank = int(np.isin(FOLDS[bank_rows], [EF]).sum()); f4_in_bank = int(np.isin(FOLDS[bank_rows], [IVF]).sum())
summary["checks"]["fold0_not_in_train_or_bank"] = {"pass": f0_in_bank == 0 and not np.isin(FOLDS[TRAIN_ROWS], [EF]).any(),
                                                   "fold0_in_bank": f0_in_bank, "fold4_in_bank": f4_in_bank, "bank_n": int(len(bank_rows))}
log(f"CHECK fold0 not in train/bank: {'PASS' if f0_in_bank == 0 else 'FAIL'} (bank {len(bank_rows)} rows, folds {sorted(set(FOLDS[bank_rows].tolist()))})")
bank = torch.from_numpy(bank_np).to(DEV)
bank_pos = -np.ones(len(df), np.int64); bank_pos[bank_rows] = np.arange(len(bank_rows))
Zt_val = Zt_all[[pos_all[int(r)] for r in IV_ROWS]]
TIM["teacher_embed_s"] = time.time() - t

# Procrustes head init from clean folds-1-3 GAP features
t = time.time()
Hs = l2n(extract_raw(student.backbone, bank_rows, SS))
W = procrustes(Hs, bank_np)
W0 = student.head.weight.detach().cpu().numpy()
cos_before = float((l2n(Hs @ W0.T) * bank_np).sum(1).mean())
student.head.weight.data.copy_(torch.from_numpy(W).to(DEV))
cos_after = float((l2n(Hs @ W.T) * bank_np).sum(1).mean())
orth_err = float(np.abs(W @ W.T - np.eye(W.shape[0])).max())
cos_val = float((l2n(l2n(extract_raw(student.backbone, IV_ROWS, SS)) @ W.T) * Zt_val).sum(1).mean())
np.save(od / "procrustes_W.npy", W)
summary["checks"]["procrustes"] = {"pass": cos_after > cos_before and orth_err < 1e-4, "cos_before_random_head": cos_before,
                                   "cos_after_procrustes_train": cos_after, "cos_after_procrustes_fold4": cos_val, "max_abs_WWt_minus_I": orth_err}
log(f"CHECK Procrustes: cos(train) random-head {cos_before:.4f} -> Procrustes {cos_after:.4f}; fold-4 {cos_val:.4f}; |WW^T-I|max {orth_err:.1e}")
TIM["procrustes_s"] = time.time() - t; dump()

# references: SSL ep40 at 960 (as the SSL arm) and Procrustes-768 (fixed W, no training); teacher (full run)
t = time.time()
def ref_eval(name, enc, rows=EVAL_ROWS, teacher_mode=False):
    if teacher_mode: E, Ed = Zt_eval, extract_t(teacher, EVAL_ROWS, degraded=True)
    else: E, Ed = extract_pair(enc, rows)
    np.save(od / f"embeddings_{name}.npy", E); np.save(od / f"embeddings_{name}_degraded.npy", Ed)
    summary["references"][name] = evaluate(rows, E, name, Ed, "dev"); summary["references"][name]["dim"] = int(E.shape[1]); dump()
if "ssl960" in REFS: ref_eval("ssl_ep40_960", student.backbone)
if "ssl768p" in REFS: ref_eval("ssl_ep40_768procrustes", Proj(student.backbone, W).to(DEV))
if "teacher" in REFS: ref_eval("teacher", None, teacher_mode=True)
TIM["ref_eval_s"] = time.time() - t; dump()

# batch probe (joint stage, teacher live)
bs, ck, probe_peak = kd_probe_batch(student, teacher, bank, bank_pos, amp)
student.backbone.set_grad_checkpointing(ck)
summary["batch"] = {"batch_size": bs, "grad_ckpt": ck, "probe_peak_gib": probe_peak}; dump()

# view-alignment check on real data (teacher view == resize of the student clean view; degradation on student only)
def view_check(n=6):
    res = []
    for k, r in enumerate(TRAIN_ROWS[:n]):
        img = np.asarray(CACHE[r]); rng = np.random.RandomState(1000 + k)
        cl, st, a, M, (c, wh, ang, hf, vf) = kd_view(img, rng, r, SS, 1.0)
        xt = teacher_input(norm_gpu(torch.from_numpy(cl).permute(2, 0, 1)[None]))
        direct = cv2.warpAffine(img, _affine(c, wh, TS, ang, hf, vf), (TS, TS), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        xd = norm_gpu(torch.from_numpy(direct).permute(2, 0, 1)[None])
        flip = cv2.warpAffine(img, _affine(c, wh, TS, ang, not hf, vf), (TS, TS), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        xf = norm_gpu(torch.from_numpy(flip).permute(2, 0, 1)[None])
        cl0, st0, a0, _, _ = kd_view(img, np.random.RandomState(1000 + k), r, SS, 0.0)     # same seed, no degradation
        res.append({"row": int(r), "teacher_vs_direct_warp_mae_norm": float((xt - xd).abs().mean()),
                    "negative_control_hflipped_mae_norm": float((xt - xf).abs().mean()),
                    "clean_identical_with_p0": bool(np.array_equal(cl, cl0)), "student_equals_clean_when_p0": bool(np.array_equal(st0, cl0)),
                    "student_minus_clean_mae_u8": float(np.abs(st.astype(np.float32) - cl).mean()), "alpha": a})
    return res
vc = view_check()
summary["checks"]["view_alignment"] = {"pass": all(v["teacher_vs_direct_warp_mae_norm"] < 0.25 * v["negative_control_hflipped_mae_norm"] for v in vc), "per_image": vc}
summary["checks"]["degrade_student_only"] = {"pass": all(v["clean_identical_with_p0"] and v["student_equals_clean_when_p0"] and v["student_minus_clean_mae_u8"] > 0.5 for v in vc),
    "note": "teacher input is always built from the clean view C in kd_step; degraded copy differs from clean; p=0 -> identical"}
log("CHECK view alignment:", json.dumps(vc))

# ----------------------------------------------------------------------------- train
E_ = CFG["train"]["epochs"]; H_ = min(KD["head_epochs"], E_)
ds = KDDS(TRAIN_ROWS, "train")
dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=CFG["data"]["num_workers"], pin_memory=DEV.type == "cuda")
steps_ep = len(dl); total2 = steps_ep * (E_ - H_); warm2 = steps_ep * KD["bb_warmup_epochs"]
log(f"=== KD: {len(TRAIN_ROWS)} imgs, bs {bs}, grad_ckpt {ck}, {steps_ep} steps/ep, {E_} ep ({H_} head-only), bank {len(bank_rows)}, "
    f"student params {sum(p.numel() for p in student.parameters())/1e6:.2f}M")
if DEV.type == "cuda": torch.cuda.reset_peak_memory_stats()
hist = []; s2 = 0; inner = {}; opt = None; scaler = torch.amp.GradScaler("cuda", enabled=amp); t_train = time.time()
for ep in range(1, E_ + 1):
    stage = 1 if ep <= H_ else 2
    if ep == 1 and stage == 1: opt = make_opt(student, 1)
    if stage == 2 and (opt is None or ep == H_ + 1): opt = make_opt(student, 2)
    if stage == 1: student.backbone.eval(); student.backbone.requires_grad_(False); student.head.train()
    else: student.train(); student.requires_grad_(True)
    ds.epoch = ep; t = time.time(); agg = {"loss": [], "cos": [], "kl": [], "gn": []}; ci, al, zs_all, tstep, sts = [], [], [], [], []
    for C, Sx, a, r in dl:
        ts = time.time()
        if stage == 2:
            bb_lr, hd_lr = stage2_lrs(s2, warm2, total2)
            for g in opt.param_groups: g["lr"] = hd_lr if g["name"] == "head" else bb_lr
            s2 += 1
        l, lc, lk, gn, cos_i, zs, st = kd_step(student, teacher, opt, scaler, C, Sx, r, bank, bank_pos, amp, stage)
        for k_, v_ in zip(["loss", "cos", "kl", "gn"], [l, lc, lk, gn]): agg[k_].append(v_)
        ci.append(cos_i); al.append(a.numpy()); zs_all.append(zs); sts.append(st)
        if DEV.type == "cuda": torch.cuda.synchronize()
        tstep.append(time.time() - ts)
    t_tr = time.time() - t; ci = np.concatenate(ci); al = np.concatenate(al)
    row = {"epoch": ep, "stage": stage, "loss": float(np.mean(agg["loss"])), "loss_cos": float(np.mean(agg["cos"])), "loss_kl": float(np.mean(agg["kl"])),
           "loss_cos_first_step": agg["cos"][0], "loss_kl_first_step": agg["kl"][0], "grad_norm": float(np.mean(agg["gn"])),
           "cos_clean_views": float(ci[al < 0].mean()) if (al < 0).any() else float("nan"),
           "cos_deg_views": float(ci[al >= 0].mean()) if (al >= 0).any() else float("nan"), "deg_frac": float((al >= 0).mean()),
           "deg_alpha_mean": float(al[al >= 0].mean()) if (al >= 0).any() else float("nan"),
           "std_z": std_metric(torch.cat(zs_all)), "lr": {g["name"]: g["lr"] for g in opt.param_groups},
           "train_s": t_tr, "img_per_s": len(ci) / max(t_tr, 1e-9), "gpu_img_per_s_median_step": bs / float(np.median(tstep)),
           **{k_: float(np.mean([s[k_] for s in sts if k_ in s])) for k_ in sts[0]}}
    if not math.isfinite(row["loss"]):
        json.dump(hist + [row], open(od / "history.json", "w"), indent=1, default=float); raise SystemExit(f"non-finite loss at epoch {ep}")
    if ep % KD["val_every"] == 0 or ep == E_:
        tv = time.time(); row.update(kd_val(student, Zt_val, bank)); row["val_s"] = time.time() - tv
    hist.append(row)
    log(f"  [KD] ep {ep}/{E_} st{stage} loss {row['loss']:.4f} cos {row['loss_cos']:.4f} kl {row['loss_kl']:.4f} | cos clean/deg views "
        f"{row['cos_clean_views']:.3f}/{row['cos_deg_views']:.3f} deg {row['deg_frac']:.2f} | p_t_self {row.get('p_t_self', float('nan')):.3f} "
        f"p_t max/top5 {row['p_t_max']:.3f}/{row['p_t_top5_mass']:.3f} top1 {row['top1_agree']:.3f} | std_z {row['std_z']:.3f} gn {row['grad_norm']:.2f} lr {json.dumps({k: f'{v:.2e}' for k, v in row['lr'].items()})} | "
        f"{row['img_per_s']:.1f} img/s (gpu {row['gpu_img_per_s_median_step']:.1f}) {t_tr:.0f}s"
        + (f" | val4 cos {row['val_cos_clean']:.4f}/{row['val_cos_deg']:.4f} kl {row['val_kl_clean']:.4f}/{row['val_kl_deg']:.4f}" if "val_cos_clean" in row else ""))
    json.dump(hist, open(od / "history.json", "w"), indent=1, default=float)
    if ep % KD["ckpt_every"] == 0 or ep == E_:
        torch.save({"student": student.state_dict(), "epoch": ep, "config": CFG, "note": "pre-BN-recal"}, od / (f"ep{ep:03d}.pth" if ep != E_ else "last_pre_recal.pth"))
    if ep in KD["inner_eval_epochs"] and ep != E_:
        E1, Ed1 = extract_pair(student, INNER_ROWS); inner[f"ep{ep}"] = evaluate(INNER_ROWS, E1, f"kd/ep{ep}", Ed1, "inner")
        json.dump(inner, open(od / "inner_evals.json", "w"), indent=1, default=float)
TIM["train_s"] = time.time() - t_train
peak = torch.cuda.max_memory_allocated() / 2**30 if DEV.type == "cuda" else float("nan")
summary["kd"].update({"history": hist, "batch_size": bs, "grad_ckpt": ck, "steps_per_epoch": steps_ep, "peak_gpu_gib_train": peak}); dump()

# pre-recal (fold-4 only) vs post-recal
E_pre = extract(student, IV_ROWS)
if KD["bn_recal"]:
    t = time.time(); rr = bn_recalibrate(student, TRAIN_ROWS, bs); TIM["bn_recal_s"] = time.time() - t
    summary["checks"]["bn_recal_changes_stats"] = {"pass": rr["mean_abs_delta_running_mean"] > 0, **rr}
    summary["kd"]["val_after_recal"] = kd_val(student, Zt_val, bank)
    summary["kd"]["fold4_pre_vs_post_recal_cos"] = float((E_pre * extract(student, IV_ROWS)).sum(1).mean())
    log("val fold4 after recal:", json.dumps(summary["kd"]["val_after_recal"]), "pre/post embedding cos", summary["kd"]["fold4_pre_vs_post_recal_cos"])
torch.save({"student": student.state_dict(), "epoch": E_, "bn_recal": KD["bn_recal"], "config": CFG,
            "arch": {"backbone": CFG["model"]["backbone"], "feature": "gap960", "head": "Linear(960,768,bias=False)+L2", "emb_dim": KD["emb_dim"]}},
           od / "kd_final.pth")
dump()

# final evals: 768-d output (the encoder) on dev (fold 0) + inner (fold 4); 960-d backbone diagnostic
t = time.time()
E, Ed = extract_pair(student, EVAL_ROWS)
assert E.shape[1] == KD["emb_dim"], E.shape
summary["checks"]["eval_uses_768"] = {"pass": E.shape[1] == 768, "dim": int(E.shape[1])}
np.save(od / "embeddings_kd768.npy", E); np.save(od / "embeddings_kd768_degraded.npy", Ed)
summary["kd"]["eval_dev_768"] = evaluate(EVAL_ROWS, E, "kd_final_768", Ed, "dev")
summary["kd"]["eval_inner_768"] = evaluate(EVAL_ROWS, E, "kd_final_768", Ed, "inner"); dump()
E9, Ed9 = extract_pair(student.backbone, EVAL_ROWS)
np.save(od / "embeddings_kd960.npy", E9); np.save(od / "embeddings_kd960_degraded.npy", Ed9)
if not SMOKE: summary["kd"]["eval_dev_960_diag"] = evaluate(EVAL_ROWS, E9, "kd_final_960_backbone_diag", Ed9, "dev")
summary["kd"]["inner_evals"] = inner
TIM["final_eval_s"] = time.time() - t

# full-run estimate from this run's measured epoch times
h2 = [x for x in hist if x["stage"] == 2] or hist; h1 = [x for x in hist if x["stage"] == 1] or hist
ep2 = float(np.median([x["train_s"] + x.get("val_s", 0) for x in h2])); ep1 = float(np.median([x["train_s"] + x.get("val_s", 0) for x in h1]))
nf = CFG_FULL_EPOCHS; hf_ = CFG["kd"].get("head_epochs_full", 3)
ev = float(np.mean([summary["references"][k]["eval_s"] for k in summary["references"]])) if summary["references"] else 250.0
summary["estimate_full"] = {"sec_per_epoch_head": ep1, "sec_per_epoch_joint": ep2, "eval_s_each_smoke": ev,
    "est_train_h": (3 * ep1 + (nf - 3) * ep2) / 3600, "est_total_h_excl_setup": (3 * ep1 + (nf - 3) * ep2 + 9 * ev * 1.5 + 600) / 3600,
    "note": "9 evaluate calls in full (3 refs, kd dev/inner/960, 1 inner mid-run, +margin); smoke evals use 3 draws (full 20) -> x1.5"}
TIM["wall_s"] = time.time() - T0; dump()
log("ESTIMATE", json.dumps(summary["estimate_full"], default=float))
log("CHECKS", json.dumps({k: v["pass"] for k, v in summary["checks"].items()}))
log("DONE", json.dumps(TIM, default=float))
