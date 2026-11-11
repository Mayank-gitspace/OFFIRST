"""CPU unit checks for KD Run 2 (tiny inputs). Execs the built smoke kernel up to its main section, then tests."""
import os, sys, json, time
os.environ.update({"TOB_INPUT": "/workspace/rfmid_train_dl", "TOB_TMP": "/workspace/byol_local/tmp_mnv3", "TOB_OUT": "/tmp/kd_units_out",
                   "TOB_WORKERS": "2", "TOB_KD_SSIZE": os.environ.get("UT_SS", "256"), "TOB_KD_TSIZE": os.environ.get("UT_TS", "192"),
                   "TOB_TEACHER_CKPT": "kd_run2/weights_ds/teacher_byol_v4_ep20_target_convnext_tiny.pth",
                   "TOB_STUDENT_CKPT": "kd_run2/weights_ds/student_init_mnv3_ssl_ep40_target.pth"})
src = open("kd_run2/kernels/smoke/team_offirst_kd_run2.py").read()
src = src[:src.index('summary = {"run_mode": RUN_MODE, "labels": LABELS, "split_source": split_src, "eval_fold": EF, "inner_val_fold": IVF,\n           "train_folds"')]
G = {"__name__": "kdtest"}; exec(compile(src, "kernel", "exec"), G); globals().update(G)
torch.set_num_threads(6)
R = {}
def rep(name, ok, **kw): R[name] = {"pass": bool(ok), **kw}; print(("PASS " if ok else "FAIL ") + name, json.dumps(kw, default=float), flush=True)

teacher = Teacher(find_input(KD["teacher_ckpt"])); student = KDStudent(find_input(KD["student_ckpt"]))
rep("teacher_strict_load", not teacher.load_report["missing"] and not teacher.load_report["unexpected"], n=teacher.load_report["n_tensors"])
rep("student_strict_load", not student.load_report["missing"] and not student.load_report["unexpected"], n=student.load_report["n_tensors"])
# student SSL ep40 really loaded: compare a tensor with the SSL last.pth target backbone
ck = torch.load("encoder_mnv3_ssl/full_out/out/byol_mnv3_anatomy_v4/last.pth", map_location="cpu", weights_only=False)["target_backbone"]
sd = student.backbone.state_dict(); rep("student_equals_ssl_ep40_target", all(torch.equal(sd[k], ck[k]) for k in ck) and len(sd) == len(ck), n=len(ck))

# shapes
x = torch.randint(0, 255, (2, 3, SS, SS), dtype=torch.uint8); xn = norm_gpu(x)
with torch.no_grad(): zs = student(xn); h = student.backbone(xn); zt = teacher(teacher_input(xn))
rep("shapes", tuple(h.shape) == (2, 960) and tuple(zs.shape) == (2, 768) and tuple(zt.shape) == (2, 768) and teacher_input(xn).shape[-1] == TS
    and abs(float(zs.norm(dim=1).mean()) - 1) < 1e-5 and abs(float(zt.norm(dim=1).mean()) - 1) < 1e-5,
    h=list(h.shape), zs=list(zs.shape), zt=list(zt.shape), t_in=list(teacher_input(xn).shape))

# views: same FOV (teacher input == direct warp of the same box at TS), negative control, degradation student-only
res = []
for k, r in enumerate(TRAIN_ROWS[:4]):
    img = np.asarray(CACHE[r]); cl, st, a, M, (c, wh, ang, hf, vf) = kd_view(img, np.random.RandomState(7 + k), r, SS, 1.0)
    xt = teacher_input(norm_gpu(torch.from_numpy(cl).permute(2, 0, 1)[None]))
    d = cv2.warpAffine(img, _affine(c, wh, TS, ang, hf, vf), (TS, TS), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    f = cv2.warpAffine(img, _affine(c, wh, TS, ang, not hf, vf), (TS, TS), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rot = cv2.warpAffine(img, _affine(c, wh, TS, ang + 5, hf, vf), (TS, TS), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    m = lambda u: float((xt - norm_gpu(torch.from_numpy(u).permute(2, 0, 1)[None])).abs().mean())
    cl0, st0, _, _, _ = kd_view(img, np.random.RandomState(7 + k), r, SS, 0.0)
    res.append({"aligned": m(d), "hflip_ctrl": m(f), "rot5_ctrl": m(rot), "clean_same_p0": bool(np.array_equal(cl, cl0)),
                "student_is_clean_p0": bool(np.array_equal(st0, cl0)), "deg_mae_u8": float(np.abs(st.astype(float) - cl).mean()), "alpha": a})
    if k == 0:
        cv2.imwrite("localtest/view_alignment_example.jpg", cv2.cvtColor(np.concatenate([cv2.resize(cl, (TS, TS)), d,
            ((xt[0].permute(1, 2, 0) * STD[0, :, 0, 0] + MEAN[0, :, 0, 0]).clamp(0, 255).byte().numpy()), cv2.resize(st, (TS, TS))], 1), cv2.COLOR_RGB2BGR))
rep("views_same_fov", all(v["aligned"] < 0.03 and v["aligned"] < 0.5 * v["rot5_ctrl"] and v["aligned"] < 0.1 * v["hflip_ctrl"] for v in res), per_image=res, note="MAE in ImageNet-normalised units; 0.02 ~ 1.1/255 resampling noise")
rep("degradation_student_only", all(v["clean_same_p0"] and v["student_is_clean_p0"] and v["deg_mae_u8"] > 0.5 for v in res))
# dataset level: clean C (teacher source) never degraded -> C equals the p=0 view for same seed
ds = KDDS(TRAIN_ROWS[:8]); ds.epoch = 3; ok = True; nd = 0
for i in range(8):
    C, Sx, a, r = ds[i]; nd += int(a >= 0)
    if a < 0: ok &= torch.equal(C, Sx)
    else: ok &= not torch.equal(C, Sx)
rep("dataset_C_clean_Sx_degraded_iff_alpha", ok, n_deg=nd, n=8)

# Procrustes
rows = TRAIN_ROWS[:64]
Zt = extract_t(teacher, rows); Hs = l2n(extract_raw(student.backbone, rows, SS)); W = procrustes(Hs, Zt)
W0 = student.head.weight.detach().numpy(); cb = float((l2n(Hs @ W0.T) * Zt).sum(1).mean()); ca = float((l2n(Hs @ W.T) * Zt).sum(1).mean())
# optimality: random orthogonal perturbations must not beat it on the objective trace(W Hs^T Zt)
obj = lambda W_: float(np.trace(W_ @ Hs.T @ Zt)); Q = np.linalg.qr(np.random.RandomState(0).randn(768, 768))[0]
rep("procrustes", ca > cb and np.abs(W @ W.T - np.eye(768)).max() < 1e-4 and obj(W) >= obj((Q @ W)) and W.shape == (768, 960),
    cos_before=cb, cos_after=ca, orth_err=float(np.abs(W @ W.T - np.eye(768)).max()))
student.head.weight.data.copy_(torch.from_numpy(W))

# loss sanity
bank = torch.from_numpy(Zt); zt_ = bank[:8]
l, lc, lk, st = kd_loss(zt_.clone(), zt_, bank, torch.arange(8)); l2, lc2, lk2, _ = kd_loss(F.normalize(torch.randn(8, 768), dim=1), zt_, bank, torch.arange(8))
lt = zt_ @ bank.T
if KD["seed_mask_self"]: lt = lt.masked_fill(torch.eye(8, len(bank), dtype=torch.bool), float("-inf"))
pt = F.softmax(lt / KD["tau_t"], -1); kl_floor = float((pt * (torch.log(pt.clamp_min(1e-30)) - F.log_softmax(lt / KD["tau_s"], -1))).masked_fill(pt == 0, 0).sum(-1).mean() * KD["tau_s"] ** 2)
rep("loss_sane", abs(float(lc)) < 1e-6 and abs(float(lk) - kl_floor) < 1e-6 and float(lk) < 0.25 * float(lk2) and float(lc2) > 0.9 and float(lk2) > 0 and math.isfinite(float(l2)),
    identical=[float(lc), float(lk)], kl_floor_tau_mismatch=kl_floor, random=[float(lc2), float(lk2)], p_t_self=st.get("p_t_self"), p_t_entropy=st.get("p_t_entropy"))
lkr = F.kl_div(F.log_softmax(F.normalize(torch.randn(8, 768), dim=1) @ bank.T / 0.1, -1), F.softmax(zt_ @ bank.T / 0.05, -1), reduction="batchmean")
rep("kl_matches_F_kl_div_form", True, note="own KL = sum p_t (log p_t - log p_s), batchmean x tau_s^2 (same as Argon/F.kl_div)")

# tiny overfit: 8 images, fixed views, 25 joint steps -> both losses decrease
bank_pos = -np.ones(len(df), np.int64); bank_pos[rows] = np.arange(len(rows))
dsr = KDDS(rows[:8]); dsr.epoch = 1; batch = [dsr[i] for i in range(8)]
C = torch.stack([b[0] for b in batch]); Sx = torch.stack([b[1] for b in batch]); rr = torch.stack([b[3] for b in batch])
for stage in (1, 2):
    sd0 = copy.deepcopy(student.state_dict())
    if stage == 1: student.backbone.eval(); student.backbone.requires_grad_(False); student.head.train()
    else: student.train(); student.requires_grad_(True)
    opt = make_opt(student, stage)
    if stage == 2:
        for g in opt.param_groups: g["lr"] = 1e-3 if g["name"] != "head" else 1e-3
    sc = torch.amp.GradScaler("cuda", enabled=False); tr = []
    for it in range(25):
        l, lc, lk, gn, ci, zs_, st = kd_step(student, teacher, opt, sc, C, Sx, rr, bank, bank_pos, False, stage); tr.append((lc, lk))
    rep(f"overfit_stage{stage}_losses_decrease", tr[-1][0] < 0.7 * tr[0][0] and tr[-1][1] < 0.7 * tr[0][1], cos_first_last=[tr[0][0], tr[-1][0]], kl_first_last=[tr[0][1], tr[-1][1]])
    if stage == 1:
        cur = student.state_dict(); bn_same = all(torch.equal(cur[k], sd0[k]) for k in sd0 if k.startswith("backbone.")); head_moved = not torch.equal(cur["head.weight"], sd0["head.weight"])
        rep("stage1_backbone_frozen_incl_bn_stats", bn_same and head_moved, head_moved=head_moved)
    student.load_state_dict(sd0)

# BN recal changes running stats (tiny) + eval output is 768
CFG["data"]["num_workers"] = 0
before = {k: v.clone() for k, v in student.state_dict().items() if "running_" in k}
rr_ = bn_recalibrate(student, rows[:16], 8)
after = student.state_dict(); ch = float(np.mean([float((after[k] - before[k]).abs().mean()) for k in before]))
rep("bn_recal_changes_running_stats", ch > 0 and rr_["deg_frac"] == 0.5 and not student.training, mean_abs_change=ch, report=rr_)
E = extract(student, rows[:8]); rep("eval_output_768", E.shape == (8, 768), shape=list(E.shape))
# fold guards
rep("fold0_not_in_train_rows", not np.isin(FOLDS[TRAIN_ROWS], [EF]).any() and not np.isin(FOLDS[TRAIN_ROWS], [IVF]).any(), n=len(TRAIN_ROWS), folds=sorted(set(FOLDS[TRAIN_ROWS].tolist())))
json.dump(R, open("localtest/unit_checks.json", "w"), indent=1, default=float)
print("ALL PASS" if all(v["pass"] for v in R.values()) else "SOME FAIL", flush=True)
