# p_t over neighbours with the real 1,152-image bank (teacher clean embeddings, in-run split folds 1-3), self masked vs not.
import numpy as np, pandas as pd, torch, torch.nn.functional as F, json
E = np.load('kd_run2/localtest/teacher_emb_640path_fp32.npy')
sp = pd.read_csv('encoder_mnv3_ssl/full_out/out/split_used.csv'); lab = pd.read_csv('/workspace/rfmid_train_dl/Training_Set/RFMiD_Training_Labels.csv')
d = lab.merge(sp, on='ID'); rows = np.where(d.fold.isin([1, 2, 3]).values)[0]; Y = d[[c for c in lab.columns if c not in ('ID', 'Disease_Risk')]].values[rows]
B = torch.from_numpy(E[rows]); Z = B.clone(); out = {}
for mask in (False, True):
    lt = Z @ B.T / 0.05
    if mask: lt = lt.masked_fill(torch.eye(len(B), dtype=torch.bool), float('-inf'))
    p = F.softmax(lt, -1); ent = -(p * torch.log(p.clamp_min(1e-30))).sum(-1)
    top1 = lt.argmax(-1).numpy()
    same = float(np.mean([(Y[i] == Y[j]).all() for i, j in enumerate(top1)])); share = float(np.mean([(Y[i] & Y[j]).any() for i, j in enumerate(top1)]))
    rnd = np.random.RandomState(0).randint(0, len(B), len(B)); same_r = float(np.mean([(Y[i] == Y[j]).all() for i, j in enumerate(rnd)]))
    j = list(d.columns).index('ARMD'); armd = Y[:, [c for c in lab.columns if c not in ('ID', 'Disease_Risk')].index('ARMD')] == 1
    pa = float((p[torch.from_numpy(armd)][:, torch.from_numpy(armd)]).sum(-1).mean())
    out['masked' if mask else 'self_included'] = {'p_t_max': float(p.max(-1).values.mean()), 'p_t_top5_mass': float(p.topk(5, -1).values.sum(-1).mean()),
        'entropy_nats': float(ent.mean()), 'eff_neighbours_exp_entropy': float(ent.exp().mean()),
        'top1_identical_labelset': same, 'top1_shares_a_label': share, 'random_identical_labelset': same_r,
        'ARMD_query_mass_on_ARMD_anchors': pa, 'ARMD_prevalence_in_bank': float(armd.mean())}
# KL masked, finite + gradient finite
zs = F.normalize(Z[:16] + 0.3 * torch.randn(16, 768), dim=1).requires_grad_(True); zt = Z[:16]
lt = (zt @ B.T) / 0.05; ls = (zs @ B.T) / 0.10; m = torch.zeros_like(lt, dtype=torch.bool); m[torch.arange(16), torch.arange(16)] = True
lt = lt.masked_fill(m, float('-inf')); ls = ls.masked_fill(m, float('-inf')); pt = F.softmax(lt, -1); lps = F.log_softmax(ls, -1)
kl = (pt * (torch.log(pt.clamp_min(1e-30)) - lps)).masked_fill(pt == 0, 0).sum(-1).mean() * 0.01; kl.backward()
out['masked_kl_value'] = float(kl); out['masked_kl_grad_finite'] = bool(torch.isfinite(zs.grad).all()); out['grad_norm'] = float(zs.grad.norm())
print(json.dumps(out, indent=1)); json.dump(out, open('kd_run2/localtest/masked_kl_bank_stats.json', 'w'), indent=1)
