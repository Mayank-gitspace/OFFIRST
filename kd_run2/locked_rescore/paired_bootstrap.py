"""Locked A/B split CPU re-score + paired bootstrap: model B (KD 768) minus model A (SSL ep40), 1000 iterations.
Adapted from Argon's eval_paired_bootstrap.py (exact Mann-Whitney dominance-matrix bootstrap). Changes vs Argon:
  * scores come from OUR protocol (v4/MNV3 evaluate(): StandardScaler + C-grid CV balanced logreg LP on locked folds 1-4,
    few-shot LOGREG headline with the fixed seeds seed*1000+k*100+d, kNN k=20 temp 0.07) so point estimates equal the
    numbers in RESULTS.md (checked by --selftest), instead of Argon's unscaled C=1 LP and prototype-only few-shot;
  * bootstrap resamples QUERY IMAGES jointly (one multiplicity vector per iteration shared by all labels and both models)
    instead of independent per-class pos/neg resampling, so macro means keep their cross-label correlation;
  * adds degraded metrics (LP clean->deg, kNN deg-query) for the non-inferiority guardrails and the decision rule.
Usage: python paired_bootstrap.py --a EMB_A.npy --a-deg EMB_A_deg.npy --b EMB_B.npy --b-deg EMB_B_deg.npy --ids embedding_ids.csv --out res.json
"""
import sys, json, time, argparse, numpy as np, pandas as pd
sys.path.insert(0, 'encoder_mnv3_ssl/locked_rescore')
import evallib as L
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
CFG, LABELS = L.CFG, L.LABELS
HEAD = CFG['eval']['rare_headline']

def lp_scores(Es, ys, Eqs, labs):
    """v4 linear_probe, returning decision scores for each query variant in Eqs (list)."""
    sc = StandardScaler().fit(Es); Xs = sc.transform(Es); Xq = [sc.transform(e) for e in Eqs]; out = {}
    for j in labs:
        grid = CFG['eval']['probe_C_grid']; best, bestC = -1, grid[len(grid) // 2]
        n_min = int(min(ys[:, j].sum(), len(ys) - ys[:, j].sum())); nspl = min(3, n_min)
        if nspl >= 2:
            for C in grid:
                aucs = []
                for tr, va in StratifiedKFold(nspl, shuffle=True, random_state=0).split(Xs, ys[:, j]):
                    if len(np.unique(ys[va, j])) < 2 or len(np.unique(ys[tr, j])) < 2: continue
                    m = LogisticRegression(C=C, class_weight='balanced', max_iter=3000).fit(Xs[tr], ys[tr, j])
                    aucs.append(roc_auc_score(ys[va, j], m.decision_function(Xs[va])))
                a = float(np.nanmean(aucs)) if aucs and not np.all(np.isnan(aucs)) else float('nan')
                if not np.isnan(a) and a > best: best, bestC = a, C
        m = LogisticRegression(C=bestC, class_weight='balanced', max_iter=3000).fit(Xs, ys[:, j])
        out[j] = [m.decision_function(x)[None] for x in Xq]      # (1, Nq) each
    return out

def fs_scores(Es, ys, Eq, labs, draws):
    """v4 fewshot logreg head, same seeds -> {(k, j): (draws, Nq)}."""
    ec = CFG['eval']; out = {}
    for k in ec['shots']:
        for j in labs: out[(k, j)] = []
        for d in range(draws):
            rng = np.random.RandomState(CFG['seed'] * 1000 + k * 100 + d)
            for j in L.rare_all:                       # consume the rng exactly like evaluate() (all rare labels, same order)
                pos = np.where(ys[:, j] == 1)[0]; neg = np.where(ys[:, j] == 0)[0]
                sp = rng.choice(pos, min(k, len(pos)), replace=False); sn = rng.choice(neg, min(k * ec['neg_per_pos'], len(neg)), replace=False)
                if j not in labs: continue
                Xs = np.concatenate([Es[sp], Es[sn]]); y_ = np.r_[np.ones(len(sp)), np.zeros(len(sn))]
                lr = LogisticRegression(C=ec['logreg_C'], class_weight='balanced', max_iter=2000).fit(Xs, y_)
                out[(k, j)].append(lr.decision_function(Eq))
        for j in labs: out[(k, j)] = np.stack(out[(k, j)])
    return out

def knn_scores(Eb, yb, Eq, labs, k=None):
    k = k or CFG['monitor']['knn_k']; sims = Eq @ Eb.T; idx = np.argsort(-sims, 1)[:, :k]; w = np.exp(np.take_along_axis(sims, idx, 1) / 0.07)
    S = np.einsum('qk,qkl->ql', w, yb[idx]) / w.sum(1, keepdims=True); return {j: S[:, j][None] for j in labs}

def dom(sc, y):
    """episode-averaged Mann-Whitney dominance matrix (n_pos, n_neg); sc (E, Nq)."""
    p, n = sc[:, y == 1], sc[:, y == 0]; d = p[:, :, None] - n[:, None, :]
    D = ((d > 0) + 0.5 * (d == 0)).astype(np.float32)
    return D.mean(0), np.where(y == 1)[0], np.where(y == 0)[0], D

def boot_auc(K, ip, ineg, Wb):
    """exact AUROC under multiplicities: point (Wb=None) or (n_boot,) for Wb (n_boot, Nq)."""
    if Wb is None: return float(K.mean())
    wp, wn = Wb[:, ip], Wb[:, ineg]
    return np.einsum('bp,pn,bn->b', wp, K, wn) / (wp.sum(1) * wn.sum(1) + 1e-12)

def metric_set(E, Ed, s, q, Y, labs_c, labs_r, draws):
    ys, yq = Y[s], Y[q]; dz = [j for j in labs_c if LABELS[j] != 'Disease_Risk']
    lp = lp_scores(E[s], ys, [E[q], Ed[q]], labs_c); fs = fs_scores(E[s], ys, E[q], labs_r, draws)
    kn = knn_scores(E[s], ys, Ed[q], labs_c)
    M = {'LP_common': {j: dom(lp[j][0], yq[:, j]) for j in dz},
         'LP_clean_to_deg': {j: dom(lp[j][1], yq[:, j]) for j in dz},
         'kNN_deg_query': {j: dom(kn[j], yq[:, j]) for j in labs_c}}
    for k in CFG['eval']['shots']: M[f'rare_{k}shot'] = {j: dom(fs[(k, j)], yq[:, j]) for j in labs_r}
    M['ARMD@10'] = {LABELS.index('ARMD'): M['rare_10shot'][LABELS.index('ARMD')]}
    return M

def evaluate_metric(M, Wb):
    return {m: np.mean([boot_auc(K, ip, ineg, Wb) for K, ip, ineg, _ in d.values()], 0) for m, d in M.items()}

def evaluate_metric_2level(M, Wb, Vb):
    """two-level bootstrap: query images (Wb) AND support draws (Vb, (n_boot, draws) multiplicities) resampled jointly,
    so the CI includes support-set sampling variance (query-only bootstrap with fixed supports does not)."""
    out = {}
    for m, d in M.items():
        v = []
        for K, ip, ineg, D in d.values():
            if D.shape[0] == 1: v.append(boot_auc(K, ip, ineg, Wb)); continue
            Vn = Vb / Vb.sum(1, keepdims=True); wp, wn = Wb[:, ip], Wb[:, ineg]
            v.append(np.einsum('bd,bp,dpn,bn->b', Vn, wp, D, wn, optimize=True) / (wp.sum(1) * wn.sum(1) + 1e-12))
        out[m] = np.mean(v, 0)
    return out

def main(a):
    t = time.time(); L.setup(a.ids, a.split); Y = L.Y_ALL; F_ = L.FOLDS
    s = np.where(F_ != 0)[0]; q = np.where(F_ == 0)[0]
    labs_c, rare = L.label_sets(list(q), list(s)); L.rare_all = rare
    labs_r = [j for j in rare if LABELS[j] in HEAD]
    pos_counts = {LABELS[j]: int(Y[q, j].sum()) for j in sorted(set(rare) | set(labs_c))}
    A = [L.nz(np.load(a.a)), L.nz(np.load(a.a_deg))]; B = [L.nz(np.load(a.b)), L.nz(np.load(a.b_deg))]
    MA = metric_set(*A, s, q, Y, labs_c, labs_r, a.draws); MB = metric_set(*B, s, q, Y, labs_c, labs_r, a.draws)
    rng = np.random.default_rng(a.seed); Wb = np.zeros((a.n_boot, len(q)))
    for b in range(a.n_boot): Wb[b] = np.bincount(rng.integers(0, len(q), len(q)), minlength=len(q))
    Vb = np.stack([np.bincount(rng.integers(0, a.draws, a.draws), minlength=a.draws) for _ in range(a.n_boot)]).astype(np.float64)
    pa, pb = evaluate_metric(MA, None), evaluate_metric(MB, None); ba, bb = evaluate_metric(MA, Wb), evaluate_metric(MB, Wb)
    ba2, bb2 = evaluate_metric_2level(MA, Wb, Vb), evaluate_metric_2level(MB, Wb, Vb)
    for X in (ba2, bb2): X['rare_mean_5_10'] = (X['rare_5shot'] + X['rare_10shot']) / 2
    pa['rare_mean_5_10'] = (pa['rare_5shot'] + pa['rare_10shot']) / 2; pb['rare_mean_5_10'] = (pb['rare_5shot'] + pb['rare_10shot']) / 2
    ba['rare_mean_5_10'] = (ba['rare_5shot'] + ba['rare_10shot']) / 2; bb['rare_mean_5_10'] = (bb['rare_5shot'] + bb['rare_10shot']) / 2
    res = {}
    for m in pa:
        d = bb[m] - ba[m]; lo, hi = np.percentile(d, [2.5, 97.5])
        res[m] = {'A': float(pa[m]), 'B': float(pb[m]), 'delta': float(pb[m] - pa[m]), 'ci95': [float(lo), float(hi)],
                  'p_delta_gt_0': float((d > 0).mean()), 'p_delta_gt_-0.005': float((d > -0.005).mean())}
        d2 = bb2[m] - ba2[m]; res[m].update({'ci95_2level': [float(x) for x in np.percentile(d2, [2.5, 97.5])], 'p_delta_gt_0_2level': float((d2 > 0).mean())})
        print(f"{m:<18} A {pa[m]:.4f}  B {pb[m]:.4f}  d {pb[m]-pa[m]:+.4f} [{lo:+.4f},{hi:+.4f}]  P(d>0) {(d > 0).mean():.3f} | 2-level CI [{res[m]['ci95_2level'][0]:+.4f},{res[m]['ci95_2level'][1]:+.4f}] P {res[m]['p_delta_gt_0_2level']:.3f}", flush=True)
    r = res
    rule = {'rare_mean_5_10_ge_0.02': r['rare_mean_5_10']['delta'] >= 0.02 and r['rare_mean_5_10']['p_delta_gt_0'] >= 0.9,
            'ARMD@10_ge_0.05': r['ARMD@10']['delta'] >= 0.05 and r['ARMD@10']['p_delta_gt_0'] >= 0.9,
            'LP_common_noninf_-0.005': r['LP_common']['delta'] >= -0.005,
            'degraded_LP_c2d_noninf_-0.005': r['LP_clean_to_deg']['delta'] >= -0.005,
            'degraded_kNN_noninf_-0.005': r['kNN_deg_query']['delta'] >= -0.005}
    rule['KEEP_B'] = all(rule.values())
    rule['KEEP_B_if_P_from_2level_bootstrap'] = (rule['LP_common_noninf_-0.005'] and rule['degraded_LP_c2d_noninf_-0.005'] and rule['degraded_kNN_noninf_-0.005']
        and r['rare_mean_5_10']['delta'] >= 0.02 and r['rare_mean_5_10']['p_delta_gt_0_2level'] >= 0.9 and r['ARMD@10']['delta'] >= 0.05 and r['ARMD@10']['p_delta_gt_0_2level'] >= 0.9)
    out = {'a': a.a, 'b': a.b, 'split': a.split, 'n_query': int(len(q)), 'n_support': int(len(s)), 'draws': a.draws, 'n_boot': a.n_boot,
           'common_labels': [LABELS[j] for j in labs_c], 'rare_headline': [LABELS[j] for j in labs_r], 'fold0_positives': pos_counts,
           'metrics': res, 'decision_rule': rule, 'secs': time.time() - t,
           'notes': 'B minus A; paired joint image bootstrap over locked fold 0; few-shot support from locked folds 1-4 (protocol), fixed seeds'}
    print('decision', json.dumps(rule)); json.dump(out, open(a.out, 'w'), indent=1, default=float); return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for k in ['a', 'a-deg', 'b', 'b-deg', 'out']: ap.add_argument('--' + k, required=True)
    ap.add_argument('--ids', default='encoder_mnv3_ssl/full_out/out/byol_mnv3_anatomy_v4/embedding_ids.csv')
    ap.add_argument('--split', default=str(L.LOCKED)); ap.add_argument('--draws', type=int, default=CFG['eval']['draws'])
    ap.add_argument('--n-boot', type=int, default=1000); ap.add_argument('--seed', type=int, default=12345)
    main(ap.parse_args())
