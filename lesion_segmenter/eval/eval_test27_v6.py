"""Score v3 hires1440 and v5 on the official 27 IDRiD SEGMENTATION test images (IDRiD_55-81).
Pixel AUPR/Dice pooled over the dataset (binned AP, 20001 bins), at model resolution and mapped back
to original resolution. Crop-level pick/hit stats with several min_score sets.
Only reads: IDRiD 'A. Segmentation' testing set. No grading, no RFMiD."""
import sys, json, copy, cv2, numpy as np, torch
sys.path.insert(0, '/workspace/team_offirst/lesion_detector')
from common import *
from inference import load_model, sync_preprocessing, pick_lesions
torch.set_num_threads(8); dev = torch.device('cpu')
L = '/workspace/team_offirst/lesion_detector/'
S = '/workspace/team_offirst/data/idrid_seg/A. Segmentation/'
OUT = '/workspace/team_offirst/v6_eval/'
classes = ['MA', 'HE', 'EX', 'SE']
names = {'MA': '1. Microaneurysms', 'HE': '2. Haemorrhages', 'EX': '3. Hard Exudates', 'SE': '4. Soft Exudates'}
IDS = [f'IDRiD_{i}' for i in range(55, 82)]
NB = 20001
THR = {'own_config': None, 'v5_config': {'MA': .94, 'HE': .5, 'EX': .905, 'SE': .5},
       'tight': {'MA': .96, 'HE': .94, 'EX': .94, 'SE': .5}}

def setup(cfg_path, ckpt, long_side=None):
    cfg = load_config(cfg_path)
    if long_side: cfg['data']['long_side'] = long_side
    sync_preprocessing(cfg, ckpt, dev); m, _ = load_model(ckpt, cfg, dev); return cfg, m
MODELS = {'v6': (L + 'config.yaml', '/workspace/team_offirst/ckpt_v6/ckpt/best.pth', None)}

def to_orig(p, meta):
    """(C,h,w) probs in resized space -> (C,H,W) original frame (zeros outside the crop)."""
    H, W = meta['orig_h'], meta['orig_w']; x0, y0, x1, y1 = meta['x0'], meta['y0'], meta['x1'], meta['y1']
    out = np.zeros((p.shape[0], H, W), np.float32)
    up = np.stack([cv2.resize(p[c].astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) for c in range(p.shape[0])])
    ax0, ay0, ax1, ay1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    out[:, ay0:ay1, ax0:ax1] = up[:, ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
    return out

class Acc:
    def __init__(s): s.pos = np.zeros((4, NB)); s.neg = np.zeros((4, NB)); s.inter = np.zeros(4); s.ps = np.zeros(4); s.ts = np.zeros(4)
    def add(s, p, g, c):
        b = np.clip((p * (NB - 1)).round().astype(np.int64), 0, NB - 1).ravel(); gg = g.ravel()
        s.pos[c] += np.bincount(b[gg], minlength=NB); s.neg[c] += np.bincount(b[~gg], minlength=NB)
        pr = p > 0.5; s.inter[c] += (pr & g).sum(); s.ps[c] += pr.sum(); s.ts[c] += g.sum()
    def res(s):
        ap, dice = [], []
        for c in range(4):
            tp = np.cumsum(s.pos[c][::-1]); fp = np.cumsum(s.neg[c][::-1]); P = tp[-1]
            if P == 0: ap.append(float('nan')); dice.append(float('nan')); continue
            prec = tp / np.maximum(tp + fp, 1); rec = tp / P
            ap.append(float(np.sum(np.diff(np.concatenate([[0], rec])) * prec)))   # step AP (sklearn-style)
            dice.append(float(2 * s.inter[c] / (s.ps[c] + s.ts[c])))
        return ap, dice

report = {}
for mname, (cp, ck, ls) in MODELS.items():
    cfg, m = setup(cp, ck, ls); inf = cfg['inference']
    acc_r, acc_o = Acc(), Acc(); per_img = {}; crop = {k: {} for k in THR}
    for i in IDS:
        path = S + '1. Original Images/b. Testing Set/' + i + '.jpg'
        assert 'Grading' not in path and 'Testing Set' in path
        rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        gtraw = np.stack([(lambda g: (g > 0).astype(np.uint8) if g is not None else np.zeros(rgb.shape[:2], np.uint8))(
            cv2.imread(f"{S}2. All Segmentation Groundtruths/b. Testing Set/{names[c]}/{i}_{c}.tif", 0)) for c in classes], -1)
        has = [bool(gtraw[..., k].any()) for k in range(4)]
        img_r, meta = preprocess_from_cfg(rgb, cfg['data'])
        p = sliding_window_predict(m, normalize(img_r), dev, inf['tile'], inf['overlap'], False)
        gt = preprocess_mask(gtraw, meta, img_r.shape[:2]).transpose(2, 0, 1) > 0
        po = to_orig(p, meta); go = gtraw.transpose(2, 0, 1) > 0
        for k in range(4):
            if k == 3 and not has[3]: continue      # SE: only images with SE GT
            acc_r.add(p[k], gt[k], k); acc_o.add(po[k], go[k], k)
        del po
        np.savez_compressed(f'{OUT}probs_{mname}_{i}.npz', p=(p * 255).astype(np.uint8))
        for tn, t in THR.items():
            c2 = copy.deepcopy(cfg)
            if t is not None: c2['inference']['min_score'] = t
            les, _, _ = pick_lesions(p, img_r, classes, c2, use_fov_mask=True)
            hits = [int(gt[classes.index(l['lesion_type'])][max(0, l['box_resized'][1]):l['box_resized'][3], max(0, l['box_resized'][0]):l['box_resized'][2]].any()) for l in les]
            crop[tn][i] = {'lesions': [(l['lesion_type'], round(l['confidence'], 3), l['box_resized']) for l in les], 'hits': hits, 'gt_present': dict(zip(classes, has))}
        print(mname, i, {tn: [(a, b, h) for (a, b, _), h in zip(crop[tn][i]['lesions'], crop[tn][i]['hits'])] for tn in THR}, flush=True)
    ar, dr = acc_r.res(); ao, do = acc_o.res()
    report[mname] = {'own_min_score': inf['min_score'], 'model_res': {'aupr': dict(zip(classes, ar)), 'dice05': dict(zip(classes, dr))},
                     'orig_res': {'aupr': dict(zip(classes, ao)), 'dice05': dict(zip(classes, do))}, 'crop': crop}
    json.dump(report, open(OUT + 'test27_report.json', 'w'), indent=1, default=str)
print('DONE')
