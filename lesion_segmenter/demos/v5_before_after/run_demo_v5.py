"""Before/after demo. BEFORE = previous pipeline (v3 hires1440 ckpt, bbox preprocessing, old grading-tuned
min_score). AFTER = v5 ckpt (camera-robust, circle preprocessing) + min_score tuned on RFMiD-train normals.
Images: IDRiD seg TEST DR images + RFMiD TRAINING normals from rfmid_thresh_tune.txt only."""
import sys, json, cv2, numpy as np, torch
sys.path.insert(0, 'lesion_detector')
from common import *
from inference import load_model, sync_preprocessing, pick_lesions, make_views
torch.set_num_threads(8); dev = torch.device('cpu')
L = 'lesion_detector/'
S = 'data/idrid_seg/A. Segmentation/'
R = 'data/rfmid/thresh_tune/'
classes = ['MA', 'HE', 'EX', 'SE']
names = {'MA': '1. Microaneurysms', 'HE': '2. Haemorrhages', 'EX': '3. Hard Exudates', 'SE': '4. Soft Exudates'}
COL = {'MA': (0, 0, 255), 'HE': (0, 255, 0), 'EX': (0, 255, 255), 'SE': (255, 128, 0)}
tune = set(l.strip() for l in open('data_splits/rfmid_thresh_tune.txt'))
DR = ['IDRiD_55', 'IDRiD_60', 'IDRiD_67', 'IDRiD_75']
HEALTHY = sys.argv[1].split(',')
assert all(h in tune for h in HEALTHY)

def setup(cfg_path, ckpt, long_side=None):
    cfg = load_config(cfg_path)
    if long_side: cfg['data']['long_side'] = long_side
    sync_preprocessing(cfg, ckpt, dev); m, _ = load_model(ckpt, cfg, dev); return cfg, m
P = {'before': setup(L + 'original/config_pre_v5_thresholds.yaml', 'kaggle_runs/hires1440/ckpt/best.pth', 1440),
     'after': setup(L + 'config.yaml', 'ckpt_v5/ckpt/best.pth')}

def run(tag, i, rgb, gtraw):
    cfg, m = P[tag]; inf = cfg['inference']
    img_r, meta = preprocess_from_cfg(rgb, cfg['data'])
    p = sliding_window_predict(m, normalize(img_r), dev, inf['tile'], inf['overlap'], False)
    les, cands, fov = pick_lesions(p, img_r, classes, cfg, use_fov_mask=True)
    gt = None
    if gtraw is not None:
        gt = preprocess_mask(gtraw, meta, img_r.shape[:2]).transpose(2, 0, 1) > 0
    hits = None if gt is None else [int(gt[classes.index(l['lesion_type'])][max(0, l['box_resized'][1]):l['box_resized'][3], max(0, l['box_resized'][0]):l['box_resized'][2]].any()) for l in les]
    im = cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR)
    if gt is not None:
        for k in range(4):
            cn, _ = cv2.findContours(gt[k].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(im, cn, -1, (255, 255, 255), 1)
    pm = p * fov['safe'][None]
    for k, c in enumerate(classes):
        im[cv2.dilate((pm[k] > inf['conf_threshold']).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = COL[c]
    for k, l in enumerate(les):
        a, b, c2, d = l['box_resized']; ok = '' if hits is None else (' hit' if hits[k] else ' miss')
        cv2.rectangle(im, (a, b), (c2, d), (255, 255, 255), 5)
        cv2.putText(im, f"{l['lesion_type']} {l['confidence']:.3f}{ok}", (a, max(110, b - 8)), 0, 1.1, (255, 255, 255), 3)
    title = (f"BEFORE v3+old thr: {len(les)} lesions" if tag == 'before' else f"AFTER v5+RFMiD thr: {len(les)} lesions")
    cv2.rectangle(im, (0, 0), (im.shape[1], 80), (0, 0, 0), -1); cv2.putText(im, title, (15, 55), 0, 1.5, (255, 255, 255), 3)
    return cv2.resize(im, (int(1000 * im.shape[1] / im.shape[0]), 1000)), {'lesions': [(l['lesion_type'], round(l['confidence'], 3)) for l in les], 'hits': hits}

out = {}
for i in DR + HEALTHY:
    if i.startswith('IDRiD_'):
        path = S + '1. Original Images/b. Testing Set/' + i + '.jpg'
    else:
        path = R + i + '.png'
    assert 'Grading' not in path and 'Imagenes' not in path
    bgr = cv2.imread(path); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gtraw = None
    if i.startswith('IDRiD_'):
        gtraw = np.stack([(lambda g: (g > 0).astype(np.uint8) if g is not None else np.zeros(rgb.shape[:2], np.uint8))(
            cv2.imread(f"{S}2. All Segmentation Groundtruths/b. Testing Set/{names[c]}/{i}_{c}.tif", 0)) for c in classes], -1)
    A, ra = run('before', i, rgb, gtraw); B, rb = run('after', i, rgb, gtraw)
    inp = cv2.resize(bgr, (int(1000 * bgr.shape[1] / bgr.shape[0]), 1000))
    lab = f"INPUT {i} " + ("(IDRiD seg test, DR; white = GT)" if i.startswith('IDRiD') else f"(RFMiD train normal, {bgr.shape[1]}x{bgr.shape[0]})")
    cv2.rectangle(inp, (0, 0), (inp.shape[1], 80), (0, 0, 0), -1); cv2.putText(inp, lab, (15, 55), 0, 1.2, (255, 255, 255), 3)
    row = np.concatenate([inp, A, B], 1); row = cv2.resize(row, (2700, int(2700 * row.shape[0] / row.shape[1])), interpolation=cv2.INTER_AREA)
    name = f"{i if i.startswith('IDRiD') else 'RFMiD_' + i}_before_after.png"; cv2.imwrite(name, row)
    out[i] = {'before': ra, 'after': rb, 'png': 'v5_demo/' + name}; print(i, out[i], flush=True)
json.dump(out, open('demo_report.json', 'w'), indent=1)
