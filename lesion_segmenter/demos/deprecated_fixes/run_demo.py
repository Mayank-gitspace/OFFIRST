import sys, json, cv2, numpy as np
sys.path.insert(0, 'lesion_detector')
from common import *
from inference import pick_lesions, make_views, legacy_cfg
cfg = load_config('lesion_detector/config.yaml'); cfg['data']['long_side'] = 1440
old = legacy_cfg(cfg); classes = ['MA','HE','EX','SE']; inf = cfg['inference']
sp = json.load(open('data/grade0_split.json'))
G = 'data/idrid_grade/Imagenes/Imagenes/'; S = 'data/idrid_seg/A. Segmentation/'
names = {'MA': '1. Microaneurysms', 'HE': '2. Haemorrhages', 'EX': '3. Hard Exudates', 'SE': '4. Soft Exudates'}
COL = {'MA': (0, 0, 255), 'HE': (0, 255, 0), 'EX': (0, 255, 255), 'SE': (255, 128, 0)}
sets = {'tune_healthy': sp['tune_test_grade0'], 'heldout_healthy': sp['heldout_eval'], 'dr_test': [f'IDRiD_{k}' for k in range(55, 82)]}
DEMO = ['IDRiD_029test', 'IDRiD_55', 'IDRiD_60', 'IDRiD_67', 'IDRiD_75']
def load(i, s):
    path = (S + '1. Original Images/b. Testing Set/' + i + '.jpg') if s == 'dr_test' else G + i + '.jpg'
    bgr = cv2.imread(path); img_r, meta = preprocess_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 1440, 15)
    p = np.load(f'probs/{i}.npz')['p_f16'].astype(np.float32)
    gt = None
    if s == 'dr_test':
        gt = np.zeros((4,) + img_r.shape[:2], bool)
        for k, c in enumerate(classes):
            g = cv2.imread(f"{S}2. All Segmentation Groundtruths/b. Testing Set/{names[c]}/{i}_{c}.tif", 0)
            if g is not None: gt[k] = preprocess_mask((g > 0)[..., None].astype(np.uint8), meta, img_r.shape[:2])[..., 0] > 0
    return bgr, img_r, meta, p, gt
def hits(les, gt):
    if gt is None: return None
    a = gt.any(0); return [int(a[max(0,l['box_resized'][1]):l['box_resized'][3], max(0,l['box_resized'][0]):l['box_resized'][2]].sum() > 0) for l in les]
def panel(base, p, les, nc, title, gt=None, fov=None):
    im = base.copy()
    if gt is not None:
        for k in range(4):
            cnts, _ = cv2.findContours(gt[k].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(im, cnts, -1, (255, 255, 255), 1)
    for k, c in enumerate(classes):
        im[cv2.dilate((p[k] > inf['conf_threshold']).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = COL[c]
    if fov is not None:
        cnts, _ = cv2.findContours(fov['safe'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(im, cnts, -1, (255, 255, 0), 2)
    for k, l in enumerate(les):
        a, b, c2, d = l['box_resized']; cv2.rectangle(im, (a, b), (c2, d), (255, 255, 255), 5)
        cv2.putText(im, f"{l['lesion_type']} {l['confidence']:.3f}", (a, max(30, b - 8)), 0, 1.1, (255, 255, 255), 3)
    cv2.rectangle(im, (0, 0), (im.shape[1], 80), (0, 0, 0), -1); cv2.putText(im, title, (15, 55), 0, 1.5, (255, 255, 255), 3)
    return im
summary, demo = {}, {}
for s, ids in sets.items():
    agg = {'old': [], 'new': []}
    for i in ids:
        bgr, img_r, meta, p, gt = load(i, s)
        lo, co, _ = pick_lesions(p, img_r, classes, old, use_fov_mask=False)
        ln, cn, fov = pick_lesions(p, img_r, classes, cfg, use_fov_mask=True)
        vo = make_views(lo, meta, old, i, np.random.default_rng(42)); vn = make_views(ln, meta, cfg, i, np.random.default_rng(42))
        for tag, l, c, v in (('old', lo, co, vo), ('new', ln, cn, vn)):
            agg[tag].append({'id': i, 'blobs': len(c), 'lesions': len(l), 'crops': len(v), 'hits': hits(l, gt),
                             'picked': [(x['lesion_type'], round(x['confidence'], 3)) for x in l]})
        if i in DEMO:
            demo[i] = {t: agg[t][-1] for t in agg}
            base = cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR)
            A = panel(base, p, lo, len(co), f"BEFORE (peak, top-4, no FOV): {len(co)} blobs, {len(lo)} lesions, {len(vo)} crops", gt)
            B = panel(base, p * fov['safe'][None], ln, len(cn), f"AFTER (mean, min_score, top-1/class, FOV): {len(cn)} blobs>=min, {len(ln)} lesions, {len(vn)} crops", gt, fov)
            Ipanel = base.copy(); cv2.rectangle(Ipanel, (0, 0), (Ipanel.shape[1], 80), (0, 0, 0), -1)
            cv2.putText(Ipanel, f"INPUT {i} ({'healthy grade 0' if 'test' in i else 'DR test, white = GT outlines'})", (15, 55), 0, 1.5, (255, 255, 255), 3)
            row = np.concatenate([Ipanel, A, B], 1)
            row = cv2.resize(row, (2700, int(2700 * row.shape[0] / row.shape[1])), interpolation=cv2.INTER_AREA)
            cv2.imwrite(f'{i}_before_after.png', row)
            for tag, vv in (('before', vo), ('after', vn)):
                tiles = []
                for v in vv:
                    a, b, c2, d = v['bbox']; t = cv2.resize(bgr[b:d, a:c2], (180, 180))
                    t = cv2.copyMakeBorder(t, 26, 4, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                    cv2.putText(t, f"{v['lesion_type']} L{v['lesion_idx']} v{v['view_idx']}", (4, 19), 0, 0.5, (0, 0, 0), 1); tiles.append(t)
                if not tiles:
                    t = np.full((120, 560, 3), 255, np.uint8); cv2.putText(t, f"{i} {tag}: 0 crops", (10, 70), 0, 1.0, (0, 0, 0), 2); tiles = [t]
                rows = [np.concatenate(tiles[k:k + 6], 1) for k in range(0, len(tiles), 6)]; w = max(x.shape[1] for x in rows)
                rows = [cv2.copyMakeBorder(x, 0, 0, 0, w - x.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for x in rows]
                cv2.imwrite(f'{i}_crops_{tag}.png', np.concatenate(rows, 0))
    summary[s] = {}
    for t in ('old', 'new'):
        L = agg[t]; allh = [h for x in L for h in (x['hits'] or [])]
        summary[s][t] = {'images': len(L), 'imgs_with_crops': sum(x['crops'] > 0 for x in L), 'total_lesions': sum(x['lesions'] for x in L),
                         'total_crops': sum(x['crops'] for x in L), 'mean_crops_per_img': round(np.mean([x['crops'] for x in L]), 2)}
        if allh: summary[s][t]['lesion_crops_touching_GT'] = f"{sum(allh)}/{len(allh)}"
    summary[s]['per_image'] = agg
json.dump({'thresholds': inf['min_score'], 'summary': {k: {t: v[t] for t in ('old', 'new')} for k, v in summary.items()}, 'demo': demo,
           'per_image': {k: v['per_image'] for k, v in summary.items()}}, open('report.json', 'w'), indent=1)
for k, v in summary.items(): print(k, '\n  OLD', v['old'], '\n  NEW', v['new'])
for i, v in demo.items(): print(i, 'OLD', {k: v['old'][k] for k in ('blobs', 'lesions', 'crops', 'hits')}, '| NEW', {k: v['new'][k] for k in ('blobs', 'lesions', 'crops', 'hits', 'picked')})
