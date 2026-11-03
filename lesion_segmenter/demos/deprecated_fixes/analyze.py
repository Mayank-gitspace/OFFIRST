import sys, json, cv2, numpy as np, pickle
sys.path.insert(0, 'lesion_detector')
from common import *
from inference import find_candidates
from fov_mask import apply_fov_mask, filter_candidates
cfg = load_config('lesion_detector/config.yaml'); inf = cfg['inference']; classes = ['MA','HE','EX','SE']
sp = json.load(open('data/grade0_split.json'))
G = 'data/idrid_grade/Imagenes/Imagenes/'; S = 'data/idrid_seg/A. Segmentation/'
names = {'MA': '1. Microaneurysms', 'HE': '2. Haemorrhages', 'EX': '3. Hard Exudates', 'SE': '4. Soft Exudates'}
sets = {'tune_healthy': sp['tune_test_grade0'], 'heldout_healthy': sp['heldout_eval'], 'dr_test': [f'IDRiD_{k}' for k in range(55, 82)]}
out = {}
for sname, ids in sets.items():
    for i in ids:
        path = (S + '1. Original Images/b. Testing Set/' + i + '.jpg') if sname == 'dr_test' else G + i + '.jpg'
        rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img_r, meta = preprocess_image(rgb, 1440, 15)
        p = np.load(f'probs/{i}.npz')['p_f16'].astype(np.float32)
        pm, fov = apply_fov_mask(p, img_r, inf)
        cfg2 = json.loads(json.dumps(cfg)); cfg2['inference']['blob_score'] = 'mean'
        cands = filter_candidates(find_candidates(pm, classes, cfg2), fov['safe'])
        gt = None
        if sname == 'dr_test':
            gt = np.zeros((4,) + img_r.shape[:2], bool)
            for k, c in enumerate(classes):
                g = cv2.imread(f"{S}2. All Segmentation Groundtruths/b. Testing Set/{names[c]}/{i}_{c}.tif", 0)
                if g is not None: gt[k] = preprocess_mask((g > 0)[..., None].astype(np.uint8), meta, img_r.shape[:2])[..., 0] > 0
            gtd = np.stack([cv2.dilate(g.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0 for g in gt])
            anyg = gt.any(0)
            for c in cands:
                x, y = int(round(c['cx'])), int(round(c['cy']))
                c['hit_same'] = bool(gtd[classes.index(c['lesion_type']), y, x])
                c['hit_any_center'] = bool(gtd.any(0)[y, x])
                c['gt_px_in_crop'] = int(anyg[max(0, y-64):y+64, max(0, x-64):x+64].sum())
            out[i] = {'set': sname, 'cands': cands, 'gt_classes': [c for k, c in enumerate(classes) if gt[k].any()]}
        else:
            out[i] = {'set': sname, 'cands': cands}
pickle.dump(out, open('cands.pkl', 'wb'))
print('done', len(out))
