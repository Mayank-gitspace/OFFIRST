import sys, json, cv2, numpy as np
sys.path.insert(0, 'lesion_detector')
from common import *
S = 'data/idrid_seg/A. Segmentation/'
classes = ['MA', 'HE', 'EX', 'SE']
names = {'MA': '1. Microaneurysms', 'HE': '2. Haemorrhages', 'EX': '3. Hard Exudates', 'SE': '4. Soft Exudates'}
COL = {'MA': (0, 0, 255), 'HE': (0, 255, 0), 'EX': (0, 255, 255), 'SE': (255, 128, 0)}
R = json.load(open('test27_report.json'))
D = {'v3_hires1440': {'preprocess': 'bbox', 'long_side': 1440, 'fundus_threshold': 15},
     'v5': {'preprocess': 'circle', 'long_side': 1440, 'fov_diameter': 1440, 'fov_red_threshold': 12, 'clahe': False}}
for i, tag in [(sys.argv[1], 'good'), (sys.argv[2], 'bad')]:
    bgr = cv2.imread(S + '1. Original Images/b. Testing Set/' + i + '.jpg'); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gtraw = np.stack([(lambda g: (g > 0).astype(np.uint8) if g is not None else np.zeros(rgb.shape[:2], np.uint8))(
        cv2.imread(f"{S}2. All Segmentation Groundtruths/b. Testing Set/{names[c]}/{i}_{c}.tif", 0)) for c in classes], -1)
    panels = []
    for m in ['v3_hires1440', 'v5']:
        img_r, meta = preprocess_from_cfg(rgb, D[m]); p = np.load(f'probs_{m}_{i}.npz')['p'] / 255.
        gt = preprocess_mask(gtraw, meta, img_r.shape[:2]).transpose(2, 0, 1) > 0
        for kind in ['gt', 'pred']:
            im = cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR)
            for k, c in enumerate(classes):
                msk = gt[k] if kind == 'gt' else p[k] > 0.5
                im[cv2.dilate(msk.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = COL[c]
            if kind == 'pred':
                cr = R[m]['crop']['own_config'][i]
                for (lt, cf, b), h in zip(cr['lesions'], cr['hits']):
                    b = [int(x) for x in b.strip('[]()').split(',')] if isinstance(b, str) else b
                    cv2.rectangle(im, (b[0], b[1]), (b[2], b[3]), (255, 255, 255), 5)
                    cv2.putText(im, f"{lt} {cf:.3f} {'hit' if h else 'MISS'}", (b[0], max(110, b[1] - 8)), 0, 1.1, (255, 255, 255), 3)
            t = f"{m} {'GROUND TRUTH' if kind == 'gt' else 'pred>0.5 + crops (own thr)'}"
            cv2.rectangle(im, (0, 0), (im.shape[1], 80), (0, 0, 0), -1); cv2.putText(im, t, (15, 55), 0, 1.3, (255, 255, 255), 3)
            panels.append(cv2.resize(im, (int(1000 * im.shape[1] / im.shape[0]), 1000)))
    row = np.concatenate(panels, 1); row = cv2.resize(row, (3200, int(3200 * row.shape[0] / row.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.putText(row, f"{i} ({tag})  MA red  HE green  EX yellow  SE blue", (10, row.shape[0] - 15), 0, 0.9, (255, 255, 255), 2)
    cv2.imwrite(f'{i}_{tag}_overlay.png', row); print(f'v5_test27/{i}_{tag}_overlay.png')
