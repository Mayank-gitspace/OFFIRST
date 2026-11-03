"""v6 + re-tuned thresholds (from v6_tune/thresholds_v6.json, applied in memory only; config.yaml untouched)
on RFMiD TRAINING DR=1 images. Usage: python overlays_dr.py ID1,ID2,..."""
import sys, json, csv, cv2, numpy as np, torch
sys.path.insert(0, '/workspace/team_offirst/lesion_detector')
from common import *
from inference import load_model, sync_preprocessing, pick_lesions
torch.set_num_threads(8); dev = torch.device('cpu')
CK = '/workspace/team_offirst/ckpt_v6/ckpt/best.pth'; DRD = '/workspace/team_offirst/data/rfmid/dr_train/'
classes = ['MA', 'HE', 'EX', 'SE']; COL = {'MA': (0, 0, 255), 'HE': (0, 255, 0), 'EX': (0, 255, 255), 'SE': (255, 128, 0)}
rows = {r['ID']: r for r in csv.DictReader(open('/workspace/team_offirst/data/rfmid/Training_Set/Training_Set/RFMiD_Training_Labels.csv', encoding='utf-8-sig'))}
thr = json.load(open('/workspace/team_offirst/v6_tune/thresholds_v6.json'))['min_score']
cfg = load_config('/workspace/team_offirst/lesion_detector/config.yaml'); sync_preprocessing(cfg, CK, dev)
cfg['inference']['min_score'] = thr; m, _ = load_model(CK, cfg, dev); inf = cfg['inference']
out = {}
for i in sys.argv[1].split(','):
    assert rows[i]['DR'] == '1'
    bgr = cv2.imread(DRD + i + '.png'); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img_r, meta = preprocess_from_cfg(rgb, cfg['data'])
    p = sliding_window_predict(m, normalize(img_r), dev, inf['tile'], inf['overlap'], False)
    les, cands, fov = pick_lesions(p, img_r, classes, cfg, use_fov_mask=True)
    im = cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR); pm = p * fov['safe'][None]
    for k, c in enumerate(classes):
        im[cv2.dilate((pm[k] > inf['conf_threshold']).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = COL[c]
    for l in les:
        a, b, c2, d = l['box_resized']; cv2.rectangle(im, (a, b), (c2, d), (255, 255, 255), 5)
        cv2.putText(im, f"{l['lesion_type']} {l['confidence']:.3f}", (a, max(110, b - 8)), 0, 1.1, (255, 255, 255), 3)
    cv2.rectangle(im, (0, 0), (im.shape[1], 80), (0, 0, 0), -1)
    cv2.putText(im, f"v6 + v6 thr: {len(les)} crops", (15, 55), 0, 1.5, (255, 255, 255), 3)
    inp = cv2.resize(bgr, (int(1000 * bgr.shape[1] / bgr.shape[0]), 1000))
    cv2.rectangle(inp, (0, 0), (inp.shape[1], 80), (0, 0, 0), -1)
    cv2.putText(inp, f"RFMiD train {i} DR=1 {bgr.shape[1]}x{bgr.shape[0]}", (15, 55), 0, 1.2, (255, 255, 255), 3)
    row = np.concatenate([inp, cv2.resize(im, (int(1000 * im.shape[1] / im.shape[0]), 1000))], 1)
    cv2.putText(row, "prob>conf_thr: MA red HE green EX yellow SE blue; boxes = picked crops", (10, row.shape[0] - 15), 0, 0.9, (255, 255, 255), 2)
    fn = f'/workspace/team_offirst/v6_dr_check/RFMiD_{i}_DR_v6_overlay.png'; cv2.imwrite(fn, row)
    out[i] = {'size': f"{bgr.shape[1]}x{bgr.shape[0]}", 'crops': [(l['lesion_type'], round(l['confidence'], 3)) for l in les], 'png': fn}; print(i, out[i], flush=True)
json.dump(out, open('/workspace/team_offirst/v6_dr_check/overlays_report.json', 'w'), indent=1)
