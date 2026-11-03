"""Camera-shortcut check: v6 + re-tuned thresholds on RFMiD TRAINING images with DR=1 vs the 341 tune normals.
Only RFMiD Training split (HF ctmedtech/RFMID Training_Set or Kaggle awsaf49). Never val/test."""
import sys, json, csv, cv2, numpy as np, torch
from pathlib import Path
from PIL import Image
from collections import Counter
sys.path.insert(0, '/workspace/team_offirst/lesion_detector')
from common import load_config
from inference import load_model, sync_preprocessing
from tune_thresholds import top_per_class, guard_path
torch.set_num_threads(8); dev = torch.device('cpu')
CK = '/workspace/team_offirst/ckpt_v6/ckpt/best.pth'; DRD = Path('/workspace/team_offirst/data/rfmid/dr_train'); HD = Path('/workspace/team_offirst/data/rfmid/thresh_tune')
OUT = Path('/workspace/team_offirst/v6_dr_check'); guard_path(DRD)
classes = ['MA', 'HE', 'EX', 'SE']
thr = json.load(open('/workspace/team_offirst/v6_tune/thresholds_v6.json'))['min_score']
rows = {r['ID']: r for r in csv.DictReader(open('/workspace/team_offirst/data/rfmid/Training_Set/Training_Set/RFMiD_Training_Labels.csv', encoding='utf-8-sig'))}
dr_ids = [i for i, r in rows.items() if r['DR'] == '1']
tune = set(l.strip() for l in open('/workspace/team_offirst/data_splits/rfmid_thresh_tune.txt') if l.strip())
assert not (set(dr_ids) & tune)
cfg = load_config('/workspace/team_offirst/lesion_detector/config.yaml'); sync_preprocessing(cfg, CK, dev); m, _ = load_model(CK, cfg, dev)
cache_p = OUT / 'dr_cache.json'; cache = json.load(open(cache_p)) if cache_p.exists() else {}
for n, i in enumerate(dr_ids):
    if i in cache: continue
    p = DRD / f'{i}.png'
    if not p.exists(): print('missing', i); continue
    r = top_per_class(m, {'id': i, 'image': p, 'masks': None}, classes, cfg, dev, False, cfg['data'], cfg['inference'])
    w, h = Image.open(p).size
    cache[i] = {'res': r, 'size': f"{w}x{h}"}
    if n % 20 == 0: json.dump(cache, open(cache_p, 'w')); print(n, flush=True)
json.dump(cache, open(cache_p, 'w'))
hc = json.load(open('/workspace/team_offirst/v6_tune/cache.json'))
def stats(items):
    n = len(items); any_ = sum(any(v['res'][c] and v['res'][c]['score'] >= thr[c] for c in classes) for v in items)
    return {'n': n, 'any_crop_rate': round(any_ / max(n, 1), 3), **{c: round(sum(bool(v['res'][c] and v['res'][c]['score'] >= thr[c]) for v in items) / max(n, 1), 3) for c in classes}}
sizes = {}
for i in tune:
    w, h = Image.open(HD / f'{i}.png').size; sizes[i] = f"{w}x{h}"
H = [{'res': hc[f'healthy/rfmid_{i}'], 'size': sizes[i]} for i in tune]
D = list(cache.values())
rep = {'thresholds': thr, 'healthy_all': stats(H), 'dr_all': stats(D), 'per_camera': {}}
for s in sorted(set(v['size'] for v in H + D)):
    rep['per_camera'][s] = {'healthy': stats([v for v in H if v['size'] == s]), 'dr': stats([v for v in D if v['size'] == s])}
json.dump(rep, open(OUT / 'dr_check_report.json', 'w'), indent=1); print(json.dumps(rep, indent=1))
