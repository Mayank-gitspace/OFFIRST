import sys, json, os, cv2, numpy as np, torch
sys.path.insert(0, 'lesion_detector')
from common import *
from inference import load_model
cfg = load_config('lesion_detector/config.yaml'); cfg['data']['long_side'] = 1440
inf = cfg['inference']; torch.set_num_threads(8); dev = torch.device('cpu')
m, classes = load_model('kaggle_runs/hires1440/ckpt/best.pth', cfg, dev)
sp = json.load(open('data/grade0_split.json'))
G = 'data/idrid_grade/Imagenes/Imagenes/'; S = 'data/idrid_seg/A. Segmentation/1. Original Images/b. Testing Set/'
jobs = [(i, G + i + '.jpg') for i in sp['tune_test_grade0'] + sp['heldout_eval']] + \
       [(f'IDRiD_{k}', S + f'IDRiD_{k}.jpg') for k in range(55, 82)]
for name, path in jobs:
    out = f'probs/{name}.npz'
    if os.path.exists(out): continue
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img_r, meta = preprocess_image(rgb, 1440, cfg['data']['fundus_threshold'])
    p = sliding_window_predict(m, normalize(img_r), dev, inf['tile'], inf['overlap'], False)
    np.savez_compressed(out, probs=(p * 255).round().astype(np.uint8), p_f16=p.astype(np.float16), meta=json.dumps(meta))
    print(name, flush=True)
