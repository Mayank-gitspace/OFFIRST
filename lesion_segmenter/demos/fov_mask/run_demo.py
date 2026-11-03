import sys, json, os, time, cv2, numpy as np, torch
sys.path.insert(0, 'lesion_detector')
from common import *
from inference import load_model, pick_lesions, make_views
OUT = 'fov_mask_demo'
cfg = load_config('lesion_detector/config.yaml'); cfg['data']['long_side'] = 1440
inf = cfg['inference']; dev = torch.device('cpu'); torch.set_num_threads(8)
m, classes = load_model('kaggle_runs/hires1440/ckpt/best.pth', cfg, dev)
SEG = 'user_test/t/A. Segmentation/'
names = {'MA': '1. Microaneurysms', 'HE': '2. Haemorrhages', 'EX': '3. Hard Exudates', 'SE': '4. Soft Exudates'}
IMGS = {'IDRiD_029test_healthy': 'user_test/grad/Imagenes/Imagenes/IDRiD_029test.jpg',
        'IDRiD_55_DR': SEG + '1. Original Images/b. Testing Set/IDRiD_55.jpg'}
COL = {'MA': (0, 0, 255), 'HE': (0, 255, 0), 'EX': (0, 255, 255), 'SE': (255, 128, 0)}  # BGR
report = {}
for tag, path in IMGS.items():
    bgr = cv2.imread(path); assert bgr is not None, path
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img_r, meta = preprocess_image(rgb, 1440, cfg['data']['fundus_threshold'])
    cache = f'{OUT}/{tag}_probs.npz'
    if os.path.exists(cache):
        p = np.load(cache)['probs']
    else:
        t = time.time(); p = sliding_window_predict(m, normalize(img_r), dev, inf['tile'], inf['overlap'], False)
        print(tag, 'inference s', round(time.time() - t, 1)); np.savez_compressed(cache, probs=p)
    # GT (DR image only) in resized space
    gt = None
    if tag.endswith('DR'):
        gt = np.zeros((4,) + img_r.shape[:2], bool)
        for i, c in enumerate(classes):
            g = cv2.imread(f"{SEG}2. All Segmentation Groundtruths/b. Testing Set/{names[c]}/IDRiD_55_{c}.tif", 0)
            if g is not None: gt[i] = preprocess_mask((g > 0)[..., None].astype(np.uint8), meta, img_r.shape[:2])[..., 0] > 0
    les0, c0, _ = pick_lesions(p, img_r, classes, cfg, use_fov_mask=False)
    les1, c1, fov = pick_lesions(p, img_r, classes, cfg, use_fov_mask=True)
    safe, ret = fov['safe'], fov['retina']
    dist = cv2.distanceTransform(ret, cv2.DIST_L2, 5)   # px from the retina rim
    def desc(c):
        x, y = int(round(c['cx'])), int(round(c['cy']))
        d = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items() if k != 'box_resized'}
        d['rim_dist_px'] = round(float(dist[y, x]), 1); d['in_safe'] = bool(safe[y, x])
        if gt is not None:
            b = c.get('box_resized') or [x - 64, y - 64, x + 64, y + 64]
            d['gt_px_in_crop'] = int(gt[:, max(0,b[1]):b[3], max(0,b[0]):b[2]].sum())
        return d
    v0 = make_views(les0, meta, cfg, tag, np.random.default_rng(42))
    v1 = make_views(les1, meta, cfg, tag, np.random.default_rng(42))
    r = {'radius_px': round(fov['radius_px'], 1), 'erode_px': fov['erode_px'], 'img_r_shape': list(img_r.shape[:2]),
         'before': {'candidate_blobs': len(c0), 'lesions_picked': len(les0), 'crops': len(v0),
                    'candidates': [desc(c) for c in c0], 'picked': [desc(c) for c in les0]},
         'after': {'candidate_blobs': len(c1), 'lesions_picked': len(les1), 'crops': len(v1),
                   'candidates': [desc(c) for c in c1], 'picked': [desc(c) for c in les1]}}
    r['before']['cands_outside_safe'] = sum(not d['in_safe'] for d in r['before']['candidates'])
    report[tag] = r
    print(tag, {k: r[k] for k in ('radius_px', 'erode_px')}, 'before', len(c0), len(les0), len(v0), 'after', len(c1), len(les1), len(v1))
    # ---------- visuals (resized-fundus space)
    base = cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR)
    ov = base.copy(); band = (ret > 0) & (safe == 0)
    ov[band] = (0.3 * ov[band] + 0.7 * np.array([255, 0, 255])).astype(np.uint8)
    cnts, _ = cv2.findContours(safe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(ov, cnts, -1, (255, 255, 0), 3)
    def blobs(pm, cands, les, title):
        im = base.copy()
        for i, c in enumerate(classes):
            mk = cv2.dilate((pm[i] > inf['conf_threshold']).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            im[mk] = COL[c]
        for c in cands: cv2.circle(im, (int(c['cx']), int(c['cy'])), 22, COL[c['lesion_type']], 3)
        for k, l in enumerate(les):
            a, b, cc, d = l['box_resized']; cv2.rectangle(im, (a, b), (cc, d), (255, 255, 255), 5)
            cv2.putText(im, f"{k+1}:{l['lesion_type']}", (a, max(30, b - 8)), 0, 1.2, (255, 255, 255), 3)
        cv2.drawContours(im, cnts, -1, (255, 255, 0), 2)
        cv2.putText(im, title, (20, 60), 0, 1.6, (255, 255, 255), 4)
        return im
    pm1 = p * safe[None]
    panels = [base.copy(), ov, blobs(p, c0, les0, f"BEFORE: {len(c0)} blobs, {len(les0)} picked, {len(v0)} crops"),
              blobs(pm1, c1, les1, f"AFTER: {len(c1)} blobs, {len(les1)} picked, {len(v1)} crops")]
    cv2.putText(panels[0], f"INPUT {tag}", (20, 60), 0, 1.6, (255, 255, 255), 4)
    cv2.putText(panels[1], f"FOV mask: rim band (magenta) erode {fov['erode_px']}px", (20, 60), 0, 1.4, (255, 255, 255), 4)
    grid = np.concatenate([np.concatenate(panels[:2], 1), np.concatenate(panels[2:], 1)], 0)
    grid = cv2.resize(grid, (2400, int(2400 * grid.shape[0] / grid.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f'{OUT}/{tag}_fov_demo.png', grid)
    # crops for after
    for name, vv in (('before', v0), ('after', v1)):
        tiles = []
        for v in vv:
            a, b, cc, d = v['bbox']; t = cv2.resize(bgr[b:d, a:cc], (200, 200))
            t = cv2.copyMakeBorder(t, 26, 4, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(t, f"{v['lesion_type']} L{v['lesion_idx']} v{v['view_idx']}", (4, 19), 0, 0.55, (0, 0, 0), 1); tiles.append(t)
        if tiles:
            rows = [np.concatenate(tiles[i:i + 6], 1) for i in range(0, len(tiles), 6)]; w = max(x.shape[1] for x in rows)
            rows = [cv2.copyMakeBorder(x, 0, 0, 0, w - x.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for x in rows]
            cv2.imwrite(f'{OUT}/{tag}_crops_{name}.png', np.concatenate(rows, 0))
json.dump(report, open(f'{OUT}/report.json', 'w'), indent=1)
