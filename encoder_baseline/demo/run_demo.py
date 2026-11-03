import os, sys, json, numpy as np, cv2, torch, torch.nn as nn, timm, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
R=Path(os.environ.get('TOB_ROOT', '.')); OUT=R/'encoder_baseline/demo'
ck=torch.load(R/'encoder_baseline/full_run_v2/out/convnext_tiny/best.pth',map_location='cpu',weights_only=False)
LABELS=ck['labels']; print('weights:',ck['weights'],'epoch:',ck['epoch'],'val mAUROC',ck['metrics']['mean_auroc'])
csvcols=[c for c in pd.read_csv(R/'encoder_baseline/full_run_v2/out/split_5fold_seed42.csv',nrows=1).columns if c not in ('ID','fold')]
assert csvcols==LABELS, (csvcols, LABELS)
class Net(nn.Module):
    def __init__(s,name,n):
        super().__init__(); s.backbone=timm.create_model(name,pretrained=False,num_classes=0); s.head=nn.Linear(s.backbone.num_features,n)
    def forward(s,x): return s.head(s.backbone(x))
m=Net('convnext_tiny',len(LABELS)); print(m.load_state_dict(ck['model'])); m.eval()
S=512
def crop_fundus(img, thr=15):
    h, w = img.shape[:2]; sc = 512.0 / max(h, w)
    small = cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
    g = small.max(axis=2); mask = (g > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    ys, xs = np.where(mask > 0)
    if len(xs) < 100: return img
    x0, x1 = int(xs.min() / sc), int((xs.max() + 1) / sc); y0, y1 = int(ys.min() / sc), int((ys.max() + 1) / sc)
    img = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    h, w = img.shape[:2]; mm = max(h, w); out = np.zeros((mm, mm, 3), img.dtype)
    out[(mm - h) // 2:(mm - h) // 2 + h, (mm - w) // 2:(mm - w) // 2 + w] = img
    return out
MEAN=np.array([0.485,0.456,0.406],np.float32)*255; STD=np.array([0.229,0.224,0.225],np.float32)*255
res={}
for tag,path,title in [(a.split('|')[0],a.split('|')[1],a.split('|')[2]) for a in sys.argv[1:]]:
    raw=cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB); print(tag,'raw',raw.shape)
    im=cv2.resize(crop_fundus(raw),(S,S),interpolation=cv2.INTER_AREA)
    x=torch.from_numpy(((im.astype(np.float32)-MEAN)/STD).transpose(2,0,1).copy())[None]
    with torch.no_grad(): p=torch.sigmoid(m(x)).numpy()[0]
    order=np.argsort(-p); res[tag]={LABELS[i]:float(p[i]) for i in order}
    fig,ax=plt.subplots(1,2,figsize=(12,5.5)); ax[0].imshow(im); ax[0].axis('off'); ax[0].set_title(f'{title}\n(preprocessed 512px)')
    t5=order[:5]; ax[1].barh([LABELS[i] for i in t5][::-1],p[t5][::-1],color=['tab:red' if p[i]>0.3 else 'tab:gray' for i in t5][::-1])
    for j,i in enumerate(t5[::-1]): ax[1].text(p[i]+0.01,j,f'{p[i]:.3f}',va='center')
    ax[1].set_xlim(0,1.1); ax[1].axvline(0.3,ls='--',c='k',lw=0.8); ax[1].set_title(f'ConvNeXt-T baseline ({ck["weights"]} wts, ep {ck["epoch"]}) top-5 sigmoid')
    plt.tight_layout(); fp=OUT/f'{tag}_demo.png'; plt.savefig(fp,dpi=110); plt.close(); print('saved',fp)
json.dump(res,open(OUT/'demo_probs.json','w'),indent=1)
for t,r in res.items():
    print('\n==',t); [print(f'  {k:14s} {v:.4f}') for k,v in r.items()]
