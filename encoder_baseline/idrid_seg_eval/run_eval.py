import os, json, numpy as np, cv2, torch, torch.nn as nn, timm, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
torch.set_num_threads(os.cpu_count())
# Project root (repo root by default); override with TOB_ROOT. Data paths below are relative to it.
R=Path(os.environ.get('TOB_ROOT', Path(__file__).resolve().parents[2])); OUT=R/'encoder_baseline/idrid_seg_eval'
ck=torch.load(R/'encoder_baseline/full_run_v2/out/convnext_tiny/best.pth',map_location='cpu',weights_only=False)
LABELS=ck['labels']
class Net(nn.Module):
    def __init__(s,name,n):
        super().__init__(); s.backbone=timm.create_model(name,pretrained=False,num_classes=0); s.head=nn.Linear(s.backbone.num_features,n)
    def forward(s,x): f=s.backbone(x); return s.head(f),f
m=Net('convnext_tiny',len(LABELS)); m.load_state_dict(ck['model']); m.eval()
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
def infer(paths):
    P,F=[],[]
    for i in range(0,len(paths),8):
        xs=[]
        for p in paths[i:i+8]:
            assert 'idrid_grade' not in p and 'Grading' not in p
            raw=cv2.cvtColor(cv2.imread(p),cv2.COLOR_BGR2RGB)
            im=cv2.resize(crop_fundus(raw),(S,S),interpolation=cv2.INTER_AREA)
            xs.append(((im.astype(np.float32)-MEAN)/STD).transpose(2,0,1))
        with torch.no_grad(): lo,f=m(torch.from_numpy(np.stack(xs)))
        P.append(torch.sigmoid(lo).numpy()); F.append(f.numpy()); print(i,flush=True)
    return np.concatenate(P),np.concatenate(F)
# IDRiD seg
SEG=R/'data/idrid_seg/A. Segmentation'
ids,paths,sets=[],[],[]
for sub in ['a. Training Set','b. Testing Set']:
    for f in sorted(os.listdir(SEG/'1. Original Images'/sub)):
        if f.endswith('.jpg'): ids.append(f[:-4]); paths.append(str(SEG/'1. Original Images'/sub/f)); sets.append(sub)
LES={'MA':'1. Microaneurysms','HE':'2. Haemorrhages','EX':'3. Hard Exudates','SE':'4. Soft Exudates'}
area={k:[] for k in list(LES)+['total']}
for i,sub in zip(ids,sets):
    union=None; 
    for k,d in LES.items():
        fp=SEG/'2. All Segmentation Groundtruths'/sub/d/f'{i}_{k}.tif'
        if fp.exists():
            mk=cv2.imread(str(fp),cv2.IMREAD_UNCHANGED); mk=(mk[...,:3].max(axis=2) if mk.ndim==3 else mk)>0
        else: mk=None
        if mk is not None and union is None: union=np.zeros_like(mk)
        if mk is not None: union|=mk
        area[k].append(100*mk.mean() if mk is not None else 0.0)
    area['total'].append(100*union.mean())
for k in area: area[k]=np.array(area[k])
# RFMiD fold-0 val
s=pd.read_csv(R/'encoder_baseline/full_run_v2/out/split_5fold_seed42.csv'); L=[c for c in s.columns if c not in('ID','fold')]
v=s[s.fold==0]
loc={}
for d in ['data/rfmid/thresh_tune','data/rfmid/dr_train','data/rfmid/normals_unz']:
    for f in os.listdir(R/d):
        if f.endswith('.png'): loc.setdefault(int(f[:-4]),str(R/d/f))
nid=[i for i in v[v[L].sum(axis=1)==0].ID if i in loc]; did=[i for i in v[v.DR==1].ID if i in loc]
print('normals',len(nid),'DR',len(did),'IDRiD',len(ids))
Pi,Fi=infer(paths); Pn,Fn=infer([loc[i] for i in nid]); Pd,Fd=infer([loc[i] for i in did])
np.savez(OUT/'preds_embs.npz',Pi=Pi,Fi=Fi,Pn=Pn,Fn=Fn,Pd=Pd,Fd=Fd,ids=ids,nid=nid,did=did)
jDR,jRisk=LABELS.index('DR'),LABELS.index('Disease_Risk')
def stats(x): return dict(n=int(len(x)),min=float(x.min()),median=float(np.median(x)),mean=float(x.mean()),max=float(x.max()),pct_gt_0p5=float(100*(x>0.5).mean()))
def auc(pos,neg): return float(roc_auc_score(np.r_[np.ones(len(pos)),np.zeros(len(neg))],np.r_[pos,neg]))
rep={'n_idrid_seg':len(ids),'n_idrid_seg_train':sets.count('a. Training Set'),'n_idrid_seg_test':sets.count('b. Testing Set'),
 'n_rfmid_fold0_val_normals':len(nid),'n_rfmid_fold0_val_DR':len(did),
 'checkpoint':{'epoch':ck['epoch'],'weights':ck['weights']}}
rep['dist']={'idrid_seg':{'DR':stats(Pi[:,jDR]),'Disease_Risk':stats(Pi[:,jRisk])},
             'rfmid_val_normals':{'DR':stats(Pn[:,jDR]),'Disease_Risk':stats(Pn[:,jRisk])},
             'rfmid_val_DR':{'DR':stats(Pd[:,jDR]),'Disease_Risk':stats(Pd[:,jRisk])}}
rep['auroc']={'idrid_seg_vs_normals_DR':auc(Pi[:,jDR],Pn[:,jDR]),'idrid_seg_vs_normals_Disease_Risk':auc(Pi[:,jRisk],Pn[:,jRisk]),
              'rfmid_valDR_vs_normals_DR_in_domain':auc(Pd[:,jDR],Pn[:,jDR]),'rfmid_valDR_vs_normals_Disease_Risk_in_domain':auc(Pd[:,jRisk],Pn[:,jRisk])}
rep['spearman']={}
for lab,j in [('DR',jDR),('Disease_Risk',jRisk)]:
    for k in area:
        r,p=spearmanr(Pi[:,j],area[k]); rep['spearman'][f'{lab}_vs_{k}']={'rho':float(r),'p':float(p)}
o=np.argsort(Pi[:,jDR])[:8]
rep['lowest_DR']=[dict(id=ids[i],set=sets[i][:1],DR=float(Pi[i,jDR]),Disease_Risk=float(Pi[i,jRisk]),**{f'{k}_pct':round(float(area[k][i]),3) for k in area}) for i in o]
rep['lesion_area_pct_summary']={k:dict(median=float(np.median(area[k])),max=float(area[k].max())) for k in area}
# few-shot
nz=lambda X:X/np.linalg.norm(X,axis=1,keepdims=True)
Ei,En=nz(Fi),nz(Fn); rng=np.random.default_rng(0)
def probe(Xp,Xn,k,draws=10):
    a=[]
    for _ in range(draws):
        ip=rng.permutation(len(Xp)); ineg=rng.permutation(len(Xn))
        Xs=np.r_[Xp[ip[:k]],Xn[ineg[:k]]]; ys=np.r_[np.ones(k),np.zeros(k)]
        Xq=np.r_[Xp[ip[k:]],Xn[ineg[k:]]]; yq=np.r_[np.ones(len(Xp)-k),np.zeros(len(Xn)-k)]
        clf=LogisticRegression(C=1.0,max_iter=2000).fit(Xs,ys); a.append(roc_auc_score(yq,clf.decision_function(Xq)))
    return dict(mean=float(np.mean(a)),std=float(np.std(a)))
rep['fewshot_idrid_vs_normals']={f'k{k}':probe(Ei,En,k) for k in (1,5,10)}
heavy=area['EX']>1.0
rep['fewshot_heavyEX_within_idrid']={'n_heavy':int(heavy.sum()),'n_not':int((~heavy).sum()),'k5':probe(Ei[heavy],Ei[~heavy],5)}
r,p=spearmanr(Pi[:,LABELS.index('EX')] if 'EX' in LABELS else Pi[:,jDR],area['EX']) if 'EX' in LABELS else (np.nan,np.nan)
dup=pd.read_csv(R/'data_splits/idrid_seg_duplicates_in_grading.tsv',sep='\t',comment='#')
rep['note_duplicates']=f'{len(dup)} IDRiD seg images are pixel-identical to IDRiD grading images (data_splits/idrid_seg_duplicates_in_grading.tsv); grading folder never read.'
json.dump(rep,open(OUT/'report.json','w'),indent=1)
pd.DataFrame(dict(id=ids,set=sets,DR=Pi[:,jDR],Disease_Risk=Pi[:,jRisk],**{f'{k}_pct':area[k] for k in area})).to_csv(OUT/'idrid_seg_per_image.csv',index=False)
fig,ax=plt.subplots(1,2,figsize=(12,4.5)); b=np.linspace(0,1,26)
ax[0].hist(Pn[:,jDR],b,alpha=.6,label=f'RFMiD fold-0 val normals (n={len(nid)})'); ax[0].hist(Pi[:,jDR],b,alpha=.6,label=f'IDRiD seg, all DR (n={len(ids)})')
ax[0].hist(Pd[:,jDR],b,histtype='step',lw=1.5,color='k',label=f'RFMiD fold-0 val DR (n={len(did)})')
ax[0].set_xlabel('DR prob'); ax[0].set_ylabel('count'); ax[0].legend(fontsize=8); ax[0].set_title('ConvNeXt-T baseline: DR prob')
ax[1].scatter(area['total'],Pi[:,jDR],c=['tab:blue' if s_[0]=='a' else 'tab:orange' for s_ in sets],s=18)
ax[1].set_xscale('log'); ax[1].set_xlabel('total lesion area % (MA+HE+EX+SE union, log)'); ax[1].set_ylabel('DR prob')
ax[1].set_title(f"IDRiD seg: Spearman rho={rep['spearman']['DR_vs_total']['rho']:.2f} (blue=train, orange=test)")
plt.tight_layout(); plt.savefig(OUT/'idrid_seg_eval.png',dpi=110)
print(json.dumps(rep,indent=1))
# camera-confound control: IDRiD seg DR vs RFMiD val DR (both diseased) -> separability = camera/domain signal
Ed=nz(Fd); rep['fewshot_camera_control_idridDR_vs_rfmidDR']={f'k{k}':probe(Ei,Ed,k) for k in (1,5,10)}
json.dump(rep,open(OUT/'report.json','w'),indent=1)
print(rep['fewshot_camera_control_idridDR_vs_rfmidDR'])
