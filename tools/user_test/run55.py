import sys,json,cv2,numpy as np,torch
sys.path.insert(0,'.')
from common import *
from inference import load_model,find_candidates,nms_topk,make_views
cfg=load_config('config.yaml'); cfg['data']['long_side']=1440
dev=torch.device('cpu'); m,classes=load_model('kaggle_runs/hires1440/ckpt/best.pth',cfg,dev)
B='user_test/t/A. Segmentation/'
bgr=cv2.imread(B+'1. Original Images/b. Testing Set/IDRiD_55.jpg'); print('size',bgr.shape)
rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); d=cfg['data'];inf=cfg['inference']
img_r,meta=preprocess_image(rgb,1440,d['fundus_threshold'])
p=sliding_window_predict(m,normalize(img_r),dev,inf['tile'],inf['overlap'],False)
Hr,Wr=img_r.shape[:2]
names={'MA':'1. Microaneurysms','HE':'2. Haemorrhages','EX':'3. Hard Exudates','SE':'4. Soft Exudates'}
gt=np.zeros((4,)+bgr.shape[:2],bool)
for i,c in enumerate(classes):
  g=cv2.imread(f"{B}2. All Segmentation Groundtruths/b. Testing Set/{names[c]}/IDRiD_55_{c}.tif",0)
  if g is not None: gt[i]=g>0
  print(c,'gt px',int(gt[i].sum()),'pred max',round(float(p[i].max()),3),'pred px>0.4',int((p[i]>0.4).sum()))
les=nms_topk(find_candidates(p,classes,cfg),inf['crop_size'],Wr,Hr,inf['nms_iou'],inf['max_lesions_per_image'])
views=make_views(les,meta,cfg,'IDRiD_55',np.random.default_rng(42))
anyg=gt.any(0); out=[]
import os; os.makedirs('user_test/crops55',exist_ok=True)
tiles=[]
for v in views:
  x1,y1,x2,y2=v['bbox']; cr=bgr[y1:y2,x1:x2]
  hit={c:int(gt[i,y1:y2,x1:x2].sum()) for i,c in enumerate(classes)}
  v['gt_hits']=hit; out.append(v)
  t=cv2.resize(cr,(220,220)); lab=f"{v['lesion_type']} L{v['lesion_idx']} v{v['view_idx']} {'HIT' if sum(hit.values()) else 'MISS'}"
  t=cv2.copyMakeBorder(t,26,4,2,2,cv2.BORDER_CONSTANT,value=(255,255,255)); cv2.putText(t,lab,(4,19),0,0.55,(0,0,0),1)
  tiles.append(t)
for v in out: print(v['lesion_type'],v['lesion_idx'],v['view_idx'],v['bbox'],v['gt_hits'])
rows=[np.concatenate(tiles[i:i+3],1) for i in range(0,len(tiles),3)]
w=max(r.shape[1] for r in rows); rows=[cv2.copyMakeBorder(r,0,0,0,w-r.shape[1],cv2.BORDER_CONSTANT,value=(255,255,255)) for r in rows]
cv2.imwrite('user_test/crops55.png',np.concatenate(rows,0))
# overlay pred vs gt, with crop boxes
cols=[(0,0,255),(0,255,0),(0,255,255),(255,128,0)]
pv=bgr.copy(); gv=bgr.copy()
for i in range(4):
  pm=cv2.resize((p[i]>0.4).astype(np.uint8),None,fx=1,fy=1)
  # map pred to original via meta: place resized back into bbox
  x0,y0,x1_,y1_=(meta['x0'],meta['y0'],meta['x1'],meta['y1'])
  pm=cv2.resize(pm,(x1_-x0,y1_-y0),interpolation=cv2.INTER_NEAREST).astype(bool)
  full=np.zeros(bgr.shape[:2],bool); full[y0:y1_,x0:x1_]=pm
  pv[full]=cols[i]; gv[gt[i]]=cols[i]
for v in out:
  x1,y1,x2,y2=v['bbox']
  if v['view_idx']==1: cv2.rectangle(pv,(x1,y1),(x2,y2),(255,255,255),6)
vis=np.concatenate([gv,pv],1); vis=cv2.resize(vis,(1600,int(1600*vis.shape[0]/vis.shape[1])))
cv2.imwrite('user_test/overlay55.png',vis)
print(meta)
