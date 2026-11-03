import sys,cv2,numpy as np,torch
sys.path.insert(0,'.')
from common import *
from inference import load_model,find_candidates,nms_topk,make_views
cfg=load_config('config.yaml'); cfg['data']['long_side']=1440
m,classes=load_model('kaggle_runs/hires1440/ckpt/best.pth',cfg,torch.device('cpu'))
bgr=cv2.imread('user_test/grad/Imagenes/Imagenes/IDRiD_029test.jpg'); print(bgr.shape)
rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); inf=cfg['inference']
img_r,meta=preprocess_image(rgb,1440,cfg['data']['fundus_threshold'])
p=sliding_window_predict(m,normalize(img_r),torch.device('cpu'),inf['tile'],inf['overlap'],False)
for i,c in enumerate(classes): print(c,'max',round(float(p[i].max()),3),'px>0.4',int((p[i]>0.4).sum()))
Hr,Wr=img_r.shape[:2]
cands=find_candidates(p,classes,cfg); print('candidate blobs',len(cands))
les=nms_topk(cands,inf['crop_size'],Wr,Hr,inf['nms_iou'],inf['max_lesions_per_image'])
for l in les: print(l['lesion_type'],round(l['confidence'],3),l['area'])
views=make_views(les,meta,cfg,'h',np.random.default_rng(42))
cols=[(0,0,255),(0,255,0),(0,255,255),(255,128,0)]; pv=bgr.copy()
x0,y0,x1,y1=meta['x0'],meta['y0'],meta['x1'],meta['y1']
for i in range(4):
  pm=cv2.resize((p[i]>0.4).astype(np.uint8),(x1-x0,y1-y0),interpolation=cv2.INTER_NEAREST).astype(bool)
  f=np.zeros(bgr.shape[:2],bool); f[y0:y1,x0:x1]=pm; pv[cv2.dilate(f.astype(np.uint8),np.ones((9,9)))>0]=cols[i]
for v in views:
  if v['view_idx']==1: a,b,c,d=v['bbox']; cv2.rectangle(pv,(a,b),(c,d),(255,255,255),8)
vis=np.concatenate([bgr,pv],1); vis=cv2.resize(vis,(1600,int(1600*vis.shape[0]/vis.shape[1])))
cv2.imwrite('user_test/healthy.png',vis)
tiles=[]
for v in views:
  a,b,c,d=v['bbox']; t=cv2.resize(bgr[b:d,a:c],(220,220)); t=cv2.copyMakeBorder(t,26,4,2,2,cv2.BORDER_CONSTANT,value=(255,255,255))
  cv2.putText(t,f"{v['lesion_type']} L{v['lesion_idx']} v{v['view_idx']}",(4,19),0,0.55,(0,0,0),1); tiles.append(t)
if tiles:
  rows=[np.concatenate(tiles[i:i+3],1) for i in range(0,len(tiles),3)]; w=max(r.shape[1] for r in rows)
  rows=[cv2.copyMakeBorder(r,0,0,0,w-r.shape[1],cv2.BORDER_CONSTANT,value=(255,255,255)) for r in rows]
  cv2.imwrite('user_test/healthy_crops.png',np.concatenate(rows,0))
print('views',len(views))
