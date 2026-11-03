import sys,json,cv2,numpy as np,torch
sys.path.insert(0,'.')
from common import *
from inference import load_model,find_candidates,nms_topk
cfg=load_config('config.yaml'); cfg['data']['long_side']=1440
dev=torch.device('cpu'); m,classes=load_model('kaggle_runs/hires1440/ckpt/best.pth',cfg,dev)
bgr=cv2.imread('user_test/img.jpg'); rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
d=cfg['data'];inf=cfg['inference']
img_r,meta=preprocess_image(rgb,1440,d['fundus_threshold'])
p=sliding_window_predict(m,normalize(img_r),dev,inf['tile'],inf['overlap'],False)
print(classes,[round(float(p[i].max()),3) for i in range(4)],[int((p[i]>inf.get('threshold',0.4)).sum()) for i in range(4)])
Hr,Wr=img_r.shape[:2]
les=nms_topk(find_candidates(p,classes,cfg),inf['crop_size'],Wr,Hr,inf['nms_iou'],inf['max_lesions_per_image'])
print(json.dumps(les,default=str)[:800])
cols={0:(255,0,0),1:(0,255,0),2:(255,255,0),3:(0,128,255)}
ov=img_r.copy()
for i,c in cols.items():
  mk=p[i]>0.4; ov[mk]=(0.4*ov[mk]+0.6*np.array(c)).astype(np.uint8)
vis=np.concatenate([img_r,ov],1); vis=cv2.resize(vis,(1400,int(1400*vis.shape[0]/vis.shape[1])))
cv2.imwrite('user_test/result.png',cv2.cvtColor(vis,cv2.COLOR_RGB2BGR))
