import sys,cv2,numpy as np,copy
sys.path.insert(0,'lesion_detector')
from common import *; from inference import find_candidates,nms_topk
from fov_mask import retina_mask
cfg=load_config('lesion_detector/config.yaml'); inf=cfg['inference']; classes=['MA','HE','EX','SE']
SEG='user_test/t/A. Segmentation/'
IM={'IDRiD_029test_healthy':'user_test/grad/Imagenes/Imagenes/IDRiD_029test.jpg','IDRiD_55_DR':SEG+'1. Original Images/b. Testing Set/IDRiD_55.jpg'}
for tag,path in IM.items():
  rgb=cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB); img_r,meta=preprocess_image(rgb,1440,15)
  p=np.load(f'{tag}_probs.npz')['probs']; ret=retina_mask(img_r,12); r=np.sqrt(ret.sum()/np.pi)
  for frame in (False,True):
    for f in (0.025,0.05,0.075,0.10):
      e=int(round(f*r)); k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*e+1,2*e+1))
      if frame: safe=cv2.erode(cv2.copyMakeBorder(ret,1,1,1,1,cv2.BORDER_CONSTANT,value=0),k)[1:-1,1:-1]
      else: safe=cv2.erode(ret,k)
      c=[x for x in find_candidates(p*safe[None],classes,cfg) if safe[int(round(x['cy'])),int(round(x['cx']))]]
      les=nms_topk(c,inf['crop_size'],img_r.shape[1],img_r.shape[0],inf['nms_iou'],4)
      print(tag,'frame_edge' if frame else 'disc_rim ', f'frac={f} erode={e}px blobs={len(c)} picked={[(l["lesion_type"],int(l["cx"]),int(l["cy"])) for l in les]}')
