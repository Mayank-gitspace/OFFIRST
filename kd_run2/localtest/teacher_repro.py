# Check: teacher (v4 ep20 target) loads strictly and reproduces phase0 embeddings / locked LP.
import os,json,time,numpy as np,pandas as pd,cv2,torch,timm
torch.set_num_threads(6)
ck=torch.load('kd_run2/weights_ds/teacher_byol_v4_ep20_target_convnext_tiny.pth',map_location='cpu',weights_only=False)
m=timm.create_model('convnext_tiny',pretrained=False,num_classes=0); r=m.load_state_dict(ck['state_dict'],strict=True)
print('teacher load missing',r.missing_keys,'unexpected',r.unexpected_keys, 'out dim', m.num_features); m.eval()
MEAN=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)*255; STD=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)*255
C5=np.load('/workspace/byol_local/tmp/cache_512_1920.npy',mmap_mode='r'); C6=np.load('/workspace/byol_local/tmp_mnv3/cache_640_1920.npy',mmap_mode='r')
ids=pd.read_csv('encoder_mnv3_ssl/full_out/out/byol_mnv3_anatomy_v4/embedding_ids.csv'); assert (ids.ID.values==np.arange(1,1921)).all()
sp=pd.read_csv('/workspace/offirst_upload/encoder_byol/full_run/split_used.csv').set_index('ID').loc[ids.ID].fold.values
ref=np.load('encoder_byol/v4/phase0_rescore/emb_ep20_target_full.npy')
@torch.no_grad()
def emb(C,rows,fp16=False):
    E=[]
    for s in range(0,len(rows),32):
        x=torch.from_numpy(np.stack([cv2.resize(np.asarray(C[i]),(384,384),interpolation=cv2.INTER_AREA) for i in rows[s:s+32]])).permute(0,3,1,2).float()
        x=((x-MEAN)/STD).contiguous(memory_format=torch.channels_last)
        if fp16:
            with torch.autocast('cpu',dtype=torch.bfloat16): e=m(x).float()
        else: e=m(x)
        E.append(e.numpy())
    E=np.concatenate(E); return E/np.linalg.norm(E,axis=1,keepdims=True)
f0=np.where(sp==0)[0]; t=time.time()
E5=emb(C5,f0); c=(E5*ref[f0]).sum(1); print(f'512-cache path vs phase0 ref fold0: cos min {c.min():.6f} mean {c.mean():.6f} ({time.time()-t:.0f}s)')
allr=np.arange(1920); t=time.time(); E6=emb(C6,allr); print(f'640-cache (kernel path) all 1920 in {time.time()-t:.0f}s')
c=(E6*ref).sum(1); print(f'640-cache vs phase0 ref all: cos min {c.min():.4f} mean {c.mean():.4f}')
np.save('teacher_emb_640path_fp32.npy',E6.astype(np.float32))
