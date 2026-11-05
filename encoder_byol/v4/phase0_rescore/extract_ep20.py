# CPU feature extraction for v4 best.pth (ep20): online + target backbones; global 384 + disc/mac 224 regions (v4 RegionDS logic)
import os,math,numpy as np,pandas as pd,cv2,torch,timm,time
torch.set_num_threads(os.cpu_count())
O='../full_out/out'; ck=torch.load(f'{O}/byol_anatomy_v4/best.pth',map_location='cpu',weights_only=False); assert ck['epoch']==20
C5=np.load('/workspace/byol_local/tmp/cache_512_1920.npy',mmap_mode='r')
BOX=pd.read_csv(f'{O}/box_table_used.csv'); BA=BOX[["disc_cx","disc_cy","disc_diam","mac_cx","mac_cy"]].values.astype(np.float64)
MEAN=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)*255; STD=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)*255
def aff(c,w,out):
    T1=np.array([[1,0,-c[0]],[0,1,-c[1]],[0,0,1]],float); Sm=np.diag([out/w,out/w,1.]); T2=np.array([[1,0,out/2],[0,1,out/2],[0,0,1]],float); return (T2@Sm@T1)[:2]
def img(i,kind):
    im=np.asarray(C5[i])
    if kind=='full': return cv2.resize(im,(384,384),interpolation=cv2.INTER_AREA)
    S=im.shape[0]; dcx,dcy,dd,mcx,mcy=BA[i]*S
    cx,cy=(dcx,dcy) if kind=='disc' else (mcx,mcy); w=dd*2.5
    return cv2.warpAffine(im,aff((cx,cy),w,224),(224,224),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
def bb(sd):
    m=timm.create_model('convnext_tiny',pretrained=False,num_classes=0); m.load_state_dict(sd); return m.eval().to(memory_format=torch.channels_last)
nets={'online':bb({k[9:]:v for k,v in ck['online'].items() if k.startswith('backbone.')}),'target':bb(ck['target_backbone'])}
@torch.no_grad()
def ext(kind):
    E={n:[] for n in nets}; t=time.time()
    for s in range(0,1920,32):
        x=torch.from_numpy(np.stack([img(i,kind) for i in range(s,min(s+32,1920))])).permute(0,3,1,2).float()
        x=((x-MEAN)/STD).contiguous(memory_format=torch.channels_last)
        for n,m in nets.items(): E[n].append(m(x).numpy())
        if s%320==0: print(kind,s,f'{time.time()-t:.0f}s',flush=True)
    for n in E:
        e=np.concatenate(E[n]); e=(e/(np.linalg.norm(e,axis=1,keepdims=True)+1e-12)).astype(np.float32); np.save(f'emb_ep20_{n}_{kind}.npy',e)
for k in ['full','disc','macula']: ext(k)
print('DONE')
