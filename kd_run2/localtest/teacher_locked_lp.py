import sys,json,numpy as np; sys.path.insert(0,'encoder_mnv3_ssl/locked_rescore'); import evallib as L
ids='encoder_mnv3_ssl/full_out/out/byol_mnv3_anatomy_v4/embedding_ids.csv'; L.setup(ids,L.LOCKED)
for nm,p in [('phase0_ref','encoder_byol/v4/phase0_rescore/emb_ep20_target_full.npy'),('kernel640path','teacher_emb_640path_fp32.npy')]:
    b=L.brief(L.run(np.load(p),nm)); print(nm,json.dumps({k:v for k,v in b.items() if k not in('common','rare')}),flush=True)
