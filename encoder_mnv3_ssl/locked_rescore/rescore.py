# Re-score MNV3 BYOL ep40 (selected) embeddings with the v4 phase0 eval funcs (byte-identical to v4 kernel), CPU only.
import sys,json,numpy as np; sys.path.insert(0,'encoder_mnv3_ssl/locked_rescore')
import evallib as L
from pathlib import Path
D=Path('encoder_mnv3_ssl/full_out/out/byol_mnv3_anatomy_v4')
ids=D/'embedding_ids.csv'; own=Path('encoder_mnv3_ssl/full_out/out/split_used.csv')
res={}
for split_name,sp in [('own_split_sanity',own),('locked',L.LOCKED)]:
    L.setup(ids,sp)
    for net in ['target','online']:
        E=np.load(D/f'embeddings_{net}.npy'); r=L.run(E,f'mnv3_ep40_{net}_{split_name}')
        res[f'{net}_{split_name}']=L.brief(r)
        print(net,split_name,json.dumps({k:v for k,v in L.brief(r).items() if k not in('common','rare')}))
json.dump(res,open('encoder_mnv3_ssl/locked_rescore/rescore.json','w'),indent=1,default=float)
