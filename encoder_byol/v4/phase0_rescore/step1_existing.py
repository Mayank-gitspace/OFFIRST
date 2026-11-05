import sys,json; sys.path.insert(0,'.')
import evallib as L, numpy as np
from pathlib import Path
res={}
BL=Path('/workspace/byol_local/out2')
import pandas as pd
# sanity: A/B embedding_ids folds == locked split?
jobs=[('v4_ep20_online_locked',L.OUT4/'byol_anatomy_v4','embeddings_best_online.npy',L.LOCKED),
      ('v4_ep120_target_locked',L.OUT4/'byol_anatomy_v4','embeddings_target.npy',L.LOCKED),
      ('v4_ep120_multiregion_concat_locked',L.OUT4/'byol_anatomy_v4','embeddings_multiregion_online.npy',L.LOCKED),
      ('v4_ep20_online_v4split(repro)',L.OUT4/'byol_anatomy_v4','embeddings_best_online.npy',L.OUT4/'split_used.csv')]
for tag,d,f,sp in jobs:
    L.setup(d/'embedding_ids.csv',sp); L.FOLDS; 
    # evaluate uses module globals
    r=L.run(np.load(d/f),tag); res[tag]={'brief':L.brief(r),'full':r}
    print(tag,json.dumps({k:v for k,v in L.brief(r).items() if k not in('common','rare')}),flush=True)
json.dump(res,open('step1_existing.json','w'),indent=1,default=float)
