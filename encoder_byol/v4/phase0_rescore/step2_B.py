import sys,json,numpy as np; sys.path.insert(0,'.')
import evallib as L
net=sys.argv[1]; ro=sys.argv[2]
L.setup(L.OUT4/'byol_anatomy_v4/embedding_ids.csv')
f=lambda k:np.load(f'emb_Blast_{net}_{k}.npy')
if ro=='full': E=f('full')
elif ro=='l2mean': E=(f('full')+f('disc')+f('macula'))/3   # each already L2; re-L2 in run()
tag=f'B_last_{net}_{ro}_locked'; r=L.run(E,tag)
json.dump({'brief':L.brief(r),'full':r},open(f'step2_B_{net}_{ro}.json','w'),indent=1,default=float)
print(tag,json.dumps({k:v for k,v in L.brief(r).items() if k not in('common','rare')}))
