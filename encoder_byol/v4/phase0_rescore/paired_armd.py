# Paired few-shot logreg: v4 ep20 (target/online global) vs B last target. Exactly the kernel's seeded draws
# (rng consumed across all rare labels in order), same query fold; 1000x query-bootstrap CI on the AUROC delta.
import sys,json,numpy as np; sys.path.insert(0,'.')
import evallib as L
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
L.setup(L.OUT4/'byol_anatomy_v4/embedding_ids.csv'); F=L.FOLDS; Y=L.Y_ALL; ec=L.CFG['eval']
S=np.where(F!=0)[0]; Q=np.where(F==0)[0]
_,rare=L.label_sets(list(Q),list(S)); HL=['ODC','ODP','ARMD']
E={'B':np.load('emb_Blast_target_full.npy'),'v4ep20_target':np.load('emb_ep20_target_full.npy'),'v4ep20_online':np.load('emb_ep20_online_full.npy')}
rb=np.random.RandomState(0); BOOT=[]
while len(BOOT)<1000:
    b=rb.choice(len(Q),len(Q))
    if all(0<Y[Q[b],L.LABELS.index(l)].sum()<len(b) for l in HL): BOOT.append(b)
res={}
for k in ec['shots']:
    scores={n:{l:[] for l in HL} for n in E}
    for d in range(ec['draws']):
        rng=np.random.RandomState(L.CFG['seed']*1000+k*100+d)
        for j in rare:
            ys=Y[S,j]; pos=np.where(ys==1)[0]; neg=np.where(ys==0)[0]
            sp=rng.choice(pos,min(k,len(pos)),replace=False); sn=rng.choice(neg,min(k*ec['neg_per_pos'],len(neg)),replace=False)
            if L.LABELS[j] not in HL: continue
            idx=S[np.r_[sp,sn]]; y_=np.r_[np.ones(len(sp)),np.zeros(len(sn))]
            for n,Em in E.items():
                scores[n][L.LABELS[j]].append(LogisticRegression(C=ec['logreg_C'],class_weight='balanced',max_iter=2000).fit(Em[idx],y_).decision_function(Em[Q]))
    from scipy.stats import rankdata
    def fauc(y,s):
        r=rankdata(s); p=y.sum(); n=len(y)-p; return (r[y==1].sum()-p*(p+1)/2)/(p*n)
    allb=np.arange(len(Q)); CACHE={}
    def auc(n,l,b,bi=-1):
        key=(n,l,bi)
        if key not in CACHE:
            yq=Y[Q[b],L.LABELS.index(l)]; CACHE[key]=np.mean([fauc(yq,s[b]) for s in scores[n][l]])
        return CACHE[key]
    for l in HL+['headline']:
        labs=HL if l=='headline' else [l]
        pt={n:np.mean([auc(n,x,allb) for x in labs]) for n in E}
        r={'point':{n:round(v,4) for n,v in pt.items()}}
        for n in ['v4ep20_target','v4ep20_online']:
            dd=np.array([np.mean([auc(n,x,b,bi)-auc('B',x,b,bi) for x in labs]) for bi,b in enumerate(BOOT)])
            r[f'delta_{n}_vs_B']=round(pt[n]-pt['B'],4); r[f'ci95_{n}']=[round(float(np.percentile(dd,2.5)),4),round(float(np.percentile(dd,97.5)),4)]; r[f'p_v4_lt_B_{n}']=round(float((dd<0).mean()),3)
        res[f'k={k}_{l}']=r; print(f'k={k}',l,json.dumps(r),flush=True)
json.dump(res,open('paired_vs_B.json','w'),indent=1)
