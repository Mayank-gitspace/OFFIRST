import pickle, numpy as np, json
d = pickle.load(open('cands.pkl','rb')); cl=['MA','HE','EX','SE']
def top(v,c):
    xs=[x for x in v['cands'] if x['lesion_type']==c]
    return max(xs,key=lambda x:x['mean_prob']) if xs else None
res={}
for c in cl:
    print('==',c)
    for t in np.round(np.arange(0.90,0.991,0.005),3):
        row={}
        for s in ['tune_healthy','heldout_healthy']:
            vs=[v for v in d.values() if v['set']==s]
            row[s]=sum(1 for v in vs if (tp:=top(v,c)) and tp['mean_prob']>=t)/len(vs)
        dr=[v for v in d.values() if v['set']=='dr_test']
        picks=[top(v,c) for v in dr]; picks=[p for p in picks if p and p['mean_prob']>=t]
        row['dr_pick_rate']=len(picks)/len(dr)
        row['dr_pick_hit_same']=np.mean([p['hit_same'] for p in picks]) if picks else float('nan')
        row['dr_pick_hit_any']=np.mean([p['gt_px_in_crop']>0 for p in picks]) if picks else float('nan')
        res[(c,t)]=row
        print(t, {k:round(v,2) for k,v in row.items()})
pickle.dump(res,open('sweep.pkl','wb'))
