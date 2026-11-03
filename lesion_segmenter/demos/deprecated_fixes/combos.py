import pickle, numpy as np
d = pickle.load(open('cands.pkl','rb')); cl=['MA','HE','EX','SE']
def picks(v,thr):
    out=[]
    for c in cl:
        xs=[x for x in v['cands'] if x['lesion_type']==c and x['mean_prob']>=thr[c]]
        if xs: out.append(max(xs,key=lambda x:x['mean_prob']))
    return out
combos={'strict':dict(MA=.97,HE=.955,EX=.94,SE=.965),'balanced':dict(MA=.965,HE=.95,EX=.94,SE=.965),
        'lenient':dict(MA=.96,HE=.945,EX=.93,SE=.95)}
for n,thr in combos.items():
    r={}
    for s in ['tune_healthy','heldout_healthy','dr_test']:
        vs=[v for v in d.values() if v['set']==s]; P=[picks(v,thr) for v in vs]
        r[s]=dict(imgs_with_lesion=f"{sum(1 for p in P if p)}/{len(vs)}", lesions_total=sum(map(len,P)))
        if s=='dr_test':
            allp=[x for p in P for x in p]
            r[s]['pick_hit_same_class']=f"{sum(x['hit_same'] for x in allp)}/{len(allp)}"
            r[s]['pick_crop_has_any_gt']=f"{sum(x['gt_px_in_crop']>0 for x in allp)}/{len(allp)}"
            r[s]['per_class_picks']={c:sum(x['lesion_type']==c for x in allp) for c in cl}
    print(n,thr); [print('  ',k,v) for k,v in r.items()]
# legacy for comparison: top-4 global by peak, no fov? (cands are post-FOV) -> report peak top4 count
