import json,sys,glob,os
base=sys.argv[1]
def load(mode):
    p=glob.glob(f"{base}/{mode}/**/summary.json",recursive=True)
    return json.load(open(p[0])) if p else None
S={m:load(m) for m in ['random','anatomy']}
rows={}
for m,s in S.items():
    if not s: print(m,'NO SUMMARY'); continue
    md=s['modes'][m]
    for ev in ['eval_online','eval_target']:
        if ev in md: rows[f"BYOL-{m}/{ev[5:]}"]=md[ev]
    for k,v in s.get('references',{}).items(): rows.setdefault(k,v)
    h=md.get('history',[])
    print(f"\n== {m}: epochs={len(h)} wall_h={s['timings'].get('wall_s',0)/3600:.2f} timings={ {k:round(v) for k,v in s['timings'].items() if k.endswith('total_s')} }")
    for e in h:
        if e['epoch'] in (1,2,3,5,10,20,30,50,75,100,125,150) or e is h[-1]:
            print(f"  ep{e['epoch']:>3} loss={e['loss']:.3f} std_h={e['std_h']:.3f} std_z={e['std_z']:.3f} knn_mAUROC={e.get('knn_mauroc',float('nan')):.3f}")
    print("  min std_h:", min(e['std_h'] for e in h) if h else None)
print("\n== Linear probe (fold0) mean AUROC common / Disease_Risk")
for k,r in rows.items():
    lp=r.get('linear_probe',{}); print(f"  {k:40s} {lp.get('mean_auroc_common',float('nan')):.3f}  {lp.get('disease_risk_auroc',float('nan')):.3f}")
for clf in ['logreg','prototype','knn']:
  for K in [1,5,10]:
    print(f"\n== Few-shot rare k={K} {clf}: per label mean±std ; all_rare")
    for k,r in rows.items():
        f=r.get('fewshot_rare',{}).get(f'k={K}',{}).get(clf)
        if not f: continue
        pl=' '.join(f"{l}:{v[0]:.2f}±{v[1]:.2f}" for l,v in f['per_label'].items())
        print(f"  {k:32s} ALL {f['all_rare_mean']:.3f}±{f['all_rare_std_over_draws']:.3f} | {pl}")
