"""Independent score/label audit and refit of cached Putra embeddings.
AI-assisted reviewer code; source notebooks are read, never executed.
"""
from pathlib import Path
import json,hashlib,concurrent.futures
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve,roc_curve
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]; CH=ROOT/'artifacts/CH6_ANALYSIS_FULL'; OUT=ROOT/'review'
RAW=ROOT/'artifacts/raw_recovery'
rows=[]
def check(name,a,b,tol=1e-12):
    rows.append(dict(check=name,actual=float(a),expected=float(b),error=float(abs(a-b)),passed=bool(np.isclose(a,b,atol=tol,rtol=0))))
def metrics(y,s):
    p,r,_=precision_recall_curve(y,s); f,t,_=roc_curve(y,s)
    return dict(AP=average_precision_score(y,s),AUC=roc_auc_score(y,s),P_at_R90=p[r>=.9].max(),FPR_at_TPR90=f[np.flatnonzero(t>=.9)[0]])
def weighted_pr(y,s,w):
    order=np.argsort(-s,kind='stable'); y=y[order];s=s[order];w=w[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
    tp=np.cumsum(w*y)[ends];alln=np.cumsum(w)[ends]
    keep=alln>0;tp=tp[keep];alln=alln[keep]
    precision=tp/alln;recall=tp/tp[-1]
    return np.sum(np.diff(np.r_[0,recall])*precision),precision[recall>=.9].max()

meta=pd.read_parquet(RAW/'role_meta/FINAL.parquet');y=meta.y.to_numpy(int)
saved=pd.read_csv(CH/'thesis_tables/TABLE_FF4_DOM_FUSION_CASCADE_FINAL.csv')
q=saved[saved.system.eq('TRI_RUT_200000')&saved.scenario.eq('OFFICIAL_TEST')&np.isclose(saved.target_fpr,.005)]
freeze=json.loads((CH/'audit/ARCHITECTURE_FREEZE.json').read_text())['cascade']
sd=CH/'scores/ff4/TRI_RUT_200000'
scores={name:np.load(sd/f'FINAL_{name}.npy').astype(np.float64) for name in ['url','text','dom','gated','equal']}
# Historical notebook compared float32 arrays with scalar thresholds in NumPy 1.x.
# Explicit float32 bounds reproduce the recorded tie behavior across NumPy versions.
u32=scores['url'].astype(np.float32)
c=scores['gated'].copy();c[u32>=np.float32(freeze['url_high'])]=1e9;c[u32<=np.float32(freeze['url_low'])]=-1e9;scores['cascade']=c
for _,r in q.iterrows():
    s=scores[r['mode']];p=s>=r.threshold
    for col,v in dict(tp=sum(p&(y==1)),fp=sum(p&(y==0)),tn=sum(~p&(y==0)),fn=sum(~p&(y==1)),**metrics(y,s)).items():check('TRI official '+r['mode']+' '+col,v,r[col])
pd.DataFrame(rows).to_csv(OUT/'raw_score_checks.csv',index=False)
assert all(r['passed'] for r in rows),'Metadata order failed independent score/label agreement'
print('Official score and confusion metrics matched in six modes.',flush=True)
# Reproduce exact original RNG order: 1000 FULL followed by 1000 CASCADE.
rng=np.random.default_rng(20260817);pos=np.flatnonzero(y);neg=np.flatnonzero(y==0)
boot=[];summary=[];expected=pd.read_csv(CH/'thesis_tables/TABLE_FINAL_BOOTSTRAP_AP_P90.csv')
for name,s in [('FULL',scores['gated']),('CASCADE',c)]:
    vals=[]
    # Sort scores once. Accumulation over weights preserves score ties.
    order=np.argsort(-s,kind='stable');ss=s[order];yy=y[order]
    ends=np.r_[np.flatnonzero(ss[1:]!=ss[:-1]),len(s)-1]
    for b in range(1000):
        ip=rng.choice(pos,len(pos),replace=True);inn=rng.choice(neg,len(neg),replace=True)
        ids=np.r_[ip,inn];w=np.bincount(ids,minlength=len(y))[order]
        tp=np.cumsum(w*yy)[ends];n=np.cumsum(w)[ends];keep=n>0
        tp=tp[keep];n=n[keep];pr=tp/n;rec=tp/len(pos)
        ap=np.sum(np.diff(np.r_[0,rec])*pr);p90=pr[rec>=.9].max()
        if b==0:
            check(name+' weighted bootstrap AP validation',ap,average_precision_score(y[ids],s[ids]))
            check(name+' weighted bootstrap PR validation',p90,metrics(y[ids],s[ids])['P_at_R90'])
        vals.append((ap,p90));boot.append(dict(system=name,replicate=b,AP=ap,P_at_R90=p90))
    v=np.array(vals);r=expected[expected.system.eq(name)].iloc[0]
    valsdict=dict(AP=average_precision_score(y,s),AP_ci95_lo=np.percentile(v[:,0],2.5),AP_ci95_hi=np.percentile(v[:,0],97.5),P_at_R90=metrics(y,s)['P_at_R90'],P_at_R90_ci95_lo=np.percentile(v[:,1],2.5),P_at_R90_ci95_hi=np.percentile(v[:,1],97.5))
    for col,vv in valsdict.items():check(name+' bootstrap '+col,vv,r[col])
    summary.append(dict(system=name,**valsdict));print(name,'bootstrap complete',flush=True)
pd.DataFrame(boot).to_csv(OUT/'raw_bootstrap_replicates.csv',index=False)
pd.DataFrame(summary).to_csv(OUT/'raw_bootstrap_summary.csv',index=False)
pd.DataFrame(rows).to_csv(OUT/'raw_score_checks.csv',index=False)

# Fit all 120 independent run/budget/representation combinations from caches.
tasks=[];refit=[]
for folder,refpath,variants in [
 ('putra_r0_vs_rut_label_efficiency',ROOT/'artifacts/empirical_supplements/PUTRA_DIRECT_RESULTS/results/PUTRA_R0_VS_RUT_N10_RUN_METRICS.csv',['R0','RUT']),
 ('putra_target_adaptation_v1',ROOT/'artifacts/PUTRA_TARGET_ADAPTATION_RESULTS_SMALL/results/PUTRA_TARGET_ADAPTATION_N10_RUN_METRICS.csv',['E1_TARGET_RUT'])]:
    p=RAW/folder;early=pd.read_parquet(p/'data/putra_early_pool.parquet');late=pd.read_parquet(p/'data/putra_late_test.parquet')
    ids=(early.url.astype(str)+'|'+early.scan_dt.astype(str)+'|'+early.text.astype(str).str.slice(0,256)).tolist()
    yt=early.label.to_numpy(int);yv=late.label.to_numpy(int);ref=pd.read_csv(refpath)
    cache={v:{sp:np.concatenate([np.load(p/f'embeddings/{v}_URL_{sp}.npy'),np.load(p/f'embeddings/{v}_TEXT_{sp}.npy')],axis=1).astype(np.float32) for sp in ['EARLY','LATE']} for v in variants}
    for run in range(10):
        seed=9101+run;ranked={lab:sorted(np.flatnonzero(yt==lab),key=lambda i:int(hashlib.sha256(f'PUTRA_TARGET|{seed}|{lab}|{ids[i]}'.encode()).hexdigest()[:16],16)) for lab in [0,1]}
        previous=set()
        for budget in [200,500,1000,2000]:
            idx=np.sort(np.r_[ranked[0][:budget//2],ranked[1][:budget//2]])
            assert previous.issubset(set(idx));previous=set(idx)
            h=hashlib.sha256(''.join(ids[i]+'\n' for i in idx).encode()).hexdigest()
            for variant in variants:
                exp=ref[ref.run.eq(run)&ref.budget.eq(budget)&ref.variant.eq(variant)].iloc[0]
                assert h==exp.sample_hash,(run,budget,variant,'hash mismatch')
                tasks.append((folder,run,budget,variant,idx,yt,yv,cache[variant],exp))
def fit(task):
    folder,run,budget,var,idx,yt,yv,x,exp=task
    model=LogisticRegression(C=1.0,solver='liblinear',class_weight='balanced',max_iter=2000,random_state=142+run*20)
    model.fit(x['EARLY'][idx],yt[idx]);s=model.decision_function(x['LATE']).astype(np.float32).astype(float)
    met=metrics(yv,s)
    return dict(folder=folder,run=run,budget=budget,variant=var,**met),exp
with threadpool_limits(limits=1),concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    for n,(met,exp) in enumerate(pool.map(fit,tasks)):
        refit.append(met)
        for col in ['AP','AUC','P_at_R90','FPR_at_TPR90']:check(f"Putra {met['variant']} {met['run']} {met['budget']} {col}",met[col],exp[col],1e-8)
        if n%20==19:print('Putra fits',n+1,'of',len(tasks),flush=True)
pd.DataFrame(refit).to_csv(OUT/'raw_putra_refit_metrics.csv',index=False)
pd.DataFrame(rows).to_csv(OUT/'raw_score_checks.csv',index=False)
out={'checks':len(rows),'passed':sum(r['passed'] for r in rows),'failures':[r for r in rows if not r['passed']],'label_provenance':'results (4).zip / phreshphish_v8_4_FINAL/role_meta/FINAL.parquet; all six official mode confusion matrices and ranking metrics independently agree','putra':'new CPU fits from saved frozen embeddings; encoder training not rerun'}
(OUT/'raw_score_summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out),flush=True)
