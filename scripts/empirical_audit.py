"""Independent arithmetic audit; never executes training notebooks.

Run from any directory: python scripts/empirical_audit.py
Requires numpy, pandas, scipy. Inputs are archived run-level result rows, confusion
counts and score arrays. Missing instance labels and raw timing samples are stated
explicitly: derived statistics cannot validate absent upstream measurements.
"""
from pathlib import Path
import itertools, json, math, hashlib
import numpy as np
import pandas as pd
from scipy.stats import t, beta

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts'
CH = ART / 'CH6_ANALYSIS_FULL'
SUP = ART / 'empirical_supplements'
OUT = ROOT / 'review'
OUT.mkdir(exist_ok=True)
LOG = []
FILES = {}

def read(path):
    path=Path(path)
    FILES[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)

def save(name, rows):
    d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    d.to_csv(OUT / ('empirical_'+name+'.csv'), index=False)
    return d

def record(name, actual, expected, atol=1e-11):
    a,b=np.asarray(actual,float),np.asarray(expected,float)
    okay=bool(a.shape==b.shape and np.allclose(a,b,atol=atol,rtol=0,equal_nan=True))
    err=float(np.nanmax(np.abs(a-b))) if a.shape==b.shape and a.size else None
    LOG.append(dict(check=name,passed=okay,max_abs_error=err))
    return okay

def holm(p):
    p=np.asarray(p,float); order=np.argsort(p,kind='stable'); out=np.empty_like(p)
    out[order]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    return out

def flip(d, one_sided=False):
    d=np.asarray(d,float)
    signed=np.array(list(itertools.product((-1,1),repeat=len(d))),float) @ d / len(d)
    return float(np.mean(signed>=d.mean()-1e-15)) if one_sided else float(np.mean(abs(signed)>=abs(d.mean())-1e-15))

def describe(d):
    d=np.asarray(d,float); mean=d.mean(); half=t.ppf(.975,len(d)-1)*d.std(ddof=1)/math.sqrt(len(d))
    return dict(n=len(d),mean_delta=float(mean),ci95_lo=float(mean-half),ci95_hi=float(mean+half),positive_pairs=int((d>0).sum()),negative_pairs=int((d<0).sum()),p_raw=flip(d))

def counts_check(d,name):
    if not {'tp','tn','fp','fn'}.issubset(d): return
    q=d.dropna(subset=['tp','tn','fp','fn'])
    for metric,values in {'tpr':q.tp/(q.tp+q.fn),'fpr':q.fp/(q.fp+q.tn),'precision':q.tp/(q.tp+q.fp).clip(lower=1),'f1':2*q.tp/(2*q.tp+q.fp+q.fn).clip(lower=1)}.items():
        if metric in q: record(name+' '+metric+' from integer counts',q[metric],values)
    if 'fpr_ci_lo' in q:
        n=q.fp+q.tn
        lo=np.where(q.fp.eq(0),0,beta.ppf(.025,q.fp,n-q.fp+1))
        hi=np.where(q.fp.eq(n),1,beta.ppf(.975,q.fp+1,n-q.fp))
        record(name+' Clopper Pearson lower',q.fpr_ci_lo,lo)
        record(name+' Clopper Pearson upper',q.fpr_ci_hi,hi)

def main():
    # FF1: 40 run rows; derive basic metrics from integer confusion counts.
    ff1=read(CH/'results/FF1_MODALITY_N5_ALL.csv'); counts_check(ff1,'FF1')
    agg=save('ff1',ff1.groupby(['probe','rep'],as_index=False).agg(n=('run','size'),tpr=('tpr','mean'),fpr=('fpr','mean'),FPR_at_TPR90=('FPR_at_TPR90','mean'),tpr_sd=('tpr','std')))
    ff1stats=[]
    for probe,q in ff1.groupby('probe'):
        p=q.pivot(index='run',columns='rep',values='tpr')
        ff1stats.append(dict(probe=probe,contrast='RUT-R0',**describe(p.RUT-p.R0)))
        ff1stats.append(dict(probe=probe,contrast='RUT-RT-RU+R0',**describe(p.RUT-p.RT-p.RU+p.R0)))
    save('ff1_contrasts',ff1stats)
    # Full internal learning curves and both possible B95 interpretations.
    curves={}
    for role in ['DEV','FINAL']:
        d=read(CH/f'results/FF2_LABEL_CURVE_{role}_ALL.csv'); counts_check(d,'FF2 '+role)
        q=d[np.isclose(d.target_fpr,.005)]
        if role=='FINAL': q=q[q.scenario.eq('OFFICIAL_TEST')]
        curves[role]=save('ff2_'+role.lower(),q.groupby(['probe','rep','budget'],as_index=False).agg(n=('run','size'),tpr=('tpr','mean'),tpr_sd=('tpr','std'),fpr=('fpr','mean'),AP=('AP','mean')))
    b95=[]
    for role,d in curves.items():
        for (probe,rep),q in d.groupby(['probe','rep']):
            endpoint=float(q.loc[q.budget.eq(200000),'tpr'].iloc[0]); hit=q[q.tpr>=.95*endpoint]
            b95.append(dict(role=role,probe=probe,rep=rep,endpoint_tpr=endpoint,target_tpr=.95*endpoint,B95=int(hit.budget.min())))
    save('b95',b95)
    # N10 equal budget uses historical five and new five linear probes, NOT the
    # similarly named integrated deep-model N10 table in CH6.
    eq=read(SUP/'FINAL_SSL_LABEL_SUFFICIENCY_N10_FF1_FF3_RESOURCE_LINK_RESULTS_v5/results/N10_MATCHED_ANALYSIS_ALL.csv')
    counts_check(eq,'N10 matched')
    rows=[]
    for sc,q in eq[eq.budget.eq(20000)&eq.rep.isin(['R0','RUT'])].groupby('scenario'):
        p=q.pivot(index='run',columns='rep',values='tpr')
        f=q.pivot(index='run',columns='rep',values='fpr')
        rows.append(dict(scenario=sc,R0_tpr=p.R0.mean(),RUT_tpr=p.RUT.mean(),delta_fpr=(f.RUT-f.R0).mean(),**describe(p.RUT-p.R0)))
    equal=pd.DataFrame(rows); equal['p_holm_secondary']=np.nan
    ix=equal.scenario.ne('OFFICIAL_TEST'); equal.loc[ix,'p_holm_secondary']=holm(equal.loc[ix,'p_raw'])
    save('ff3_equal_budget',equal)
    cross=SUP/'FINAL_SSL_CROSS_PARADIGM_LABEL_BENCHMARK_N10_RESULTS_v1'
    linear=read(cross/'results/N10_RUT_FULL_GRID_AND_R0_200K.csv')
    xgb=read(cross/'results/XGB_200K_N10_ALL_SCENARIOS.csv')
    counts_check(linear,'FF3 linear'); counts_check(xgb,'FF3 XGB')
    save('ff3_heatmap',linear[linear.rep.eq('RUT')].groupby(['budget','scenario'],as_index=False).agg(n=('run','size'),tpr=('tpr','mean'),fpr=('fpr','mean')))
    save('ff3_references',pd.concat([linear[linear.rep.eq('R0')].assign(reference='R0_200K'),xgb.assign(reference='XGB_200K')]).groupby(['reference','scenario'],as_index=False).agg(n=('tpr','size'),tpr=('tpr','mean'),fpr=('fpr','mean')))
    tests=[]
    for margin in [0,.005,.01,.02]:
        for budget,q in linear[linear.rep.eq('RUT')].groupby('budget'):
            for sc,a in q.groupby('scenario'):
                a=a.sort_values('run')
                r=linear[linear.rep.eq('R0')&linear.scenario.eq(sc)].sort_values('run')
                record(f'paired runs B{budget} {sc}',a.run,r.run)
                delta=a.tpr.to_numpy()-r.tpr.to_numpy()
                tests.append(dict(reference='R0_200K',margin=margin,budget=budget,scenario=sc,mean_delta=delta.mean(),delta_fpr=a.fpr.mean()-r.fpr.mean(),p_ni=flip(delta+margin,True)))
                x=xgb[xgb.scenario.eq(sc)]; va=a.tpr.var(ddof=1)/len(a); vb=x.tpr.var(ddof=1)/len(x)
                se=math.sqrt(va+vb); df=(va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(x)-1))
                delta=a.tpr.mean()-x.tpr.mean()
                tests.append(dict(reference='XGB_200K',margin=margin,budget=budget,scenario=sc,mean_delta=delta,delta_fpr=a.fpr.mean()-x.fpr.mean(),p_ni=t.sf((delta+margin)/se,df)))
    tests=save('ff3_ni_components',tests)
    glob=tests.groupby(['reference','margin','budget'],as_index=False).agg(p_iut=('p_ni','max'),worst_mean_delta=('mean_delta','min'),max_mean_delta=('mean_delta','max'),min_delta_fpr=('delta_fpr','min'),max_delta_fpr=('delta_fpr','max'))
    glob['p_holm']=glob.groupby(['reference','margin']).p_iut.transform(lambda p:holm(p.to_numpy()))
    glob['supported']=glob.p_holm<.05
    save('ff3_ni_global',glob); save('ff3_worst_gaps',glob[np.isclose(glob.margin,.01)])
    # Putra: independently integrate each run's AP curve before pairing.
    for tag,path,baseline,adapt in [
        ('putra_direct',SUP/'PUTRA_DIRECT_RESULTS/results/PUTRA_R0_VS_RUT_N10_RUN_METRICS.csv','R0','RUT'),
        ('putra_target',ART/'PUTRA_TARGET_ADAPTATION_RESULTS_SMALL/results/PUTRA_TARGET_ADAPTATION_N10_RUN_METRICS.csv','E0_SOURCE_RUT','E1_TARGET_RUT')]:
        d=read(path); parts=[]; areas=[]
        for (run,budget),q in d.groupby(['run','budget']):
            record(f'{tag} paired rows run={run} B={budget}',q.n_train,np.repeat(budget,len(q)))
            LOG.append(dict(check=f'{tag} paired selection hash run={run} B={budget}',passed=q.sample_hash.nunique()==1,max_abs_error=None))
        for (run,var),q in d.groupby(['run','variant']):
            q=q.sort_values('budget'); x=np.log(q.budget.to_numpy(float)); y=q.AP.to_numpy(float)
            area=np.sum(np.diff(x)*(y[1:]+y[:-1])/2)/(x[-1]-x[0])
            areas.append(dict(run=run,variant=var,AP_AULC=area))
        areas=save(tag+'_aulc_runs',areas); p=areas.pivot(index='run',columns='variant',values='AP_AULC')
        save(tag+'_primary',[dict(baseline=baseline,adapted=adapt,baseline_mean=p[baseline].mean(),adapted_mean=p[adapt].mean(),**describe(p[adapt]-p[baseline]))])
        for budget,q in d.groupby('budget'):
            p=q.pivot(index='run',columns='variant',values='AP')
            parts.append(dict(budget=budget,baseline_mean=p[baseline].mean(),adapted_mean=p[adapt].mean(),**describe(p[adapt]-p[baseline])))
        parts=pd.DataFrame(parts);parts['p_holm']=holm(parts.p_raw)
        save(tag+'_budgets',parts)
        save(tag+'_all_metric_means',d.groupby(['budget','variant'],as_index=False)[['AP','AUC','P_at_R90','FPR_at_TPR90']].mean())
    # Sequential FF4: 10 paired repetitions, not independent webpages.
    ff4=read(SUP/'FINAL_FF4_SEQUENTIAL_ARCHITECTURE_ABLATION_N10_RESULTS_v1/results/FF4_N10_ALL_RUNS.csv')
    counts_check(ff4,'FF4 sequential')
    save('ff4_components',ff4.groupby('variant',as_index=False).agg(n=('replicate','size'),tpr=('tpr','mean'),tpr_sd=('tpr','std'),fpr=('fpr','mean'),FPR_at_TPR90=('FPR_at_TPR90','mean')))
    p=ff4.pivot(index='replicate',columns='variant',values='tpr'); rows=[]
    variants=['P0_URL','P1_DUAL_EQUAL','P2_TRI_EQUAL','P3_TRI_GATED']
    for a,b in zip(variants[1:],variants[:-1]):rows.append(dict(contrast=a+'-'+b,**describe(p[a]-p[b])))
    rows=pd.DataFrame(rows);rows['p_holm']=holm(rows.p_raw);save('ff4_contrasts',rows)
    for replicate,q in ff4.groupby('replicate'):
        LOG.append(dict(check=f'FF4 paired selection hash {replicate}',passed=q.selection_sha256.nunique()==1,max_abs_error=None))
    # Distinguish scenario macro means from OFFICIAL_TEST population rates.
    frontend=read(SUP/'FINAL_SSL_GUIDED_ADAPTIVE_RESOURCE_SYSTEM_RESULTS_v2_ORDER_FIX/results/CASCADE_FINAL_N5_ALL.csv')
    counts_check(frontend,'FF4 frontend')
    cols=['escalation_rate','full_path_avoided_rate','delta_tpr_vs_full_pp','delta_fpr_vs_full_pp','tpr','fpr']
    save('ff4_frontend_scenarios',frontend.groupby(['allowed_tpr_loss_pp','stream','scenario'],as_index=False)[cols].mean())
    save('ff4_frontend_macro',frontend.groupby(['allowed_tpr_loss_pp','stream'],as_index=False)[cols].mean())
    save('ff4_frontend_official',frontend[frontend.scenario.eq('OFFICIAL_TEST')].groupby(['allowed_tpr_loss_pp','stream'],as_index=False)[cols].mean())
    # Full system score availability: direct CAL threshold and routing checks.
    ff4full=read(CH/'thesis_tables/TABLE_FF4_DOM_FUSION_CASCADE_FINAL.csv');counts_check(ff4full,'FF4 full per-run')
    save('ff4_system',ff4full[ff4full.system.eq('TRI_RUT_200000')&np.isclose(ff4full.target_fpr,.005)])
    freeze=json.loads((CH/'audit/ARCHITECTURE_FREEZE.json').read_text())
    low,high=freeze['cascade']['url_low'],freeze['cascade']['url_high']
    sd=CH/'scores/ff4/TRI_RUT_200000'
    cu=np.load(sd/'CAL_url.npy').astype(float);fu=np.load(sd/'FINAL_url.npy').astype(float)
    cg=np.load(sd/'CAL_gated.npy').astype(float)
    # Preserve historical NumPy float32 scalar-coercion at routing boundaries.
    low32=float(np.float32(low));high32=float(np.float32(high))
    cc=cg.copy();cc[cu>=high32]=1e9;cc[cu<=low32]=-1e9
    routed=(fu>low32)&(fu<high32)
    routing=float(routed.mean())
    save('routing',[dict(final_n=len(fu),escalated_n=int(routed.sum()),escalation_rate=routing,early_exit_rate=1-routing,float64_alternative_escalation_rate=float(((fu>low)&(fu<high)).mean()))])
    thresholds=[]
    for mode,cs in [('gated',cg),('cascade',cc)]:
        th=np.nextafter(np.sort(cs)[-int(.005*len(cs))],np.inf)
        q=ff4full[ff4full.system.eq('TRI_RUT_200000')&ff4full['mode'].eq(mode)&np.isclose(ff4full.target_fpr,.005)]
        record('CAL threshold '+mode,q.threshold,np.repeat(th,len(q)))
        thresholds.append(dict(mode=mode,threshold=th,cal_fp=int((cs>=th).sum()),cal_fpr=float((cs>=th).mean())))
    save('cal_thresholds',thresholds)
    ops=read(CH/'thesis_tables/TABLE_OPERATIONAL_BENCHMARK.csv').set_index('mode')
    full=ops.loc['full']; full=full[full.system.eq('TRI_RUT_200K_FULL')].iloc[0]
    cas=ops.loc['cascade']
    save('runtime_derived',[dict(full_latency_ms=full.latency_b1_median_ms,cascade_latency_ms=cas.latency_b1_median_ms,latency_reduction_pct=100*(1-cas.latency_b1_median_ms/full.latency_b1_median_ms),full_pages_s=full.throughput_pages_s_mean,cascade_pages_s=cas.throughput_pages_s_mean,speedup=cas.throughput_pages_s_mean/full.throughput_pages_s_mean,raw_timing_samples_available=False)])
    # Prior-shift precision transformation needs only prevalence and precision;
    # avoids incorrectly assigning maximum precision to an exact 90% recall.
    boot=read(CH/'thesis_tables/TABLE_FINAL_BOOTSTRAP_AP_P90.csv');prior=76800/168060;rows=[]
    for _,row in boot.iterrows():
        likelihood_ratio=(1-row.P_at_R90)/row.P_at_R90*prior/(1-prior)
        for prevalence in [prior,.05,.01]:
            precision=prevalence/(prevalence+(1-prevalence)*likelihood_ratio)
            rows.append(dict(system=row.system,prevalence=prevalence,precision=precision,source_precision=row.P_at_R90,interpretation='prior_shift_at_selected_PR_operating_point'))
    save('base_rate',rows)
    save('checks',LOG)
    (OUT/'empirical_input_hashes.json').write_text(json.dumps(FILES,indent=2),encoding='utf-8')
    summary=dict(checks=len(LOG),passed=sum(x['passed'] for x in LOG),failed=[x for x in LOG if not x['passed']],limitations=['This aggregate audit is supplemented by raw_score_audit.py, which uses recovered FINAL labels to reconstruct scores, ROC/AP and bootstrap intervals.','Raw per-page latency samples and per-repeat timing durations were not exported. Runtime level values are table claims; ratio/reduction arithmetic is independently reproduced.','This script reconstructs Putra AP-AULC and inference from historical per-run metrics. Additional CPU refits from recovered embeddings are documented in closure_summary.json.','N10 replication measures downstream variation at fixed TAPT checkpoints and fixed evaluation population; not independent pretraining, temporal or deployment replications.'])
    (OUT/'empirical_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
