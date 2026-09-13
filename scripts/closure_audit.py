"""KI-unterstützter unabhängiger Ergänzungsaudit; keine historischen Ergebnisse überschreiben."""
from pathlib import Path
import json, itertools, math, hashlib
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'review'

def paired(d):
    d=np.asarray(d,float);n=len(d);se=d.std(ddof=1)/np.sqrt(n)
    permutations=np.array(list(itertools.product((-1,1),repeat=n)))@d/n
    return dict(n=n,mean_delta_pp=100*d.mean(),ci95_lo_pp=100*(d.mean()-t.ppf(.975,n-1)*se),
                ci95_hi_pp=100*(d.mean()+t.ppf(.975,n-1)*se),
                p_exact=float(np.mean(abs(permutations)>=abs(d.mean())-1e-15)),
                positive=int(sum(d>0)),negative=int(sum(d<0)))

# Sämtliche einfachen FF1-Kontraste plus faktorielle Interaktion aus Integer-Counts.
ff1=pd.read_csv(ROOT/'artifacts/CH6_ANALYSIS_FULL/results/FF1_MODALITY_N5_ALL.csv')
ff1['tpr_recomputed']=ff1.tp/(ff1.tp+ff1.fn)
assert np.allclose(ff1.tpr,ff1.tpr_recomputed,rtol=0,atol=1e-14)
all_ff1=[]
for probe,q in ff1.groupby('probe'):
    p=q.pivot(index='run',columns='rep',values='tpr_recomputed')
    for base,adapted in itertools.combinations(['R0','RT','RU','RUT'],2):
        all_ff1.append(dict(probe=probe,contrast=adapted+'-'+base,**paired(p[adapted]-p[base])))
    all_ff1.append(dict(probe=probe,contrast='RUT-RT-RU+R0',**paired(p.RUT-p.RT-p.RU+p.R0)))
pd.DataFrame(all_ff1).to_csv(OUT/'FF1_ALL_CONTRASTS.csv',index=False)

# N10-Einzelmittel und ihre t-Intervalle (von gepaarten Differenzintervallen getrennt).
supp=ROOT/'artifacts/empirical_supplements'
eq=pd.read_csv(supp/'FINAL_SSL_LABEL_SUFFICIENCY_N10_FF1_FF3_RESOURCE_LINK_RESULTS_v5/results/N10_MATCHED_ANALYSIS_ALL.csv')
meanrows=[]
for (rep,budget,scenario),q in eq.groupby(['rep','budget','scenario']):
    assert len(q)==10
    for metric in ['tpr','fpr']:
        d=q[metric].to_numpy();half=t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
        meanrows.append(dict(rep=rep,budget=budget,scenario=scenario,metric=metric,n=len(d),mean_pct=100*d.mean(),ci95_lo_pct=100*(d.mean()-half),ci95_hi_pct=100*(d.mean()+half)))
pd.DataFrame(meanrows).to_csv(OUT/'closure_n10_means_ci.csv',index=False)

# Finale Systemabstände sind deskriptive Unterschiede desselben eingefrorenen Modells.
sys=pd.read_csv(OUT/'empirical_ff4_system.csv');gaps=[]
for scenario,q in sys.groupby('scenario'):
    q=q.set_index('mode');q['tpr_raw']=q.tp/(q.tp+q.fn);q['fpr_raw']=q.fp/(q.fp+q.tn)
    for new,old in [('gated','equal'),('cascade','gated')]:
        gaps.append(dict(scenario=scenario,contrast=new+'-'+old,tpr_delta_pp=100*(q.loc[new,'tpr_raw']-q.loc[old,'tpr_raw']),fpr_delta_pp=100*(q.loc[new,'fpr_raw']-q.loc[old,'fpr_raw'])))
pd.DataFrame(gaps).to_csv(OUT/'closure_ff4_system_gaps.csv',index=False)

# Bayes-Umrechnung separat mit P*TP/FP-Äquivalent gegen Primäraudit prüfen.
base=pd.read_csv(OUT/'empirical_base_rate.csv')
pi0=76800/168060
base['precision_bayes']=base.source_precision*(base.prevalence/pi0)/(base.source_precision*(base.prevalence/pi0)+(1-base.source_precision)*(1-base.prevalence)/(1-pi0))
assert np.allclose(base.precision,base.precision_bayes,rtol=0,atol=1e-14)
base.to_csv(OUT/'closure_base_rate.csv',index=False)

raw=pd.read_csv(OUT/'raw_putra_refit_metrics.csv')
aulc=[]
for (variant,run),q in raw.groupby(['variant','run']):
    q=q.sort_values('budget');x=np.log10(q.budget.to_numpy())
    assert list(q.budget)==[200,500,1000,2000]
    aulc.append(dict(variant=variant,run=run,AP_AULC=float(np.trapz(q.AP,x)/(x[-1]-x[0]))))
aulc=pd.DataFrame(aulc);aulc.to_csv(OUT/'closure_refit_aulc.csv',index=False)
pv=aulc.pivot(index='run',columns='variant',values='AP_AULC')
results=[dict(contrast='RUT-R0',**paired(pv.RUT-pv.R0)),dict(contrast='E1_TARGET_RUT-RUT',**paired(pv.E1_TARGET_RUT-pv.RUT))]
pd.DataFrame(results).to_csv(OUT/'closure_refit_inference.csv',index=False)
errors=pd.read_csv(OUT/'raw_score_checks.csv')
err=errors[errors['check'].str.startswith('Putra ')].copy()
err['metric']=err['check'].str.split().str[-1]
err=err.groupby('metric',as_index=False).agg(checks=('error','size'),different_at_1e8=('passed',lambda s:int((~s).sum())),max_abs_error=('error','max'))
err['max_abs_error_pp']=100*err.max_abs_error
err.to_csv(OUT/'closure_refit_errors.csv',index=False)

disjoint=ROOT/'artifacts/audit_supplements/tapt_disjoint/FINAL_OOD_TAPT_DISJOINTNESS_AUDIT_v1/OOD_TAPT_DISJOINTNESS_AUDIT.csv'
d=pd.read_csv(disjoint)
assert (d.overlap_rows==0).all()
rates=d[d.reference.eq('SSL')].copy()
rates['missing_n']=rates.scenario_rows-rates.nonempty_rows
rates['missing_pct']=100*rates.missing_n/rates.scenario_rows
rates.to_csv(OUT/'closure_ood_missing.csv',index=False)
report={
 'refit_inference':results,
 'refit_errors':err.to_dict('records'),
 'raw_metric_checks':len(errors),'raw_metric_passes':int(errors.passed.sum()),
 'raw_non_putra_checks':int((~errors['check'].str.startswith('Putra ')).sum()),
 'raw_non_putra_all_pass':bool(errors.loc[~errors['check'].str.startswith('Putra '),'passed'].all()),
 'ood_missing':rates[['scenario_check','missing_n','missing_pct']].to_dict('records'),
 'ood_scope':'Rechenkontrolle der exportierten Audit-Zählungen; Identitätsüberschneidungen mangels sämtlicher Rohrollen nicht neu berechnet.',
 'unsupported_claim_removed':'99.618% vollständige URLs bei 128 Token: kein zuordenbarer 200k-Roh- oder Zählexport gefunden.',
 'historical_iid_excluded':'3051er-Historienbestand überschneidet sich mit SSL/DEV/CAL, wird nicht als aktueller FF3-Test verwendet.',
 'gpu_scope':'Kein erneutes Encoder-/T4-Training; Zeitquantile und absolute Durchsätze nur gegen Code und Aggregate geprüft.',
}
# Rollenlabels, Zeitgrenze und exakte SHA-Disjunktheit aus wiedergewonnenen Metadaten.
role_dir=ROOT/'artifacts/raw_recovery/role_meta'
roles={p.stem:pd.read_parquet(p) for p in role_dir.glob('*.parquet')}
role_checks=[]
for name,q in roles.items():
    role_checks.append(dict(role=name,n=len(q),negative=int(sum(q.y==0)),positive=int(sum(q.y==1)),duplicate_sha=int(q.sha256.duplicated().sum())))
    assert not q.sha256.duplicated().any()
for a,b in itertools.combinations(roles,2):
    overlap=len(set(roles[a].sha256)&set(roles[b].sha256))
    role_checks.append(dict(pair=a+' / '+b,overlap=overlap));assert overlap==0
dates=pd.to_datetime(roles['FINAL'].date,utc=True);cut=dates.quantile(.75)
role_checks.append(dict(role='FINAL temporal',first=str(dates.min()),last=str(dates.max()),cut_q75=str(cut),late_n=int(sum(dates>=cut)),ties_at_cut=int(sum(dates==cut))))
(OUT/'closure_role_audit.json').write_text(json.dumps(role_checks,indent=2),encoding='utf-8')
report['independent_role_scope']='CAL, DEV, SSL und FINAL: sämtliche Labels/Counts, SHA-Duplikate, sechs paarweise SHA-Schnittmengen und FINAL-Zeitquantil unabhängig aus Metadaten geprüft. SUP und exakte Domain-/Template-Identitäten nur gegen archivierte Audits geprüft.'
(OUT/'closure_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=True,indent=2))
