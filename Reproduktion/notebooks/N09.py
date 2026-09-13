# KI-Unterstuetzung: siehe Erklaerung der Arbeit.
# Ursprungsdatei: phreshphish_FINAL_RQ_CLOSURE_20K_200K_SSL_DOM_FUSION_v7_2_FINAL(3).ipynb
# CELL 1
# 00 — Environment / configuration / atomic helpers
import os,gc,re,json,math,time,random,hashlib,zipfile,shutil,warnings,itertools
from pathlib import Path
from contextlib import nullcontext
from collections import Counter
import numpy as np,pandas as pd
import torch,torch.nn as nn,torch.nn.functional as F
from torch.utils.data import IterableDataset,DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score,roc_curve,precision_recall_curve
from scipy.stats import beta as beta_dist,t as student_t
warnings.filterwarnings('ignore')

if not torch.cuda.is_available():
 raise RuntimeError('STOP: GPU erforderlich')
DEVICE=torch.device('cuda')
AMP=True

# ------------------------------------------------------------------
# Scientific / engineering freeze
# ------------------------------------------------------------------
MASTER_SEED=20260813
SCI20K_SEED=20260813
DOWNSTREAM_SEEDS=[42,62,82,102,122]
ENGINEERING_SEED=82

SCI_LABEL_ROWS=20000
HIGH_LABEL_ROWS=200000

URL_MODEL_ID='bert-base-uncased'
TEXT_MODEL_ID='roberta-base'
URL_MAX_LEN=128
TEXT_MAX_LEN=256
PROJ_DIM=256

TARGET_FPRS=[.0001,.0005,.001,.0025,.005,.01,.02]
PRIMARY_FPR=.005

# Corrected contrastive
SSL_ROWS=200000
CONTRASTIVE_EPOCHS=1
CONTRASTIVE_LAST_N=2
CONTRASTIVE_LR=1e-5
CONTRASTIVE_TEMP=.10
CONTRASTIVE_QUEUE=2048
URL_MASK_P=.08
TEXT_MASK_P=.12
HYBRID_SAME_WEIGHT=.70
HYBRID_CROSS_WEIGHT=.30
SSL_BATCH_CANDIDATES=[16,8,4]
SSL_RESUME_EVERY=2000

# Deep supervised
DEEP_EPOCHS=1
DEEP_LAST_N=4
ENCODER_LR=1e-5
HEAD_LR=2e-4
WEIGHT_DECAY=.01
WARMUP_RATIO=.05
AUX_TOTAL_WEIGHT=.30
DEEP_RESUME_EVERY=1000
DUAL_BATCH_CANDIDATES=[16,8,4]
TRI_BATCH_CANDIDATES=[8,4,2]

# DOM
DOM_MAX_NODES=512
DOM_DIM=128
DOM_LAYERS=3
DOM_COLS=['dom_tag','dom_parent_idx','dom_depth','dom_attr_count','dom_child_count']

# Cascade gate, frozen before FINAL
CASCADE_MAX_TPR_LOSS_PP=1.0
CASCADE_MIN_FULL_REDUCTION=.30

ROOT=Path('/kaggle/working/phreshphish_FINAL_RQ_CLOSURE_20K_200K_v7_2_FINAL') if Path('/kaggle/working').exists() else Path('/mnt/data/phreshphish_FINAL_RQ_CLOSURE_20K_200K_v7_2_FINAL')
CKPT=ROOT/'checkpoints';EMB=ROOT/'embeddings';SCORES=ROOT/'scores';RESULTS=ROOT/'results';AUDIT=ROOT/'audit';PARTS=ROOT/'parts'
for p in [ROOT,CKPT,EMB,SCORES,RESULTS,AUDIT,PARTS]:p.mkdir(parents=True,exist_ok=True)

def seed_all(seed):
 random.seed(int(seed));np.random.seed(int(seed));torch.manual_seed(int(seed))
 if torch.cuda.is_available():torch.cuda.manual_seed_all(int(seed))

def atomic_json(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8');os.replace(tmp,path)

def atomic_torch(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=Path(str(path)+'.tmp');torch.save(obj,tmp);os.replace(tmp,path)

def amp_ctx():
 return torch.autocast('cuda',dtype=torch.float16) if AMP else nullcontext()

seed_all(MASTER_SEED)
CONFIG={
 'version':'v7_2_FINAL_RQ_CLOSURE_20K_200K',
 'scientific_primary_label_rows':SCI_LABEL_ROWS,
 'scientific_primary_role':'20k balanced labeled subset of the 200k SSL pool',
 'high_supervision_label_rows':HIGH_LABEL_ROWS,
 'high_supervision_role':'secondary R0-vs-R1 label-scaling condition + engineering artifact',
 'legacy_4k_used':False,
 'downstream_seeds':DOWNSTREAM_SEEDS,
 'historical_final_results_known':True,
 'threshold_domain':'raw_fp32_logits',
 'ff1_representations':['R0','R1','R2C','R3C','CSAME'],
 'ff2_budgets':[SCI_LABEL_ROWS,HIGH_LABEL_ROWS],
 'ff4_dom_retested_at_labels':HIGH_LABEL_ROWS
}
EXPERIMENT_REGISTRY=[
 {'rq':'FF1','id':'FF1_REP_20K','labels':20000,'representations':['R0','R1','R2C','R3C','CSAME'],
  'downstream':['LINEAR','MLP'],'seeds':DOWNSTREAM_SEEDS,'primary_metric':'TPR@0.5% FPR',
  'purpose':'downstream utility of SSL representations'},
 {'rq':'FF2','id':'FF2_MECHANISM_20K','labels':20000,'representations':['R0','R1','R2C','R3C','CSAME'],
  'downstream':['LINEAR','MLP'],'seeds':DOWNSTREAM_SEEDS,
  'purpose':'dependence of SSL utility on downstream mechanism'},
 {'rq':'FF2','id':'FF2_LABEL_SCALE_DEEP','labels':[20000,200000],'representations':['R0','R1'],
  'downstream':['matched Deep DUAL'],'seeds':[ENGINEERING_SEED],
  'purpose':'change of DAPT utility with labeled supervision'},
 {'rq':'FF3','id':'FF3_SHIFT_20K','labels':20000,'representations':['R0','R1','R2C','R3C','CSAME'],
  'downstream':['LINEAR','MLP'],'seeds':DOWNSTREAM_SEEDS,
  'scenarios':['OFFICIAL_TEST','DOMAIN_OOD_EXACT','TEMPLATE_OOD_EXACT','DOMAIN_TEMPLATE_OOD_EXACT','LATE_TEST_Q4'],
  'fpr_targets':TARGET_FPRS,'purpose':'representation robustness under shift and operating constraints'},
 {'rq':'FF4','id':'FF4_ENGINEERING_20K_200K','labels':[20000,200000],
  'representations':['20k SSL champion held constant across both label conditions'],
  'downstream':['DUAL','TRI','URL-only','TEXT-only','DOM-only','equal fusion','gated fusion','200k cascade'],
  'seeds':[ENGINEERING_SEED],
  'purpose':'DOM/fusion utility at 20k vs 200k plus final 200k cascade/resources'}
]
atomic_json(AUDIT/'EXPERIMENT_REGISTRY.json',EXPERIMENT_REGISTRY)

atomic_json(AUDIT/'V7_PROTOCOL_FREEZE.json',CONFIG)
print(torch.cuda.get_device_name(0))
print(json.dumps(CONFIG,indent=2))


# CELL 2
# 01 — Resolve DATA/META roots, DAPT/DOM assets, optional completed v6 corrected SSL
import pyarrow.parquet as pq

SEARCH_ROOT=Path('/kaggle/input') if Path('/kaggle/input').exists() else Path('/mnt/data')
FREEZE_ID='PHRESHPHISH_FINAL_DATA_FREEZE_v3_HF_SHARDED_DOM'
ROLE_REL={
 'SUP':Path('roles/supervised_pool'),
 'DEV':Path('roles/development'),
 'CAL':Path('roles/fpr_calibration_benign'),
 'SSL':Path('roles/ssl_pool_200k'),
 'FINAL':Path('roles/final_test')
}

# Correct metadata sidecar
markers=[]
for p in SEARCH_ROOT.rglob('FINAL_DATA_FREEZE_COMPLETE.json'):
 try:o=json.loads(p.read_text())
 except:continue
 if o.get('status')=='COMPLETE' and o.get('freeze_id')==FREEZE_ID:
  markers.append((p,o))
if len(markers)!=1:
 raise RuntimeError(f'STOP expected exactly one corrected freeze marker, found {len(markers)}')
FREEZE_FILE,FREEZE_INFO=markers[0]
META_ROOT=FREEZE_FILE.parent
MANIFEST_ROOT=META_ROOT/'manifests'
FREEZE_AUDIT=META_ROOT/'audit'

reqmeta=[
 MANIFEST_ROOT/'train_role_manifest_PRIVATE_WITH_LABELS.parquet',
 MANIFEST_ROOT/'final_test_exact_ood_flags_SEALED.parquet',
 MANIFEST_ROOT/'final_test_manifest_SEALED.parquet'
]
miss=[str(x) for x in reqmeta if not x.exists()]
if miss:raise RuntimeError(f'STOP metadata missing {miss}')

# Large role root
roots=[]
for r in SEARCH_ROOT.rglob('roles'):
 if r.is_dir():
  root=r.parent
  if all((root/v).exists() for v in ROLE_REL.values()):roots.append(root)
roots=list({str(x.resolve()):x for x in roots}.values())
if len(roots)!=1:raise RuntimeError(f'STOP expected one role root, found {[str(x) for x in roots]}')
DATA_ROOT=roots[0]
ROLE_DIR={k:DATA_ROOT/v for k,v in ROLE_REL.items()}

def nrows(d):
 fs=sorted(Path(d).glob('*.parquet'))
 if not fs:raise RuntimeError(f'no parquet in {d}')
 return sum(pq.ParquetFile(p).metadata.num_rows for p in fs)
COUNTS={k:nrows(v) for k,v in ROLE_DIR.items()}
EXPECTED={'SUP':4000,'DEV':20000,'CAL':50000,'SSL':200000,'FINAL':168060}
if COUNTS!=EXPECTED:raise RuntimeError(f'STOP role counts {COUNTS}')

# Old full checkpoint root containing DAPT and DOM SSL
old=[]
for p in SEARCH_ROOT.rglob('phreshphish_FINAL_DEEP_TRIMODAL_SSL_v4_3_LABELSET_ALIGNMENT_FIX'):
 if not p.is_dir():continue
 score=sum(x.exists() for x in [
  p/'checkpoints/R1_DAPT_TEXT/COMPLETE.json',
  p/'checkpoints/DOM_MASKED_SSL/encoder.pt',
  p/'audit/DOM_VOCAB.json'
 ])
 if score:old.append((score,p))
if not old:raise RuntimeError('STOP attach bigresults/full v4.3 checkpoint dataset')
old.sort(key=lambda x:x[0],reverse=True)
OLD_ROOT=old[0][1]
DAPT_DIR=OLD_ROOT/'checkpoints/R1_DAPT_TEXT'
DOM_ENCODER_PATH=OLD_ROOT/'checkpoints/DOM_MASKED_SSL/encoder.pt'
DOM_VOCAB_PATH=OLD_ROOT/'audit/DOM_VOCAB.json'
for p in [DAPT_DIR/'COMPLETE.json',DOM_ENCODER_PATH,DOM_VOCAB_PATH]:
 if not p.exists():raise RuntimeError(f'STOP missing required old scientific asset {p}')

# Optional completed v6 corrected contrastive dirs.
def find_completed_dir(dirname):
 hits=[]
 for done in SEARCH_ROOT.rglob(f'{dirname}/COMPLETE.json'):
  try:o=json.loads(done.read_text())
  except:continue
  if o.get('status')=='COMPLETE' and (done.parent/'url').exists() and (done.parent/'text').exists():
   hits.append(done.parent)
 return sorted(hits,key=lambda p:len(str(p)))[0] if hits else None

V6_R2C=find_completed_dir('R2C_CONTRASTIVE_REPAIRED')
V6_R3C=find_completed_dir('R3C_DAPT_CONTRASTIVE_REPAIRED')

# Front-loaded sealed manifest check
if pq.ParquetFile(MANIFEST_ROOT/'final_test_exact_ood_flags_SEALED.parquet').metadata.num_rows!=168060:
 raise RuntimeError('STOP OOD manifest row mismatch')

atomic_json(AUDIT/'INPUT_RESOLUTION.json',{
 'status':'PASS','data_root':str(DATA_ROOT),'meta_root':str(META_ROOT),
 'old_root':str(OLD_ROOT),'counts':COUNTS,
 'v6_r2c':None if V6_R2C is None else str(V6_R2C),
 'v6_r3c':None if V6_R3C is None else str(V6_R3C)
})
print({'INPUT_RESOLUTION':'PASS','counts':COUNTS,'R2C_REUSE':str(V6_R2C),'R3C_REUSE':str(V6_R3C)})


# CELL 3
# 02 — Transformer / metrics / data-integrity utilities
from transformers import AutoTokenizer,AutoModel,get_linear_schedule_with_warmup
HF_TOKEN=os.environ.get('HF_TOKEN')
if not HF_TOKEN and Path('/kaggle/working').exists():
 try:
  from kaggle_secrets import UserSecretsClient
  HF_TOKEN=UserSecretsClient().get_secret('HF_TOKEN')
 except:pass

def resolve(mid):
 hits=[]
 for p in SEARCH_ROOT.rglob('config.json'):
  if mid.split('/')[-1].lower() in str(p.parent).lower():hits.append(p.parent)
 return str(sorted(hits,key=lambda x:len(str(x)))[0]) if hits else mid

URL_SRC=resolve(URL_MODEL_ID);TEXT_SRC=resolve(TEXT_MODEL_ID)
url_tok=AutoTokenizer.from_pretrained(URL_SRC,token=HF_TOKEN)
text_tok=AutoTokenizer.from_pretrained(TEXT_SRC,token=HF_TOKEN)
_u=AutoModel.from_pretrained(URL_SRC,token=HF_TOKEN);_t=AutoModel.from_pretrained(TEXT_SRC,token=HF_TOKEN)
URL_H=_u.config.hidden_size;TEXT_H=_t.config.hidden_size
del _u,_t;gc.collect()

def freeze_last(model,n):
 for p in model.parameters():p.requires_grad=False
 layers=getattr(getattr(model,'encoder',None),'layer',None)
 if layers is None:layers=getattr(getattr(model,'transformer',None),'layer',None)
 if layers is None:raise RuntimeError(type(model))
 for layer in layers[-n:]:
  for p in layer.parameters():p.requires_grad=True
 if getattr(model,'pooler',None) is not None:
  for p in model.pooler.parameters():p.requires_grad=True

def masked_mean(h,m):
 mm=m.unsqueeze(-1).to(h.dtype);return (h*mm).sum(1)/mm.sum(1).clamp_min(1)
def pooled(out,mask):return masked_mean(out.last_hidden_state,mask)

def y01(s):
 y=s.astype(str).str.lower().map({'benign':0,'phish':1})
 if y.isna().any():raise RuntimeError(f'unknown labels: {s[y.isna()].unique()}')
 return y.astype(int).to_numpy()

# Tie-aware conservative threshold; works on raw logits.
def thr_fpr(neg_scores,f):
 s=np.asarray(neg_scores,dtype=np.float64)
 if not np.isfinite(s).all():raise RuntimeError('non-finite threshold scores')
 k=int(math.floor(float(f)*len(s)))
 if k<=0:return float(np.nextafter(s.max(),np.inf))
 desc=np.sort(s)[::-1];b=float(desc[k-1])
 return b if int((s>=b).sum())<=k else float(np.nextafter(b,np.inf))

def ci_fpr(fp,n):
 if n==0:return (np.nan,np.nan)
 return (0. if fp==0 else float(beta_dist.ppf(.025,fp,n-fp+1)),
         1. if fp==n else float(beta_dist.ppf(.975,fp+1,n-fp)))

def op(y,score,th):
 y=np.asarray(y);s=np.asarray(score);p=s>=th
 tp=int(((p==1)&(y==1)).sum());fp=int(((p==1)&(y==0)).sum())
 tn=int(((p==0)&(y==0)).sum());fn=int(((p==0)&(y==1)).sum())
 tpr=tp/max(tp+fn,1);fpr=fp/max(fp+tn,1);prec=tp/max(tp+fp,1);f1=2*prec*tpr/max(prec+tpr,1e-12)
 lo,hi=ci_fpr(fp,fp+tn)
 return {'tpr':tpr,'fpr':fpr,'precision':prec,'f1':f1,'tp':tp,'fp':fp,'tn':tn,'fn':fn,
         'fp_per_1000':1000*fpr,'fpr_ci_lo':lo,'fpr_ci_hi':hi}

def precision_at_recall(y,s,target=.90):
 p,r,_=precision_recall_curve(y,s);ra=r[::-1];pa=p[::-1]
 grid=np.linspace(0,1,2001);pi=np.interp(grid,ra,pa);j=int(np.argmin(np.abs(grid-target)))
 return float(pi[j])

def curves(y,s):
 fpr,tpr,_=roc_curve(y,s)
 def fat(target):
  ii=np.where(tpr>=target)[0];return float(fpr[ii[0]]) if len(ii) else np.nan
 return {'AP':float(average_precision_score(y,s)),'AUC':float(roc_auc_score(y,s)),
         'P_at_R90':precision_at_recall(y,s,.90),'FPR_at_TPR90':fat(.90),'FPR_at_TPR95':fat(.95)}

def read_role(key,cols):
 out=[]
 for p in sorted(ROLE_DIR[key].glob('*.parquet')):
  names=set(pq.ParquetFile(p).schema_arrow.names)
  missing=[x for x in cols if x not in names]
  if missing:raise RuntimeError(f'{p}: missing {missing}')
  out.append(pd.read_parquet(p,columns=cols))
 return pd.concat(out,ignore_index=True)


# Physical-row layout for memory-safe contiguous DOM retrieval from CAL/FINAL.
ROLE_LAYOUT={}
for _key in ['CAL','FINAL']:
 _layout=[];_off=0
 for _p in sorted(ROLE_DIR[_key].glob('*.parquet')):
  _n=pq.ParquetFile(_p).metadata.num_rows
  _layout.append((_p,_off,_off+_n));_off+=_n
 ROLE_LAYOUT[_key]=_layout

def read_role_slice(key,start,end,cols):
 if key not in ROLE_LAYOUT:raise KeyError(key)
 if not (0<=start<=end<=COUNTS[key]):raise ValueError((key,start,end))
 parts=[]
 for p,a,b in ROLE_LAYOUT[key]:
  lo=max(start,a);hi=min(end,b)
  if lo>=hi:continue
  pf=pq.ParquetFile(p)
  names=set(pf.schema_arrow.names);missing=[x for x in cols if x not in names]
  if missing:raise RuntimeError(f'{p}: missing slice cols {missing}')
  tab=pf.read(columns=cols).slice(lo-a,hi-lo)
  parts.append(tab.to_pandas())
 out=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=cols)
 if len(out)!=(end-start):raise RuntimeError(f'role slice mismatch {key} {start}:{end} -> {len(out)}')
 return out

# Private labels for exactly the 200k SSL pool.
priv=pd.read_parquet(MANIFEST_ROOT/'train_role_manifest_PRIVATE_WITH_LABELS.parquet',columns=['sha256','label','role'])
priv['sha256']=priv.sha256.astype(str).str.lower()
sslmeta=priv[priv.role.eq('SSL_POOL')][['sha256','label']].copy().reset_index(drop=True)
if len(sslmeta)!=200000 or sslmeta.sha256.duplicated().any():raise RuntimeError('SSL private manifest invalid')
sslmeta['y']=y01(sslmeta.label)

# Verify physical SSL pool exactly matches hidden-label manifest.
physical=[]
for p in sorted(ROLE_DIR['SSL'].glob('*.parquet')):
 q=pd.read_parquet(p,columns=['sha256']);physical.extend(q.sha256.astype(str).str.lower().tolist())
if len(physical)!=200000 or len(set(physical))!=200000:raise RuntimeError('physical SSL SHA invalid')
if set(physical)!=set(sslmeta.sha256):raise RuntimeError('physical/private SSL mismatch')
SSL_LABEL_MAP=dict(zip(sslmeta.sha256,sslmeta.y.astype(int)))

# Deterministic balanced 20k subset from SSL pool; NO legacy SUP4k.
def keyed_rank(sha,seed):
 return hashlib.sha256(f'{seed}|{sha}'.encode()).hexdigest()
sel=[]
for cls in [0,1]:
 q=sslmeta[sslmeta.y.eq(cls)].copy()
 q['rank']=[keyed_rank(x,SCI20K_SEED) for x in q.sha256]
 q=q.sort_values('rank').head(SCI_LABEL_ROWS//2)
 sel.append(q[['sha256','y']])
SCI20_META=pd.concat(sel,ignore_index=True)
SCI20_META=SCI20_META.sort_values('sha256').reset_index(drop=True)
SCI20_SHA=set(SCI20_META.sha256)
if len(SCI20_META)!=20000 or int(SCI20_META.y.sum())!=10000:raise RuntimeError('SCI20K subset invalid')

# Materialize the 20k URL/text rows only.
parts=[]
for p in sorted(ROLE_DIR['SSL'].glob('*.parquet')):
 q=pd.read_parquet(p,columns=['sha256','url','text'])
 q['sha256']=q.sha256.astype(str).str.lower();q=q[q.sha256.isin(SCI20_SHA)]
 if len(q):parts.append(q)
SCI20=pd.concat(parts,ignore_index=True).merge(SCI20_META,on='sha256',validate='one_to_one')
SCI20=SCI20.sort_values('sha256').reset_index(drop=True)
if len(SCI20)!=20000 or int(SCI20.y.sum())!=10000:raise RuntimeError('SCI20K materialization invalid')

DEV=read_role('DEV',['sha256','url','text','label','date']+DOM_COLS)
DEV['sha256']=DEV.sha256.astype(str).str.lower();DEVY=y01(DEV.label)
CAL=read_role('CAL',['sha256','url','text','label','date'])
CAL['sha256']=CAL.sha256.astype(str).str.lower();CALY=y01(CAL.label)
if len(CAL)!=50000 or int(CALY.sum())!=0:raise RuntimeError('CAL invalid')
FINAL=read_role('FINAL',['sha256','url','text','label','date'])
FINAL['sha256']=FINAL.sha256.astype(str).str.lower();FINALY=y01(FINAL.label)
if len(FINAL)!=168060 or (int((FINALY==0).sum()),int((FINALY==1).sum()))!=(91260,76800):raise RuntimeError('FINAL invalid')
if any(c not in DEV.columns for c in DOM_COLS):raise RuntimeError('DEV DOM columns missing')
# Front-load a tiny CAL/FINAL slice to catch nested Arrow/schema problems before training.
for _key in ['CAL','FINAL']:
 _probe=read_role_slice(_key,0,min(8,COUNTS[_key]),['sha256']+DOM_COLS)
 if len(_probe)==0 or any(c not in _probe.columns for c in DOM_COLS):
  raise RuntimeError(f'{_key} DOM slice preflight failed')
print({'DOM_DATA_PREFLIGHT':'PASS','DEV_rows':len(DEV),'CAL_FINAL_mode':'chunked parquet slices'})


# Leakage/disjointness at row hash level.
sets={'SCI20':set(SCI20.sha256),'DEV':set(DEV.sha256),'CAL':set(CAL.sha256),'FINAL':set(FINAL.sha256)}
for a,b in itertools.combinations(sets,2):
 if sets[a]&sets[b]:raise RuntimeError(f'STOP SHA overlap {a} {b}: {len(sets[a]&sets[b])}')

# Development partitions: 10k/10k for representation threshold/eval; plus frozen 5k ENG_TUNE/ENG_META.
idx=np.arange(len(DEV))
REP_CAL,REP_EVAL=train_test_split(idx,test_size=.5,stratify=DEVY,random_state=20260821)
a,b=train_test_split(idx,test_size=.5,stratify=DEVY,random_state=20260812)
_,_=a,b
ENG_TUNE,ENG_META=train_test_split(b,test_size=.5,stratify=DEVY[b],random_state=20260814)
ENG_TUNE=np.sort(ENG_TUNE);ENG_META=np.sort(ENG_META)
REP_CAL=np.sort(REP_CAL);REP_EVAL=np.sort(REP_EVAL)

flags=pd.read_parquet(MANIFEST_ROOT/'final_test_exact_ood_flags_SEALED.parquet')
flags['sha256']=flags.sha256.astype(str).str.lower();flags=flags.set_index('sha256').loc[FINAL.sha256].reset_index()
dt=pd.to_datetime(FINAL.date,errors='coerce');cut=dt.dropna().quantile(.75);late=(dt>=cut).fillna(False).to_numpy()
FINAL_MASKS={
 'OFFICIAL_TEST':np.ones(len(FINAL),dtype=bool),
 'DOMAIN_OOD_EXACT':flags.domain_ood_exact.to_numpy(bool),
 'TEMPLATE_OOD_EXACT':flags.template_ood_exact.to_numpy(bool),
 'DOMAIN_TEMPLATE_OOD_EXACT':flags.domain_template_ood_exact.to_numpy(bool),
 'LATE_TEST_Q4':late
}

atomic_json(AUDIT/'DATA_PROTOCOL.json',{
 'status':'PASS','scientific_20k':{'rows':len(SCI20),'benign':10000,'phish':10000,'source':'SSL_POOL hidden labels'},
 'high_supervision_rows':200000,'legacy_SUP4k_used':False,
 'dev_rows':len(DEV),'cal_rows':len(CAL),'final_rows':len(FINAL),
 'sha_disjointness':'PASS'
})
print({'DATA_PROTOCOL':'PASS','SCI20':len(SCI20),'HIGH':200000,'legacy_4k_used':False})


# CELL 4
# 03 — Corrected contrastive checkpoints: reuse v6 or rebuild R2C/R3C; train C-SAME
def h64(x):
 if x is None:return 0
 x=str(x)
 if not x:return 0
 return int(hashlib.sha256(x.encode('utf-8',errors='ignore')).hexdigest()[:15],16)

class SSLIter(IterableDataset):
 def __init__(self,seed,max_rows=SSL_ROWS):
  self.files=sorted(ROLE_DIR['SSL'].glob('*.parquet'));self.seed=seed;self.max_rows=max_rows
 def __iter__(self):
  rng=np.random.default_rng(self.seed);files=list(self.files);rng.shuffle(files);n=0
  for p in files:
   q=pd.read_parquet(p,columns=['url','text','domain','template_hash'])
   ii=np.arange(len(q));rng.shuffle(ii)
   for j in ii:
    r=q.iloc[j]
    yield {'url':str(r.url or ''),'text':str(r.text or ''),
           'domain_h':h64(r.domain),'template_h':h64(r.template_hash)}
    n+=1
    if n>=self.max_rows:return

def token_view(batch,tok,p):
 out={k:v.clone() for k,v in batch.items()}
 ids=out['input_ids'];att=out['attention_mask'].bool();special=torch.zeros_like(att)
 for sid in tok.all_special_ids:special|=ids.eq(int(sid))
 mask=(torch.rand(ids.shape,device=ids.device)<p)&att&(~special)
 ids[mask]=int(tok.mask_token_id);out['input_ids']=ids
 return out

def family_mask(qd,qt,cd,ct,b):
 same_d=(qd[:,None].eq(cd[None,:]))&qd[:,None].ne(0)
 same_t=(qt[:,None].eq(ct[None,:]))&qt[:,None].ne(0)
 m=same_d|same_t
 ii=torch.arange(b,device=qd.device);m[ii,ii]=False
 return m

def masked_ce(query,positive,queue,qd,qt,qdq,qtk,temp):
 cand=positive if queue is None else torch.cat([positive,queue],0)
 cd=qd if qdq is None else torch.cat([qd,qdq],0)
 ct=qt if qtk is None else torch.cat([qt,qtk],0)
 logits=query@cand.T/temp
 logits=logits.masked_fill(family_mask(qd,qt,cd,ct,len(query)),-1e4)
 return F.cross_entropy(logits,torch.arange(len(query),device=query.device))

class ContrastiveDual(nn.Module):
 def __init__(self,text_source):
  super().__init__()
  self.u=AutoModel.from_pretrained(URL_SRC,token=HF_TOKEN)
  self.t=AutoModel.from_pretrained(text_source,token=HF_TOKEN)
  freeze_last(self.u,CONTRASTIVE_LAST_N);freeze_last(self.t,CONTRASTIVE_LAST_N)
  self.up=nn.Sequential(nn.Linear(URL_H,PROJ_DIM),nn.GELU(),nn.Linear(PROJ_DIM,PROJ_DIM))
  self.tp=nn.Sequential(nn.Linear(TEXT_H,PROJ_DIM),nn.GELU(),nn.Linear(PROJ_DIM,PROJ_DIM))
 def enc_u(self,b):return F.normalize(self.up(pooled(self.u(**b),b['attention_mask'])),dim=-1)
 def enc_t(self,b):return F.normalize(self.tp(pooled(self.t(**b),b['attention_mask'])),dim=-1)

def train_contrastive(name,text_source,mode):
 out=CKPT/name;done=out/'COMPLETE.json';resume=out/'resume.pt'
 if done.exists():return out
 # Import complete v6 source when available.
 src=None
 if name=='R2C_CONTRASTIVE_REPAIRED':src=V6_R2C
 if name=='R3C_DAPT_CONTRASTIVE_REPAIRED':src=V6_R3C
 if src is not None:
  shutil.copytree(src,out,dirs_exist_ok=True)
  print({'IMPORTED_CONTRASTIVE':name,'from':str(src)});return out

 out.mkdir(parents=True,exist_ok=True);last=None
 for bs in SSL_BATCH_CANDIDATES:
  try:
   seed_all(MASTER_SEED);m=ContrastiveDual(text_source).to(DEVICE)
   pars=[p for p in m.parameters() if p.requires_grad]
   opt=torch.optim.AdamW(pars,lr=CONTRASTIVE_LR,weight_decay=.01)
   scaler=torch.cuda.amp.GradScaler(enabled=AMP)
   qu=qt=qdq=qtk=None;skip=0;losses=[]
   if resume.exists():
    try:
     ck=torch.load(resume,map_location=DEVICE)
     if int(ck.get('batch_size',-1))==bs and ck.get('mode')==mode:
      m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
      qu=None if ck.get('qu') is None else ck['qu'].to(DEVICE)
      qt=None if ck.get('qt') is None else ck['qt'].to(DEVICE)
      qdq=None if ck.get('qdq') is None else ck['qdq'].to(DEVICE)
      qtk=None if ck.get('qtk') is None else ck['qtk'].to(DEVICE)
      skip=int(ck.get('next_batch',0));losses=list(ck.get('losses',[]))
      print({name+'_RESUME':skip})
    except Exception as e:print({'SSL_RESUME_IGNORED':repr(e)})
   ds=SSLIter(MASTER_SEED)
   def coll(rows):
    u=url_tok([r['url'] for r in rows],padding=True,truncation=True,max_length=URL_MAX_LEN,return_tensors='pt')
    t=text_tok([r['text'] for r in rows],padding=True,truncation=True,max_length=TEXT_MAX_LEN,return_tensors='pt')
    d=torch.tensor([r['domain_h'] for r in rows],dtype=torch.long)
    h=torch.tensor([r['template_h'] for r in rows],dtype=torch.long)
    return u,t,d,h
   dl=DataLoader(ds,batch_size=bs,collate_fn=coll,num_workers=0)
   m.train()
   for j,(u,t,dh,th) in enumerate(dl):
    if j<skip:continue
    u={k:v.to(DEVICE) for k,v in u.items()};t={k:v.to(DEVICE) for k,v in t.items()}
    dh=dh.to(DEVICE);th=th.to(DEVICE)
    u1=token_view(u,url_tok,URL_MASK_P);u2=token_view(u,url_tok,URL_MASK_P)
    t1=token_view(t,text_tok,TEXT_MASK_P);t2=token_view(t,text_tok,TEXT_MASK_P)
    opt.zero_grad(set_to_none=True)
    with amp_ctx():
     zu1=m.enc_u(u1);zu2=m.enc_u(u2);zt1=m.enc_t(t1);zt2=m.enc_t(t2)
     lu=(masked_ce(zu1,zu2,qu,dh,th,qdq,qtk,CONTRASTIVE_TEMP)+masked_ce(zu2,zu1,qu,dh,th,qdq,qtk,CONTRASTIVE_TEMP))/2
     lt=(masked_ce(zt1,zt2,qt,dh,th,qdq,qtk,CONTRASTIVE_TEMP)+masked_ce(zt2,zt1,qt,dh,th,qdq,qtk,CONTRASTIVE_TEMP))/2
     if mode=='hybrid':
      lx=(masked_ce(zu1,zt1,qt,dh,th,qdq,qtk,CONTRASTIVE_TEMP)+masked_ce(zt1,zu1,qu,dh,th,qdq,qtk,CONTRASTIVE_TEMP))/2
      loss=(HYBRID_SAME_WEIGHT/2)*lu+(HYBRID_SAME_WEIGHT/2)*lt+HYBRID_CROSS_WEIGHT*lx
     elif mode=='same_only':
      loss=.5*lu+.5*lt
     else:raise KeyError(mode)
    if not torch.isfinite(loss):raise RuntimeError(f'nonfinite SSL loss {name}')
    scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(pars,1.)
    scaler.step(opt);scaler.update()
    qu=(zu2.detach() if qu is None else torch.cat([qu,zu2.detach()],0))[-CONTRASTIVE_QUEUE:]
    qt=(zt2.detach() if qt is None else torch.cat([qt,zt2.detach()],0))[-CONTRASTIVE_QUEUE:]
    qdq=(dh.detach() if qdq is None else torch.cat([qdq,dh.detach()],0))[-CONTRASTIVE_QUEUE:]
    qtk=(th.detach() if qtk is None else torch.cat([qtk,th.detach()],0))[-CONTRASTIVE_QUEUE:]
    losses.append(float(loss.detach().cpu()));nxt=j+1
    if nxt%SSL_RESUME_EVERY==0:
     atomic_torch(resume,{'model':m.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),
      'qu':qu.cpu(),'qt':qt.cpu(),'qdq':qdq.cpu(),'qtk':qtk.cpu(),'next_batch':nxt,'batch_size':bs,'mode':mode,'losses':losses[-500:]})
     print({name+'_batch':nxt,'loss100':float(np.mean(losses[-100:]))})
   (out/'url').mkdir(exist_ok=True);(out/'text').mkdir(exist_ok=True)
   m.u.save_pretrained(out/'url');m.t.save_pretrained(out/'text')
   url_tok.save_pretrained(out/'url');text_tok.save_pretrained(out/'text')
   atomic_json(done,{'status':'COMPLETE','mode':mode,'rows':SSL_ROWS,'batch_size':bs,'mean_loss':float(np.mean(losses))})
   resume.unlink(missing_ok=True);del m,dl,ds,qu,qt,qdq,qtk;gc.collect();torch.cuda.empty_cache();last=None;break
  except torch.cuda.OutOfMemoryError as e:
   last=e;gc.collect();torch.cuda.empty_cache();print({'SSL_OOM':name,'batch':bs})
 if last is not None:raise last
 return out

R2C_DIR=train_contrastive('R2C_CONTRASTIVE_REPAIRED',TEXT_SRC,'hybrid')
R3C_DIR=train_contrastive('R3C_DAPT_CONTRASTIVE_REPAIRED',str(DAPT_DIR),'hybrid')
CSAME_DIR=train_contrastive('CSAME_R1_CONTRASTIVE',str(DAPT_DIR),'same_only')

REP_SRC={
 'R0':(URL_SRC,TEXT_SRC),
 'R1':(URL_SRC,str(DAPT_DIR)),
 'R2C':(str(R2C_DIR/'url'),str(R2C_DIR/'text')),
 'R3C':(str(R3C_DIR/'url'),str(R3C_DIR/'text')),
 'CSAME':(str(CSAME_DIR/'url'),str(CSAME_DIR/'text'))
}
atomic_json(AUDIT/'SSL_REPRESENTATION_SOURCES.json',{k:list(v) for k,v in REP_SRC.items()})
print('SSL representation sources ready')


# CELL 5
# 04 — FF1: 20k representation utility, Linear + MLP, 5 downstream seeds, FINAL/OOD
@torch.no_grad()
def embed_df(rep,df,out_path,batch=64):
 out=Path(out_path);done=Path(str(out)+'.complete.json');prog=Path(str(out)+'.progress.json')
 shape=(len(df),URL_H+TEXT_H)
 if out.exists() and done.exists():
  a=np.load(out,mmap_mode='r')
  if a.shape==shape:return a
 us,ts=REP_SRC[rep]
 u=AutoModel.from_pretrained(us,token=HF_TOKEN).to(DEVICE).eval()
 t=AutoModel.from_pretrained(ts,token=HF_TOKEN).to(DEVICE).eval()
 start=0
 if out.exists() and prog.exists():
  try:start=int(json.loads(prog.read_text()).get('next_row',0))
  except:start=0
 if start==0:
  out.unlink(missing_ok=True);done.unlink(missing_ok=True)
  a=np.lib.format.open_memmap(out,mode='w+',dtype=np.float16,shape=shape);a.flush();del a
 a=np.load(out,mmap_mode='r+')
 for st in range(start,len(df),5000):
  en=min(st+5000,len(df));x=df.iloc[st:en];buf=[]
  for q0 in range(0,len(x),batch):
   q=x.iloc[q0:q0+batch]
   ub=url_tok(q.url.fillna('').astype(str).tolist(),padding=True,truncation=True,max_length=URL_MAX_LEN,return_tensors='pt')
   tb=text_tok(q.text.fillna('').astype(str).tolist(),padding=True,truncation=True,max_length=TEXT_MAX_LEN,return_tensors='pt')
   ub={k:v.to(DEVICE) for k,v in ub.items()};tb={k:v.to(DEVICE) for k,v in tb.items()}
   with amp_ctx():
    eu=pooled(u(**ub),ub['attention_mask']);et=pooled(t(**tb),tb['attention_mask'])
   buf.append(torch.cat([eu,et],1).float().cpu().numpy())
  a[st:en]=np.concatenate(buf).astype(np.float16);a.flush();atomic_json(prog,{'next_row':en})
  print({'EMBED':rep,'dataset':out.stem,'rows':en,'total':len(df)})
 del a,u,t;gc.collect();torch.cuda.empty_cache();atomic_json(done,{'status':'COMPLETE','shape':shape});prog.unlink(missing_ok=True)
 return np.load(out,mmap_mode='r')

class MLPProbe(nn.Module):
 def __init__(self,d):
  super().__init__();self.net=nn.Sequential(nn.Linear(d,256),nn.GELU(),nn.Dropout(.15),nn.Linear(256,1))
 def forward(self,x):return self.net(x).squeeze(-1)

def fit_mlp(X,y,seed):
 seed_all(seed);m=MLPProbe(X.shape[1]).to(DEVICE)
 opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-3)
 xx=torch.tensor(np.asarray(X,dtype=np.float32));yy=torch.tensor(y,dtype=torch.float32)
 dl=DataLoader(TensorDataset(xx,yy),batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(seed))
 for _ in range(12):
  m.train()
  for a,b in dl:
   a=a.to(DEVICE);b=b.to(DEVICE);opt.zero_grad(set_to_none=True)
   l=F.binary_cross_entropy_with_logits(m(a),b);l.backward();opt.step()
 return m.eval()

@torch.no_grad()
def mlp_logits(m,X):
 out=[]
 for st in range(0,len(X),4096):
  out.append(m(torch.tensor(np.asarray(X[st:st+4096],dtype=np.float32),device=DEVICE)).float().cpu().numpy())
 return np.concatenate(out)

def train_probes(rep,Xtrain,ytrain):
 d=CKPT/'PROBES_20K'/rep;d.mkdir(parents=True,exist_ok=True);models=[]
 for seed in DOWNSTREAM_SEEDS:
  lp=d/f'LINEAR_seed{seed}.npz'
  if lp.exists():
   z=np.load(lp);coef=z['coef'];intercept=z['intercept']
   lin=('LINEAR',seed,coef,intercept)
  else:
   clf=LogisticRegression(C=1,max_iter=2000,solver='liblinear',class_weight='balanced',random_state=seed).fit(Xtrain,ytrain)
   np.savez(lp,coef=clf.coef_.astype(np.float32),intercept=clf.intercept_.astype(np.float32))
   lin=('LINEAR',seed,clf.coef_.astype(np.float32),clf.intercept_.astype(np.float32))
  mp=d/f'MLP_seed{seed}.pt'
  m=MLPProbe(Xtrain.shape[1]).to(DEVICE)
  if mp.exists():m.load_state_dict(torch.load(mp,map_location=DEVICE))
  else:
   del m;m=fit_mlp(Xtrain,ytrain,seed);atomic_torch(mp,m.state_dict())
  models.append(lin);models.append(('MLP',seed,mp,None))
 return models

def linear_logits(X,coef,intercept):
 return (np.asarray(X,dtype=np.float32)@coef.T).reshape(-1)+float(intercept.reshape(-1)[0])

# Score all trained probes while streaming representation embeddings; store raw logits only.
@torch.no_grad()
def score_probe_dataset(rep,df,name,models,batch=64):
 out=SCORES/'FF1_20K'/rep;out.mkdir(parents=True,exist_ok=True)
 done=out/f'{name}.complete.json'
 paths={(typ,seed):out/f'{name}_{typ}_seed{seed}.npy' for typ,seed,_,_ in models}
 if done.exists() and all(p.exists() for p in paths.values()):return paths
 us,ts=REP_SRC[rep];u=AutoModel.from_pretrained(us,token=HF_TOKEN).to(DEVICE).eval();t=AutoModel.from_pretrained(ts,token=HF_TOKEN).to(DEVICE).eval()
 # Load MLPs once.
 mlps={}
 for typ,seed,a,b in models:
  if typ=='MLP':
   m=MLPProbe(URL_H+TEXT_H).to(DEVICE);m.load_state_dict(torch.load(a,map_location=DEVICE));m.eval();mlps[seed]=m
 arrays={}
 for key,p in paths.items():
  p.unlink(missing_ok=True);arrays[key]=np.lib.format.open_memmap(p,mode='w+',dtype=np.float32,shape=(len(df),))
 for st in range(0,len(df),5000):
  en=min(st+5000,len(df));x=df.iloc[st:en];Z=[]
  for q0 in range(0,len(x),batch):
   q=x.iloc[q0:q0+batch]
   ub=url_tok(q.url.fillna('').astype(str).tolist(),padding=True,truncation=True,max_length=URL_MAX_LEN,return_tensors='pt')
   tb=text_tok(q.text.fillna('').astype(str).tolist(),padding=True,truncation=True,max_length=TEXT_MAX_LEN,return_tensors='pt')
   ub={k:v.to(DEVICE) for k,v in ub.items()};tb={k:v.to(DEVICE) for k,v in tb.items()}
   with amp_ctx():eu=pooled(u(**ub),ub['attention_mask']);et=pooled(t(**tb),tb['attention_mask'])
   Z.append(torch.cat([eu,et],1).float())
  Zt=torch.cat(Z,0);Zn=Zt.cpu().numpy()
  for typ,seed,a,b in models:
   if typ=='LINEAR':arrays[(typ,seed)][st:en]=linear_logits(Zn,a,b)
   else:arrays[(typ,seed)][st:en]=mlps[seed](Zt).float().cpu().numpy()
  for arr in arrays.values():arr.flush()
  print({'FF1_SCORE':rep,'dataset':name,'rows':en,'total':len(df)})
 for arr in arrays.values():arr.flush()
 del arrays,u,t,mlps;gc.collect();torch.cuda.empty_cache()
 atomic_json(done,{'status':'COMPLETE','rows':len(df),'raw_logits':True})
 return paths

FF1_ROWS=[]
for rep in ['R0','R1','R2C','R3C','CSAME']:
 Xtr=np.asarray(embed_df(rep,SCI20,EMB/f'{rep}_SCI20.npy'),dtype=np.float32)
 Xdev=np.asarray(embed_df(rep,DEV,EMB/f'{rep}_DEV.npy'),dtype=np.float32)
 models=train_probes(rep,Xtr,SCI20.y.to_numpy(int))
 # DEV results.
 for typ,seed,a,b in models:
  if typ=='LINEAR':
   lc=linear_logits(Xdev[REP_CAL],a,b);le=linear_logits(Xdev[REP_EVAL],a,b)
  else:
   m=MLPProbe(Xdev.shape[1]).to(DEVICE);m.load_state_dict(torch.load(a,map_location=DEVICE));m.eval()
   lc=mlp_logits(m,Xdev[REP_CAL]);le=mlp_logits(m,Xdev[REP_EVAL]);del m;torch.cuda.empty_cache()
  yc=DEVY[REP_CAL];ye=DEVY[REP_EVAL]
  for f in TARGET_FPRS:
   th=thr_fpr(lc[yc==0],f)
   FF1_ROWS.append({'rep':rep,'probe':typ,'seed':seed,'dataset':'DEV_EVAL','scenario':'DEV_EVAL','target_fpr':f,'threshold':th,**op(ye,le,th)})
  FF1_ROWS.append({'rep':rep,'probe':typ,'seed':seed,'dataset':'DEV_EVAL','scenario':'DEV_EVAL','target_fpr':np.nan,'threshold':np.nan,**curves(ye,le)})
 # CAL/FINAL streaming.
 calpaths=score_probe_dataset(rep,CAL,'CAL',models)
 finpaths=score_probe_dataset(rep,FINAL,'FINAL',models)
 for typ,seed,_,_ in models:
  cs=np.load(calpaths[(typ,seed)],mmap_mode='r');fs=np.load(finpaths[(typ,seed)],mmap_mode='r')
  for f in TARGET_FPRS:
   th=thr_fpr(np.asarray(cs),f)
   for sn,mask in FINAL_MASKS.items():
    FF1_ROWS.append({'rep':rep,'probe':typ,'seed':seed,'dataset':'FINAL','scenario':sn,'target_fpr':f,'threshold':th,**op(FINALY[mask],np.asarray(fs)[mask],th)})
  for sn,mask in FINAL_MASKS.items():
   FF1_ROWS.append({'rep':rep,'probe':typ,'seed':seed,'dataset':'FINAL','scenario':sn,'target_fpr':np.nan,'threshold':np.nan,**curves(FINALY[mask],np.asarray(fs)[mask])})
 del Xtr,Xdev;gc.collect()

FF1=pd.DataFrame(FF1_ROWS)
FF1.to_csv(RESULTS/'RQ_FF1_20K_REPRESENTATION_ALL_SEEDS.csv',index=False)
AGG=FF1.groupby(['rep','probe','dataset','scenario','target_fpr'],dropna=False).agg(
 tpr_mean=('tpr','mean'),tpr_std=('tpr','std'),fpr_mean=('fpr','mean'),fpr_std=('fpr','std'),
 AP_mean=('AP','mean'),AP_std=('AP','std'),P_at_R90_mean=('P_at_R90','mean'),P_at_R90_std=('P_at_R90','std')
).reset_index()
AGG.to_csv(RESULTS/'RQ_FF1_20K_REPRESENTATION_AGG.csv',index=False)
display(AGG[(AGG.scenario=='OFFICIAL_TEST')&(np.isclose(AGG.target_fpr.fillna(-1),PRIMARY_FPR))])


# ------------------------------------------------------------------
# 20k SSL representation champion: DEVELOPMENT only.
# Mean across both downstream mechanisms and all five seeds.
# ------------------------------------------------------------------
DEV_PRIMARY=AGG[
 (AGG.dataset=='DEV_EVAL')&
 (AGG.scenario=='DEV_EVAL')&
 np.isclose(AGG.target_fpr.fillna(-1),PRIMARY_FPR)
].copy()

REP_DEV_SCORE=DEV_PRIMARY.groupby('rep',as_index=False).agg(
 tpr_mean_across_probes=('tpr_mean','mean'),
 fpr_mean_across_probes=('fpr_mean','mean')
)

DEV_RANK=AGG[
 (AGG.dataset=='DEV_EVAL')&
 (AGG.scenario=='DEV_EVAL')&
 (AGG.target_fpr.isna())
].groupby('rep',as_index=False).agg(
 P_at_R90_mean_across_probes=('P_at_R90_mean','mean'),
 AP_mean_across_probes=('AP_mean','mean')
)

REP_DEV_SCORE=REP_DEV_SCORE.merge(DEV_RANK,on='rep',validate='one_to_one')
REP_DEV_SCORE=REP_DEV_SCORE.sort_values(
 ['tpr_mean_across_probes','P_at_R90_mean_across_probes','AP_mean_across_probes'],
 ascending=False
).reset_index(drop=True)

SSL20_CHAMP=str(REP_DEV_SCORE.iloc[0].rep)
REP_DEV_SCORE.to_csv(RESULTS/'SSL20_REPRESENTATION_SELECTION_DEV.csv',index=False)
atomic_json(AUDIT/'SSL20_REPRESENTATION_FREEZE.json',{
 'status':'FROZEN_ON_DEV_ONLY',
 'selected_representation':SSL20_CHAMP,
 'label_condition':20000,
 'selection_population':'mean across LINEAR+MLP and five downstream seeds',
 'rule':'TPR@0.5% FPR > P@R90 > AP',
 'held_constant_for_DOM_20K_200K':True,
 'final_used':False
})
print({'SSL20_CHAMP':SSL20_CHAMP})
display(REP_DEV_SCORE)


# CELL 6
# 05 — DOM encoder + fair Deep DUAL/TRI model
DOM_VOCAB=json.loads(DOM_VOCAB_PATH.read_text())
PAD,UNK,MASK=0,1,2

def bucket(v):
 try:return min(max(int(v),0),31)
 except:return 0
def _rv(r,k):
 if isinstance(r,dict):return r.get(k)
 try:return r[k]
 except:return getattr(r,k,None)

def graph_row(r):
 def arr(k):
  x=_rv(r,k);return x if isinstance(x,(list,np.ndarray)) else []
 tags=arr('dom_tag');par=arr('dom_parent_idx');dep=arr('dom_depth');att=arr('dom_attr_count');chi=arr('dom_child_count')
 n=min(len(tags),DOM_MAX_NODES)
 tag=np.array([DOM_VOCAB.get(str(x),UNK) for x in tags[:n]],np.int64)
 depth=np.array([bucket(x) for x in dep[:n]],np.int64);attr=np.array([bucket(x) for x in att[:n]],np.int64);child=np.array([bucket(x) for x in chi[:n]],np.int64)
 edges=[]
 for ii,pp in enumerate(par[:n]):
  try:pp=int(pp)
  except:continue
  if ii>0 and 0<=pp<n:edges.extend([(pp,ii),(ii,pp)])
 return tag,depth,attr,child,np.asarray(edges,np.int64)

def graph_batch(rows):
 T=[];D=[];A=[];C=[];E=[];B=[];off=0
 for bi,r in enumerate(rows):
  t,d,a,c,e=graph_row(r);n=len(t);T.append(t);D.append(d);A.append(a);C.append(c);B.append(np.full(n,bi,np.int64))
  if len(e):E.append(e+off)
  off+=n
 def cat(xs):return torch.tensor(np.concatenate(xs) if xs else np.array([],np.int64),dtype=torch.long,device=DEVICE)
 edge=torch.tensor(np.concatenate(E,0).T if E else np.zeros((2,0),np.int64),dtype=torch.long,device=DEVICE)
 return {'tag':cat(T),'depth':cat(D),'attr':cat(A),'child':cat(C),'edge':edge,'batch':cat(B),'n_graphs':len(rows)}

class GCNLayer(nn.Module):
 def __init__(self,d):super().__init__();self.s=nn.Linear(d,d);self.n=nn.Linear(d,d);self.norm=nn.LayerNorm(d)
 def forward(self,x,e):
  if e.numel()==0:return self.norm(x+F.gelu(self.s(x)))
  src,dst=e;agg=torch.zeros_like(x);deg=torch.zeros((len(x),1),device=x.device,dtype=x.dtype)
  agg.index_add_(0,dst,x[src]);deg.index_add_(0,dst,torch.ones((len(dst),1),device=x.device,dtype=x.dtype))
  return self.norm(x+F.gelu(self.s(x)+self.n(agg/deg.clamp_min(1))))

class DOMEncoder(nn.Module):
 def __init__(self):
  super().__init__();self.tag=nn.Embedding(len(DOM_VOCAB),96,padding_idx=PAD);self.depth=nn.Embedding(32,16);self.attr=nn.Embedding(32,8);self.child=nn.Embedding(32,8)
  self.inp=nn.Linear(128,DOM_DIM);self.layers=nn.ModuleList([GCNLayer(DOM_DIM) for _ in range(DOM_LAYERS)]);self.att=nn.Linear(DOM_DIM,1)
 def nodes(self,g):
  x=self.inp(torch.cat([self.tag(g['tag']),self.depth(g['depth']),self.attr(g['attr']),self.child(g['child'])],1))
  for l in self.layers:x=l(x,g['edge'])
  return x
 def forward(self,g):
  x=self.nodes(g);n=g['n_graphs'];num=torch.zeros((n,DOM_DIM),device=x.device);den=torch.zeros((n,1),device=x.device)
  w=torch.exp(torch.clamp(self.att(x).squeeze(-1),-10,10));num.index_add_(0,g['batch'],x*w[:,None]);den.index_add_(0,g['batch'],w[:,None])
  return num/den.clamp_min(1e-8)

class DeepFusion(nn.Module):
 def __init__(self,rep,use_dom):
  super().__init__();self.rep=rep;self.use_dom=use_dom
  us,ts=REP_SRC[rep]
  self.u=AutoModel.from_pretrained(us,token=HF_TOKEN);self.t=AutoModel.from_pretrained(ts,token=HF_TOKEN)
  freeze_last(self.u,DEEP_LAST_N);freeze_last(self.t,DEEP_LAST_N)
  self.ua=nn.Linear(URL_H,PROJ_DIM);self.ta=nn.Linear(TEXT_H,PROJ_DIM)
  self.dom=DOMEncoder() if use_dom else None
  if use_dom:self.dom.load_state_dict(torch.load(DOM_ENCODER_PATH,map_location='cpu'))
  self.da=nn.Linear(DOM_DIM,PROJ_DIM) if use_dom else None
  self.gate=nn.Sequential(nn.Linear(PROJ_DIM,64),nn.GELU(),nn.Linear(64,1))
  nm=3 if use_dom else 2
  self.head=nn.Sequential(nn.Linear(PROJ_DIM*(nm+1),512),nn.GELU(),nn.Dropout(.2),nn.Linear(512,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,1))
  self.uaux=nn.Linear(PROJ_DIM,1);self.taux=nn.Linear(PROJ_DIM,1);self.daux=nn.Linear(PROJ_DIM,1) if use_dom else None

 def encode(self,u,t,g=None):
  zu=F.normalize(self.ua(pooled(self.u(**u),u['attention_mask'])),dim=-1)
  zt=F.normalize(self.ta(pooled(self.t(**t),t['attention_mask'])),dim=-1)
  zs=[zu,zt]
  if self.use_dom:zs.append(F.normalize(self.da(self.dom(g)),dim=-1))
  return zs

 def forward(self,u,t,g=None,fusion_mode='gated'):
  zs=self.encode(u,t,g);st=torch.stack(zs,1)
  gw=torch.softmax(self.gate(st).squeeze(-1),1)
  if fusion_mode=='gated':weighted=(st*gw[:,:,None]).sum(1)
  elif fusion_mode=='equal':weighted=st.mean(1)
  else:raise KeyError(fusion_mode)
  main=self.head(torch.cat(zs+[weighted],1)).squeeze(-1)
  aux={'url':self.uaux(zs[0]).squeeze(-1),'text':self.taux(zs[1]).squeeze(-1)}
  if self.use_dom:aux['dom']=self.daux(zs[2]).squeeze(-1)
  return main,aux,gw

def _wb(logits,y,w=None):
 raw=F.binary_cross_entropy_with_logits(logits,y,reduction='none')
 if w is None:return raw.mean()
 w=w.to(raw.dtype);return (raw*w).sum()/w.sum().clamp_min(1e-12)

def deep_loss(main,aux,y,w=None):
 return _wb(main,y,w)+AUX_TOTAL_WEIGHT*torch.stack([_wb(a,y,w) for a in aux.values()]).mean()


# CELL 7
# 06 — Exact 20k / 200k supervised stream and matched Deep training
DOM_COLS=['dom_tag','dom_parent_idx','dom_depth','dom_attr_count','dom_child_count']

class DeepTrainIter(IterableDataset):
 def __init__(self,budget,seed,use_dom):
  self.budget=budget;self.seed=seed;self.use_dom=use_dom;self.files=sorted(ROLE_DIR['SSL'].glob('*.parquet'))
  self.allowed=SCI20_SHA if budget==SCI_LABEL_ROWS else None
  if budget not in [SCI_LABEL_ROWS,HIGH_LABEL_ROWS]:raise ValueError(budget)
 def __iter__(self):
  rng=np.random.default_rng(self.seed);files=list(self.files);rng.shuffle(files);n=0
  for p in files:
   cols=['sha256','url','text']+(DOM_COLS if self.use_dom else [])
   q=pd.read_parquet(p,columns=cols);q['sha256']=q.sha256.astype(str).str.lower()
   if self.allowed is not None:q=q[q.sha256.isin(self.allowed)]
   ii=np.arange(len(q));rng.shuffle(ii)
   for j in ii:
    r=q.iloc[j];sha=str(r.sha256)
    y=SSL_LABEL_MAP.get(sha)
    if y is None:raise RuntimeError(f'missing SSL label {sha}')
    yield r,int(y);n+=1
    if n>=self.budget:return

def deep_collate(rows,use_dom):
 rs=[x[0] for x in rows];yy=torch.tensor([x[1] for x in rows],dtype=torch.float32)
 u=url_tok([str(r.url or '') for r in rs],padding=True,truncation=True,max_length=URL_MAX_LEN,return_tensors='pt')
 t=text_tok([str(r.text or '') for r in rs],padding=True,truncation=True,max_length=TEXT_MAX_LEN,return_tensors='pt')
 g=graph_batch(rs) if use_dom else None
 return u,t,g,yy

def train_deep(tag,rep,budget,use_dom):
 out=CKPT/'DEEP'/tag;done=out/'COMPLETE.json';resume=out/'resume.pt';state=out/'model_state.pt'
 if done.exists() and state.exists():
  print({'REUSE_DEEP':tag});return out
 out.mkdir(parents=True,exist_ok=True)
 cands=TRI_BATCH_CANDIDATES if use_dom else DUAL_BATCH_CANDIDATES;last=None
 for bs in cands:
  try:
   seed_all(ENGINEERING_SEED);m=DeepFusion(rep,use_dom).to(DEVICE)
   enc=[];head=[]
   for n,p in m.named_parameters():
    if not p.requires_grad:continue
    (enc if n.startswith('u.') or n.startswith('t.') else head).append(p)
   opt=torch.optim.AdamW([{'params':enc,'lr':ENCODER_LR,'weight_decay':WEIGHT_DECAY},{'params':head,'lr':HEAD_LR,'weight_decay':WEIGHT_DECAY}])
   steps=math.ceil(budget/bs)*DEEP_EPOCHS
   sched=get_linear_schedule_with_warmup(opt,int(WARMUP_RATIO*steps),steps)
   scaler=torch.cuda.amp.GradScaler(enabled=AMP);skip=0;losses=[]
   if resume.exists():
    try:
     ck=torch.load(resume,map_location=DEVICE)
     if int(ck.get('batch_size',-1))==bs and ck.get('rep')==rep and int(ck.get('budget',-1))==budget and bool(ck.get('use_dom'))==use_dom:
      m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler']);scaler.load_state_dict(ck['scaler'])
      skip=int(ck.get('next_batch',0));losses=list(ck.get('losses',[]));print({tag+'_RESUME':skip})
    except Exception as e:print({'DEEP_RESUME_IGNORED':tag,'err':repr(e)})
   ds=DeepTrainIter(budget,ENGINEERING_SEED,use_dom)
   coll=lambda rows:deep_collate(rows,use_dom)
   dl=DataLoader(ds,batch_size=bs,collate_fn=coll,num_workers=0)
   m.train()
   for j,(u,t,g,y) in enumerate(dl):
    if j<skip:continue
    u={k:v.to(DEVICE) for k,v in u.items()};t={k:v.to(DEVICE) for k,v in t.items()};y=y.to(DEVICE)
    opt.zero_grad(set_to_none=True)
    with amp_ctx():main,aux,_=m(u,t,g);loss=deep_loss(main,aux,y)
    if not torch.isfinite(loss):raise RuntimeError(f'nonfinite deep loss {tag}')
    scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
    scaler.step(opt);scaler.update();sched.step();losses.append(float(loss.detach().cpu()));nxt=j+1
    if nxt%DEEP_RESUME_EVERY==0:
     atomic_torch(resume,{'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'scaler':scaler.state_dict(),
                         'next_batch':nxt,'batch_size':bs,'rep':rep,'budget':budget,'use_dom':use_dom,'losses':losses[-500:]})
     print({tag+'_batch':nxt,'of':steps,'loss100':float(np.mean(losses[-100:]))})
   atomic_torch(state,m.state_dict())
   atomic_json(done,{'status':'COMPLETE','tag':tag,'rep':rep,'budget':budget,'use_dom':use_dom,'batch_size':bs,
                     'mean_loss':float(np.mean(losses)),'trainable_params':int(sum(p.numel() for p in m.parameters() if p.requires_grad))})
   resume.unlink(missing_ok=True);del m,dl,ds;gc.collect();torch.cuda.empty_cache();last=None;break
  except torch.cuda.OutOfMemoryError as e:
   last=e;gc.collect();torch.cuda.empty_cache();print({'DEEP_OOM':tag,'batch':bs})
 if last is not None:raise last
 return out

# FF2: exact matched label-scaling comparison.
D_R0_20=train_deep('DUAL_R0_20K','R0',SCI_LABEL_ROWS,False)
D_R1_20=train_deep('DUAL_R1_20K','R1',SCI_LABEL_ROWS,False)
D_R0_200=train_deep('DUAL_R0_200K','R0',HIGH_LABEL_ROWS,False)
D_R1_200=train_deep('DUAL_R1_200K','R1',HIGH_LABEL_ROWS,False)
print('Matched R0/R1 label-scaling training complete')

# FF4 holds the 20k SSL champion constant at both 20k and 200k.
if SSL20_CHAMP=='R0':
 CHAMP_DUAL_20=D_R0_20;CHAMP_DUAL_200=D_R0_200
elif SSL20_CHAMP=='R1':
 CHAMP_DUAL_20=D_R1_20;CHAMP_DUAL_200=D_R1_200
else:
 CHAMP_DUAL_20=train_deep(f'DUAL_{SSL20_CHAMP}_20K',SSL20_CHAMP,SCI_LABEL_ROWS,False)
 CHAMP_DUAL_200=train_deep(f'DUAL_{SSL20_CHAMP}_200K',SSL20_CHAMP,HIGH_LABEL_ROWS,False)

print({'FF4_FIXED_REPRESENTATION':SSL20_CHAMP,
       'DUAL20':str(CHAMP_DUAL_20),'DUAL200':str(CHAMP_DUAL_200)})



# CELL 8
# 07 — DEV raw-logit scoring; select 200k representation on DEV, then retest DOM at 200k
def deep_eval_df(model,df,ids,use_dom,batch,return_equal=True):
 ids=np.asarray(ids);out={'gated':[],'equal':[],'url':[],'text':[],'dom':[],'gate':[]}
 model.eval()
 with torch.no_grad():
  for st in range(0,len(ids),batch):
   ii=ids[st:st+batch];x=df.iloc[ii]
   u=url_tok(x.url.fillna('').astype(str).tolist(),padding=True,truncation=True,max_length=URL_MAX_LEN,return_tensors='pt')
   t=text_tok(x.text.fillna('').astype(str).tolist(),padding=True,truncation=True,max_length=TEXT_MAX_LEN,return_tensors='pt')
   u={k:v.to(DEVICE) for k,v in u.items()};t={k:v.to(DEVICE) for k,v in t.items()}
   g=graph_batch([x.iloc[j] for j in range(len(x))]) if use_dom else None
   # IMPORTANT: convert logits to FP32 before any sigmoid; we never need sigmoid for ranking/thresholding.
   with amp_ctx():main,aux,gw=model(u,t,g,'gated')
   out['gated'].append(main.float().cpu().numpy());out['url'].append(aux['url'].float().cpu().numpy());out['text'].append(aux['text'].float().cpu().numpy());out['gate'].append(gw.float().cpu().numpy())
   if use_dom:out['dom'].append(aux['dom'].float().cpu().numpy())
   if return_equal:
    with amp_ctx():eq,_,_=model(u,t,g,'equal')
    out['equal'].append(eq.float().cpu().numpy())
 for k in ['gated','equal','url','text']:
  out[k]=np.concatenate(out[k]) if out[k] else None
 out['dom']=np.concatenate(out['dom']) if out['dom'] else None
 out['gate']=np.concatenate(out['gate']) if out['gate'] else None
 return out

def load_deep(dir,rep,use_dom):
 meta=json.loads((dir/'COMPLETE.json').read_text());m=DeepFusion(rep,use_dom).to(DEVICE);m.load_state_dict(torch.load(dir/'model_state.pt',map_location=DEVICE));return m,int(meta['batch_size'])

def dev_summary(tag,dir,rep,use_dom):
 m,bs=load_deep(dir,rep,use_dom)
 a=deep_eval_df(m,DEV,ENG_TUNE,use_dom,bs);b=deep_eval_df(m,DEV,ENG_META,use_dom,bs)
 yt=DEVY[ENG_TUNE];ym=DEVY[ENG_META];th=thr_fpr(a['gated'][yt==0],PRIMARY_FPR)
 row={'tag':tag,'rep':rep,'budget':json.loads((dir/'COMPLETE.json').read_text())['budget'],'use_dom':use_dom,'threshold':th,
      **op(ym,b['gated'],th),**curves(ym,b['gated']),
      'gate_url_mean':float(b['gate'][:,0].mean()),'gate_text_mean':float(b['gate'][:,1].mean())}
 if use_dom:row['gate_dom_mean']=float(b['gate'][:,2].mean())
 del m;gc.collect();torch.cuda.empty_cache()
 return row

pre=[
 dev_summary('DUAL_R0_20K',D_R0_20,'R0',False),
 dev_summary('DUAL_R1_20K',D_R1_20,'R1',False),
 dev_summary('DUAL_R0_200K',D_R0_200,'R0',False),
 dev_summary('DUAL_R1_200K',D_R1_200,'R1',False)
]
PRE=pd.DataFrame(pre)
PRE.to_csv(RESULTS/'RQ_FF2_LABEL_SCALE_DEV.csv',index=False)
display(PRE)

# FF4 matched DOM comparison with representation held constant.
TRI20_DIR=train_deep(f'TRI_{SSL20_CHAMP}_20K',SSL20_CHAMP,SCI_LABEL_ROWS,True)
TRI200_DIR=train_deep(f'TRI_{SSL20_CHAMP}_200K',SSL20_CHAMP,HIGH_LABEL_ROWS,True)

DUAL20_ROW=dev_summary(f'DUAL_{SSL20_CHAMP}_20K',CHAMP_DUAL_20,SSL20_CHAMP,False)
TRI20_ROW=dev_summary(f'TRI_{SSL20_CHAMP}_20K',TRI20_DIR,SSL20_CHAMP,True)
DUAL200_ROW=dev_summary(f'DUAL_{SSL20_CHAMP}_200K',CHAMP_DUAL_200,SSL20_CHAMP,False)
TRI200_ROW=dev_summary(f'TRI_{SSL20_CHAMP}_200K',TRI200_DIR,SSL20_CHAMP,True)

DOM_DEV=pd.DataFrame([DUAL20_ROW,TRI20_ROW,DUAL200_ROW,TRI200_ROW])
DOM_DEV.to_csv(RESULTS/'RQ_FF4_DOM_20K_200K_DEV.csv',index=False)
display(DOM_DEV)

def dom_gate(dualrow,trirow):
 delta_pp=100*(trirow['tpr']-dualrow['tpr'])
 fpr90_good=bool(trirow['FPR_at_TPR90']<=.90*dualrow['FPR_at_TPR90'])
 return {
  'use_dom':bool(delta_pp>=.5 or fpr90_good),
  'delta_tpr_pp':float(delta_pp),
  'tri_fpr90_le_90pct_dual':fpr90_good,
  'dual_tpr':float(dualrow['tpr']),
  'tri_tpr':float(trirow['tpr']),
  'dual_FPR_at_TPR90':float(dualrow['FPR_at_TPR90']),
  'tri_FPR_at_TPR90':float(trirow['FPR_at_TPR90'])
 }

DOM_GATE_20=dom_gate(DUAL20_ROW,TRI20_ROW)
DOM_GATE_200=dom_gate(DUAL200_ROW,TRI200_ROW)

# Final system architecture is selected from the 200k engineering condition only.
USE_DOM=bool(DOM_GATE_200['use_dom'])
SELECTED_ARCH='TRI' if USE_DOM else 'DUAL'
SELECTED_DIR=TRI200_DIR if USE_DOM else CHAMP_DUAL_200
DEEP_REP=SSL20_CHAMP

atomic_json(AUDIT/'DOM_20K_200K_GATE_FREEZE.json',{
 'status':'FROZEN_ON_DEV_ONLY',
 'representation':SSL20_CHAMP,
 'representation_held_constant':True,
 'gate_rule':'+0.5pp TPR@0.5% FPR OR FPR@TPR90 <= 90% of DUAL',
 '20k':DOM_GATE_20,
 '200k':DOM_GATE_200,
 'final_artifact_architecture':SELECTED_ARCH,
 'final_used':False
})
print({'FF4_REP':SSL20_CHAMP,
       'DOM_20K_DELTA_PP':DOM_GATE_20['delta_tpr_pp'],
       'DOM_200K_DELTA_PP':DOM_GATE_200['delta_tpr_pp'],
       'FINAL_ARCH':SELECTED_ARCH})


# CELL 9
# 08 — FF4 branch/fusion ablations at 20k AND 200k; cascade/resources at 200k
def ablation_dev(tag,dir,rep,use_dom):
 m,bs=load_deep(dir,rep,use_dom)
 et=deep_eval_df(m,DEV,ENG_TUNE,use_dom,bs)
 em=deep_eval_df(m,DEV,ENG_META,use_dom,bs)
 yt=DEVY[ENG_TUNE];ym=DEVY[ENG_META]
 rows=[]
 for mode in ['url','text','gated','equal']+(['dom'] if use_dom else []):
  th=thr_fpr(et[mode][yt==0],PRIMARY_FPR)
  rows.append({
   'tag':tag,'rep':rep,'use_dom':use_dom,
   'budget':json.loads((dir/'COMPLETE.json').read_text())['budget'],
   'mode':mode,'threshold':th,
   **op(ym,em[mode],th),**curves(ym,em[mode])
  })
 del m;gc.collect();torch.cuda.empty_cache()
 return rows

ABL_DEV=pd.DataFrame(
 ablation_dev(f'DUAL_{SSL20_CHAMP}_20K',CHAMP_DUAL_20,SSL20_CHAMP,False)+
 ablation_dev(f'TRI_{SSL20_CHAMP}_20K',TRI20_DIR,SSL20_CHAMP,True)+
 ablation_dev(f'DUAL_{SSL20_CHAMP}_200K',CHAMP_DUAL_200,SSL20_CHAMP,False)+
 ablation_dev(f'TRI_{SSL20_CHAMP}_200K',TRI200_DIR,SSL20_CHAMP,True)
)
ABL_DEV.to_csv(RESULTS/'RQ_FF4_FUSION_BRANCH_20K_200K_DEV.csv',index=False)
display(ABL_DEV)

# Final 200k artifact and cascade.
selected_use_dom=SELECTED_ARCH=='TRI'
MSEL,BSSEL=load_deep(SELECTED_DIR,SSL20_CHAMP,selected_use_dom)
ET=deep_eval_df(MSEL,DEV,ENG_TUNE,selected_use_dom,BSSEL)
EM=deep_eval_df(MSEL,DEV,ENG_META,selected_use_dom,BSSEL)
yt=DEVY[ENG_TUNE];ym=DEVY[ENG_META]

full_thr=thr_fpr(ET['gated'][yt==0],PRIMARY_FPR)
url_high=thr_fpr(ET['url'][yt==0],.001)
pos=np.sort(ET['url'][yt==1])
k=max(1,int(math.floor(.01*len(pos))))
url_low=float(np.nextafter(pos[k-1],-np.inf))

full_meta=op(ym,EM['gated'],full_thr)
esc=(EM['url']>url_low)&(EM['url']<url_high)
cas=EM['gated'].copy()
cas[EM['url']>=url_high]=1e9
cas[EM['url']<=url_low]=-1e9
cas_meta=op(ym,cas,full_thr)
loss_pp=100*(full_meta['tpr']-cas_meta['tpr'])

USE_CASCADE=bool(
 loss_pp<=CASCADE_MAX_TPR_LOSS_PP and
 esc.mean()<=1-CASCADE_MIN_FULL_REDUCTION and
 cas_meta['fpr']<=full_meta['fpr']+.001
)

latids=ENG_META[:min(512,len(ENG_META))]
vals=[]
for _ in range(3):
 torch.cuda.synchronize();t0=time.perf_counter()
 _=deep_eval_df(MSEL,DEV,latids,selected_use_dom,BSSEL,False)
 torch.cuda.synchronize();vals.append(time.perf_counter()-t0)
lat_ms=1000*float(np.median(vals))/len(latids)
model_bytes=(SELECTED_DIR/'model_state.pt').stat().st_size

CASCADE={
 'use':USE_CASCADE,
 'url_low':url_low,'url_high':url_high,
 'dev_meta_escalation_rate':float(esc.mean()),
 'dev_meta_full_reduction':float(1-esc.mean()),
 'tpr_loss_pp':float(loss_pp),
 'full_meta':full_meta,'cascade_meta':cas_meta
}
SYSTEM={
 'status':'FROZEN_ON_DEV_ONLY',
 'representation':SSL20_CHAMP,
 'architecture':SELECTED_ARCH,
 'budget':HIGH_LABEL_ROWS,
 'dom_gate_20k':DOM_GATE_20,
 'dom_gate_200k':DOM_GATE_200,
 'cascade':CASCADE,
 'latency_dev_ms_page_full':lat_ms,
 'model_state_bytes':model_bytes,
 'threshold_domain':'raw_fp32_logits',
 'final_used':False
}
atomic_json(AUDIT/'V7_2_SYSTEM_FREEZE.json',SYSTEM)
print(json.dumps(SYSTEM,indent=2))


# CELL 10
# 09 — FP32-logit CAL/FINAL scoring: FF2 label scaling + FF4 DOM/fusion/cascade
def score_deep_cache(tag,dir,rep,use_dom,df,name):
 base=SCORES/'DEEP'/tag;base.mkdir(parents=True,exist_ok=True)
 modes=['gated','equal','url','text']+(['dom'] if use_dom else [])
 paths={k:base/f'{name}_{k}_logit.npy' for k in modes};done=base/f'{name}.complete.json';prog=base/f'{name}.progress.json'
 if done.exists() and all(p.exists() for p in paths.values()):return paths
 m,bs=load_deep(dir,rep,use_dom)
 start=0
 if prog.exists() and all(p.exists() for p in paths.values()):
  try:start=int(json.loads(prog.read_text()).get('next_row',0))
  except:start=0
 if start==0:
  for p in list(paths.values())+[done,prog]:p.unlink(missing_ok=True)
  arrs={k:np.lib.format.open_memmap(p,mode='w+',dtype=np.float32,shape=(len(df),)) for k,p in paths.items()}
 else:arrs={k:np.load(p,mmap_mode='r+') for k,p in paths.items()}
 for st in range(start,len(df),10000):
  en=min(st+10000,len(df))
  if use_dom:
   if name not in ['CAL','FINAL']:raise RuntimeError(f'chunked DOM scoring unsupported dataset {name}')
   x=df.iloc[st:en].copy().reset_index(drop=True)
   ddom=read_role_slice(name,st,en,['sha256']+DOM_COLS)
   ddom['sha256']=ddom.sha256.astype(str).str.lower()
   if not np.array_equal(x.sha256.astype(str).str.lower().to_numpy(),ddom.sha256.to_numpy()):
    raise RuntimeError(f'{name} DOM slice SHA alignment failed at {st}:{en}')
   for c in DOM_COLS:x[c]=ddom[c].tolist()
   ids=np.arange(len(x));z=deep_eval_df(m,x,ids,True,bs)
  else:
   ids=np.arange(st,en);z=deep_eval_df(m,df,ids,False,bs)
  for k in modes:arrs[k][st:en]=z[k].astype(np.float32);arrs[k].flush()
  atomic_json(prog,{'next_row':en});print({'DEEP_SCORE':tag,'dataset':name,'rows':en,'total':len(df)})
 for a in arrs.values():a.flush()
 del arrs,m;gc.collect();torch.cuda.empty_cache();atomic_json(done,{'status':'COMPLETE','raw_fp32_logits':True,'rows':len(df)});prog.unlink(missing_ok=True)
 return paths

DEEP_VARIANTS={
 'DUAL_R0_20K':(D_R0_20,'R0',False),
 'DUAL_R1_20K':(D_R1_20,'R1',False),
 'DUAL_R0_200K':(D_R0_200,'R0',False),
 'DUAL_R1_200K':(D_R1_200,'R1',False),
 f'DUAL_{SSL20_CHAMP}_20K':(CHAMP_DUAL_20,SSL20_CHAMP,False),
 f'TRI_{SSL20_CHAMP}_20K':(TRI20_DIR,SSL20_CHAMP,True),
 f'DUAL_{SSL20_CHAMP}_200K':(CHAMP_DUAL_200,SSL20_CHAMP,False),
 f'TRI_{SSL20_CHAMP}_200K':(TRI200_DIR,SSL20_CHAMP,True)
}

DEEP_ROWS=[]
CACHE={}
for tag,(d,rep,use_dom) in DEEP_VARIANTS.items():
 cp=score_deep_cache(tag,d,rep,use_dom,CAL,'CAL')
 fp=score_deep_cache(tag,d,rep,use_dom,FINAL,'FINAL')
 CACHE[tag]=(cp,fp)
 cs=np.asarray(np.load(cp['gated'],mmap_mode='r'))
 fs=np.asarray(np.load(fp['gated'],mmap_mode='r'))
 for f in TARGET_FPRS:
  th=thr_fpr(cs,f)
  for sn,mask in FINAL_MASKS.items():
   DEEP_ROWS.append({
    'tag':tag,'rep':rep,'use_dom':use_dom,
    'budget':json.loads((d/'COMPLETE.json').read_text())['budget'],
    'mode':'gated','scenario':sn,'target_fpr':f,'threshold':th,
    **op(FINALY[mask],fs[mask],th)
   })
 for sn,mask in FINAL_MASKS.items():
  DEEP_ROWS.append({
   'tag':tag,'rep':rep,'use_dom':use_dom,
   'budget':json.loads((d/'COMPLETE.json').read_text())['budget'],
   'mode':'gated','scenario':sn,'target_fpr':np.nan,'threshold':np.nan,
   **curves(FINALY[mask],fs[mask])
  })

DEEP_FINAL=pd.DataFrame(DEEP_ROWS)
DEEP_FINAL.to_csv(RESULTS/'RQ_FF2_FF3_DEEP_20K_200K_FINAL.csv',index=False)

# FF4: all branch/fusion modes for DUAL/TRI at 20k AND 200k.
FF4_ROWS=[]
ff4_tags=[
 f'DUAL_{SSL20_CHAMP}_20K',f'TRI_{SSL20_CHAMP}_20K',
 f'DUAL_{SSL20_CHAMP}_200K',f'TRI_{SSL20_CHAMP}_200K'
]
for tag in ff4_tags:
 d,rep,use_dom=DEEP_VARIANTS[tag]
 cp,fp=CACHE[tag]
 modes=['url','text','gated','equal']+(['dom'] if use_dom else [])
 for mode in modes:
  cs=np.asarray(np.load(cp[mode],mmap_mode='r'))
  fs=np.asarray(np.load(fp[mode],mmap_mode='r'))
  for f in TARGET_FPRS:
   th=thr_fpr(cs,f)
   for sn,mask in FINAL_MASKS.items():
    FF4_ROWS.append({
     'system':tag,'rep':rep,
     'budget':json.loads((d/'COMPLETE.json').read_text())['budget'],
     'use_dom':use_dom,'mode':mode,'scenario':sn,
     'target_fpr':f,'threshold':th,
     **op(FINALY[mask],fs[mask],th)
    })
  for sn,mask in FINAL_MASKS.items():
   FF4_ROWS.append({
    'system':tag,'rep':rep,
    'budget':json.loads((d/'COMPLETE.json').read_text())['budget'],
    'use_dom':use_dom,'mode':mode,'scenario':sn,
    'target_fpr':np.nan,'threshold':np.nan,
    **curves(FINALY[mask],fs[mask])
   })

# Cascade only for the final 200k selected architecture.
selected_tag=f'{SELECTED_ARCH}_{SSL20_CHAMP}_200K'
cp,fp=CACHE[selected_tag]
if USE_CASCADE:
 full_cal=np.asarray(np.load(cp['gated'],mmap_mode='r'))
 url_cal=np.asarray(np.load(cp['url'],mmap_mode='r'))
 full_fin=np.asarray(np.load(fp['gated'],mmap_mode='r'))
 url_fin=np.asarray(np.load(fp['url'],mmap_mode='r'))
 ccal=full_cal.copy();cfin=full_fin.copy()
 ccal[url_cal>=url_high]=1e9;ccal[url_cal<=url_low]=-1e9
 cfin[url_fin>=url_high]=1e9;cfin[url_fin<=url_low]=-1e9
 for f in TARGET_FPRS:
  th=thr_fpr(ccal,f)
  for sn,mask in FINAL_MASKS.items():
   FF4_ROWS.append({
    'system':selected_tag,'rep':SSL20_CHAMP,'budget':HIGH_LABEL_ROWS,
    'use_dom':selected_use_dom,'mode':'cascade','scenario':sn,
    'target_fpr':f,'threshold':th,
    **op(FINALY[mask],cfin[mask],th)
   })
 for sn,mask in FINAL_MASKS.items():
  FF4_ROWS.append({
   'system':selected_tag,'rep':SSL20_CHAMP,'budget':HIGH_LABEL_ROWS,
   'use_dom':selected_use_dom,'mode':'cascade','scenario':sn,
   'target_fpr':np.nan,'threshold':np.nan,
   **curves(FINALY[mask],cfin[mask])
  })
 FINAL_ESC=float(((url_fin>url_low)&(url_fin<url_high)).mean())
else:
 FINAL_ESC=1.0

FF4=pd.DataFrame(FF4_ROWS)
FF4.to_csv(RESULTS/'RQ_FF4_DOM_FUSION_20K_200K_AND_CASCADE_FINAL.csv',index=False)

display(DEEP_FINAL[
 (DEEP_FINAL.scenario=='OFFICIAL_TEST')&
 np.isclose(DEEP_FINAL.target_fpr.fillna(-1),PRIMARY_FPR)
])
display(FF4[
 (FF4.scenario=='OFFICIAL_TEST')&
 np.isclose(FF4.target_fpr.fillna(-1),PRIMARY_FPR)
])


# CELL 11
# 10 — Paired seed statistics + direct FF1–FF4 evidence exports
def exact_signflip_p(diff):
 d=np.asarray(diff,float);d=d[np.isfinite(d)]
 if len(d)==0:return np.nan
 obs=abs(d.mean());vals=[]
 for signs in itertools.product([-1,1],repeat=len(d)):
  vals.append(abs(np.mean(d*np.asarray(signs))))
 return float(np.mean(np.asarray(vals)>=obs-1e-15))

def mean_ci(diff):
 d=np.asarray(diff,float);d=d[np.isfinite(d)]
 if len(d)<2:return (float(np.mean(d)) if len(d) else np.nan,np.nan,np.nan)
 m=float(d.mean());se=float(d.std(ddof=1)/math.sqrt(len(d)));q=float(student_t.ppf(.975,len(d)-1))
 return m,m-q*se,m+q*se

# FF1 paired against R0 at primary FPR, FINAL and DEV.
stats=[]
for dataset,scenario in [('DEV_EVAL','DEV_EVAL'),('FINAL','OFFICIAL_TEST'),
                         ('FINAL','DOMAIN_OOD_EXACT'),('FINAL','TEMPLATE_OOD_EXACT'),
                         ('FINAL','DOMAIN_TEMPLATE_OOD_EXACT'),('FINAL','LATE_TEST_Q4')]:
 for probe in ['LINEAR','MLP']:
  base=FF1[(FF1.rep=='R0')&(FF1.probe==probe)&(FF1.dataset==dataset)&(FF1.scenario==scenario)&np.isclose(FF1.target_fpr.fillna(-1),PRIMARY_FPR)]
  bdict=dict(zip(base.seed,base.tpr))
  for rep in ['R1','R2C','R3C','CSAME']:
   q=FF1[(FF1.rep==rep)&(FF1.probe==probe)&(FF1.dataset==dataset)&(FF1.scenario==scenario)&np.isclose(FF1.target_fpr.fillna(-1),PRIMARY_FPR)]
   qdict=dict(zip(q.seed,q.tpr));common=sorted(set(bdict)&set(qdict));diff=np.array([qdict[s]-bdict[s] for s in common])
   m,lo,hi=mean_ci(diff)
   stats.append({'dataset':dataset,'scenario':scenario,'probe':probe,'contrast':f'{rep}-R0','n':len(common),
                 'mean_delta_tpr_pp':100*m,'ci95_lo_pp':100*lo,'ci95_hi_pp':100*hi,
                 'positive_seeds':int((diff>0).sum()),'exact_signflip_p':exact_signflip_p(diff)})
STATS=pd.DataFrame(stats);STATS.to_csv(RESULTS/'RQ_FF1_PAIRED_STATS_20K.csv',index=False)

# Compact research-question tables.

# FF1: 20k representation utility on OFFICIAL_TEST at the primary operating point.
FF1_PRIMARY=AGG[
 (AGG.dataset=='FINAL')&
 (AGG.scenario=='OFFICIAL_TEST')&
 np.isclose(AGG.target_fpr.fillna(-1),PRIMARY_FPR)
].copy()
FF1_PRIMARY.to_csv(RESULTS/'RQ_FF1_PRIMARY_20K.csv',index=False)

# FF2a: same 20k representation experiment, explicitly shaped for downstream-mechanism comparison.
FF2_MECHANISM_20K=FF1_PRIMARY.copy()
FF2_MECHANISM_20K.to_csv(RESULTS/'RQ_FF2_DOWNSTREAM_MECHANISM_20K.csv',index=False)

# FF2b: matched Deep R0/R1 at 20k and 200k.
FF2_LABEL_SCALE=DEEP_FINAL[
 (DEEP_FINAL.scenario=='OFFICIAL_TEST')&
 np.isclose(DEEP_FINAL.target_fpr.fillna(-1),PRIMARY_FPR)&
 (~DEEP_FINAL.use_dom)&
 (DEEP_FINAL.tag.isin(['DUAL_R0_20K','DUAL_R1_20K','DUAL_R0_200K','DUAL_R1_200K']))
].copy()
FF2_LABEL_SCALE.to_csv(RESULTS/'RQ_FF2_LABEL_SCALE_R0_R1_20K_200K.csv',index=False)

# FF3 PRIMARY: all 20k representation variants, probes, seeds, shifts and full FPR grid.
# This directly mirrors the wording of FF3 ("untersuchten Repräsentationsvarianten").
FF3_REP_20K=FF1[
 (FF1.dataset=='FINAL')&
 (FF1.scenario.isin(['OFFICIAL_TEST','DOMAIN_OOD_EXACT','TEMPLATE_OOD_EXACT',
                     'DOMAIN_TEMPLATE_OOD_EXACT','LATE_TEST_Q4']))
].copy()
FF3_REP_20K.to_csv(RESULTS/'RQ_FF3_20K_REPRESENTATIONS_SHIFT_LOWFPR.csv',index=False)

FF3_REP_20K_AGG=FF3_REP_20K.groupby(
 ['rep','probe','scenario','target_fpr'],dropna=False
).agg(
 tpr_mean=('tpr','mean'),tpr_std=('tpr','std'),
 fpr_mean=('fpr','mean'),fpr_std=('fpr','std'),
 AP_mean=('AP','mean'),AP_std=('AP','std'),
 P_at_R90_mean=('P_at_R90','mean'),P_at_R90_std=('P_at_R90','std')
).reset_index()
FF3_REP_20K_AGG.to_csv(RESULTS/'RQ_FF3_20K_REPRESENTATIONS_SHIFT_LOWFPR_AGG.csv',index=False)

# FF3 SECONDARY operational result: selected 200k DUAL system over the same scenarios/FPR grid.
FF3_HIGH_200K=DEEP_FINAL[
 DEEP_FINAL.tag.eq(f'DUAL_{DEEP_REP}_200K')
].copy()
FF3_HIGH_200K.to_csv(RESULTS/'RQ_FF3_SECONDARY_200K_SYSTEM_SHIFT_LOWFPR.csv',index=False)

# FF4 DOM comparison at the SAME frozen representation under 20k and 200k labels.
dom_tags=[
 f'DUAL_{SSL20_CHAMP}_20K',f'TRI_{SSL20_CHAMP}_20K',
 f'DUAL_{SSL20_CHAMP}_200K',f'TRI_{SSL20_CHAMP}_200K'
]
DOM_PRIMARY=DEEP_FINAL[
 (DEEP_FINAL.tag.isin(dom_tags))&
 (DEEP_FINAL.scenario=='OFFICIAL_TEST')&
 np.isclose(DEEP_FINAL.target_fpr.fillna(-1),PRIMARY_FPR)
].copy()
DOM_PRIMARY.to_csv(RESULTS/'RQ_FF4_DOM_PRIMARY_20K_200K.csv',index=False)

FUSION_PRIMARY=FF4[
 (FF4.scenario=='OFFICIAL_TEST')&
 np.isclose(FF4.target_fpr.fillna(-1),PRIMARY_FPR)
].copy()
FUSION_PRIMARY.to_csv(RESULTS/'RQ_FF4_FUSION_PRIMARY_20K_200K.csv',index=False)

# Direct R0->R1 deltas for FF2 at matched Deep 20k/200k.
ff2_delta=[]
for budget in [SCI_LABEL_ROWS,HIGH_LABEL_ROWS]:
 r0=FF2_LABEL_SCALE[(FF2_LABEL_SCALE.budget==budget)&(FF2_LABEL_SCALE.rep=='R0')]
 r1=FF2_LABEL_SCALE[(FF2_LABEL_SCALE.budget==budget)&(FF2_LABEL_SCALE.rep=='R1')]
 if len(r0)==1 and len(r1)==1:
  ff2_delta.append({
   'budget':budget,
   'R0_tpr':float(r0.iloc[0].tpr),'R1_tpr':float(r1.iloc[0].tpr),
   'delta_tpr_pp':100*float(r1.iloc[0].tpr-r0.iloc[0].tpr),
   'R0_fpr':float(r0.iloc[0].fpr),'R1_fpr':float(r1.iloc[0].fpr),
   'R0_precision':float(r0.iloc[0].precision),'R1_precision':float(r1.iloc[0].precision)
  })
FF2_DELTA=pd.DataFrame(ff2_delta)
FF2_DELTA.to_csv(RESULTS/'RQ_FF2_R0_R1_DIRECT_DELTAS.csv',index=False)

evidence={
 'status':'COMPLETE',
 'protocol':{
  'primary_scientific_condition':'20k labels',
  'high_supervision_condition':'200k labels',
  'legacy_4k_used':False,
  'downstream_seeds':DOWNSTREAM_SEEDS,
  'threshold_domain':'raw_fp32_logits',
  'historical_final_known':True
 },
 'FF1':{
  'claim_scope':'downstream utility of SSL representations at 20k labels',
  'primary_file':'RQ_FF1_PRIMARY_20K.csv',
  'all_seed_file':'RQ_FF1_20K_REPRESENTATION_ALL_SEEDS.csv',
  'paired_stats_file':'RQ_FF1_PAIRED_STATS_20K.csv'
 },
 'FF2':{
  'claim_scope':'downstream mechanism at 20k and matched R0-vs-R1 label scaling from 20k to 200k',
  'mechanism_file':'RQ_FF2_DOWNSTREAM_MECHANISM_20K.csv',
  'label_scale_file':'RQ_FF2_LABEL_SCALE_R0_R1_20K_200K.csv',
  'direct_delta_file':'RQ_FF2_R0_R1_DIRECT_DELTAS.csv'
 },
 'FF3':{
  'primary_claim_scope':'20k representation variants under temporal/domain/template shifts and low-FPR grid',
  'primary_file':'RQ_FF3_20K_REPRESENTATIONS_SHIFT_LOWFPR.csv',
  'primary_aggregate_file':'RQ_FF3_20K_REPRESENTATIONS_SHIFT_LOWFPR_AGG.csv',
  'secondary_200k_system_file':'RQ_FF3_SECONDARY_200K_SYSTEM_SHIFT_LOWFPR.csv'
 },
 'FF4':{
  'claim_scope':'same frozen 20k SSL champion: DOM/fusion at 20k and 200k; cascade/resources at 200k',
  'dom_file':'RQ_FF4_DOM_PRIMARY_20K_200K.csv',
  'fusion_primary_file':'RQ_FF4_FUSION_PRIMARY_20K_200K.csv',
  'fusion_cascade_file':'RQ_FF4_DOM_FUSION_20K_200K_AND_CASCADE_FINAL.csv',
  'selected_representation':SSL20_CHAMP,
  'selected_architecture':SELECTED_ARCH,
  'cascade_used':USE_CASCADE,
  'dev_full_latency_ms_page':SYSTEM['latency_dev_ms_page_full'],
  'model_state_bytes':SYSTEM['model_state_bytes'],
  'final_full_execution_reduction':float(1-FINAL_ESC) if USE_CASCADE else 0.0
 },
 'interpretation_guardrails':[
  'High absolute performance of an R1-initialized 200k model is not itself evidence of SSL utility.',
  'SSL utility at 200k must be inferred only from the matched R1-vs-R0 comparison.',
  'FF3 primary inference concerns the 20k representation variants; the 200k system is secondary operational evidence.',
  'DOM/fusion comparisons at 20k and 200k hold the frozen 20k SSL champion constant.',
  'C-SAME is a diagnostic contrastive variant and must not be presented as preregistered.',
  'Earlier FINAL results were historically known before v7.1.'
 ]
}
atomic_json(RESULTS/'RESEARCH_QUESTION_EVIDENCE_v7_2.json',evidence)
print(json.dumps(evidence,indent=2))


# CELL 12
# 11 — Final integrity checks + compact results package
# No legacy SUP role may have entered any v7 scientific/deep training code path.
if CONFIG['legacy_4k_used'] is not False:raise RuntimeError('legacy 4k invariant failed')
if SCI_LABEL_ROWS!=20000 or HIGH_LABEL_ROWS!=200000:raise RuntimeError('label budget freeze changed')
if not (RESULTS/'RESEARCH_QUESTION_EVIDENCE_v7_2.json').exists():raise RuntimeError('RQ evidence missing')

complete={
 'status':'COMPLETE',
 'version':'v7_2_FINAL_RQ_CLOSURE_20K_200K',
 'legacy_4k_used':False,
 'scientific_label_rows':SCI_LABEL_ROWS,
 'high_supervision_rows':HIGH_LABEL_ROWS,
 'selected_representation':SSL20_CHAMP,
 'selected_architecture':SELECTED_ARCH,
 'cascade_used':USE_CASCADE,
 'raw_fp32_logit_thresholding':True
}
atomic_json(ROOT/'FINAL_V7_2_RQ_CLOSURE_COMPLETE.json',complete)

zp=Path('/kaggle/working/phreshphish_FINAL_RQ_CLOSURE_20K_200K_v7_2_RESULTS.zip') if Path('/kaggle/working').exists() else ROOT.parent/'phreshphish_FINAL_RQ_CLOSURE_20K_200K_v7_2_RESULTS.zip'
with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for base in [RESULTS,AUDIT]:
  for p in base.rglob('*'):
   if p.is_file():z.write(p,arcname=str(p.relative_to(ROOT)))
 z.write(ROOT/'FINAL_V7_2_RQ_CLOSURE_COMPLETE.json',arcname='FINAL_V7_2_RQ_CLOSURE_COMPLETE.json')
with zipfile.ZipFile(zp) as z:
 bad=z.testzip()
 if bad:raise RuntimeError(f'bad result zip {bad}')
print(complete);print({'RESULTS_ZIP':str(zp),'SAVE_FULL_KAGGLE_OUTPUT_FOR_CHECKPOINTS':True})
