# KI-Unterstuetzung: siehe Erklaerung der Arbeit.
# Ursprungsdatei: phreshphish_FINAL_DEEP_TRIMODAL_SSL_SYSTEM_v5_1_STABILITY_AUDITED(2).ipynb
# CELL 9
# 00 — Environment / configuration
import os,gc,re,json,math,time,random,hashlib,zipfile,shutil,warnings
from pathlib import Path
from collections import Counter
from contextlib import nullcontext
import numpy as np,pandas as pd
import torch,torch.nn as nn,torch.nn.functional as F
from torch.utils.data import IterableDataset,DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score,roc_curve,precision_recall_curve
from scipy.stats import beta as beta_dist
warnings.filterwarnings('ignore')

def seed_all(seed):
 random.seed(int(seed));np.random.seed(int(seed));torch.manual_seed(int(seed))
 if torch.cuda.is_available():torch.cuda.manual_seed_all(int(seed))
if not torch.cuda.is_available(): raise RuntimeError('STOP: GPU erforderlich')
DEVICE=torch.device('cuda'); AMP=True
SSL_SEED=20260812
DOWNSTREAM_SEEDS=[42,52,62,72,82,242,252,262,272,282]
ENGINEERING_SEEDS=[42,62,82]
URL_MODEL_ID='bert-base-uncased'; TEXT_MODEL_ID='roberta-base'
URL_MAX_LEN=128; TEXT_MAX_LEN=256; SSL_MAX_ROWS=200000
DAPT_EPOCHS=1; DAPT_BATCH_CANDIDATES=[16,8,4]; DAPT_LR=5e-5; MLM_PROB=.15
CONTRASTIVE_EPOCHS=1; CONTRASTIVE_BATCH_CANDIDATES=[16,8,4]; CONTRASTIVE_LR=2e-5
CONTRASTIVE_TEMP=.07; PROJ_DIM=256; QUEUE_SIZE=4096; SSL_LAST_N=4
LABEL_BUDGETS={'B10':400,'B25R':1000,'B100':4000}
TARGET_FPRS=[.0001,.0005,.001,.0025,.005,.01,.02]; PRIMARY_FPR=.005
TARGET_TPRS=[.90,.95]; PAUC_MAX=[.005,.01,.02]
# DOM SSL / Deep engineering
DOM_MAX_NODES=512; DOM_VOCAB_MAX=1024; DOM_DIM=128; DOM_LAYERS=3; DOM_MASK_P=.30
DOM_SSL_EPOCHS=1; DOM_SSL_BATCH=64; DOM_SSL_LR=2e-3
E2E_EPOCHS=3; E2E_HARDNEG_EPOCHS=1; E2E_BATCH_CANDIDATES=[16,8,4]; E2E_LR=2e-5; E2E_LAST_N=2
AUX_TOTAL_WEIGHT=.30; HARDNEG_WEIGHT=5.; HARDNEG_QUANTILE=.80
ENGINEERING_OBJECTIVE_ID='FAIR_AUX_TOTAL_030_WEIGHTNORM_V1'
DOM_GATE_MIN_PP=.5; DOM_GATE_REL_FPR90=.90
CASCADE_MAX_TPR_LOSS_PP=1.0; CASCADE_MIN_REDUCTION=.30
DEV_SPLIT_SEED=20260812
# Lightweight crash recovery. Deliberately coarse to avoid excessive Kaggle I/O.
RESUME_EVERY_TRANSFORMER_STEPS=4000
RESUME_EVERY_DOM_STEPS=1000
CACHE_PROGRESS_ROWS=20000
FINAL_SCORE_PROGRESS_ROWS=10000

def atomic_json(path,obj):
 path=Path(path);tmp=Path(str(path)+'.tmp')
 tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8');os.replace(tmp,path)

def atomic_torch(path,obj):
 path=Path(path);tmp=Path(str(path)+'.tmp')
 torch.save(obj,tmp);os.replace(tmp,path)

def emb_done(out):
 return Path(str(out)+'.complete.json')
def emb_progress(out):
 return Path(str(out)+'.progress.json')

def valid_emb(out,shape):
 out=Path(out);done=emb_done(out)
 if not out.exists() or not done.exists():return False
 try:
  meta=json.loads(done.read_text());a=np.load(out,mmap_mode='r')
  ok=tuple(a.shape)==tuple(shape) and a.dtype==np.float16 and tuple(meta.get('shape',[]))==tuple(shape)
  del a;return bool(ok)
 except:return False
DOWNLOAD_OFFICIAL_BENCHMARK_INDEX=True
PHRESHPHISH_DATASET_ID='phreshphish/phreshphish'; PINNED_REVISION='eabec4b7a66324b79cc8a0ad856d1731dc26fe1a'
ROOT=Path('/kaggle/working/phreshphish_FINAL_DEEP_TRIMODAL_SSL_v5_1_STABILITY_AUDITED') if Path('/kaggle/working').exists() else Path('/mnt/data/phreshphish_FINAL_DEEP_TRIMODAL_SSL_v5_1_STABILITY_AUDITED')
CKPT=ROOT/'checkpoints'; EMB=ROOT/'embeddings'; SCORES=ROOT/'scores'; RESULTS=ROOT/'results'; AUDIT=ROOT/'audit'; ARTIFACT=ROOT/'software_artifact'

for p in [ROOT,CKPT,EMB,SCORES,RESULTS,AUDIT,ARTIFACT]: p.mkdir(parents=True,exist_ok=True)

# ------------------------------------------------------------------
# Cross-session recovery from attached v4.3 output.
# IMPORTANT: only artifacts UNAFFECTED by the v5 engineering-loss patch are imported.
# Old DUAL/TRI/HARDNEG/SYSTEM_FREEZE checkpoints are intentionally NOT imported.
# ------------------------------------------------------------------
MODEL_RESUME_SOURCE=None
if Path('/kaggle/input').exists():
 candidates=[]
 for p in Path('/kaggle/input').rglob('phreshphish_FINAL_DEEP_TRIMODAL_SSL_v4_3_LABELSET_ALIGNMENT_FIX'):
  if not p.is_dir():continue
  score=0
  for rel in [
   'checkpoints/R1_DAPT_TEXT/COMPLETE.json',
   'checkpoints/R2_CONTRASTIVE/COMPLETE.json',
   'checkpoints/R3_DAPT_CONTRASTIVE/COMPLETE.json',
   'checkpoints/DOM_MASKED_SSL/COMPLETE.json',
  ]:
   score+=int((p/rel).exists())
  score+=sum(1 for q in (p/'embeddings').glob('R[0-3]_*.*') if q.is_file()) if (p/'embeddings').exists() else 0
  if score>0:candidates.append((score,p))

 if candidates:
  candidates=sorted(candidates,key=lambda x:x[0],reverse=True)
  top=candidates[0][0]
  best=[p for score,p in candidates if score==top]
  if len(best)>1:
   raise RuntimeError(f'STOP multiple equally complete v4.3 scientific resume roots: {[str(x) for x in best]}')
  src=best[0];MODEL_RESUME_SOURCE=src
  print({'SCIENTIFIC_RESUME_SNAPSHOT_FOUND':str(src),'completion_score':top})

  def copy_tree_if(srcp,dstp):
   srcp=Path(srcp);dstp=Path(dstp)
   if srcp.exists():
    if srcp.is_dir():shutil.copytree(srcp,dstp,dirs_exist_ok=True)
    else:
     dstp.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(srcp,dstp)

  for rel in [
   'checkpoints/R1_DAPT_TEXT',
   'checkpoints/R2_CONTRASTIVE',
   'checkpoints/R3_DAPT_CONTRASTIVE',
   'checkpoints/DOM_MASKED_SSL',
  ]:
   copy_tree_if(src/rel,ROOT/rel)

  # Only R0-R3 SUP/DEV embeddings and their complete/progress markers.
  if (src/'embeddings').exists():
   for p0 in (src/'embeddings').glob('*'):
    if p0.is_file() and re.match(r'R[0-3]_(SUP|DEV)\.npy(\.complete\.json|\.progress\.json)?$',p0.name):
     copy_tree_if(p0,EMB/p0.name)

  atomic_json(AUDIT/'SCIENTIFIC_RESUME_IMPORT.json',{
   'status':'COMPLETE',
   'source':str(src),
   'completion_score':top,
   'imported_scope':['R1_DAPT','R2_CONTRASTIVE','R3_CONTRASTIVE','DOM_SSL','R0-R3_SUP_DEV_EMBEDDINGS'],
   'explicitly_not_imported':['old_engineering_seeds','old_ENG_SELECTED','old_SYSTEM_FREEZE']
  })
  print({'SCIENTIFIC_RESUME_IMPORT':'COMPLETE'})
else:
 print('No attached v4.3 scientific resume snapshot found; normal execution.')

# Lightweight startup runtime self-test.
_STARTUP_PROBE=AUDIT/'STARTUP_SELFTEST.json'
seed_all(SSL_SEED)
atomic_json(_STARTUP_PROBE,{
 'status':'PASS',
 'seed_function_callable':callable(seed_all),
 'root':str(ROOT),
 'root_name_ok':ROOT.name=='phreshphish_FINAL_DEEP_TRIMODAL_SSL_v5_1_STABILITY_AUDITED',
 'resume_source':None if MODEL_RESUME_SOURCE is None else str(MODEL_RESUME_SOURCE)
})
_startup_obj=json.loads(_STARTUP_PROBE.read_text())
if _startup_obj.get('status')!='PASS' or not _startup_obj.get('root_name_ok'):
 raise RuntimeError(f'STOP startup selftest failed: {_startup_obj}')
print({'STARTUP_SELFTEST':'PASS','root':str(ROOT)})

CONFIG={'SSL_SEED':SSL_SEED,'DOWNSTREAM_SEEDS':DOWNSTREAM_SEEDS,'ENGINEERING_SEEDS':ENGINEERING_SEEDS,
 'URL_MODEL_ID':URL_MODEL_ID,'TEXT_MODEL_ID':TEXT_MODEL_ID,'SSL_MAX_ROWS':SSL_MAX_ROWS,
 'LABEL_BUDGETS':LABEL_BUDGETS,'TARGET_FPRS':TARGET_FPRS,'PRIMARY_FPR':PRIMARY_FPR,
 'scientific_representations':['R0','R1','R2','R3'],'engineering':['DUAL','TRI_DOM_SSL'],
 'version':'v5_1_STABILITY_AUDITED','metric_P_at_R90':'PhreshPhish-style PR interpolation',
 'engineering_gate':'ENG_TUNE thresholds -> ENG_META evaluation; seed-aggregated',
 'engineering_objective':ENGINEERING_OBJECTIVE_ID,
 'aux_total_weight':AUX_TOTAL_WEIGHT,
 'hardnegative_weight_normalization':'sum(weighted BCE)/sum(weights)'}

RUNTIME_PLAN={
 'basis':'observed Tesla T4 timings from prior v4.3 run; planning estimate only',
 'with_v43_scientific_resume_hours':{'low':5.0,'high':7.0},
 'without_resume_hours':{'low':8.0,'high':10.0},
 'components_minutes_estimate':{
  'fair_engineering_rerun_6_seeds':'65-85',
  'CAL_FINAL_embeddings_R0_R3':'100-125',
  'scientific_N10_and_statistics':'30-60',
  'final_deep_scoring':'75-100',
  'benchmark_bootstrap_exports':'15-35',
  'overhead':'15-30'
 }
}
atomic_json(AUDIT/'RUNTIME_PLAN.json',RUNTIME_PLAN)

(AUDIT/'EXPERIMENT_CONFIG.json').write_text(json.dumps(CONFIG,indent=2))
print(torch.cuda.get_device_name(0));print(json.dumps(CONFIG,indent=2))

# CELL 10
# 01 — Resolve exact FINAL DATA FREEZE v3 as ROLE ROOT + METADATA SIDECAR ROOT
import pyarrow.parquet as pq

SEARCH_ROOT=Path('/kaggle/input') if Path('/kaggle/input').exists() else Path('/mnt/data')
EXPECTED_FREEZE_ID='PHRESHPHISH_FINAL_DATA_FREEZE_v3_HF_SHARDED_DOM'
EXPECTED_COUNTS={'SUP':4000,'DEV':20000,'CAL':50000,'SSL':200000,'FINAL':168060}

ROLE_REL={
 'SUP':Path('roles/supervised_pool'),
 'DEV':Path('roles/development'),
 'CAL':Path('roles/fpr_calibration_benign'),
 'SSL':Path('roles/ssl_pool_200k'),
 'FINAL':Path('roles/final_test'),
}

META_REQUIRED=[
 Path('manifests/supervised_pool_manifest.parquet'),
 Path('manifests/final_test_exact_ood_flags_SEALED.parquet'),
 Path('manifests/final_test_manifest_SEALED.parquet'),
 Path('audit/LABELSET_AUDIT.csv'),
 Path('audit/DATA_FREEZE_SPEC.json'),
 Path('labelsets'),
]

def _valid_marker(p):
 try:o=json.loads(Path(p).read_text())
 except Exception as e:return None,f'invalid_json:{e!r}'
 if o.get('status')!='COMPLETE':return None,'not_complete'
 if o.get('freeze_id')!=EXPECTED_FREEZE_ID:return None,f"wrong_freeze_id:{o.get('freeze_id')!r}"
 return o,'ok'

def _role_root_ok(root):
 root=Path(root)
 missing=[str(rel) for rel in ROLE_REL.values() if not (root/rel).exists()]
 return len(missing)==0,missing

def _meta_root_ok(root):
 root=Path(root)
 missing=[str(rel) for rel in META_REQUIRED if not (root/rel).exists()]
 return len(missing)==0,missing

# ------------------------------------------------------------
# A. Exactly one valid corrected COMPLETE marker.
# ------------------------------------------------------------
marker_candidates=[]
marker_diag=[]
for p in SEARCH_ROOT.rglob('FINAL_DATA_FREEZE_COMPLETE.json'):
 o,why=_valid_marker(p);marker_diag.append({'path':str(p),'status':why})
 if o is not None:marker_candidates.append((p,o))

uniq={}
for p,o in marker_candidates:uniq[str(p.resolve())]=(p,o)
marker_candidates=list(uniq.values())

if len(marker_candidates)!=1:
 raise RuntimeError(
  f'STOP expected exactly one valid corrected COMPLETE marker; found={len(marker_candidates)}; '
  f'{[str(p) for p,_ in marker_candidates]}; diagnostics={marker_diag[:20]}'
 )

FREEZE_FILE,FREEZE_INFO=marker_candidates[0]

# ------------------------------------------------------------
# B. Metadata root: prefer the marker parent (corrected handoff).
# ------------------------------------------------------------
META_ROOT=None
ok,missing=_meta_root_ok(FREEZE_FILE.parent)
if ok:
 META_ROOT=FREEZE_FILE.parent
else:
 meta_candidates=[]
 for manifests in SEARCH_ROOT.rglob('manifests'):
  if not manifests.is_dir():continue
  root=manifests.parent
  mok,mmiss=_meta_root_ok(root)
  if mok:meta_candidates.append(root)
 md={}
 for x in meta_candidates:md[str(x.resolve())]=x
 meta_candidates=list(md.values())
 if len(meta_candidates)==1:META_ROOT=meta_candidates[0]
 else:
  raise RuntimeError(
   f'STOP corrected metadata root unresolved. marker_parent={FREEZE_FILE.parent}; '
   f'missing_there={missing}; candidates={[str(x) for x in meta_candidates]}'
  )

# ------------------------------------------------------------
# C. Role root: independently locate the one root with all five role dirs.
# It may intentionally be a DIFFERENT Kaggle dataset.
# ------------------------------------------------------------
role_candidates=[]
role_diag=[]
for roles in SEARCH_ROOT.rglob('roles'):
 if not roles.is_dir():continue
 root=roles.parent
 rok,rmiss=_role_root_ok(root)
 role_diag.append({'root':str(root),'missing':rmiss})
 if rok:role_candidates.append(root)

rd={}
for x in role_candidates:rd[str(x.resolve())]=x
role_candidates=list(rd.values())

# If metadata root itself also has full roles, prefer same-root.
if _role_root_ok(META_ROOT)[0]:
 DATA_ROOT=META_ROOT
 INPUT_MODE='SAME_ROOT'
else:
 if len(role_candidates)!=1:
  raise RuntimeError(
   f'STOP role root unresolved; candidates={[str(x) for x in role_candidates]}; '
   f'diagnostics={role_diag[:20]}'
  )
 DATA_ROOT=role_candidates[0]
 INPUT_MODE='SIDECAR_META_OVERLAY'

# ------------------------------------------------------------
# D. Bind paths: roles from DATA_ROOT; all frozen metadata from META_ROOT.
# ------------------------------------------------------------
ROLE_ROOT=DATA_ROOT/'roles'
MANIFEST_ROOT=META_ROOT/'manifests'
LABELSET_ROOT=META_ROOT/'labelsets'
FREEZE_AUDIT=META_ROOT/'audit'
ROLE_DIR={k:DATA_ROOT/v for k,v in ROLE_REL.items()}

print({
 'DATA_FREEZE_INPUT':INPUT_MODE,
 'marker':str(FREEZE_FILE),
 'role_root':str(DATA_ROOT),
 'metadata_root':str(META_ROOT)
})

# ------------------------------------------------------------
# E. Front-load ALL dependencies needed later, before GPU work.
# ------------------------------------------------------------
def nrows(d):
 fs=sorted(Path(d).glob('*.parquet'))
 if not fs:raise RuntimeError(f'STOP no parquet files in {d}')
 return sum(pq.ParquetFile(p).metadata.num_rows for p in fs)

counts={k:nrows(v) for k,v in ROLE_DIR.items()}
if counts!=EXPECTED_COUNTS:
 raise RuntimeError(f'STOP role counts {counts} != {EXPECTED_COUNTS}')

# Corrected audit spec must be arithmetically consistent.
spec=json.loads((FREEZE_AUDIT/'DATA_FREEZE_SPEC.json').read_text())
exp=spec.get('expected',{})
audit_test={
 'rows':int(exp.get('test_rows',-1)),
 'benign':int(exp.get('test_benign',-1)),
 'phish':int(exp.get('test_phish',-1)),
}
if audit_test!={'rows':168060,'benign':91260,'phish':76800}:
 raise RuntimeError(f'STOP corrected DATA_FREEZE_SPEC test counts invalid: {audit_test}')
if audit_test['benign']+audit_test['phish']!=audit_test['rows']:
 raise RuntimeError('STOP corrected DATA_FREEZE_SPEC arithmetic inconsistent')

# Supervised manifest exists NOW, not 4 hours later.
sup_manifest=MANIFEST_ROOT/'supervised_pool_manifest.parquet'
if pq.ParquetFile(sup_manifest).metadata.num_rows!=4000:
 raise RuntimeError('STOP supervised_pool_manifest rows != 4000')

# Exact OOD flags exist NOW and cover full FINAL.
ood_path=MANIFEST_ROOT/'final_test_exact_ood_flags_SEALED.parquet'
ood_pf=pq.ParquetFile(ood_path)
if ood_pf.metadata.num_rows!=168060:
 raise RuntimeError(f'STOP exact OOD manifest rows={ood_pf.metadata.num_rows} != 168060')
ood_cols=set(ood_pf.schema_arrow.names)
ood_required={'sha256','domain_ood_exact','template_ood_exact','domain_template_ood_exact'}
if not ood_required.issubset(ood_cols):
 raise RuntimeError(f'STOP exact OOD manifest fields missing: {sorted(ood_required-ood_cols)}')

final_manifest=MANIFEST_ROOT/'final_test_manifest_SEALED.parquet'
if pq.ParquetFile(final_manifest).metadata.num_rows!=168060:
 raise RuntimeError('STOP final_test_manifest rows != 168060')

# 30 frozen labelsets must exist before training.
label_files=sorted(LABELSET_ROOT.glob('B*_seed*.npy'))
if len(label_files)!=30:
 raise RuntimeError(f'STOP expected 30 frozen labelset files, found={len(label_files)}')

# Logical nested DOM schema + SSL leakage guard.
sslcols=set()
for p in ROLE_DIR['SSL'].glob('*.parquet'):
 sslcols.update(pq.ParquetFile(p).schema_arrow.names)
if 'label' in sslcols:raise RuntimeError('STOP: SSL label leakage')
required_ssl={
 'sha256','url','text','dom_tag','dom_parent_idx','dom_depth',
 'dom_attr_count','dom_child_count','ssl_rank'
}
if not required_ssl.issubset(sslcols):
 raise RuntimeError(f'STOP SSL fields missing: {sorted(required_ssl-sslcols)}')

DATA_FREEZE_HASH=hashlib.sha256(FREEZE_FILE.read_bytes()).hexdigest()

atomic_json(AUDIT/'DATA_FREEZE_LINK.json',{
 'marker':str(FREEZE_FILE),
 'role_root':str(DATA_ROOT),
 'metadata_root':str(META_ROOT),
 'sha256':DATA_FREEZE_HASH,
 'counts':counts,
 'freeze':FREEZE_INFO,
 'input_mode':INPUT_MODE,
 'preflight':{
  'corrected_test_counts':audit_test,
  'supervised_manifest_rows':4000,
  'exact_ood_manifest_rows':168060,
  'final_manifest_rows':168060,
  'labelset_files':30,
 }
})

print({
 'DATA_FREEZE':'PASS',
 'input_mode':INPUT_MODE,
 'counts':counts,
 'corrected_test_counts':audit_test,
 'OOD_MANIFEST_PREFLIGHT':'PASS',
 'role_root':str(DATA_ROOT),
 'metadata_root':str(META_ROOT)
})


atomic_json(AUDIT/'METHODOLOGY_PATCH_v5.json',{
 'status':'FROZEN_BEFORE_CAL_FINAL',
 'reason':'remove modality-count confound in DUAL vs TRI auxiliary supervision',
 'old_auxiliary_objective':'main + 0.15 * SUM(auxiliary_head_losses)',
 'new_auxiliary_objective':'main + 0.30 * MEAN(auxiliary_head_losses)',
 'hardnegative_change':'same per-example weights applied to main and auxiliary heads; normalized by weight sum',
 'affected_scope':['FF4 engineering DUAL/TRI/HARDNEG only'],
 'unaffected_scope':['R0-R3','DAPT','contrastive SSL','DOM SSL pretraining','labelsets','development partitions','CAL','FINAL'],
 'posthoc_parameter_tuning':False
})
print({'METHODOLOGY_PATCH_v5':'FROZEN','CAL_FINAL_NOT_USED_FOR_PATCH':True})


# CELL 11
# 02 — Transformers / provenance / generic utilities
from transformers import AutoTokenizer,AutoModel,AutoModelForMaskedLM,DataCollatorForLanguageModeling
from transformers.optimization import get_linear_schedule_with_warmup
HF_TOKEN=os.environ.get('HF_TOKEN')
if not HF_TOKEN and Path('/kaggle/working').exists():
 try:
  from kaggle_secrets import UserSecretsClient
  HF_TOKEN=UserSecretsClient().get_secret('HF_TOKEN')
 except: pass
def resolve(mid):
 hits=[]
 for p in SEARCH_ROOT.rglob('config.json'):
  if mid.split('/')[-1].lower() in str(p.parent).lower(): hits.append(p.parent)
 return (str(sorted(hits,key=lambda x:len(str(x)))[0]),'LOCAL') if hits else (mid,'HF')
URL_SRC,URL_KIND=resolve(URL_MODEL_ID); TEXT_SRC,TEXT_KIND=resolve(TEXT_MODEL_ID)
url_tok=AutoTokenizer.from_pretrained(URL_SRC,token=HF_TOKEN); text_tok=AutoTokenizer.from_pretrained(TEXT_SRC,token=HF_TOKEN)
u0=AutoModel.from_pretrained(URL_SRC,token=HF_TOKEN); t0=AutoModel.from_pretrained(TEXT_SRC,token=HF_TOKEN)
URL_H=u0.config.hidden_size; TEXT_H=t0.config.hidden_size; REP_DIM=URL_H+TEXT_H
(AUDIT/'MODEL_PROVENANCE.json').write_text(json.dumps({'url':{'source':URL_SRC,'kind':URL_KIND,'hidden':URL_H,'commit':getattr(u0.config,'_commit_hash',None)},'text':{'source':TEXT_SRC,'kind':TEXT_KIND,'hidden':TEXT_H,'commit':getattr(t0.config,'_commit_hash',None)}},indent=2))
del u0,t0;gc.collect()
def amp_ctx(): return torch.autocast('cuda',dtype=torch.float16) if AMP else nullcontext()
def masked_mean(h,m):
 m=m.unsqueeze(-1).to(h.dtype);return (h*m).sum(1)/m.sum(1).clamp_min(1)
def pooled(out,mask): return masked_mean(out.last_hidden_state,mask)
def freeze_last(model,n):
 for p in model.parameters():p.requires_grad=False
 layers=getattr(getattr(model,'encoder',None),'layer',None)
 if layers is None: layers=getattr(getattr(model,'transformer',None),'layer',None)
 if layers is None: raise RuntimeError(type(model))
 for layer in layers[-n:]:
  for p in layer.parameters():p.requires_grad=True
 if getattr(model,'pooler',None) is not None:
  for p in model.pooler.parameters():p.requires_grad=True
class ParquetIter(IterableDataset):
 def __init__(self,d,cols,seed,max_rows):self.files=sorted(Path(d).glob('*.parquet'));self.cols=cols;self.seed=seed;self.max_rows=max_rows
 def __iter__(self):
  rng=np.random.default_rng(self.seed);files=list(self.files);rng.shuffle(files);n=0
  for p in files:
   df=pd.read_parquet(p,columns=self.cols);idx=np.arange(len(df));rng.shuffle(idx)
   for i in idx:
    yield {x:df.iloc[i][x] for x in self.cols};n+=1
    if n>=self.max_rows:return
print({'URL_SRC':URL_SRC,'TEXT_SRC':TEXT_SRC,'REP_DIM':REP_DIM})

# CELL 12
# 03 — Read only NON-FINAL data
# Exact SUP ordering from frozen supervised_pool_manifest + deterministic DEV partitions + labelset audit

def logical_names(p):
 return set(pq.ParquetFile(p).schema_arrow.names)

def read_role(key,cols=None):
 fs=sorted(ROLE_DIR[key].glob('*.parquet'));parts=[]
 for p in fs:
  names=logical_names(p)
  use=None if cols is None else [x for x in cols if x in names]
  missing=[] if cols is None else [x for x in cols if x not in names]
  if missing:raise RuntimeError(f'STOP missing columns in {p.name}: {missing}')
  parts.append(pd.read_parquet(p,columns=use))
 if not parts:raise RuntimeError(f'STOP empty role {key}')
 return pd.concat(parts,ignore_index=True)

def y01(s):
 y=s.astype(str).str.lower().map({'benign':0,'phish':1})
 if y.isna().any():
  raise RuntimeError(f'STOP unknown labels: {sorted(s[y.isna()].astype(str).unique())}')
 return y.astype(int).to_numpy()

basecols=[
 'sha256','url','text','label',
 'dom_tag','dom_parent_idx','dom_depth','dom_node_count','dom_max_depth',
 'dom_attr_count','dom_child_count','struct_token_count'
]

# ------------------------------------------------------------
# A. Load raw materialized roles.
# ------------------------------------------------------------
sup_raw=read_role('SUP',basecols)
dev=read_role('DEV',basecols)

sup_raw['sha256']=sup_raw.sha256.astype(str).str.lower()
dev['sha256']=dev.sha256.astype(str).str.lower()

if len(sup_raw)!=4000:
 raise RuntimeError(f'STOP SUP rows {len(sup_raw)} != 4000')
if sup_raw.sha256.duplicated().any():
 raise RuntimeError('STOP duplicate SHA inside materialized SUP')
if dev.sha256.duplicated().any():
 raise RuntimeError('STOP duplicate SHA inside DEV')

# ------------------------------------------------------------
# B. Reconstruct the EXACT index basis used by Data Freeze Builder.
# Builder semantics:
#   sup = SUPERVISED_POOL.sort_values("sha256").reset_index(drop=True)
#   sup["supervised_index"] = np.arange(...)
# Labelset .npy files index THIS ordering.
# ------------------------------------------------------------
sup_manifest_path=MANIFEST_ROOT/'supervised_pool_manifest.parquet'
if not sup_manifest_path.exists():
 raise RuntimeError(f'STOP missing frozen supervised manifest: {sup_manifest_path}')

sm=pd.read_parquet(sup_manifest_path)
required_manifest={'sha256','supervised_index','label'}
missing=required_manifest-set(sm.columns)
if missing:
 raise RuntimeError(f'STOP supervised manifest missing {sorted(missing)}')

sm=sm.copy()
sm['sha256']=sm.sha256.astype(str).str.lower()
if len(sm)!=4000:
 raise RuntimeError(f'STOP supervised manifest rows {len(sm)} != 4000')
if sm.sha256.duplicated().any():
 raise RuntimeError('STOP duplicate SHA in supervised manifest')
if sm.supervised_index.duplicated().any():
 raise RuntimeError('STOP duplicate supervised_index in manifest')

expected_idx=np.arange(4000,dtype=np.int64)
actual_idx=np.sort(sm.supervised_index.astype(np.int64).to_numpy())
if not np.array_equal(actual_idx,expected_idx):
 raise RuntimeError('STOP supervised_index is not exact 0..3999')

if set(sm.sha256)!=set(sup_raw.sha256):
 only_manifest=sorted(set(sm.sha256)-set(sup_raw.sha256))[:10]
 only_role=sorted(set(sup_raw.sha256)-set(sm.sha256))[:10]
 raise RuntimeError(
  f'STOP SUP SHA mismatch manifest vs materialized role; '
  f'only_manifest={only_manifest}, only_role={only_role}'
 )

# Join role data onto the frozen manifest, then restore supervised_index order.
# Keep manifest label separately to verify materialized labels.
sm_key=sm[['sha256','supervised_index','label']].rename(columns={'label':'manifest_label'})
sup=sup_raw.merge(sm_key,on='sha256',how='inner',validate='one_to_one')
sup=sup.sort_values('supervised_index').reset_index(drop=True)

if not np.array_equal(sup.supervised_index.astype(np.int64).to_numpy(),expected_idx):
 raise RuntimeError('STOP failed to reconstruct supervised_index order')

role_label=sup.label.astype(str).str.lower().to_numpy()
manifest_label=sup.manifest_label.astype(str).str.lower().to_numpy()
if not np.array_equal(role_label,manifest_label):
 bad=np.where(role_label!=manifest_label)[0][:10]
 raise RuntimeError(f'STOP label mismatch SUP role vs manifest at indices {bad.tolist()}')

# Drop audit-only alignment columns from model input after validation.
sup=sup.drop(columns=['manifest_label'])
sy=y01(sup.label)
dy=y01(dev.label)

if (int((sy==0).sum()),int((sy==1).sum()))!=(2000,2000):
 raise RuntimeError('STOP SUP class counts')
if (int((dy==0).sum()),int((dy==1).sum()))!=(10000,10000):
 raise RuntimeError('STOP DEV class counts')

atomic_json(AUDIT/'SUPERVISED_INDEX_ALIGNMENT.json',{
 'status':'PASS',
 'rows':len(sup),
 'ordering_basis':'manifests/supervised_pool_manifest.parquet::supervised_index',
 'sha_set_equal':True,
 'role_manifest_labels_equal':True,
 'first_sha256':str(sup.sha256.iloc[0]),
 'last_sha256':str(sup.sha256.iloc[-1])
})
print({
 'SUPERVISED_INDEX_ALIGNMENT':'PASS',
 'rows':len(sup),
 'benign':int((sy==0).sum()),
 'phish':int((sy==1).sum())
})

# ------------------------------------------------------------
# C. Deterministic DEV partitions.
# ------------------------------------------------------------
idx=np.arange(len(dev))
a,b=train_test_split(
 idx,test_size=.5,stratify=dy,random_state=DEV_SPLIT_SEED
)
sc,se=train_test_split(
 a,test_size=.5,stratify=dy[a],random_state=DEV_SPLIT_SEED+1
)
et,em=train_test_split(
 b,test_size=.5,stratify=dy[b],random_state=DEV_SPLIT_SEED+2
)

PARTS={
 'SCREEN_CAL':np.sort(sc),
 'SCREEN_EVAL':np.sort(se),
 'ENG_TUNE':np.sort(et),
 'ENG_META':np.sort(em)
}

for k,v in PARTS.items():
 if len(v)!=5000 or dy[v].sum()!=2500:
  raise RuntimeError(f'STOP invalid DEV partition {k}')

np.savez(AUDIT/'DEV_PARTITIONS.npz',**PARTS)

# ------------------------------------------------------------
# D. Frozen labelsets.
# IMPORTANT: indices are positions in the aligned `sup`.
# ------------------------------------------------------------
label_audit_path=FREEZE_AUDIT/'LABELSET_AUDIT.csv'
if not label_audit_path.exists():
 raise RuntimeError(f'STOP missing LABELSET_AUDIT.csv: {label_audit_path}')

label_audit=pd.read_csv(label_audit_path)
LABELSETS={}
label_checks=[]

for seed in DOWNSTREAM_SEEDS:
 LABELSETS[seed]={}
 for bgt,n in LABEL_BUDGETS.items():
  p=LABELSET_ROOT/f'{bgt}_seed{seed}.npy'
  if not p.exists():
   raise RuntimeError(f'STOP missing labelset {p}')

  arr=np.load(p)
  if arr.ndim!=1:
   raise RuntimeError(f'STOP labelset not 1D: {p}')
  arr=arr.astype(np.int64,copy=False)

  if len(arr)!=n or len(np.unique(arr))!=n or arr.min()<0 or arr.max()>=len(sup):
   raise RuntimeError(f'STOP invalid labelset {p}')

  yy=sy[arr]
  benign=int((yy==0).sum())
  phish=int((yy==1).sum())
  if benign!=n//2 or phish!=n//2:
   raise RuntimeError(
    f'STOP unbalanced labelset AFTER manifest alignment {p}: '
    f'benign={benign}, phish={phish}, expected={n//2}/{n//2}'
   )

  hh=hashlib.sha256(np.ascontiguousarray(arr.astype(np.int32)).tobytes()).hexdigest()
  q=label_audit[
   (label_audit.seed.astype(int)==int(seed)) &
   (label_audit.budget.astype(str)==str(bgt))
  ]
  if len(q)!=1:
   raise RuntimeError(f'STOP labelset audit row missing/duplicate for {bgt} seed{seed}')

  expected_hash=str(q.iloc[0].idx_sha256)
  expected_benign=int(q.iloc[0].benign)
  expected_phish=int(q.iloc[0].phish)

  if hh!=expected_hash:
   raise RuntimeError(
    f'STOP labelset hash mismatch {p}: runtime={hh}, audit={expected_hash}'
   )
  if benign!=expected_benign or phish!=expected_phish:
   raise RuntimeError(
    f'STOP labelset class audit mismatch {p}: '
    f'runtime={benign}/{phish}, audit={expected_benign}/{expected_phish}'
   )

  # Keep int32 semantics exactly as frozen.
  LABELSETS[seed][bgt]=arr.astype(np.int32)

  label_checks.append({
   'seed':int(seed),'budget':bgt,'n':int(n),
   'benign':benign,'phish':phish,
   'sha256':hh
  })

atomic_json(AUDIT/'LABELSET_RUNTIME_VALIDATION.json',{
 'status':'PASS',
 'index_basis':'aligned supervised_pool_manifest.supervised_index',
 'checks':label_checks
})

print({
 'DEV_PARTS':{k:len(v) for k,v in PARTS.items()},
 'LABELSETS':'PASS',
 'validated_labelsets':len(label_checks)
})


# CELL 13
# 04 — Autograd preflight + R1 DAPT (rolling mid-epoch resume)
# preflight
m=AutoModelForMaskedLM.from_pretrained(TEXT_SRC,token=HF_TOKEN).to(DEVICE);x=text_tok(['verify account','bank login'],padding=True,return_tensors='pt');lab=x['input_ids'].clone();lab[:,1::2]=-100;x={k:v.to(DEVICE) for k,v in x.items()}
with amp_ctx():loss=m(**x,labels=lab.to(DEVICE)).loss
loss.backward();del m,x,lab;gc.collect();torch.cuda.empty_cache();atomic_json(AUDIT/'AUTOGRAD_PREFLIGHT.json',{'status':'PASS'})

DAPT_DIR=CKPT/'R1_DAPT_TEXT';DONE=DAPT_DIR/'COMPLETE.json';DAPT_RESUME=DAPT_DIR/'resume.pt'

def run_dapt(bs):
 seed_all(SSL_SEED);DAPT_DIR.mkdir(parents=True,exist_ok=True)
 model=AutoModelForMaskedLM.from_pretrained(TEXT_SRC,token=HF_TOKEN).to(DEVICE);model.train()
 ds=ParquetIter(ROLE_DIR['SSL'],['text'],SSL_SEED,SSL_MAX_ROWS);coll=DataCollatorForLanguageModeling(tokenizer=text_tok,mlm_probability=MLM_PROB)
 def cc(rows):
  e=text_tok([str(r['text'] or '') for r in rows],padding=True,truncation=True,max_length=TEXT_MAX_LEN,return_tensors='pt')
  return coll([{k:e[k][i] for k in e} for i in range(len(rows))])
 dl=DataLoader(ds,batch_size=bs,collate_fn=cc,num_workers=0)
 opt=torch.optim.AdamW(model.parameters(),lr=DAPT_LR,weight_decay=.01)
 steps=math.ceil(SSL_MAX_ROWS/bs)*DAPT_EPOCHS
 sched=get_linear_schedule_with_warmup(opt,int(.06*steps),steps)
 scaler=torch.cuda.amp.GradScaler(enabled=AMP)

 start_ep=0;skip_batches=0;ls=[]
 if DAPT_RESUME.exists():
  try:
   ck=torch.load(DAPT_RESUME,map_location=DEVICE)
   if int(ck.get('batch',-1))==bs:
    model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler']);scaler.load_state_dict(ck['scaler'])
    start_ep=int(ck.get('epoch',0));skip_batches=int(ck.get('next_batch',0));ls=list(ck.get('losses',[]))
    print({'DAPT_RESUME':True,'epoch':start_ep,'next_batch':skip_batches})
  except Exception as e:
   print('Invalid DAPT resume ignored:',repr(e));start_ep=0;skip_batches=0;ls=[]

 for ep in range(start_ep,DAPT_EPOCHS):
  # ParquetIter is deterministic for the frozen seed, so skipped batches reproduce the same order.
  for j,b in enumerate(dl):
   if ep==start_ep and j<skip_batches:continue
   b={k:v.to(DEVICE) for k,v in b.items()};opt.zero_grad(set_to_none=True)
   with amp_ctx():l=model(**b).loss
   scaler.scale(l).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update();sched.step()
   ls.append(float(l.detach().cpu()))
   nxt=j+1
   if nxt%RESUME_EVERY_TRANSFORMER_STEPS==0:
    atomic_torch(DAPT_RESUME,{'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),
                              'scaler':scaler.state_dict(),'epoch':ep,'next_batch':nxt,'batch':bs,'losses':ls[-500:]})
    print({'DAPT_checkpoint_batch':nxt,'loss':float(np.mean(ls[-100:]))})
  # epoch boundary
  atomic_torch(DAPT_RESUME,{'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),
                            'scaler':scaler.state_dict(),'epoch':ep+1,'next_batch':0,'batch':bs,'losses':ls[-500:]})
  skip_batches=0

 model.base_model.save_pretrained(DAPT_DIR);text_tok.save_pretrained(DAPT_DIR)
 atomic_json(DONE,{'status':'COMPLETE','seed':SSL_SEED,'rows':SSL_MAX_ROWS,'epochs':DAPT_EPOCHS,'batch':bs,'loss':float(np.mean(ls))})
 DAPT_RESUME.unlink(missing_ok=True)
 del model,dl,ds;gc.collect();torch.cuda.empty_cache()

if not DONE.exists():
 last=None
 for bs in DAPT_BATCH_CANDIDATES:
  try:run_dapt(bs);last=None;break
  except torch.cuda.OutOfMemoryError as e:
   last=e;gc.collect();torch.cuda.empty_cache()
 if last:raise last
else:print('Reuse completed DAPT')
print(json.loads(DONE.read_text()))


# CELL 14
# 05 — R2/R3 contrastive URL↔HTML-text SSL with FIFO negatives (rolling resume)
class DualSSL(nn.Module):
 def __init__(self,tsrc):
  super().__init__();self.u=AutoModel.from_pretrained(URL_SRC,token=HF_TOKEN);self.t=AutoModel.from_pretrained(tsrc,token=HF_TOKEN);freeze_last(self.u,SSL_LAST_N);freeze_last(self.t,SSL_LAST_N);self.up=nn.Sequential(nn.Linear(URL_H,PROJ_DIM),nn.GELU(),nn.Linear(PROJ_DIM,PROJ_DIM));self.tp=nn.Sequential(nn.Linear(TEXT_H,PROJ_DIM),nn.GELU(),nn.Linear(PROJ_DIM,PROJ_DIM))
 def forward(self,u,t):
  hu=pooled(self.u(**u),u['attention_mask']);ht=pooled(self.t(**t),t['attention_mask']);return F.normalize(self.up(hu),dim=-1),F.normalize(self.tp(ht),dim=-1)

def contrastive_loss(zu,zt,qu,qt):
 ct=torch.cat([zt,qt],0) if qt is not None else zt;cu=torch.cat([zu,qu],0) if qu is not None else zu;lab=torch.arange(len(zu),device=DEVICE)
 return (F.cross_entropy(zu@ct.T/CONTRASTIVE_TEMP,lab)+F.cross_entropy(zt@cu.T/CONTRASTIVE_TEMP,lab))/2

def train_ssl(name,tsrc,bs):
 seed_all(SSL_SEED);o=CKPT/name;o.mkdir(parents=True,exist_ok=True);resume=o/'resume.pt'
 m=DualSSL(tsrc).to(DEVICE);pars=[p for p in m.parameters() if p.requires_grad]
 opt=torch.optim.AdamW(pars,lr=CONTRASTIVE_LR,weight_decay=.01);scaler=torch.cuda.amp.GradScaler(enabled=AMP)
 ds=ParquetIter(ROLE_DIR['SSL'],['url','text'],SSL_SEED,SSL_MAX_ROWS)
 def cc(rows):
  u=url_tok([str(r['url'] or '') for r in rows],padding=True,truncation=True,max_length=URL_MAX_LEN,return_tensors='pt')
  t=text_tok([str(r['text'] or '') for r in rows],padding=True,truncation=True,max_length=TEXT_MAX_LEN,return_tensors='pt')
  return u,t
 dl=DataLoader(ds,batch_size=bs,collate_fn=cc,num_workers=0);qu=qt=None;ls=[];start_ep=0;skip_batches=0

 if resume.exists():
  try:
   ck=torch.load(resume,map_location=DEVICE)
   if int(ck.get('batch',-1))==bs:
    m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
    qu=ck.get('qu');qt=ck.get('qt')
    if qu is not None:qu=qu.to(DEVICE)
    if qt is not None:qt=qt.to(DEVICE)
    start_ep=int(ck.get('epoch',0));skip_batches=int(ck.get('next_batch',0));ls=list(ck.get('losses',[]))
    print({name+'_RESUME':True,'epoch':start_ep,'next_batch':skip_batches})
  except Exception as e:
   print('Invalid resume ignored',name,repr(e));qu=qt=None;start_ep=0;skip_batches=0;ls=[]

 for ep in range(start_ep,CONTRASTIVE_EPOCHS):
  for j,(u,t) in enumerate(dl):
   if ep==start_ep and j<skip_batches:continue
   u={k:v.to(DEVICE) for k,v in u.items()};t={k:v.to(DEVICE) for k,v in t.items()};opt.zero_grad(set_to_none=True)
   with amp_ctx():zu,zt=m(u,t);l=contrastive_loss(zu,zt,qu,qt)
   scaler.scale(l).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(pars,1.);scaler.step(opt);scaler.update()
   qu=(zu.detach() if qu is None else torch.cat([qu,zu.detach()],0))[-QUEUE_SIZE:]
   qt=(zt.detach() if qt is None else torch.cat([qt,zt.detach()],0))[-QUEUE_SIZE:]
   ls.append(float(l.detach().cpu()));nxt=j+1
   if nxt%RESUME_EVERY_TRANSFORMER_STEPS==0:
    atomic_torch(resume,{'model':m.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),
                         'qu':None if qu is None else qu.detach().cpu(),'qt':None if qt is None else qt.detach().cpu(),
                         'epoch':ep,'next_batch':nxt,'batch':bs,'losses':ls[-500:]})
    print({name+'_checkpoint_batch':nxt,'loss':float(np.mean(ls[-100:]))})
  atomic_torch(resume,{'model':m.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),
                       'qu':None if qu is None else qu.detach().cpu(),'qt':None if qt is None else qt.detach().cpu(),
                       'epoch':ep+1,'next_batch':0,'batch':bs,'losses':ls[-500:]})
  skip_batches=0

 (o/'url').mkdir(parents=True,exist_ok=True);(o/'text').mkdir(parents=True,exist_ok=True)
 m.u.save_pretrained(o/'url');m.t.save_pretrained(o/'text');url_tok.save_pretrained(o/'url');text_tok.save_pretrained(o/'text')
 atomic_json(o/'COMPLETE.json',{'status':'COMPLETE','seed':SSL_SEED,'rows':SSL_MAX_ROWS,'batch':bs,'loss':float(np.mean(ls))})
 resume.unlink(missing_ok=True)
 del m,dl,ds,qu,qt;gc.collect();torch.cuda.empty_cache()

def ensure(name,tsrc):
 if (CKPT/name/'COMPLETE.json').exists():
  print('Reuse completed',name);return
 last=None
 for bs in CONTRASTIVE_BATCH_CANDIDATES:
  try:train_ssl(name,tsrc,bs);last=None;break
  except torch.cuda.OutOfMemoryError as e:
   last=e;gc.collect();torch.cuda.empty_cache()
 if last:raise last

ensure('R2_CONTRASTIVE',TEXT_SRC);ensure('R3_DAPT_CONTRASTIVE',str(DAPT_DIR));print('R2/R3 complete')


# CELL 15
# 06 — Representation embeddings (chunked memmap) + metrics/probes
REP={'R0':{'u':URL_SRC,'t':TEXT_SRC},'R1':{'u':URL_SRC,'t':str(DAPT_DIR)},'R2':{'u':str(CKPT/'R2_CONTRASTIVE'/'url'),'t':str(CKPT/'R2_CONTRASTIVE'/'text')},'R3':{'u':str(CKPT/'R3_DAPT_CONTRASTIVE'/'url'),'t':str(CKPT/'R3_DAPT_CONTRASTIVE'/'text')}}
@torch.no_grad()
def write_emb(rep,name,df,batch=64):
 out=EMB/f'{rep}_{name}.npy';shape=(len(df),REP_DIM)
 if valid_emb(out,shape):
  print('Reuse complete embedding',out.name);return

 prog=emb_progress(out);start_row=0
 if out.exists() and prog.exists():
  try:
   meta=json.loads(prog.read_text());a=np.load(out,mmap_mode='r')
   if tuple(a.shape)==shape and a.dtype==np.float16:start_row=int(meta.get('next_row',0))
   del a
  except:start_row=0

 if start_row==0:
  out.unlink(missing_ok=True);emb_done(out).unlink(missing_ok=True)
  mm=np.lib.format.open_memmap(out,mode='w+',dtype=np.float16,shape=shape);mm.flush();del mm
  atomic_json(prog,{'next_row':0,'shape':list(shape)})
 else:print({'resume_embedding':out.name,'next_row':start_row})

 um=AutoModel.from_pretrained(REP[rep]['u'],token=HF_TOKEN).to(DEVICE).eval()
 tm=AutoModel.from_pretrained(REP[rep]['t'],token=HF_TOKEN).to(DEVICE).eval()
 mm=np.load(out,mmap_mode='r+');last_mark=start_row

 for j in range(start_row,len(df),batch):
  x=df.iloc[j:j+batch]
  u=url_tok(x.url.fillna('').astype(str).tolist(),padding=True,truncation=True,max_length=URL_MAX_LEN,return_tensors='pt')
  t=text_tok(x.text.fillna('').astype(str).tolist(),padding=True,truncation=True,max_length=TEXT_MAX_LEN,return_tensors='pt')
  u={k:v.to(DEVICE) for k,v in u.items()};t={k:v.to(DEVICE) for k,v in t.items()}
  with amp_ctx():
   a=F.normalize(pooled(um(**u),u['attention_mask']),dim=-1)
   b=F.normalize(pooled(tm(**t),t['attention_mask']),dim=-1)
   z=torch.cat([a,b],-1)
  mm[j:j+len(x)]=z.float().cpu().numpy().astype(np.float16)
  nxt=j+len(x)
  if nxt-last_mark>=CACHE_PROGRESS_ROWS or nxt==len(df):
   mm.flush();atomic_json(prog,{'next_row':nxt,'shape':list(shape)});last_mark=nxt
   print({'embedding':out.name,'rows':nxt,'total':len(df)})

 mm.flush();del mm,um,tm;gc.collect();torch.cuda.empty_cache()
 atomic_json(emb_done(out),{'status':'COMPLETE','shape':list(shape),'dtype':'float16','rep':rep,'split':name})
 prog.unlink(missing_ok=True)

def thr_fpr(s,f):
 # Largest allowable number of benign scores at/above threshold, with tie-safe fallback.
 s=np.asarray(s,float);k=int(math.floor(f*len(s)))
 if k<=0:return float(np.nextafter(s.max(),np.inf))
 desc=np.sort(s)[::-1];boundary=float(desc[k-1])
 # If the boundary has no tie overflow, allow exactly k false positives.
 if int((s>=boundary).sum())<=k:return boundary
 # Otherwise move just above the tied boundary; realized FPR becomes conservative.
 return float(np.nextafter(boundary,np.inf))
def pauc(y,s,m):
 f,t,_=roc_curve(y,s);j=np.searchsorted(f,m,side='right');x=np.r_[f[:j],m];yy=np.r_[t[:j],np.interp(m,f,t)];return float(np.trapz(yy,x))
def precision_at_recall(y,s,target=.90):
 # Match the public PhreshPhish benchmark notebook:
 # reverse recall to ascending order, interpolate precision on a 1000-point recall grid,
 # then select the grid point closest to the requested recall.
 p,r,t=precision_recall_curve(y,s);ra=r[::-1];pa=p[::-1];grid=np.linspace(0.,1.,1000);pi=np.interp(grid,ra,pa);gi=int(np.argmin(np.abs(grid-target)));precision=float(pi[gi]);realized_recall=float(grid[gi])
 # Threshold is auxiliary only; PhreshPhish P@R itself is defined from the interpolated curve.
 j=int(np.argmin(np.abs(r[:-1]-realized_recall))) if len(t) else -1;th=float(t[j]) if j>=0 else float('nan')
 return {'precision':precision,'recall':realized_recall,'threshold':th}
def fpr_at_tpr(y,s,target):
 f,t,_=roc_curve(y,s);ok=np.where(t>=target)[0];return float(np.min(f[ok])) if len(ok) else float('nan')
def binom_ci(k,n,alpha=.05):
 if not n:return (float('nan'),float('nan'))
 lo=0. if k==0 else beta_dist.ppf(alpha/2,k,n-k+1);hi=1. if k==n else beta_dist.ppf(1-alpha/2,k+1,n-k)
 return float(lo),float(hi)
def op(y,s,th):
 p=s>=th;neg=y==0;pos=y==1;fp=int((p&neg).sum());tn=int((~p&neg).sum());tp=int((p&pos).sum());fn=int((~p&pos).sum())
 tpr=tp/max(tp+fn,1);fpr=fp/max(fp+tn,1);prec=tp/max(tp+fp,1);f1=2*prec*tpr/max(prec+tpr,1e-15);lo,hi=binom_ci(fp,fp+tn)
 return {'tpr':tpr,'fpr':fpr,'precision':prec,'f1':f1,'tp':tp,'fp':fp,'tn':tn,'fn':fn,'fp_per_1000':1000*fpr,'fpr_ci_lo':lo,'fpr_ci_hi':hi}
def curves(y,s):
 pr=precision_at_recall(y,s,.90);out={'AP':float(average_precision_score(y,s)),'AUC':float(roc_auc_score(y,s)),
 'P_at_R90':pr['precision'],'R_at_P_at_R90':pr['recall'],'threshold_P_at_R90':pr['threshold']}
 for m in PAUC_MAX:out[f'pAUC_{m}']=pauc(y,s,m)
 for t in TARGET_TPRS:out[f'FPR_at_TPR_{t}']=fpr_at_tpr(y,s,t)
 return out
class Probe(nn.Module):
 def __init__(self,d):super().__init__();self.n=nn.Sequential(nn.Linear(d,256),nn.GELU(),nn.Dropout(.2),nn.Linear(256,64),nn.GELU(),nn.Linear(64,1))
 def forward(self,x):return self.n(x).squeeze(-1)
def fit_probe(X,y,seed):
 seed_all(seed);m=Probe(X.shape[1]).to(DEVICE);opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-3);ds=TensorDataset(torch.tensor(X,dtype=torch.float32),torch.tensor(y,dtype=torch.float32));dl=DataLoader(ds,batch_size=min(256,len(ds)),shuffle=True,generator=torch.Generator().manual_seed(seed))
 for _ in range(35):
  for x,z in dl:x=x.to(DEVICE);z=z.to(DEVICE);opt.zero_grad(set_to_none=True);l=F.binary_cross_entropy_with_logits(m(x),z);l.backward();opt.step()
 return m.eval()
@torch.no_grad()
def pred_probe(m,X):
 o=[]
 for i in range(0,len(X),2048):o.append(torch.sigmoid(m(torch.tensor(X[i:i+2048],dtype=torch.float32,device=DEVICE))).cpu().numpy())
 return np.concatenate(o)
for r in REP:write_emb(r,'SUP',sup);write_emb(r,'DEV',dev)
print('SUP/DEV embeddings ready')

# CELL 16
# 07 — Scientific representation screen → ARCHITECTURE_FREEZE
rows=[];screen_seed=42;tr=LABELSETS[screen_seed]['B25R']
for r in REP:
 xs=np.load(EMB/f'{r}_SUP.npy',mmap_mode='r').astype(np.float32);xd=np.load(EMB/f'{r}_DEV.npy',mmap_mode='r').astype(np.float32)
 for typ in ['LINEAR','MLP']:
  if typ=='LINEAR':
   m=LogisticRegression(C=1,max_iter=1500,solver='liblinear',class_weight='balanced',random_state=screen_seed).fit(xs[tr],sy[tr]);sca=m.predict_proba(xd[PARTS['SCREEN_CAL']])[:,1];sev=m.predict_proba(xd[PARTS['SCREEN_EVAL']])[:,1]
  else:
   m=fit_probe(xs[tr],sy[tr],screen_seed);sca=pred_probe(m,xd[PARTS['SCREEN_CAL']]);sev=pred_probe(m,xd[PARTS['SCREEN_EVAL']]);del m;gc.collect();torch.cuda.empty_cache()
  yc=dy[PARTS['SCREEN_CAL']];ye=dy[PARTS['SCREEN_EVAL']];th=thr_fpr(sca[yc==0],PRIMARY_FPR);rows.append({'rep':r,'probe':typ,'threshold':th,**op(ye,sev,th),**curves(ye,sev)})
SCR=pd.DataFrame(rows);SCR.to_csv(RESULTS/'REPRESENTATION_SCREEN.csv',index=False);display(SCR)
lin=SCR[SCR.probe=='LINEAR'].sort_values(['tpr','P_at_R90','pAUC_0.005','AP'],ascending=False);CHAMP=str(lin.iloc[0].rep)
ARCH={'status':'FROZEN','selected_representation':CHAMP,'selection':'SCREEN only; max linear TPR@0.5%, tie P@R=.90, pAUC/AP','final_test_touched':False,'calibration_touched':False}
(AUDIT/'ARCHITECTURE_FREEZE.json').write_text(json.dumps(ARCH,indent=2));print(ARCH)

# CELL 17
# 08 — DOM vocabulary + masked DOM SSL + deep gated fusion
# DOM vocabulary is fitted ONLY on SSL/SUP/DEV, never calibration/final test.
DOM_VOCAB_PATH=AUDIT/'DOM_VOCAB.json'
if DOM_VOCAB_PATH.exists():DOM_VOCAB=json.loads(DOM_VOCAB_PATH.read_text())
else:
 cnt=Counter()
 for df in [sup,dev]:
  for x in df.dom_tag:
   if isinstance(x,(list,np.ndarray)):cnt.update(map(str,x[:DOM_MAX_NODES]))
 for p in sorted(ROLE_DIR['SSL'].glob('*.parquet')):
  d=pd.read_parquet(p,columns=['dom_tag','ssl_rank']);d=d[d.ssl_rank<SSL_MAX_ROWS]
  for x in d.dom_tag:
   if isinstance(x,(list,np.ndarray)):cnt.update(map(str,x[:DOM_MAX_NODES]))
 keep=[x for x,_ in cnt.most_common(DOM_VOCAB_MAX-3)];DOM_VOCAB={'<PAD>':0,'<UNK>':1,'<MASK>':2};DOM_VOCAB.update({x:i+3 for i,x in enumerate(keep)})
 DOM_VOCAB_PATH.write_text(json.dumps(DOM_VOCAB,indent=2))
PAD,UNK,MASK=0,1,2

def bucket(v):
 try:return min(max(int(v),0),31)
 except:return 0

def _rv(r,k):
 if isinstance(r,dict):return r.get(k)
 try:return r[k]
 except:return getattr(r,k,None)
def graph_row(r,mask_p=0.,rng=None):
 def arr(k):
  x=_rv(r,k)
  return x if isinstance(x,(list,np.ndarray)) else []
 tags=arr('dom_tag');par=arr('dom_parent_idx');dep=arr('dom_depth');att=arr('dom_attr_count');chi=arr('dom_child_count')
 n=min(len(tags),DOM_MAX_NODES);tag=np.array([DOM_VOCAB.get(str(x),UNK) for x in tags[:n]],np.int64);orig=tag.copy();depth=np.array([bucket(x) for x in dep[:n]],np.int64);attr=np.array([bucket(x) for x in att[:n]],np.int64);child=np.array([bucket(x) for x in chi[:n]],np.int64)
 edges=[]
 for ii,pp in enumerate(par[:n]):
  try:pp=int(pp)
  except:continue
  if ii>0 and 0<=pp<n:edges.extend([(pp,ii),(ii,pp)])
 masked=np.zeros(n,bool)
 if mask_p and n:
  rng=np.random.default_rng() if rng is None else rng;masked=rng.random(n)<mask_p
  if not masked.any():masked[rng.integers(0,n)]=True
  tag[masked]=MASK
 return tag,orig,depth,attr,child,np.asarray(edges,np.int64),masked

def graph_batch(rows,mask_p=0.,seed=None):
 rng=np.random.default_rng(seed);T=[];O=[];D=[];A=[];C=[];E=[];B=[];M=[];off=0
 for bi,r in enumerate(rows):
  t,o,d,a,c,e,m=graph_row(r,mask_p,rng);n=len(t);T.append(t);O.append(o);D.append(d);A.append(a);C.append(c);M.append(m);B.append(np.full(n,bi,np.int64));
  if len(e):E.append(e+off)
  off+=n
 def cat(xs):return torch.tensor(np.concatenate(xs) if xs else np.array([],np.int64),dtype=torch.long,device=DEVICE)
 edge=torch.tensor(np.concatenate(E,0).T if E else np.zeros((2,0),np.int64),dtype=torch.long,device=DEVICE)
 return {'tag':cat(T),'orig':cat(O),'depth':cat(D),'attr':cat(A),'child':cat(C),'edge':edge,'batch':cat(B),'mask':torch.tensor(np.concatenate(M) if M else [],dtype=torch.bool,device=DEVICE),'n_graphs':len(rows)}

class GCNLayer(nn.Module):
 def __init__(self,d):super().__init__();self.s=nn.Linear(d,d);self.n=nn.Linear(d,d);self.norm=nn.LayerNorm(d)
 def forward(self,x,e):
  if e.numel()==0:return self.norm(x+F.gelu(self.s(x)))
  src,dst=e;agg=torch.zeros_like(x);deg=torch.zeros((len(x),1),device=x.device,dtype=x.dtype);agg.index_add_(0,dst,x[src]);deg.index_add_(0,dst,torch.ones((len(dst),1),device=x.device,dtype=x.dtype));nei=agg/deg.clamp_min(1)
  return self.norm(x+F.gelu(self.s(x)+self.n(nei)))
class DOMEncoder(nn.Module):
 def __init__(self):
  super().__init__();self.tag=nn.Embedding(len(DOM_VOCAB),96,padding_idx=PAD);self.depth=nn.Embedding(32,16);self.attr=nn.Embedding(32,8);self.child=nn.Embedding(32,8);self.inp=nn.Linear(128,DOM_DIM);self.layers=nn.ModuleList([GCNLayer(DOM_DIM) for _ in range(DOM_LAYERS)]);self.att=nn.Linear(DOM_DIM,1)
 def nodes(self,g):
  x=self.inp(torch.cat([self.tag(g['tag']),self.depth(g['depth']),self.attr(g['attr']),self.child(g['child'])],1))
  for l in self.layers:x=l(x,g['edge'])
  return x
 def forward(self,g):
  x=self.nodes(g);n=g['n_graphs'];num=torch.zeros((n,DOM_DIM),device=x.device);den=torch.zeros((n,1),device=x.device);w=torch.exp(torch.clamp(self.att(x).squeeze(-1),-10,10));num.index_add_(0,g['batch'],x*w[:,None]);den.index_add_(0,g['batch'],w[:,None]);return num/den.clamp_min(1e-8)
class DOMMAE(nn.Module):
 def __init__(self):super().__init__();self.enc=DOMEncoder();self.dec=nn.Linear(DOM_DIM,len(DOM_VOCAB))
 def forward(self,g):return self.dec(self.enc.nodes(g))

DOM_DIR=CKPT/'DOM_MASKED_SSL';DOM_DONE=DOM_DIR/'COMPLETE.json';DOM_RESUME=DOM_DIR/'resume.pt'
DOM_SSL_COLS=['dom_tag','dom_parent_idx','dom_depth','dom_attr_count','dom_child_count']
if not DOM_DONE.exists():
 seed_all(SSL_SEED);DOM_DIR.mkdir(parents=True,exist_ok=True);m=DOMMAE().to(DEVICE);opt=torch.optim.AdamW(m.parameters(),lr=DOM_SSL_LR,weight_decay=1e-3);losses=[];start_ep=0;skip_step=0
 if DOM_RESUME.exists():
  try:
   ck=torch.load(DOM_RESUME,map_location=DEVICE);m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);start_ep=int(ck.get('epoch',0));skip_step=int(ck.get('next_step',0));losses=list(ck.get('losses',[]));print({'DOM_RESUME':True,'epoch':start_ep,'next_step':skip_step})
  except Exception as e:print('Invalid DOM resume ignored:',repr(e));start_ep=0;skip_step=0;losses=[]
 for ep in range(start_ep,DOM_SSL_EPOCHS):
  dsdom=ParquetIter(ROLE_DIR['SSL'],DOM_SSL_COLS,SSL_SEED+ep,SSL_MAX_ROWS)
  class _DOMCollate:
   def __init__(self,base):self.base=base;self.step=0
   def __call__(self,rows):
    g=graph_batch(rows,DOM_MASK_P,self.base+self.step);self.step+=1;return g
  dom_coll=_DOMCollate(SSL_SEED+ep*1000000)
  dl=DataLoader(dsdom,batch_size=DOM_SSL_BATCH,collate_fn=dom_coll,num_workers=0)
  step=0
  for g in dl:
   if ep==start_ep and step<skip_step:step+=1;continue
   # Re-mask deterministically from step by reconstructing the batch is expensive; the streamed
   # order is deterministic and crash-resume is operational rather than bitwise-identical.
   logits=m(g);mask=g['mask']
   if mask.any():
    loss=F.cross_entropy(logits[mask],g['orig'][mask]);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();losses.append(float(loss.detach().cpu()))
   step+=1
   if step%RESUME_EVERY_DOM_STEPS==0:
    atomic_torch(DOM_RESUME,{'model':m.state_dict(),'optimizer':opt.state_dict(),'epoch':ep,'next_step':step,'losses':losses[-500:]});print({'DOM_checkpoint_step':step,'loss':float(np.mean(losses[-100:]))})
  atomic_torch(DOM_RESUME,{'model':m.state_dict(),'optimizer':opt.state_dict(),'epoch':ep+1,'next_step':0,'losses':losses[-500:]});skip_step=0;del dl,dsdom;gc.collect();print({'DOM_SSL_epoch':ep+1,'loss':float(np.mean(losses[-100:]))})
 atomic_torch(DOM_DIR/'encoder.pt',m.enc.state_dict());atomic_json(DOM_DONE,{'status':'COMPLETE','rows':SSL_MAX_ROWS,'loss':float(np.mean(losses)),'vocab':len(DOM_VOCAB),'streamed':True});DOM_RESUME.unlink(missing_ok=True);del m;gc.collect();torch.cuda.empty_cache()
else:print('Reuse completed DOM SSL')
print(json.loads(DOM_DONE.read_text()))

def eng_sources(rep):
 if rep=='R0':return URL_SRC,TEXT_SRC
 if rep=='R1':return URL_SRC,str(DAPT_DIR)
 if rep=='R2':return str(CKPT/'R2_CONTRASTIVE'/'url'),str(CKPT/'R2_CONTRASTIVE'/'text')
 if rep=='R3':return str(CKPT/'R3_DAPT_CONTRASTIVE'/'url'),str(CKPT/'R3_DAPT_CONTRASTIVE'/'text')
 raise KeyError(rep)
ENG_U,ENG_T=eng_sources(CHAMP)

class DeepFusion(nn.Module):
 def __init__(self,use_dom):
  super().__init__();self.use_dom=use_dom;self.u=AutoModel.from_pretrained(ENG_U,token=HF_TOKEN);self.t=AutoModel.from_pretrained(ENG_T,token=HF_TOKEN);freeze_last(self.u,E2E_LAST_N);freeze_last(self.t,E2E_LAST_N);self.ua=nn.Linear(URL_H,PROJ_DIM);self.ta=nn.Linear(TEXT_H,PROJ_DIM);self.dom=DOMEncoder() if use_dom else None;
  if use_dom:self.dom.load_state_dict(torch.load(DOM_DIR/'encoder.pt',map_location='cpu'));
  self.da=nn.Linear(DOM_DIM,PROJ_DIM) if use_dom else None;self.gate=nn.Sequential(nn.Linear(PROJ_DIM,64),nn.GELU(),nn.Linear(64,1));nm=3 if use_dom else 2;self.head=nn.Sequential(nn.Linear(PROJ_DIM*(nm+1),512),nn.GELU(),nn.Dropout(.2),nn.Linear(512,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,1));self.uaux=nn.Linear(PROJ_DIM,1);self.taux=nn.Linear(PROJ_DIM,1);self.daux=nn.Linear(PROJ_DIM,1) if use_dom else None
 def forward(self,u,t,g=None):
  zu=F.normalize(self.ua(pooled(self.u(**u),u['attention_mask'])),dim=-1);zt=F.normalize(self.ta(pooled(self.t(**t),t['attention_mask'])),dim=-1);zs=[zu,zt]
  if self.use_dom:zs.append(F.normalize(self.da(self.dom(g)),dim=-1))
  st=torch.stack(zs,1);gw=torch.softmax(self.gate(st).squeeze(-1),1);weighted=(st*gw[:,:,None]).sum(1);main=self.head(torch.cat(zs+[weighted],1)).squeeze(-1);aux=[self.uaux(zs[0]).squeeze(-1),self.taux(zs[1]).squeeze(-1)]
  if self.use_dom:aux.append(self.daux(zs[2]).squeeze(-1))
  return main,aux,gw

def deep_batches(df,ids,bs,seed=None,use_dom=False):
 ids=np.asarray(ids).copy()
 if seed is not None:np.random.default_rng(seed).shuffle(ids)
 for st in range(0,len(ids),bs):
  ii=ids[st:st+bs];x=df.iloc[ii];u=url_tok(x.url.fillna('').astype(str).tolist(),padding=True,truncation=True,max_length=URL_MAX_LEN,return_tensors='pt');t=text_tok(x.text.fillna('').astype(str).tolist(),padding=True,truncation=True,max_length=TEXT_MAX_LEN,return_tensors='pt');u={k:v.to(DEVICE) for k,v in u.items()};t={k:v.to(DEVICE) for k,v in t.items()};g=graph_batch([x.iloc[j] for j in range(len(x))]) if use_dom else None;yield ii,u,t,g,torch.tensor(y01(x.label),dtype=torch.float32,device=DEVICE)
def _weighted_bce(logits,y,w=None):
 raw=F.binary_cross_entropy_with_logits(logits,y,reduction='none')
 if w is None:return raw.mean()
 w=w.to(raw.dtype)
 return (raw*w).sum()/w.sum().clamp_min(1e-12)

def deep_loss(main,aux,y,w=None):
 # Fair modality-count invariant objective:
 # total auxiliary contribution stays exactly AUX_TOTAL_WEIGHT
 # whether DUAL has 2 heads or TRI has 3.
 main_loss=_weighted_bce(main,y,w)
 if len(aux)==0:return main_loss
 aux_loss=torch.stack([_weighted_bce(a,y,w) for a in aux]).mean()
 return main_loss+AUX_TOTAL_WEIGHT*aux_loss

def _trainable_names(m):
 return {n for n,p in m.named_parameters() if p.requires_grad}

def trainable_overlay(m):
 names=_trainable_names(m);sd=m.state_dict()
 return {k:v.detach().cpu().clone() for k,v in sd.items() if k in names}

def load_trainable_overlay(m,state):
 missing,unexpected=m.load_state_dict(state,strict=False)
 if unexpected:raise RuntimeError(f'STOP overlay unexpected keys: {unexpected[:10]}')
 need=_trainable_names(m)
 missing_trainable=sorted(need & set(missing))
 if missing_trainable:raise RuntimeError(f'STOP overlay missing trainable keys: {missing_trainable[:10]}')
 return m

def model_param_audit(m):
 return {
  'total_params':int(sum(p.numel() for p in m.parameters())),
  'trainable_params':int(sum(p.numel() for p in m.parameters() if p.requires_grad))
 }
@torch.no_grad()
def pred_deep(m,use_dom,df,ids,bs):
 m.eval();scores=[];urls=[];gates=[]
 for _,u,t,g,y in deep_batches(df,ids,bs,None,use_dom):
  main,aux,gw=m(u,t,g);scores.append(torch.sigmoid(main).float().cpu().numpy());urls.append(torch.sigmoid(aux[0]).float().cpu().numpy());gates.append(gw.float().cpu().numpy())
 return np.concatenate(scores),{'url':np.concatenate(urls),'gates':np.concatenate(gates)}


# CELL 18
# 09 — FINAL FAIR Engineering: DUAL/TRI + HARDNEG
# New objective id prevents any reuse of the confounded v4.3 engineering runs.

ET=PARTS['ENG_TUNE'];EM=PARTS['ENG_META'];yt=dy[ET];ym=dy[EM]
ENG_RUN_ROOT=CKPT/f'ENG_{ENGINEERING_OBJECTIVE_ID}'
ENG_RUN_ROOT.mkdir(parents=True,exist_ok=True)
SELECTED_DIR=CKPT/f'ENG_SELECTED_{ENGINEERING_OBJECTIVE_ID}'
SELECTED_DIR.mkdir(parents=True,exist_ok=True)

def rank_tuple(y,s,threshold=None):
 c=curves(y,s);o=op(y,s,threshold) if threshold is not None else {}
 return (o.get('tpr',float('nan')),c['P_at_R90'],c['AP']),{'threshold':threshold,**o,**c}

def gate_summary(g,use_dom,prefix):
 g=np.asarray(g,float)
 out={
  f'{prefix}_gate_url_mean':float(g[:,0].mean()),
  f'{prefix}_gate_text_mean':float(g[:,1].mean()),
 }
 if use_dom and g.shape[1]>=3:
  out[f'{prefix}_gate_dom_mean']=float(g[:,2].mean())
 else:
  out[f'{prefix}_gate_dom_mean']=np.nan
 return out

def _fresh_deep(use_dom):
 return DeepFusion(use_dom).to(DEVICE)

def _base_split(seed):
 allidx=np.arange(len(sup))
 tr,mine=train_test_split(allidx,test_size=.20,stratify=sy,random_state=seed+7000)
 return np.asarray(tr),np.asarray(mine)

def prepare_base80(use_dom,arch,seed,bs,sd):
 base_done=sd/'BASE80_COMPLETE.json'
 base_overlay=sd/'BASE80_overlay.pt'
 hard_file=sd/'hard_indices.npy'
 resume=sd/'BASE80_RESUME.pt'
 tr,mine=_base_split(seed)

 if base_done.exists() and base_overlay.exists() and hard_file.exists():
  info=json.loads(base_done.read_text())
  hard=np.load(hard_file).astype(np.int64)
  print({'reuse_base80':f'{arch}_{seed}','hard_n':len(hard)})
  return base_overlay,hard,info

 seed_all(seed)
 m=_fresh_deep(use_dom)
 pars=[p for p in m.parameters() if p.requires_grad]
 opt=torch.optim.AdamW(pars,lr=E2E_LR,weight_decay=.01)
 scaler=torch.cuda.amp.GradScaler(enabled=AMP)
 start_ep=0

 if resume.exists():
  try:
   ck=torch.load(resume,map_location='cpu')
   load_trainable_overlay(m,ck['overlay'])
   opt.load_state_dict(ck['optimizer'])
   start_ep=int(ck.get('next_epoch',0))
   print({'resume_base80':f'{arch}_{seed}','next_epoch':start_ep})
  except Exception as e:
   print({'invalid_base80_resume_ignored':repr(e)})
   start_ep=0

 for ep in range(start_ep,E2E_EPOCHS):
  m.train()
  for _,u,t,g,y in deep_batches(sup,tr,bs,seed+ep,use_dom):
   opt.zero_grad(set_to_none=True)
   with amp_ctx():
    main,aux,_=m(u,t,g);loss=deep_loss(main,aux,y)
   scaler.scale(loss).backward()
   scaler.unscale_(opt)
   torch.nn.utils.clip_grad_norm_(pars,1)
   scaler.step(opt);scaler.update()
  atomic_torch(resume,{
   'overlay':trainable_overlay(m),
   'optimizer':opt.state_dict(),
   'next_epoch':ep+1,
   'arch':arch,'seed':seed,'batch':bs,
   'objective':ENGINEERING_OBJECTIVE_ID
  })
  print({'engineering_base80_epoch_complete':f'{arch}_{seed}','epoch':ep+1})

 sm,_=pred_deep(m,use_dom,sup,mine,bs)
 yy=sy[mine];ben=np.where(yy==0)[0]
 cut=float(np.quantile(sm[ben],HARDNEG_QUANTILE))
 hard=np.asarray(mine)[ben[sm[ben]>=cut]].astype(np.int64)

 atomic_torch(base_overlay,trainable_overlay(m))
 np.save(hard_file,hard)
 info={
  'status':'COMPLETE','arch':arch,'seed':seed,'batch':bs,
  'objective':ENGINEERING_OBJECTIVE_ID,
  'hard_benign_n':int(len(hard)),'hard_cut':cut,
  **model_param_audit(m)
 }
 atomic_json(base_done,info)
 resume.unlink(missing_ok=True)
 del m,opt,scaler;gc.collect();torch.cuda.empty_cache()
 return base_overlay,hard,info

def train_mode_from_base(use_dom,arch,mode,seed,bs,sd,base_overlay,hard,base_info):
 done=sd/f'{mode}_COMPLETE.json'
 overlay_file=sd/f'{mode}_overlay.pt'

 if done.exists():
  row=json.loads(done.read_text())['row']
  print({'reuse_fair_engineering_mode':f'{arch}_{seed}_{mode}'})
  return row,overlay_file if overlay_file.exists() else None

 seed_all(seed+9000)
 m=_fresh_deep(use_dom)
 load_trainable_overlay(m,torch.load(base_overlay,map_location='cpu'))
 opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=E2E_LR,weight_decay=.01)
 scaler=torch.cuda.amp.GradScaler(enabled=AMP)
 allidx=np.arange(len(sup))
 hardset=set(int(x) for x in np.asarray(hard).tolist())
 m.train()

 for ep in range(E2E_HARDNEG_EPOCHS):
  for ii,u,t,g,y in deep_batches(sup,allidx,bs,seed+9000+ep,use_dom):
   w=None
   if mode=='HARDNEG':
    w=torch.ones(len(ii),dtype=torch.float32,device=DEVICE)
    for j,k in enumerate(ii):
     if int(k) in hardset:w[j]=HARDNEG_WEIGHT
   opt.zero_grad(set_to_none=True)
   with amp_ctx():
    main,aux,_=m(u,t,g);loss=deep_loss(main,aux,y,w)
   scaler.scale(loss).backward()
   scaler.unscale_(opt)
   torch.nn.utils.clip_grad_norm_(m.parameters(),1)
   scaler.step(opt);scaler.update()

 # ENG_TUNE fixes the threshold. ENG_META evaluates the frozen threshold.
 st,ext=pred_deep(m,use_dom,dev,ET,bs)
 th=thr_fpr(st[yt==0],PRIMARY_FPR)
 mt,exm=pred_deep(m,use_dom,dev,EM,bs)
 _,det_t=rank_tuple(yt,st,th)
 _,det_m=rank_tuple(ym,mt,th)

 row={
  'arch':arch,'mode':mode,'seed':seed,'batch':bs,
  'engineering_objective':ENGINEERING_OBJECTIVE_ID,
  'aux_total_weight':AUX_TOTAL_WEIGHT,
  'hardnegative_weight':HARDNEG_WEIGHT if mode=='HARDNEG' else 1.0,
  'hard_benign_n':base_info['hard_benign_n'],
  'hard_cut':base_info['hard_cut'],
  'total_params':base_info['total_params'],
  'trainable_params':base_info['trainable_params'],
  **gate_summary(ext['gates'],use_dom,'tune'),
  **gate_summary(exm['gates'],use_dom,'meta'),
  **{f'tune_{k}':v for k,v in det_t.items()},
  **{f'meta_{k}':v for k,v in det_m.items()}
 }

 atomic_torch(overlay_file,trainable_overlay(m))
 atomic_json(done,{'status':'COMPLETE','row':row})
 del m,opt,scaler;gc.collect();torch.cuda.empty_cache()
 print({'saved_fair_engineering_mode':f'{arch}_{seed}_{mode}'})
 return row,overlay_file

# ------------------------------------------------------------------
# Train/reuse all 6 arch×seed candidates and both modes.
# A fully completed seed never re-enters BASE80 on resume.
# ------------------------------------------------------------------
rows=[]
overlay_lookup={}
for use_dom,arch in [(False,'DUAL'),(True,'TRI')]:
 for seed in ENGINEERING_SEEDS:
  sd=ENG_RUN_ROOT/f'{arch}_seed{seed}'
  sd.mkdir(parents=True,exist_ok=True)
  completed_modes={mode:(sd/f'{mode}_COMPLETE.json') for mode in ['REGULAR','HARDNEG']}

  if all(p.exists() for p in completed_modes.values()):
   for mode,pdone in completed_modes.items():
    obj=json.loads(pdone.read_text())
    if obj.get('status')!='COMPLETE':
     raise RuntimeError(f'STOP invalid engineering COMPLETE marker: {pdone}')
    row=obj['row']
    if row.get('engineering_objective')!=ENGINEERING_OBJECTIVE_ID:
     raise RuntimeError(f'STOP stale engineering objective in {pdone}')
    rows.append(row)
    ov=sd/f'{mode}_overlay.pt'
    if ov.exists():overlay_lookup[(arch,mode,seed)]=ov
   print({'reuse_complete_fair_engineering_seed':f'{arch}_{seed}'})
   continue

  err=None
  for bs in E2E_BATCH_CANDIDATES:
   try:
    base_overlay,hard,binfo=prepare_base80(use_dom,arch,seed,bs,sd)
    err=None
    break
   except torch.cuda.OutOfMemoryError as e:
    err=e;gc.collect();torch.cuda.empty_cache()
  if err:raise err

  for mode in ['REGULAR','HARDNEG']:
   row,ov=train_mode_from_base(use_dom,arch,mode,seed,bs,sd,base_overlay,hard,binfo)
   rows.append(row)
   if ov is not None:overlay_lookup[(arch,mode,seed)]=ov

  if all((sd/f'{mode}_COMPLETE.json').exists() for mode in ['REGULAR','HARDNEG']):
   (sd/'BASE80_overlay.pt').unlink(missing_ok=True)
   (sd/'hard_indices.npy').unlink(missing_ok=True)
   (sd/'BASE80_RESUME.pt').unlink(missing_ok=True)

TUNE=pd.DataFrame(rows)
TUNE.to_csv(RESULTS/'DEEP_ENGINEERING_FAIR_SEED_RESULTS.csv',index=False)

# Independent ENG_META aggregation.
AGG=TUNE.groupby(['arch','mode']).agg(
 n_seeds=('seed','count'),
 meta_tpr_mean=('meta_tpr','mean'),
 meta_tpr_sd=('meta_tpr','std'),
 meta_fpr_mean=('meta_fpr','mean'),
 meta_precision_mean=('meta_precision','mean'),
 meta_P_at_R90_mean=('meta_P_at_R90','mean'),
 meta_AP_mean=('meta_AP','mean'),
 meta_FPR_at_TPR90_mean=('meta_FPR_at_TPR_0.9','mean'),
 meta_gate_url_mean=('meta_gate_url_mean','mean'),
 meta_gate_text_mean=('meta_gate_text_mean','mean'),
 meta_gate_dom_mean=('meta_gate_dom_mean','mean'),
 total_params=('total_params','max'),
 trainable_params=('trainable_params','max'),
).reset_index()
AGG.to_csv(RESULTS/'DEEP_ENGINEERING_FAIR_META_AGGREGATE.csv',index=False)
display(AGG)

# DOM gate isolates architectural effect using REGULAR only.
dr=AGG[(AGG.arch=='DUAL')&(AGG['mode']=='REGULAR')].iloc[0]
tr=AGG[(AGG.arch=='TRI')&(AGG['mode']=='REGULAR')].iloc[0]
delta=100*(tr.meta_tpr_mean-dr.meta_tpr_mean)
fprgood=(
 np.isfinite(tr.meta_FPR_at_TPR90_mean) and
 np.isfinite(dr.meta_FPR_at_TPR90_mean) and
 tr.meta_FPR_at_TPR90_mean<=DOM_GATE_REL_FPR90*dr.meta_FPR_at_TPR90_mean
)
USE_DOM=bool(delta>=DOM_GATE_MIN_PP or fprgood)
SELECTED_ARCH='TRI' if USE_DOM else 'DUAL'

# Within accepted architecture choose mode on seed-aggregated ENG_META.
pool=AGG[AGG.arch==SELECTED_ARCH].sort_values(
 ['meta_tpr_mean','meta_P_at_R90_mean','meta_AP_mean'],
 ascending=False
)
SELECTED_MODE=str(pool.iloc[0]['mode'])

# Seed is chosen only on ENG_TUNE AFTER architecture/mode are frozen.
sp=TUNE[(TUNE.arch==SELECTED_ARCH)&(TUNE['mode']==SELECTED_MODE)].sort_values(
 ['tune_tpr','tune_P_at_R90','tune_AP'],
 ascending=False
)
SELECTED_SEED=int(sp.iloc[0].seed)
SELECTED_BATCH=int(sp.iloc[0].batch)

selected_overlay=overlay_lookup.get((SELECTED_ARCH,SELECTED_MODE,SELECTED_SEED))
if selected_overlay is None or not Path(selected_overlay).exists():
 selected_overlay=ENG_RUN_ROOT/f'{SELECTED_ARCH}_seed{SELECTED_SEED}'/f'{SELECTED_MODE}_overlay.pt'

selected_meta_path=SELECTED_DIR/'meta.json'
selected_state_path=SELECTED_DIR/'model_state.pt'
reuse_selected=False
if selected_meta_path.exists() and selected_state_path.exists():
 try:
  prev=json.loads(selected_meta_path.read_text())
  reuse_selected=(
   prev.get('objective')==ENGINEERING_OBJECTIVE_ID and
   str(prev.get('arch'))==str(SELECTED_ARCH) and
   str(prev.get('mode'))==str(SELECTED_MODE) and
   int(prev.get('seed',-1))==int(SELECTED_SEED)
  )
 except Exception:
  reuse_selected=False

use_dom=SELECTED_ARCH=='TRI'
m=_fresh_deep(use_dom)
if reuse_selected:
 m.load_state_dict(torch.load(selected_state_path,map_location='cpu'))
 print({'REUSE_SELECTED_FAIR_MODEL':'PASS','arch':SELECTED_ARCH,'mode':SELECTED_MODE,'seed':SELECTED_SEED})
else:
 if not Path(selected_overlay).exists():
  raise RuntimeError(f'STOP selected fair-engineering overlay unavailable: {selected_overlay}')
 load_trainable_overlay(m,torch.load(selected_overlay,map_location='cpu'))
 atomic_torch(selected_state_path,m.state_dict())
 atomic_torch(SELECTED_DIR/'model_overlay.pt',trainable_overlay(m))
 atomic_json(selected_meta_path,{
  'arch':SELECTED_ARCH,'mode':SELECTED_MODE,'seed':SELECTED_SEED,
  'batch':SELECTED_BATCH,'objective':ENGINEERING_OBJECTIVE_ID,
  'aux_total_weight':AUX_TOTAL_WEIGHT
 })

# Selected development scores for cascade gate.
s_et,e_et=pred_deep(m,use_dom,dev,ET,SELECTED_BATCH)
s_em,e_em=pred_deep(m,use_dom,dev,EM,SELECTED_BATCH)

# After standalone selected model exists, non-selected large overlays may be deleted.
for ov in ENG_RUN_ROOT.rglob('*_overlay.pt'):
 if ov.resolve()!=Path(selected_overlay).resolve():
  ov.unlink(missing_ok=True)

atomic_json(AUDIT/'FAIR_ENGINEERING_FREEZE.json',{
 'status':'FROZEN_BEFORE_CAL_FINAL',
 'objective':ENGINEERING_OBJECTIVE_ID,
 'aux_total_weight':AUX_TOTAL_WEIGHT,
 'hardnegative_weight':HARDNEG_WEIGHT,
 'selected_arch':SELECTED_ARCH,
 'selected_mode':SELECTED_MODE,
 'selected_seed':SELECTED_SEED,
 'DOM_regular_mean_delta_pp':float(delta),
 'DOM_gate':USE_DOM,
 'DOM_gate_fpr90_condition':bool(fprgood),
 'selection_data':'Development only'
})

print({
 'FAIR_ENGINEERING':'COMPLETE',
 'DOM_gate':USE_DOM,
 'DOM_regular_mean_delta_pp':float(delta),
 'selected_arch':SELECTED_ARCH,
 'selected_mode':SELECTED_MODE,
 'final_seed_from_ENG_TUNE':SELECTED_SEED,
 'objective':ENGINEERING_OBJECTIVE_ID
})


# CELL 19
# 10 — URL-first cascade gate + SYSTEM_FREEZE (ENG_TUNE -> ENG_META)
# Full-system operating threshold and URL routing thresholds come exclusively from ENG_TUNE.
full_thr=thr_fpr(s_et[yt==0],PRIMARY_FPR);high=thr_fpr(e_et['url'][yt==0],.001);ps=np.sort(e_et['url'][yt==1]);k=max(1,int(math.floor(.01*len(ps))));low=float(np.nextafter(ps[k-1],-np.inf))
# Independent gate on ENG_META.
fullop=op(ym,s_em,full_thr);esc=(e_em['url']>low)&(e_em['url']<high);cas=s_em.copy();cas[e_em['url']>=high]=1.;cas[e_em['url']<=low]=0.;casop=op(ym,cas,full_thr);losspp=100*(fullop['tpr']-casop['tpr']);USE_CASCADE=bool(losspp<=CASCADE_MAX_TPR_LOSS_PP and esc.mean()<=1-CASCADE_MIN_REDUCTION and casop['fpr']<=fullop['fpr']+.001)
# Development latency of selected full deep path.
lat_ids=EM[:min(512,len(EM))];vals=[]
for _ in range(3):
 torch.cuda.synchronize();t0=time.perf_counter();_=pred_deep(m,SELECTED_ARCH=='TRI',dev,lat_ids,SELECTED_BATCH);torch.cuda.synchronize();vals.append(time.perf_counter()-t0)
lat_ms=1000*float(np.median(vals))/len(lat_ids);model_bytes=(SELECTED_DIR/'model_state.pt').stat().st_size
CAS={'use':USE_CASCADE,'threshold_source':'ENG_TUNE','evaluation_source':'ENG_META','url_low':low,'url_high':high,'dev_meta_escalation_rate':float(esc.mean()),'tpr_loss_pp':losspp,'full_meta':fullop,'cascade_meta':casop}
SYSTEM={'status':'FROZEN','selected_representation':CHAMP,'architecture':SELECTED_ARCH,'hardnegative':SELECTED_MODE=='HARDNEG','selected_seed':SELECTED_SEED,'dom_ssl_used':SELECTED_ARCH=='TRI','cascade':CAS,'latency_dev_ms_page':lat_ms,'model_state_bytes':model_bytes,'selection_data':'DEVELOPMENT only; thresholds on ENG_TUNE, gates on ENG_META','engineering_objective':ENGINEERING_OBJECTIVE_ID,'aux_total_weight':AUX_TOTAL_WEIGHT,'calibration_touched':False,'final_test_touched':False}
(AUDIT/'SYSTEM_FREEZE.json').write_text(json.dumps(SYSTEM,indent=2));(RESULTS/'CASCADE_GATE.json').write_text(json.dumps(CAS,indent=2));print(json.dumps(SYSTEM,indent=2))


# CELL 20
# 11 — FINAL PHASE: load calibration + official test only after SYSTEM_FREEZE
if not (AUDIT/'SYSTEM_FREEZE.json').exists():raise RuntimeError('STOP system freeze missing')
corecols=['sha256','url','text','label','date','domain','template_hash','dom_tag','dom_parent_idx','dom_depth','dom_attr_count','dom_child_count','dom_node_count','dom_max_depth','struct_token_count']
cal=read_role('CAL',corecols);final=read_role('FINAL',corecols);cy=y01(cal.label);fy=y01(final.label)
if len(cal)!=50000 or int(cy.sum())!=0:raise RuntimeError('STOP calibration role')
if len(final)!=168060 or (int((fy==0).sum()),int((fy==1).sum()))!=(91260,76800):raise RuntimeError(f'STOP final class counts: benign={(fy==0).sum()}, phish={(fy==1).sum()}')
if final.sha256.astype(str).duplicated().any():raise RuntimeError('STOP duplicate SHA inside FINAL')
ood_runtime_path=MANIFEST_ROOT/'final_test_exact_ood_flags_SEALED.parquet'
if not ood_runtime_path.exists():raise RuntimeError(f'STOP OOD manifest disappeared after startup preflight: {ood_runtime_path}')
print({'FINAL_METADATA_SOURCE':str(MANIFEST_ROOT),'OOD_FLAGS':str(ood_runtime_path)})
flags=pd.read_parquet(ood_runtime_path);flags['sha256']=flags.sha256.astype(str);final['sha256']=final.sha256.astype(str)
if len(flags)!=len(final) or set(flags.sha256)!=set(final.sha256):raise RuntimeError('STOP OOD manifest FINAL mismatch')
flags=flags.set_index('sha256').loc[final.sha256].reset_index();dt=pd.to_datetime(final.date,errors='coerce');cut=dt.dropna().quantile(.75) if dt.notna().sum() else None;late=(dt>=cut).fillna(False).to_numpy() if cut is not None else np.zeros(len(final),bool)
MASKS={'OFFICIAL_TEST':np.ones(len(final),bool),'DOMAIN_OOD_EXACT':flags.domain_ood_exact.to_numpy(bool),'TEMPLATE_OOD_EXACT':flags.template_ood_exact.to_numpy(bool),'DOMAIN_TEMPLATE_OOD_EXACT':flags.domain_template_ood_exact.to_numpy(bool),'LATE_TEST_Q4':late}
(AUDIT/'FINAL_SCENARIOS.json').write_text(json.dumps({'late_cut':None if cut is None else str(cut),'scenarios':{k:{'n':int(v.sum()),'benign':int(((fy==0)&v).sum()),'phish':int(((fy==1)&v).sum())} for k,v in MASKS.items()}},indent=2))
for r in REP:write_emb(r,'CAL',cal);write_emb(r,'FINAL',final)
print({'CAL':len(cal),'FINAL':len(final),'FINAL_LABELS':{'benign':int((fy==0).sum()),'phish':int((fy==1).sum())}})


# CELL 21
# 12 — Scientific N10 final evaluation + paired stats (per-combination resume)
PARTDIR=RESULTS/'science_parts';PARTDIR.mkdir(parents=True,exist_ok=True)
for r in REP:
 xs=np.load(EMB/f'{r}_SUP.npy',mmap_mode='r').astype(np.float32);xc=np.load(EMB/f'{r}_CAL.npy',mmap_mode='r').astype(np.float32);xf=np.load(EMB/f'{r}_FINAL.npy',mmap_mode='r').astype(np.float32)
 for seed in DOWNSTREAM_SEEDS:
  for bgt in LABEL_BUDGETS:
   ii=LABELSETS[seed][bgt]
   for typ in ['LINEAR','MLP']:
    part=PARTDIR/f'{r}_{typ}_{bgt}_seed{seed}.csv'
    if part.exists():continue
    if typ=='LINEAR':model=LogisticRegression(C=1,max_iter=1500,solver='liblinear',class_weight='balanced',random_state=seed).fit(xs[ii],sy[ii]);sc=model.predict_proba(xc)[:,1];sf=model.predict_proba(xf)[:,1]
    else:model=fit_probe(xs[ii],sy[ii],seed);sc=pred_probe(model,xc);sf=pred_probe(model,xf);del model;gc.collect();torch.cuda.empty_cache()
    rr=[]
    for f in TARGET_FPRS:
     th=thr_fpr(sc,f);allowed=int(math.floor(f*len(sc)));limited=allowed<20
     for sn,mask in MASKS.items():
      if mask.sum() and len(np.unique(fy[mask]))==2:rr.append({'rep':r,'probe':typ,'seed':seed,'budget':bgt,'scenario':sn,'target_fpr':f,'calibration_allowed_fp':allowed,'resolution_limited_calibration':limited,'threshold':th,**op(fy[mask],sf[mask],th)})
    for sn,mask in MASKS.items():
     if mask.sum() and len(np.unique(fy[mask]))==2:rr.append({'rep':r,'probe':typ,'seed':seed,'budget':bgt,'scenario':sn,'target_fpr':np.nan,'calibration_allowed_fp':np.nan,'resolution_limited_calibration':False,'threshold':np.nan,**curves(fy[mask],sf[mask])})
    tmp=Path(str(part)+'.tmp');pd.DataFrame(rr).to_csv(tmp,index=False);os.replace(tmp,part);print({'SCI_PART_COMPLETE':part.name})
# Assemble all completed parts.
parts=sorted(PARTDIR.glob('*.csv'));expected_parts=len(REP)*len(DOWNSTREAM_SEEDS)*len(LABEL_BUDGETS)*2
if len(parts)!=expected_parts:raise RuntimeError(f'STOP scientific parts {len(parts)}/{expected_parts}')
SCI=pd.concat([pd.read_csv(p) for p in parts],ignore_index=True);SCI.to_csv(RESULTS/'SCIENTIFIC_N10_ALL_METRICS.csv',index=False);pri=SCI[(SCI.scenario=='OFFICIAL_TEST')&(np.isclose(SCI.target_fpr,PRIMARY_FPR,rtol=0,atol=1e-12))];SUM=pri.groupby(['rep','probe','budget']).agg(tpr_mean=('tpr','mean'),tpr_sd=('tpr','std'),fpr_mean=('fpr','mean'),precision_mean=('precision','mean')).reset_index();SUM.to_csv(RESULTS/'SCIENTIFIC_N10_PRIMARY_SUMMARY.csv',index=False);display(SUM)
def signp(d):
 d=np.asarray(d);obs=abs(d.mean());n=len(d);return float(np.mean([abs((d*np.array([1 if mask>>i&1 else -1 for i in range(n)])).mean())>=obs-1e-15 for mask in range(1<<n)]))
from scipy.stats import t as t_dist
def mean_ci95(d):
 d=np.asarray(d,float);m=float(d.mean());h=float(t_dist.ppf(.975,len(d)-1)*d.std(ddof=1)/math.sqrt(len(d))) if len(d)>1 else float('nan');return m,m-h,m+h
# ------------------------------------------------------------------
# Paired SSL-vs-R0 statistics at primary FPR across ALL final scenarios.
# Scope remains downstream-seed variability conditional on fixed SSL checkpoints.
# ------------------------------------------------------------------
PRIMARY=SCI[np.isclose(SCI.target_fpr.fillna(-1),PRIMARY_FPR,rtol=0,atol=1e-12)].copy()
RANK=SCI[SCI.target_fpr.isna()].copy()

def signp(d):
 d=np.asarray(d,float);obs=abs(d.mean());n=len(d)
 return float(np.mean([
  abs((d*np.array([1 if mask>>i&1 else -1 for i in range(n)])).mean())>=obs-1e-15
  for mask in range(1<<n)
 ]))

from scipy.stats import t as t_dist
def mean_ci95(d):
 d=np.asarray(d,float);m=float(d.mean())
 h=float(t_dist.ppf(.975,len(d)-1)*d.std(ddof=1)/math.sqrt(len(d))) if len(d)>1 else float('nan')
 return m,m-h,m+h

st=[]
for sn in sorted(PRIMARY.scenario.unique()):
 for typ in ['LINEAR','MLP']:
  for bgt in LABEL_BUDGETS:
   b=PRIMARY[
    (PRIMARY.rep=='R0')&(PRIMARY.probe==typ)&(PRIMARY.budget==bgt)&(PRIMARY.scenario==sn)
   ].set_index('seed').tpr
   for r in ['R1','R2','R3']:
    a=PRIMARY[
     (PRIMARY.rep==r)&(PRIMARY.probe==typ)&(PRIMARY.budget==bgt)&(PRIMARY.scenario==sn)
    ].set_index('seed').tpr
    seeds=sorted(set(a.index)&set(b.index))
    d=np.array([a.loc[z]-b.loc[z] for z in seeds],float)
    md,lo,hi=mean_ci95(d)
    st.append({
     'scenario':sn,'contrast':r+'-R0','rep':r,'probe':typ,'budget':bgt,
     'n':len(d),'mean_diff_pp':100*md,'ci95_lo_pp':100*lo,'ci95_hi_pp':100*hi,
     'wins':int((d>0).sum()),'ties':int((d==0).sum()),
     'p_signflip':signp(d),
     'scope':'downstream seeds conditional on fixed SSL checkpoint'
    })

STAT=pd.DataFrame(st)
# Holm over the complete pre-defined family of primary-FPR SSL-vs-R0 comparisons.
order=np.argsort(STAT.p_signflip.to_numpy())
adj=np.empty(len(STAT),float);run=0.
for rank,j in enumerate(order):
 run=max(run,(len(STAT)-rank)*float(STAT.p_signflip.iloc[j]))
 adj[j]=min(1.,run)
STAT['p_holm_global']=adj
STAT.to_csv(RESULTS/'SCIENTIFIC_N10_PAIRED_STATS_ALL_SCENARIOS.csv',index=False)

# Aggregate primary operating point and ranking metrics.
PAGG=PRIMARY.groupby(['rep','probe','budget','scenario']).agg(
 tpr_mean=('tpr','mean'),tpr_sd=('tpr','std'),
 fpr_mean=('fpr','mean'),precision_mean=('precision','mean'),
 f1_mean=('f1','mean'),fp_per_1000_mean=('fp_per_1000','mean')
).reset_index()

RAGG=RANK.groupby(['rep','probe','budget','scenario']).agg(
 AP_mean=('AP','mean'),AP_sd=('AP','std'),
 P_at_R90_mean=('P_at_R90','mean'),P_at_R90_sd=('P_at_R90','std'),
 FPR_at_TPR90_mean=('FPR_at_TPR_0.9','mean'),
 FPR_at_TPR95_mean=('FPR_at_TPR_0.95','mean')
).reset_index()

FF12=PAGG.merge(RAGG,on=['rep','probe','budget','scenario'],how='left')

# Mean delta to R0 within same probe/budget/scenario.
base=FF12[FF12.rep=='R0'][['probe','budget','scenario','tpr_mean','AP_mean','P_at_R90_mean']].rename(
 columns={
  'tpr_mean':'R0_tpr_mean',
  'AP_mean':'R0_AP_mean',
  'P_at_R90_mean':'R0_P_at_R90_mean'
 }
)
FF12=FF12.merge(base,on=['probe','budget','scenario'],how='left')
FF12['delta_tpr_vs_R0_pp']=100*(FF12.tpr_mean-FF12.R0_tpr_mean)
FF12['delta_AP_vs_R0']=FF12.AP_mean-FF12.R0_AP_mean
FF12['delta_P_at_R90_vs_R0']=FF12.P_at_R90_mean-FF12.R0_P_at_R90_mean
FF12.to_csv(RESULTS/'RQ_FF1_FF2_REPRESENTATION_LABEL_PROBE.csv',index=False)

# FF3: paired degradation of each scenario versus OFFICIAL_TEST.
gen=[]
for r in REP:
 for typ in ['LINEAR','MLP']:
  for bgt in LABEL_BUDGETS:
   off=PRIMARY[
    (PRIMARY.rep==r)&(PRIMARY.probe==typ)&(PRIMARY.budget==bgt)&
    (PRIMARY.scenario=='OFFICIAL_TEST')
   ].set_index('seed').tpr
   for sn in sorted(set(PRIMARY.scenario)-{'OFFICIAL_TEST'}):
    q=PRIMARY[
     (PRIMARY.rep==r)&(PRIMARY.probe==typ)&(PRIMARY.budget==bgt)&
     (PRIMARY.scenario==sn)
    ].set_index('seed').tpr
    seeds=sorted(set(off.index)&set(q.index))
    d=np.array([q.loc[z]-off.loc[z] for z in seeds],float)
    md,lo,hi=mean_ci95(d)
    gen.append({
     'rep':r,'probe':typ,'budget':bgt,'scenario':sn,'n':len(d),
     'mean_shift_vs_official_pp':100*md,
     'ci95_lo_pp':100*lo,'ci95_hi_pp':100*hi,
     'positive_seed_count':int((d>0).sum()),
     'negative_seed_count':int((d<0).sum())
    })
FF3=pd.DataFrame(gen)
FF3.to_csv(RESULTS/'RQ_FF3_GENERALIZATION_SHIFT.csv',index=False)

# Full low-FPR grid for FF3.
LOWFPR=SCI[SCI.target_fpr.notna()].groupby(
 ['rep','probe','budget','scenario','target_fpr']
).agg(
 tpr_mean=('tpr','mean'),tpr_sd=('tpr','std'),
 realized_fpr_mean=('fpr','mean'),
 precision_mean=('precision','mean'),
 fp_per_1000_mean=('fp_per_1000','mean')
).reset_index()
LOWFPR.to_csv(RESULTS/'RQ_FF3_LOW_FPR_GRID.csv',index=False)

display(
 FF12[(FF12.scenario=='OFFICIAL_TEST')&
      (FF12.budget=='B25R')][
  ['rep','probe','budget','tpr_mean','delta_tpr_vs_R0_pp','AP_mean','P_at_R90_mean']
 ]
)


# CELL 22
# 13 — Final deep system scoring / calibration / decision inputs (resumable chunks)
meta=json.loads((SELECTED_DIR/'meta.json').read_text());use_dom=SELECTED_ARCH=='TRI';m=DeepFusion(use_dom).to(DEVICE);m.load_state_dict(torch.load(SELECTED_DIR/'model_state.pt',map_location=DEVICE));bs=int(meta['batch'])

def deep_score_cache(df,name):
 scorep=SCORES/f'{name}_deep_score.npy';urlp=SCORES/f'{name}_url_score.npy';done=SCORES/f'{name}.complete.json';prog=SCORES/f'{name}.progress.json';shape=(len(df),)
 if scorep.exists() and urlp.exists() and done.exists():
  try:
   a=np.load(scorep,mmap_mode='r');b=np.load(urlp,mmap_mode='r');ok=a.shape==shape and b.shape==shape;del a,b
   if ok:return np.load(scorep,mmap_mode='r'),np.load(urlp,mmap_mode='r')
  except:pass
 start=0
 if scorep.exists() and urlp.exists() and prog.exists():
  try:start=int(json.loads(prog.read_text()).get('next_row',0))
  except:start=0
 if start==0:
  scorep.unlink(missing_ok=True);urlp.unlink(missing_ok=True);done.unlink(missing_ok=True);a=np.lib.format.open_memmap(scorep,mode='w+',dtype=np.float32,shape=shape);b=np.lib.format.open_memmap(urlp,mode='w+',dtype=np.float32,shape=shape);a.flush();b.flush();del a,b;atomic_json(prog,{'next_row':0})
 a=np.load(scorep,mmap_mode='r+');b=np.load(urlp,mmap_mode='r+');last=start
 for st in range(start,len(df),FINAL_SCORE_PROGRESS_ROWS):
  ids=np.arange(st,min(st+FINAL_SCORE_PROGRESS_ROWS,len(df)));ss,ex=pred_deep(m,use_dom,df,ids,bs);a[st:st+len(ids)]=ss.astype(np.float32);b[st:st+len(ids)]=ex['url'].astype(np.float32);nxt=st+len(ids);a.flush();b.flush();atomic_json(prog,{'next_row':nxt});print({'DEEP_SCORE':name,'rows':nxt,'total':len(df)})
 a.flush();b.flush();del a,b;atomic_json(done,{'status':'COMPLETE','rows':len(df),'model':meta});prog.unlink(missing_ok=True);return np.load(scorep,mmap_mode='r'),np.load(urlp,mmap_mode='r')

sc,uc=deep_score_cache(cal,'CAL');sf,uf=deep_score_cache(final,'FINAL')
sc=np.asarray(sc);uc=np.asarray(uc);sf=np.asarray(sf);uf=np.asarray(uf)
if USE_CASCADE:
 ssc=sc.copy();ssf=sf.copy();ssc[uc>=high]=1.;ssc[uc<=low]=0.;ssf[uf>=high]=1.;ssf[uf<=low]=0.;escal=(uf>low)&(uf<high)
else:ssc,ssf=sc,sf;escal=np.ones(len(final),bool)
np.savez_compressed(SCORES/'FINAL_DEEP_SYSTEM_SCORES.npz',sha256=final.sha256.astype(str),y=fy.astype(np.int8),score=ssf.astype(np.float32),url_score=uf.astype(np.float32),escalation=escal.astype(np.int8))
# Evaluate full selected Deep system AND optional cascade separately.
SYSTEM_SCORES={'DEEP_FULL':(sc,sf)}
if USE_CASCADE:
 SYSTEM_SCORES['DEEP_CASCADE']=(ssc,ssf)

er=[]
for sysname,(cal_score,final_score) in SYSTEM_SCORES.items():
 for f in TARGET_FPRS:
  th=thr_fpr(cal_score,f)
  allowed=int(math.floor(f*len(cal)))
  limited=allowed<20
  for sn,mask in MASKS.items():
   if mask.sum() and len(np.unique(fy[mask]))==2:
    er.append({
     'system':sysname,'scenario':sn,'target_fpr':f,
     'calibration_allowed_fp':allowed,
     'resolution_limited_calibration':limited,
     'threshold':th,
     **op(fy[mask],final_score[mask],th)
    })
 for sn,mask in MASKS.items():
  if mask.sum() and len(np.unique(fy[mask]))==2:
   er.append({
    'system':sysname,'scenario':sn,'target_fpr':np.nan,
    'calibration_allowed_fp':np.nan,
    'resolution_limited_calibration':False,
    'threshold':np.nan,
    **curves(fy[mask],final_score[mask])
   })

ENG=pd.DataFrame(er)
ENG.to_csv(RESULTS/'ENGINEERING_FINAL_METRICS_FULL_AND_CASCADE.csv',index=False)

SELECTED_SYSTEM='DEEP_CASCADE' if USE_CASCADE else 'DEEP_FULL'
selected_score=ssf if USE_CASCADE else sf
selected_cal=ssc if USE_CASCADE else sc

display(
 ENG[(ENG.scenario=='OFFICIAL_TEST')&
     ((np.isclose(ENG.target_fpr.fillna(-1),PRIMARY_FPR,rtol=0,atol=1e-12))|
      ENG.target_fpr.isna())]
)

pr=ENG[
 (ENG.system==SELECTED_SYSTEM)&
 (ENG.scenario=='OFFICIAL_TEST')&
 np.isclose(ENG.target_fpr.fillna(-1),PRIMARY_FPR,rtol=0,atol=1e-12)
].iloc[0]
rk=ENG[
 (ENG.system==SELECTED_SYSTEM)&
 (ENG.scenario=='OFFICIAL_TEST')&
 ENG.target_fpr.isna()
].iloc[0]

DEC=pd.DataFrame([{
 'system':SELECTED_ARCH+'_'+SELECTED_MODE+('_CASCADE' if USE_CASCADE else ''),
 'engineering_objective':ENGINEERING_OBJECTIVE_ID,
 'SSL_representation':CHAMP,
 'selected_seed':SELECTED_SEED,
 'uses_DOM_SSL':use_dom,
 'TPR_at_0.5pct_FPR':pr.tpr,
 'realized_FPR':pr.fpr,
 'precision_at_0.5pct':pr.precision,
 'FP_per_1000_benign':pr.fp_per_1000,
 'AP':rk.AP,
 'P_at_R90':rk.P_at_R90,
 'latency_ms_page_dev_full':SYSTEM['latency_dev_ms_page'],
 'model_state_bytes':SYSTEM['model_state_bytes'],
 'cascade_escalation_rate_final':float(escal.mean()) if USE_CASCADE else 1.0,
 'cascade_full_execution_reduction_final':float(1-escal.mean()) if USE_CASCADE else 0.0,
 'modalities':'URL+HTML text'+('+static DOM' if use_dom else '')
}])
DEC.to_csv(RESULTS/'DECISION_MATRIX_RAW_INPUTS.csv',index=False)
display(DEC)



# CELL 23
# 14 — Official PhreshPhish benchmark indices: exact benign pool + each phish subset
BENCH=pd.DataFrame()
if DOWNLOAD_OFFICIAL_BENCHMARK_INDEX and HF_TOKEN:
 try:
  from huggingface_hub import snapshot_download
  broot=Path(snapshot_download(PHRESHPHISH_DATASET_ID,repo_type='dataset',revision=PINNED_REVISION,token=HF_TOKEN,allow_patterns=['benchmark/**']))/'benchmark'
  benign_files=list(broot.rglob('benign.txt'))
  if len(benign_files)!=1:raise RuntimeError(f'expected one benign.txt, found {len(benign_files)}')
  def hashes_file(p):return list(dict.fromkeys(x.lower() for x in re.findall(r'\b[0-9a-fA-F]{64}\b',p.read_text(errors='ignore'))))
  benign_hashes=hashes_file(benign_files[0]);scoremap=pd.DataFrame({'sha256':final.sha256.astype(str).str.lower(),'y':fy,'score':selected_score}).set_index('sha256');bpresent=[h for h in benign_hashes if h in scoremap.index];ben_cov=len(bpresent)/max(len(benign_hashes),1)
  bfiles=sorted([p for p in broot.rglob('*.txt') if p.name!='benign.txt']);out=[]
  def baserate_label(p):
   # Old layout: benchmark-0.05-114.txt; new layout may use benchmark-005/...
   mm=re.search(r'benchmark-(0?\.\d+)-',p.name)
   if mm:return mm.group(1)
   mapping={'005':'0.05','010':'0.10','050':'0.50','100':'1.00','500':'5.00'}
   for part in p.parts:
    m2=re.fullmatch(r'benchmark-(005|010|050|100|500)',part)
    if m2:return mapping[m2.group(1)]
   return 'unknown'
  for j,p in enumerate(bfiles):
   ph=hashes_file(p);pp=[h for h in ph if h in scoremap.index];pcov=len(pp)/max(len(ph),1);exact=(ben_cov==1.0 and pcov==1.0)
   row={'file':str(p.relative_to(broot)),'base_rate_percent_label':baserate_label(p),'benign_listed':len(benign_hashes),'benign_mapped':len(bpresent),'benign_coverage':ben_cov,'phish_listed':len(ph),'phish_mapped':len(pp),'phish_coverage':pcov,'exact_coverage':exact}
   if exact:
    ids=bpresent+pp;q=scoremap.loc[ids];cm=curves(q.y.to_numpy(),q.score.to_numpy());row.update({'n':len(q),'realized_base_rate':float(q.y.mean()),'AP':cm['AP'],'P_at_R90':cm['P_at_R90']})
   else:row.update({'n':np.nan,'realized_base_rate':np.nan,'AP':np.nan,'P_at_R90':np.nan})
   out.append(row)
   if j and j%100==0:print({'benchmark_files':j,'of':len(bfiles)})
  BENCH=pd.DataFrame(out);BENCH.to_csv(RESULTS/'OFFICIAL_BENCHMARK_INSTANCE_METRICS.csv',index=False)
  exact_b=BENCH[BENCH.exact_coverage==True].copy()
  if len(exact_b):
   BAVG=exact_b.groupby('base_rate_percent_label').agg(n_benchmarks=('file','count'),AP_mean=('AP','mean'),AP_sd=('AP','std'),P_at_R90_mean=('P_at_R90','mean'),P_at_R90_sd=('P_at_R90','std'),realized_base_rate_mean=('realized_base_rate','mean')).reset_index();BAVG.to_csv(RESULTS/'OFFICIAL_BENCHMARK_BASE_RATE_SUMMARY.csv',index=False);display(BAVG)
  atomic_json(AUDIT/'OFFICIAL_BENCHMARK_COVERAGE.json',{'benign_listed':len(benign_hashes),'benign_coverage':ben_cov,'phish_benchmark_files':len(bfiles),'exact_benchmark_files':int(BENCH.exact_coverage.sum()) if len(BENCH) else 0,'rule':'metrics reported only at 100% SHA coverage'})
 except Exception as e:(AUDIT/'BENCHMARK_EVAL_SKIPPED.txt').write_text(repr(e));print('Benchmark evaluation skipped',repr(e))
else:print('Benchmark evaluation disabled/no HF token')

# 1000 stratified bootstrap replicates for selected final system.
# Chunked to 50 replicates per part for cross-session resume.
BOOT_DIR=RESULTS/'bootstrap_parts';BOOT_DIR.mkdir(parents=True,exist_ok=True)
BOOT_N=1000;BOOT_CHUNK=50
neg=np.where(fy==0)[0];pos=np.where(fy==1)[0]

for st in range(0,BOOT_N,BOOT_CHUNK):
 en=min(st+BOOT_CHUNK,BOOT_N)
 part=BOOT_DIR/f'boot_{st:04d}_{en-1:04d}.npy'
 if part.exists():continue
 vals=[]
 for bi in range(st,en):
  rng=np.random.default_rng(20260812+bi)
  ii=np.r_[rng.choice(neg,len(neg),replace=True),rng.choice(pos,len(pos),replace=True)]
  yy=fy[ii];zz=selected_score[ii]
  vals.append([
   average_precision_score(yy,zz),
   precision_at_recall(yy,zz,.90)['precision']
  ])
 tmp=Path(str(part)+'.tmp')
 with open(tmp,'wb') as f:np.save(f,np.asarray(vals,dtype=np.float64))
 os.replace(tmp,part)
 print({'BOOTSTRAP_PART_COMPLETE':part.name})

boot_parts=sorted(BOOT_DIR.glob('boot_*.npy'))
a=np.concatenate([np.load(p) for p in boot_parts],axis=0)
if len(a)!=BOOT_N:raise RuntimeError(f'STOP bootstrap rows {len(a)} != {BOOT_N}')

BOOT={
 'bootstrap':'1000 stratified replicates',
 'system':SELECTED_SYSTEM,
 'AP':{
  'mean':float(a[:,0].mean()),
  'lo':float(np.quantile(a[:,0],.025)),
  'hi':float(np.quantile(a[:,0],.975))
 },
 'P_at_R90':{
  'mean':float(a[:,1].mean()),
  'lo':float(np.quantile(a[:,1],.025)),
  'hi':float(np.quantile(a[:,1],.975))
 }
}
atomic_json(RESULTS/'FINAL_SYSTEM_BOOTSTRAP_CI.json',BOOT)
print(BOOT)


# CELL 24
# 15 — Research-question evidence export (descriptive, no thesis prose interpretation)

# FF1 + FF2 are already represented by RQ_FF1_FF2_REPRESENTATION_LABEL_PROBE.csv
# and SCIENTIFIC_N10_PAIRED_STATS_ALL_SCENARIOS.csv.

# FF4: fair development engineering + sealed final FULL/CASCADE metrics.
ff4_dev=AGG.copy()
ff4_dev['stage']='DEVELOPMENT_ENGINEERING'
ff4_dev['selected_arch']=SELECTED_ARCH
ff4_dev['selected_mode']=SELECTED_MODE
ff4_dev['DOM_gate']=USE_DOM
ff4_dev['DOM_regular_mean_delta_pp']=float(delta)
ff4_dev['engineering_objective']=ENGINEERING_OBJECTIVE_ID
ff4_dev.to_csv(RESULTS/'RQ_FF4_ENGINEERING_DEVELOPMENT.csv',index=False)

ff4_final=ENG.copy()
ff4_final['selected_arch']=SELECTED_ARCH
ff4_final['selected_mode']=SELECTED_MODE
ff4_final['selected_seed']=SELECTED_SEED
ff4_final['DOM_used']=use_dom
ff4_final['cascade_enabled']=USE_CASCADE
ff4_final['engineering_objective']=ENGINEERING_OBJECTIVE_ID
ff4_final.to_csv(RESULTS/'RQ_FF4_FINAL_SYSTEM.csv',index=False)

# Compact evidence JSON for tomorrow's discussion/conclusion drafting.
official_primary=FF12[FF12.scenario=='OFFICIAL_TEST'].copy()
paired_primary=STAT.copy()

def _records(df,cols=None):
 q=df if cols is None else df[cols]
 return json.loads(q.replace({np.nan:None}).to_json(orient='records'))

# Best representation by mean TPR within each pre-defined probe/budget cell.
best_cells=[]
for (probe,budget),q in official_primary.groupby(['probe','budget']):
 qq=q.sort_values(['tpr_mean','P_at_R90_mean','AP_mean'],ascending=False)
 r=qq.iloc[0]
 best_cells.append({
  'probe':probe,'budget':budget,
  'best_rep_by_mean_TPR':str(r.rep),
  'tpr_mean':float(r.tpr_mean),
  'delta_tpr_vs_R0_pp':float(r.delta_tpr_vs_R0_pp),
  'AP_mean':float(r.AP_mean),
  'P_at_R90_mean':float(r.P_at_R90_mean)
 })

# Final FULL vs CASCADE primary operating point.
ff4_primary=ENG[
 (ENG.scenario=='OFFICIAL_TEST')&
 np.isclose(ENG.target_fpr.fillna(-1),PRIMARY_FPR,rtol=0,atol=1e-12)
][['system','tpr','fpr','precision','f1','fp_per_1000']]

RQ={
 'status':'EVIDENCE_COMPLETE',
 'important_scope_note':'descriptive evidence export; interpretation belongs in Discussion/Conclusion',
 'FF1':{
  'question_focus':'downstream utility of R0-R3 representations',
  'official_test_best_cells':best_cells,
  'table':'RQ_FF1_FF2_REPRESENTATION_LABEL_PROBE.csv',
  'paired_statistics':'SCIENTIFIC_N10_PAIRED_STATS_ALL_SCENARIOS.csv'
 },
 'FF2':{
  'question_focus':'label budget and downstream probe interaction',
  'budgets':['B10','B25R','B100'],
  'probes':['LINEAR','MLP'],
  'table':'RQ_FF1_FF2_REPRESENTATION_LABEL_PROBE.csv'
 },
 'FF3':{
  'question_focus':'sealed final/OOD/late generalization and low-FPR behavior',
  'shift_table':'RQ_FF3_GENERALIZATION_SHIFT.csv',
  'low_fpr_table':'RQ_FF3_LOW_FPR_GRID.csv'
 },
 'FF4':{
  'question_focus':'DOM, fusion, hard negatives, cascade and resource observations',
  'engineering_objective':ENGINEERING_OBJECTIVE_ID,
  'selected_representation':CHAMP,
  'selected_architecture':SELECTED_ARCH,
  'selected_mode':SELECTED_MODE,
  'selected_seed':SELECTED_SEED,
  'DOM_gate':bool(USE_DOM),
  'DOM_regular_mean_delta_pp':float(delta),
  'cascade_enabled':bool(USE_CASCADE),
  'development_cascade_escalation_rate':float(CAS['dev_meta_escalation_rate']),
  'final_cascade_escalation_rate':float(escal.mean()) if USE_CASCADE else 1.0,
  'full_vs_cascade_official_primary':_records(ff4_primary),
  'development_table':'RQ_FF4_ENGINEERING_DEVELOPMENT.csv',
  'final_table':'RQ_FF4_FINAL_SYSTEM.csv'
 }
}
atomic_json(RESULTS/'RESEARCH_QUESTION_EVIDENCE.json',RQ)

# Human-readable index without making causal/statistical claims automatically.
md=f"""# Research Question Evidence Index

## FF1 – Repräsentationsnutzen
- `RQ_FF1_FF2_REPRESENTATION_LABEL_PROBE.csv`
- `SCIENTIFIC_N10_PAIRED_STATS_ALL_SCENARIOS.csv`

## FF2 – Labelverfügbarkeit und Probe
- `RQ_FF1_FF2_REPRESENTATION_LABEL_PROBE.csv`
- Budgets: B10=400, B25R=1000, B100=4000
- Probes: LINEAR, MLP

## FF3 – Generalisierung
- `RQ_FF3_GENERALIZATION_SHIFT.csv`
- `RQ_FF3_LOW_FPR_GRID.csv`
- Szenarien: OFFICIAL_TEST, DOMAIN_OOD_EXACT, TEMPLATE_OOD_EXACT,
  DOMAIN_TEMPLATE_OOD_EXACT, LATE_TEST_Q4

## FF4 – Engineering
- `RQ_FF4_ENGINEERING_DEVELOPMENT.csv`
- `RQ_FF4_FINAL_SYSTEM.csv`
- Objective: {ENGINEERING_OBJECTIVE_ID}
- Selected representation: {CHAMP}
- Selected architecture: {SELECTED_ARCH}
- Selected mode: {SELECTED_MODE}
- Cascade enabled: {USE_CASCADE}

## Methodischer Hinweis
Automatisch erzeugte Tabellen liefern die empirische Evidenz. Kausale Einordnung,
Literaturvergleich und Bewertung werden erst in Diskussion/Fazit formuliert.
"""
(RESULTS/'RESEARCH_QUESTION_EVIDENCE_INDEX.md').write_text(md,encoding='utf-8')

print({
 'RQ_EVIDENCE':'COMPLETE',
 'files':[
  'RQ_FF1_FF2_REPRESENTATION_LABEL_PROBE.csv',
  'SCIENTIFIC_N10_PAIRED_STATS_ALL_SCENARIOS.csv',
  'RQ_FF3_GENERALIZATION_SHIFT.csv',
  'RQ_FF3_LOW_FPR_GRID.csv',
  'RQ_FF4_ENGINEERING_DEVELOPMENT.csv',
  'RQ_FF4_FINAL_SYSTEM.csv',
  'RESEARCH_QUESTION_EVIDENCE.json'
 ]
})


# CELL 25
# 16 — Software artifact manifest + compact result ZIP
manifest={'artifact':'Deep Tri-Modal SSL Phishing Risk Scoring Prototype','selected_representation':CHAMP,'architecture':SELECTED_ARCH,'hardnegative':SELECTED_MODE=='HARDNEG','dom_ssl':SELECTED_ARCH=='TRI','use_cascade':USE_CASCADE,'components':{'URL':'Transformer branch','TEXT':'Transformer branch with optional DAPT/contrastive SSL','DOM':'GCN with masked self-supervised pretraining if TRI','fusion':'deep gated fusion'},'calibration':'50k benign threshold-only','primary_metric':'TPR@0.5% FPR','benchmark_metrics':['AP','P@R=.90'],'metric_implementation':'P@R=.90 uses PhreshPhish-style PR interpolation' ,'engineering_objective':ENGINEERING_OBJECTIVE_ID,'aux_total_weight':AUX_TOTAL_WEIGHT,'note':'risk-scoring prototype; autonomous blocking suitability is not assumed'}
(ARTIFACT/'SOFTWARE_ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
(ARTIFACT/'api_contract.json').write_text(json.dumps({'POST /classify':{'input':{'url':'string','html':'string'},'output':{'risk_score':'float','classification':'string','threshold_profile':'string','model_version':'string'}}},indent=2))
complete={'status':'COMPLETE','data_freeze_sha256':DATA_FREEZE_HASH,'architecture_freeze':json.loads((AUDIT/'ARCHITECTURE_FREEZE.json').read_text()),'system_freeze':json.loads((AUDIT/'SYSTEM_FREEZE.json').read_text()),'scientific_scope':'N10 downstream variability conditional on fixed SSL checkpoints','final_rows':len(final),'calibration_rows':len(cal),'metric_note':'PhreshPhish AP + P@R=.90; operational TPR@fixed FPR','engineering_objective':ENGINEERING_OBJECTIVE_ID,'rq_evidence':'results/RESEARCH_QUESTION_EVIDENCE.json'};(ROOT/'FINAL_RUN_COMPLETE.json').write_text(json.dumps(complete,indent=2))
zp=Path('/kaggle/working/phreshphish_FINAL_DEEP_TRIMODAL_SSL_v5_1_STABILITY_AUDITED_RESULTS.zip') if Path('/kaggle/working').exists() else ROOT.parent/'phreshphish_FINAL_DEEP_TRIMODAL_SSL_v5_1_STABILITY_AUDITED_RESULTS.zip'
with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for base in [RESULTS,AUDIT,ARTIFACT]:
  for p in base.rglob('*'):
   if p.is_file():z.write(p,arcname=str(p.relative_to(ROOT)))
 z.write(ROOT/'FINAL_RUN_COMPLETE.json',arcname='FINAL_RUN_COMPLETE.json')
with zipfile.ZipFile(zp) as z:
 if z.testzip():raise RuntimeError('bad zip')
print(complete);print({'RESULTS_ZIP':str(zp),'save_full_kaggle_output_for_checkpoints':True})