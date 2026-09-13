# KI-Unterstuetzung: siehe Erklaerung der Arbeit.
# Ursprungsdatei: phreshphish_v8_4_RESUME_FINAL_DOMFIX_OOMFIX(2).ipynb
# CELL 2
# 00 — Configuration
import os, gc, json, math, time, random, hashlib, warnings, itertools, threading, subprocess, statistics, shutil
from pathlib import Path
from contextlib import nullcontext, contextmanager
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, precision_recall_curve
from scipy.stats import t as student_t
import pyarrow.parquet as pq
from scipy.sparse import hstack, csr_matrix
import joblib
import xgboost as xgb

warnings.filterwarnings('ignore')
if not torch.cuda.is_available():
    raise RuntimeError('STOP: GPU required (tested target: Tesla T4).')
DEVICE=torch.device('cuda')
AMP=True

VERSION='v8_4_FINAL_COMPLETE'
MASTER_SEED=20260814

# Label curve: 5 paired replicates. Replicate 0 preserves the exact v7.2 20k ranking seed.
BUDGETS=[2000,5000,10000,20000,50000,100000,200000]
MODEL_SEEDS=[42,62,82,102,122]
LABEL_RANK_SEEDS=[20260813,20260823,20260833,20260843,20260853]
assert len(MODEL_SEEDS)==len(LABEL_RANK_SEEDS)==5

# Classical system baseline (NOT the SSL control).
# TF-IDF is fitted once on the 200k SSL training pool WITHOUT using labels.
# XGB runs three optimization seeds at every label budget; label rows are held fixed across XGB seeds.
XGB_CURVE_SEED=82
XGB_CURVE_RANK_SEED=20260813
XGB_N3_SEEDS=[42,82,122]
XGB_N3_BUDGETS=list(BUDGETS)
XGB_URL_MAX_FEATURES=25000
XGB_TEXT_MAX_FEATURES=50000
XGB_N_ESTIMATORS=400
XGB_MAX_DEPTH=6
XGB_LEARNING_RATE=.08
XGB_MIN_CHILD_WEIGHT=2
XGB_SUBSAMPLE=.90
XGB_COLSAMPLE=.65
XGB_REG_LAMBDA=2.0
XGB_MAX_BIN=128
XGB_LOW_FPR_MIN_REALIZED_RATIO=.50
XGB_BENCH_B1_N=300
XGB_BENCH_TPUT_N=5000
XGB_BENCH_REPEATS=3
XGB_BENCH_BATCH=32

# ----------------------------
# Kaggle multi-session resume
# ----------------------------
AUTO_RESUME_FROM_KAGGLE_INPUT=True
KEEP_NEW_DEEP_MODEL_STATES_AFTER_SCORING=False

# Optional staged execution. ALL is safe if the session lasts long enough; the condition-level caches make reruns cheap.
# SESSION1_CURVE_XGB skips new Deep-N3 training and later operational cells.
# SESSION2_DEEP runs through Deep-N3; previously completed curve/XGB work is reused.
# SESSION3_FINALIZE assumes curve/XGB/Deep outputs are attached and finishes operational/risk/final packaging.
SESSION_PART='ALL'  # RESUME_FINAL: complete only missing cached work, then finalize
if SESSION_PART not in {'ALL','SESSION1_CURVE_XGB','SESSION2_DEEP','SESSION3_FINALIZE'}:
    raise RuntimeError(f'Unknown SESSION_PART={SESSION_PART}')

RUN_DEEP_NEW = SESSION_PART in {'ALL','SESSION2_DEEP'}
RUN_FINALIZE = SESSION_PART in {'ALL','SESSION3_FINALIZE'}

# Deep-N3 matched end-to-end robustness check.
# Seed 82 is the already completed v7.2 run; 42 and 122 are additional runs.
DEEP_N3_SEEDS=[42,82,122]
DEEP_N3_BUDGETS=[20000,200000]
DEEP_EPOCHS=1
DEEP_BATCH=16
DEEP_ENCODER_LR=1e-5
DEEP_HEAD_LR=2e-4
DEEP_WEIGHT_DECAY=.01
DEEP_WARMUP_FRAC=.05
DEEP_AUX_TOTAL=.30
DEEP_SCORE_BATCH=32
RISK_SCORE_BATCH=32  # OOM-safe micro-batch for TRI risk-score inference on T4

TARGET_FPRS=[.0001,.0005,.001,.0025,.005,.01,.02]
PRIMARY_FPR=.005
URL_MAX_LEN=128
TEXT_MAX_LEN=256
EMBED_BATCH_URL=128
EMBED_BATCH_TEXT=64
MLP_EPOCHS=12
MLP_BATCH=256

# Technical sweet-spot diagnostics (DEV only)
RETENTION_TARGET=.95  # own engineering criterion, NOT literature constant

# Risk-band policy. Low band targets high phishing coverage on DEV; high band targets low FPR on benign CAL.
LOW_RISK_TPR_TARGET=.99
HIGH_RISK_FPR_TARGET=.005
PREVALENCE_SCENARIOS=[.001,.005,.01,.05]  # scenario assumptions only

# Operational benchmark
BENCH_B1_N=500
BENCH_TPUT_N=5000
BENCH_REPEATS=3
BENCH_BATCH=32
GPU_POLL_S=.10

# Final-system uncertainty intervals (no additional training/inference).
BOOTSTRAP_REPS=1000
BOOTSTRAP_SEED=20260814
BOOTSTRAP_BATCH=20
BOOTSTRAP_SYSTEMS=['FULL_TRI','CASCADE']

ROOT=Path('/kaggle/working/phreshphish_v8_4_FINAL') if Path('/kaggle/working').exists() else Path('/mnt/data/phreshphish_v8_4_FINAL')
EMB=ROOT/'embeddings'; CKPT=ROOT/'checkpoints'; SCORES=ROOT/'scores'; RESULTS=ROOT/'results'; AUDIT=ROOT/'audit'; META=ROOT/'role_meta'; FIG=ROOT/'figures'

def hydrate_previous_v84_output():
    """Restore exactly one prior v8.4/v8.3 workroot from /kaggle/input.

    Supports either an extracted workroot or a ZIP containing phreshphish_v8_4_FINAL/.
    Existing local files win. Multiple matching sources abort to prevent state mixing.
    """
    if not AUTO_RESUME_FROM_KAGGLE_INPUT or not Path('/kaggle/input').exists():
        return None

    input_root=Path('/kaggle/input')
    marker_names={'V8_4_WORKROOT_MARKER.json','V8_3_WORKROOT_MARKER.json'}

    direct_roots={}
    for marker_name in marker_names:
        for marker in input_root.rglob(marker_name):
            direct_roots[str(marker.parent.resolve())]=marker.parent

    zip_matches=[]
    if not direct_roots:
        import zipfile
        for zp in input_root.rglob('*.zip'):
            try:
                with zipfile.ZipFile(zp,'r') as z:
                    marker_entries=[n for n in z.namelist() if Path(n).name in marker_names and 'phreshphish_v8_4_FINAL' in n]
                    for prefix in {Path(n).parent.as_posix() for n in marker_entries}:
                        zip_matches.append((zp,prefix))
            except Exception as e:
                print({'RESUME_ZIP_SKIPPED':str(zp),'reason':repr(e)})

    if direct_roots and zip_matches:
        raise RuntimeError('Resume guard: both extracted and ZIP resume sources found. Attach exactly one.')
    if len(direct_roots)>1:
        raise RuntimeError(f'Resume guard: multiple extracted resume roots found: {list(direct_roots)}')
    if len(zip_matches)>1:
        raise RuntimeError(f'Resume guard: multiple ZIP resume roots found: {[(str(z),p) for z,p in zip_matches]}')
    if not direct_roots and not zip_matches:
        raise RuntimeError('RESUME_FINAL STOP: no prior v8.4/v8.3 workroot found under /kaggle/input. Attach the uploaded results dataset first.')

    ROOT.mkdir(parents=True,exist_ok=True)
    copied=0
    copied_bytes=0

    if direct_roots:
        src_root=next(iter(direct_roots.values()))
        for p in src_root.rglob('*'):
            if not p.is_file():
                continue
            rel=p.relative_to(src_root)
            dstp=ROOT/rel
            if dstp.exists():
                continue
            dstp.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,dstp)
            copied+=1
            copied_bytes+=p.stat().st_size
            if copied%250==0:
                print({'RESUME_COPY_FILES':copied,'GiB':round(copied_bytes/1024**3,2)})
        src_desc=str(src_root)
    else:
        import zipfile
        zp,prefix=zip_matches[0]
        prefix=prefix.rstrip('/')+'/'
        with zipfile.ZipFile(zp,'r') as z:
            members=[m for m in z.infolist() if (not m.is_dir()) and m.filename.startswith(prefix)]
            for m in members:
                rel_name=m.filename[len(prefix):]
                if not rel_name:
                    continue
                dstp=ROOT/rel_name
                if dstp.exists():
                    continue
                dstp.parent.mkdir(parents=True,exist_ok=True)
                with z.open(m,'r') as src_f, open(dstp,'wb') as dst_f:
                    shutil.copyfileobj(src_f,dst_f,length=8*1024*1024)
                copied+=1
                copied_bytes+=m.file_size
                if copied%250==0:
                    print({'RESUME_EXTRACT_FILES':copied,'GiB':round(copied_bytes/1024**3,2)})
        src_desc=f'{zp}::{prefix}'

    print({'V8_4_RESUME_HYDRATED_FROM':src_desc,'files_copied':copied,'GiB':round(copied_bytes/1024**3,2)})
    return src_desc

ROOT.mkdir(parents=True,exist_ok=True)
RESUME_SOURCE=hydrate_previous_v84_output()
for p in [ROOT,EMB,CKPT,SCORES,RESULTS,AUDIT,META,FIG]: p.mkdir(parents=True,exist_ok=True)

for _marker_name in ['V8_4_WORKROOT_MARKER.json','V8_3_WORKROOT_MARKER.json']:
    (ROOT/_marker_name).write_text(json.dumps({
        'version':VERSION,'status':'WORKROOT','session_part':SESSION_PART
    },indent=2),encoding='utf-8')

def seed_all(seed):
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed)); torch.cuda.manual_seed_all(int(seed))

def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(str(path)+'.tmp'); tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8'); os.replace(tmp,path)

def atomic_torch(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(str(path)+'.tmp'); torch.save(obj,tmp); os.replace(tmp,path)

def amp_ctx():
    return torch.autocast('cuda',dtype=torch.float16) if AMP else nullcontext()

seed_all(MASTER_SEED)
CONFIG={
    'version':VERSION,
    'session_part':SESSION_PART,
    'auto_resume_from_kaggle_input':AUTO_RESUME_FROM_KAGGLE_INPUT,
    'keep_new_deep_model_states_after_scoring':KEEP_NEW_DEEP_MODEL_STATES_AFTER_SCORING,
    'bootstrap':{'reps':BOOTSTRAP_REPS,'seed':BOOTSTRAP_SEED,'systems':BOOTSTRAP_SYSTEMS},
    'status':'POST_HOC_EXTENSION_AFTER_V7_2',
    'budgets':BUDGETS,
    'model_seeds':MODEL_SEEDS,
    'label_rank_seeds':LABEL_RANK_SEEDS,
    'primary_fpr':PRIMARY_FPR,
    'target_fprs':TARGET_FPRS,
    'retention_target_engineering_convention':RETENTION_TARGET,
    'risk_policy':{'low_dev_tpr_target':LOW_RISK_TPR_TARGET,'high_cal_fpr_target':HIGH_RISK_FPR_TARGET},
    'historical_final_known':True,
    'legacy_4k_used':False,
    'contrastive_reopened':False,
    'classical_xgb_baseline':{
        'role':'system reference, NOT SSL control',
        'curve_seed':XGB_CURVE_SEED,'n3_seeds':XGB_N3_SEEDS,'n3_budgets':XGB_N3_BUDGETS,
        'url_tfidf_max_features':XGB_URL_MAX_FEATURES,'text_tfidf_max_features':XGB_TEXT_MAX_FEATURES,
        'n_estimators':XGB_N_ESTIMATORS,'device':'cpu','raw_margin_thresholding':True,
    },
    'deep_n3':{'seeds':DEEP_N3_SEEDS,'budgets':DEEP_N3_BUDGETS,'architecture':'DUAL','reuses_v7_seed82':True,
               'encoder_lr':DEEP_ENCODER_LR,'head_lr':DEEP_HEAD_LR,'epochs':DEEP_EPOCHS,'aux_total':DEEP_AUX_TOTAL},
}
atomic_json(AUDIT/'V8_PROTOCOL.json',CONFIG)
print(torch.cuda.get_device_name(0))
print(json.dumps(CONFIG,indent=2))

# CELL 3
# 00b — Resume/checkpoint status
def checkpoint_status():
    curve_parts=list((RESULTS/'curve_parts').glob('*.csv')) if (RESULTS/'curve_parts').exists() else []
    xgb_parts=list((ROOT/'xgb_baseline'/'parts').glob('*.csv')) if (ROOT/'xgb_baseline'/'parts').exists() else []
    deep_parts=list((RESULTS/'deep_n3_parts').glob('*.csv')) if (RESULTS/'deep_n3_parts').exists() else []
    emb_done=list(EMB.glob('*.complete.json'))
    st={
        'version':VERSION,
        'session_part':SESSION_PART,
        'resume_source':RESUME_SOURCE,
        'embedding_stream_roles_complete':len(emb_done),
        'curve_condition_parts':len(curve_parts),
        'xgb_condition_parts':len(xgb_parts),
        'deep_scored_condition_parts':len(deep_parts),
        'timestamp_utc':pd.Timestamp.utcnow().isoformat(),
    }
    atomic_json(ROOT/'V8_4_RESUME_STATUS.json',st)
    print(st)
    return st

checkpoint_status()

# RESUME_FINAL hard guard: abort before expensive work unless the saved night-run state is restored.
def resume_final_guard():
    import re as _re
    curve_parts=list((RESULTS/'curve_parts').glob('*.csv')) if (RESULTS/'curve_parts').exists() else []
    xgb_parts=list((ROOT/'xgb_baseline'/'parts').glob('seed*_B*.csv')) if (ROOT/'xgb_baseline'/'parts').exists() else []
    xgb_audits=list((ROOT/'xgb_baseline'/'parts').glob('seed*_B*_audit.json')) if (ROOT/'xgb_baseline'/'parts').exists() else []
    deep_parts=list((RESULTS/'deep_n3_parts').glob('seed*_R*_B*.csv')) if (RESULTS/'deep_n3_parts').exists() else []
    deep_markers=list((RESULTS/'deep_n3_parts').glob('seed*_R*_B*_COMPLETE.json')) if (RESULTS/'deep_n3_parts').exists() else []
    emb_done=list(EMB.glob('*.complete.json'))

    missing_emb_payload=[]
    for d in emb_done:
        arr=d.with_name(d.name.replace('.complete.json','.npy'))
        if not arr.exists():
            missing_emb_payload.append(arr.name)

    expected_deep={(s,r,b) for s in DEEP_N3_SEEDS for r in ['R0','R1'] for b in DEEP_N3_BUDGETS}
    have_deep=set()
    pat=_re.compile(r'seed(\d+)_(R[01])_B(\d+)\.csv$')
    for p in deep_parts:
        m=pat.match(p.name)
        if m:
            have_deep.add((int(m.group(1)),m.group(2),int(m.group(3))))
    missing_deep=sorted(expected_deep-have_deep)

    report={
        'resume_source':RESUME_SOURCE,
        'embedding_stream_roles_complete':len(emb_done),
        'embedding_payloads_missing':missing_emb_payload,
        'curve_parts':len(curve_parts),
        'xgb_parts':len(xgb_parts),
        'xgb_audits':len(xgb_audits),
        'deep_parts':len(deep_parts),
        'deep_markers':len(deep_markers),
        'missing_deep_conditions':missing_deep,
    }
    print({'RESUME_FINAL_GUARD':report})

    problems=[]
    if RESUME_SOURCE is None: problems.append('no resume source hydrated')
    if len(emb_done)!=12: problems.append(f'embedding complete markers {len(emb_done)}/12')
    if missing_emb_payload: problems.append(f'missing embedding .npy payloads: {missing_emb_payload}')
    if len(curve_parts)!=140: problems.append(f'curve parts {len(curve_parts)}/140')
    if len(xgb_parts)!=21: problems.append(f'XGB parts {len(xgb_parts)}/21')
    if len(xgb_audits)!=21: problems.append(f'XGB audit parts {len(xgb_audits)}/21')
    if len(deep_parts)<10: problems.append(f'Deep scored parts only {len(deep_parts)}/12; expected at least 10')
    if len(deep_markers)<len(deep_parts): problems.append(f'Deep COMPLETE markers {len(deep_markers)} < Deep parts {len(deep_parts)}')

    allowed_missing={(122,'R0',200000),(122,'R1',200000)}
    if not set(missing_deep).issubset(allowed_missing):
        problems.append(f'unexpected missing Deep conditions: {missing_deep}')

    if problems:
        raise RuntimeError('RESUME_FINAL GUARD FAILED — aborting before expensive work: ' + ' | '.join(problems))
    print({'RESUME_FINAL_GUARD':'PASS','missing_deep_conditions':missing_deep,'next':'reuse caches -> complete missing Deep -> finalize'})

resume_final_guard()


# CELL 4
# 01 — Resolve required inputs and fail early
SEARCH_ROOT=Path('/kaggle/input') if Path('/kaggle/input').exists() else Path('/mnt/data')
FREEZE_ID='PHRESHPHISH_FINAL_DATA_FREEZE_v3_HF_SHARDED_DOM'
ROLE_REL={
    'DEV':Path('roles/development'),
    'CAL':Path('roles/fpr_calibration_benign'),
    'SSL':Path('roles/ssl_pool_200k'),
    'FINAL':Path('roles/final_test'),
}
DOM_COLS=['dom_tag','dom_parent_idx','dom_depth','dom_attr_count','dom_child_count']

# Corrected metadata sidecar
markers=[]
for p in SEARCH_ROOT.rglob('FINAL_DATA_FREEZE_COMPLETE.json'):
    try: o=json.loads(p.read_text())
    except: continue
    if o.get('status')=='COMPLETE' and o.get('freeze_id')==FREEZE_ID:
        markers.append((p,o))
if len(markers)!=1:
    raise RuntimeError(f'STOP: expected exactly one corrected freeze marker; found {len(markers)}. Attach the v3.2 corrected metadata sidecar.')
FREEZE_FILE,FREEZE_INFO=markers[0]
META_ROOT=FREEZE_FILE.parent
MANIFEST_ROOT=META_ROOT/'manifests'
for p in [MANIFEST_ROOT/'train_role_manifest_PRIVATE_WITH_LABELS.parquet',
          MANIFEST_ROOT/'final_test_exact_ood_flags_SEALED.parquet',
          MANIFEST_ROOT/'final_test_manifest_SEALED.parquet']:
    if not p.exists(): raise RuntimeError(f'STOP missing metadata: {p}')

# Physical role root
roots=[]
for r in SEARCH_ROOT.rglob('roles'):
    if r.is_dir():
        root=r.parent
        if all((root/v).exists() for v in ROLE_REL.values()): roots.append(root)
roots=list({str(x.resolve()):x for x in roots}.values())
if len(roots)!=1:
    raise RuntimeError(f'STOP: expected one physical role root, found {[str(x) for x in roots]}')
DATA_ROOT=roots[0]
ROLE_DIR={k:DATA_ROOT/v for k,v in ROLE_REL.items()}

def nrows(d):
    fs=sorted(Path(d).glob('*.parquet'))
    if not fs: raise RuntimeError(f'No parquet files in {d}')
    return sum(pq.ParquetFile(p).metadata.num_rows for p in fs)
COUNTS={k:nrows(v) for k,v in ROLE_DIR.items()}
EXPECTED={'DEV':20000,'CAL':50000,'SSL':200000,'FINAL':168060}
if COUNTS!=EXPECTED: raise RuntimeError(f'STOP role counts mismatch: {COUNTS}')

# Scientific R1 DAPT + DOM SSL assets from v4.3 (same assets used by v7.2)
old=[]
for p in SEARCH_ROOT.rglob('phreshphish_FINAL_DEEP_TRIMODAL_SSL_v4_3_LABELSET_ALIGNMENT_FIX'):
    if not p.is_dir(): continue
    score=sum(x.exists() for x in [p/'checkpoints/R1_DAPT_TEXT/COMPLETE.json',p/'checkpoints/DOM_MASKED_SSL/encoder.pt',p/'audit/DOM_VOCAB.json'])
    if score: old.append((score,p))
if not old: raise RuntimeError('STOP: attach the bigresults/v4.3 checkpoint dataset used by v7.2.')
old.sort(key=lambda x:x[0],reverse=True)
OLD_ROOT=old[0][1]
DAPT_DIR=OLD_ROOT/'checkpoints/R1_DAPT_TEXT'
DOM_ENCODER_PATH=OLD_ROOT/'checkpoints/DOM_MASKED_SSL/encoder.pt'
DOM_VOCAB_PATH=OLD_ROOT/'audit/DOM_VOCAB.json'

# Full v7.2 output is required for operational/risk sections. The small results ZIP is NOT enough.
v7_candidates=[]
for p in SEARCH_ROOT.rglob('V7_2_SYSTEM_FREEZE.json'):
    root=p.parent.parent
    tri=root/'checkpoints/DEEP/TRI_R1_200K/model_state.pt'
    dual=root/'checkpoints/DEEP/DUAL_R1_200K/model_state.pt'
    if tri.exists() and dual.exists(): v7_candidates.append(root)
if not v7_candidates:
    raise RuntimeError('STOP: attach the FULL v7.2 notebook output as a Kaggle input dataset. Required: checkpoints/DEEP/TRI_R1_200K and DUAL_R1_200K plus audit/V7_2_SYSTEM_FREEZE.json.')
V7_ROOT=sorted(v7_candidates,key=lambda p:len(str(p)))[0]
V7_FREEZE=json.loads((V7_ROOT/'audit/V7_2_SYSTEM_FREEZE.json').read_text())
if V7_FREEZE.get('representation')!='R1' or V7_FREEZE.get('architecture')!='TRI' or int(V7_FREEZE.get('budget',0))!=200000:
    raise RuntimeError('STOP: attached v7.2 system freeze is not the expected R1/TRI/200k system.')
TRI_DIR=V7_ROOT/'checkpoints/DEEP/TRI_R1_200K'
DUAL_DIR=V7_ROOT/'checkpoints/DEEP/DUAL_R1_200K'

# Recover original representation sources if audit exists; otherwise resolve locally / Hugging Face.
rep_sources={}
sp=V7_ROOT/'audit/SSL_REPRESENTATION_SOURCES.json'
if sp.exists(): rep_sources=json.loads(sp.read_text())

def existing_or_search(x, needle=None):
    if x and Path(str(x)).exists(): return str(x)
    needle=(needle or str(x).split('/')[-1]).lower()
    hits=[]
    for p in SEARCH_ROOT.rglob('config.json'):
        s=str(p.parent).lower()
        if needle in s and 'r1_dapt_text' not in s: hits.append(p.parent)
    if hits: return str(sorted(hits,key=lambda p:len(str(p)))[0])
    return str(x)

URL_SRC=existing_or_search((rep_sources.get('R0') or ['bert-base-uncased'])[0],'bert-base-uncased')
TEXT_BASE_SRC=existing_or_search((rep_sources.get('R0') or [None,'roberta-base'])[1],'roberta-base')
TEXT_DAPT_SRC=str(DAPT_DIR)

# Basic score-cache availability (saves ~1h in risk analysis)
tri_score_dir=V7_ROOT/'scores/DEEP/TRI_R1_200K'
CACHE_EXPECTED=[tri_score_dir/'CAL_gated_logit.npy',tri_score_dir/'CAL_url_logit.npy',tri_score_dir/'FINAL_gated_logit.npy',tri_score_dir/'FINAL_url_logit.npy']
V7_SCORE_CACHE_READY=all(p.exists() for p in CACHE_EXPECTED)

INPUT_AUDIT={
    'status':'PASS','data_root':str(DATA_ROOT),'meta_root':str(META_ROOT),'old_root':str(OLD_ROOT),'v7_root':str(V7_ROOT),
    'counts':COUNTS,'url_src':URL_SRC,'text_base_src':TEXT_BASE_SRC,'text_dapt_src':TEXT_DAPT_SRC,
    'v7_selected_score_cache_ready':V7_SCORE_CACHE_READY,
}
atomic_json(AUDIT/'INPUT_RESOLUTION.json',INPUT_AUDIT)
print(json.dumps(INPUT_AUDIT,indent=2))

# CELL 5
# 02 — Data utilities, metadata, disjointness and OOD masks
from sklearn.model_selection import train_test_split

def y01(s):
    if pd.api.types.is_numeric_dtype(s): return pd.to_numeric(s).astype(int).to_numpy()
    z=s.astype(str).str.lower().str.strip()
    pos=z.isin(['1','true','phish','phishing','malicious'])
    neg=z.isin(['0','false','benign','legitimate','legit'])
    if not (pos|neg).all(): raise RuntimeError(f'Unknown labels: {sorted(z[~(pos|neg)].unique())[:10]}')
    return pos.astype(int).to_numpy()

def read_role(key,cols):
    parts=[]
    for p in sorted(ROLE_DIR[key].glob('*.parquet')):
        parts.append(pd.read_parquet(p,columns=cols))
    out=pd.concat(parts,ignore_index=True)
    if len(out)!=COUNTS[key]: raise RuntimeError(f'Role row mismatch {key}: {len(out)}')
    return out

def read_role_slice(key,start,end,cols):
    out=[]; cur=0
    for p in sorted(ROLE_DIR[key].glob('*.parquet')):
        n=pq.ParquetFile(p).metadata.num_rows
        a=max(start-cur,0); b=min(end-cur,n)
        if a<b:
            q=pd.read_parquet(p,columns=cols).iloc[a:b].reset_index(drop=True)
            out.append(q)
        cur+=n
        if cur>=end: break
    z=pd.concat(out,ignore_index=True) if out else pd.DataFrame(columns=cols)
    if len(z)!=(end-start): raise RuntimeError(f'Role slice mismatch {key} {start}:{end} -> {len(z)}')
    return z

# Private labels for exactly the 200k SSL pool.
priv=pd.read_parquet(MANIFEST_ROOT/'train_role_manifest_PRIVATE_WITH_LABELS.parquet',columns=['sha256','label','role'])
priv['sha256']=priv.sha256.astype(str).str.lower()
sslmeta=priv[priv.role.eq('SSL_POOL')][['sha256','label']].copy().reset_index(drop=True)
if len(sslmeta)!=200000 or sslmeta.sha256.duplicated().any(): raise RuntimeError('SSL private manifest invalid')
sslmeta['y']=y01(sslmeta.label)
SSL_LABEL_MAP=dict(zip(sslmeta.sha256,sslmeta.y.astype(int)))

# Save minimal role metadata in exact physical row order. Large URL/text strings are not kept in RAM.
for key in ['SSL','DEV','CAL','FINAL']:
    out=META/f'{key}.parquet'
    if out.exists(): continue
    cols=['sha256'] + ([] if key=='SSL' else ['label']) + (['date'] if key in ['DEV','FINAL'] else [])
    q=read_role(key,cols)
    q['sha256']=q.sha256.astype(str).str.lower()
    if key=='SSL':
        q['y']=q.sha256.map(SSL_LABEL_MAP)
        if q.y.isna().any(): raise RuntimeError('SSL physical/private label mismatch')
        q['y']=q.y.astype(int)
    else: q['y']=y01(q.label)
    keep=['sha256','y']+(['date'] if 'date' in q.columns else [])
    q[keep].to_parquet(out,index=False)

SSL_META=pd.read_parquet(META/'SSL.parquet')
DEV_META=pd.read_parquet(META/'DEV.parquet')
CAL_META=pd.read_parquet(META/'CAL.parquet')
FINAL_META=pd.read_parquet(META/'FINAL.parquet')
if (len(SSL_META),len(DEV_META),len(CAL_META),len(FINAL_META))!=(200000,20000,50000,168060): raise RuntimeError('Role metadata lengths invalid')
if int(CAL_META.y.sum())!=0: raise RuntimeError('CAL must be benign-only')
if (int((FINAL_META.y==0).sum()),int((FINAL_META.y==1).sum()))!=(91260,76800): raise RuntimeError('FINAL class counts invalid')

# SHA disjointness.
sets={k:set(pd.read_parquet(META/f'{k}.parquet',columns=['sha256']).sha256) for k in ['SSL','DEV','CAL','FINAL']}
for a,b in itertools.combinations(sets,2):
    if sets[a]&sets[b]: raise RuntimeError(f'STOP SHA overlap {a} / {b}: {len(sets[a]&sets[b])}')

# Same representation development split as v7.2.
DEVY=DEV_META.y.to_numpy(int)
idx=np.arange(len(DEV_META))
REP_CAL,REP_EVAL=train_test_split(idx,test_size=.5,stratify=DEVY,random_state=20260821)
REP_CAL=np.sort(REP_CAL); REP_EVAL=np.sort(REP_EVAL)

# OOD masks aligned by SHA.
flags=pd.read_parquet(MANIFEST_ROOT/'final_test_exact_ood_flags_SEALED.parquet')
flags['sha256']=flags.sha256.astype(str).str.lower(); flags=flags.set_index('sha256').loc[FINAL_META.sha256].reset_index()
dt=pd.to_datetime(FINAL_META.date,errors='coerce'); cut=dt.dropna().quantile(.75); late=(dt>=cut).fillna(False).to_numpy()
FINAL_MASKS={
    'OFFICIAL_TEST':np.ones(len(FINAL_META),dtype=bool),
    'DOMAIN_OOD_EXACT':flags.domain_ood_exact.to_numpy(bool),
    'TEMPLATE_OOD_EXACT':flags.template_ood_exact.to_numpy(bool),
    'DOMAIN_TEMPLATE_OOD_EXACT':flags.domain_template_ood_exact.to_numpy(bool),
    'LATE_TEST_Q4':late,
}

atomic_json(AUDIT/'DATA_PROTOCOL.json',{
    'status':'PASS','ssl_rows':len(SSL_META),'dev_rows':len(DEV_META),'cal_rows':len(CAL_META),'final_rows':len(FINAL_META),
    'sha_disjointness':'PASS','legacy_4k_used':False,'late_q4_cutoff':str(cut),
    'replicate_design':'5 paired runs; same labeled rows for R0/R1 within each run; nested budgets within each run'
})
print('DATA_PROTOCOL PASS')

# CELL 6
# 03 — Shared embedding cache: URL once + base-text once + DAPT-text once
from transformers import AutoTokenizer, AutoModel

HF_TOKEN=os.environ.get('HF_TOKEN')
if not HF_TOKEN and Path('/kaggle/working').exists():
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN=UserSecretsClient().get_secret('HF_TOKEN')
    except: pass

url_tok=AutoTokenizer.from_pretrained(URL_SRC,token=HF_TOKEN)
text_tok=AutoTokenizer.from_pretrained(TEXT_BASE_SRC,token=HF_TOKEN)

def pooled(out,mask):
    h=out.last_hidden_state
    m=mask.unsqueeze(-1).to(h.dtype)
    return (h*m).sum(1)/m.sum(1).clamp_min(1.)

STREAMS={
    'URL':(URL_SRC,url_tok,'url',URL_MAX_LEN,EMBED_BATCH_URL),
    'TEXT_R0':(TEXT_BASE_SRC,text_tok,'text',TEXT_MAX_LEN,EMBED_BATCH_TEXT),
    'TEXT_R1':(TEXT_DAPT_SRC,text_tok,'text',TEXT_MAX_LEN,EMBED_BATCH_TEXT),
}
ROLE_COUNTS=COUNTS

@torch.no_grad()
def embed_stream_role(stream,role):
    src,tok,col,max_len,batch=STREAMS[stream]
    out=EMB/f'{stream}_{role}.npy'; done=EMB/f'{stream}_{role}.complete.json'; prog=EMB/f'{stream}_{role}.progress.json'
    if done.exists() and out.exists(): return out
    df=read_role(role,['sha256',col])
    n=len(df)
    model=AutoModel.from_pretrained(src,token=HF_TOKEN).to(DEVICE).eval()
    hdim=int(model.config.hidden_size)
    start=0
    if prog.exists() and out.exists():
        try: start=int(json.loads(prog.read_text()).get('next_row',0))
        except: start=0
    if start==0:
        out.unlink(missing_ok=True); prog.unlink(missing_ok=True); done.unlink(missing_ok=True)
        arr=np.lib.format.open_memmap(out,mode='w+',dtype=np.float32,shape=(n,hdim))
    else: arr=np.load(out,mmap_mode='r+')
    for st in range(start,n,batch):
        en=min(st+batch,n)
        vals=df[col].iloc[st:en].fillna('').astype(str).tolist()
        b=tok(vals,padding=True,truncation=True,max_length=max_len,return_tensors='pt')
        b={k:v.to(DEVICE) for k,v in b.items()}
        with amp_ctx(): z=pooled(model(**b),b['attention_mask'])
        arr[st:en]=z.float().cpu().numpy();
        if en%5000<batch or en==n:
            arr.flush(); atomic_json(prog,{'next_row':en}); print({'EMBED':stream,'role':role,'rows':en,'total':n})
    arr.flush(); del arr,model,df; gc.collect(); torch.cuda.empty_cache()
    atomic_json(done,{'status':'COMPLETE','rows':n,'source':str(src),'hidden_size':hdim}); prog.unlink(missing_ok=True)
    return out

for stream in ['URL','TEXT_R0','TEXT_R1']:
    for role in ['SSL','DEV','CAL','FINAL']:
        embed_stream_role(stream,role)
print('SHARED EMBEDDING CACHE COMPLETE')
checkpoint_status()

# CELL 7
# 04 — Label-subset construction + low-FPR metrics

def keyed_rank(sha,seed):
    return hashlib.sha256(f'{seed}|{sha}'.encode()).hexdigest()

def nested_budget_indices(rank_seed):
    # One deterministic ordering per class; every smaller budget is a strict subset of every larger balanced budget.
    out={}
    rank_by_class={}
    for cls in [0,1]:
        q=SSL_META[SSL_META.y.eq(cls)][['sha256']].copy()
        q['idx']=q.index.to_numpy()
        q['rank']=[keyed_rank(x,rank_seed) for x in q.sha256]
        rank_by_class[cls]=q.sort_values('rank').idx.to_numpy()
    for B in BUDGETS:
        if B==200000:
            idx=np.arange(len(SSL_META),dtype=int)
        else:
            if B%2: raise RuntimeError('Balanced budget must be even')
            n=B//2
            idx=np.sort(np.concatenate([rank_by_class[0][:n],rank_by_class[1][:n]]))
            if len(idx)!=B: raise RuntimeError(f'Budget construction failed B={B}')
        out[B]=idx
    for a,b in zip(BUDGETS[:-1],BUDGETS[1:]):
        if not set(out[a]).issubset(set(out[b])): raise RuntimeError(f'Nestedness failed {a}->{b}')
    return out

BUDGET_INDEX={rs:nested_budget_indices(rs) for rs in LABEL_RANK_SEEDS}
# Explicit anchor check: replicate 0 at 20k exactly follows v7.2 SCI20 hash rule.
anchor=BUDGET_INDEX[20260813][20000]
if len(anchor)!=20000 or int(SSL_META.y.iloc[anchor].sum())!=10000: raise RuntimeError('v7.2 20k anchor reconstruction failed')

# Record overlap between replicate label samples (diagnostic, not a failure).
overlap=[]
for B in BUDGETS[:-1]:
    for i,j in itertools.combinations(range(5),2):
        a=set(BUDGET_INDEX[LABEL_RANK_SEEDS[i]][B]); b=set(BUDGET_INDEX[LABEL_RANK_SEEDS[j]][B])
        overlap.append({'budget':B,'rep_i':i,'rep_j':j,'jaccard':len(a&b)/len(a|b),'intersection':len(a&b)})
pd.DataFrame(overlap).to_csv(AUDIT/'LABEL_REPLICATE_OVERLAP.csv',index=False)

def thr_fpr(neg_scores,f):
    s=np.asarray(neg_scores,dtype=float)
    if len(s)==0: return np.inf
    # conservative: max k=floor(f*n) false positives on calibration negatives
    k=int(math.floor(float(f)*len(s)+1e-12))
    if k<=0: return float(np.nextafter(np.max(s),np.inf))
    ss=np.sort(s)
    return float(np.nextafter(ss[-k],np.inf))

def op(y,score,th):
    y=np.asarray(y,int); score=np.asarray(score,float); p=score>=th
    tp=int(((p)&(y==1)).sum()); fp=int(((p)&(y==0)).sum()); tn=int(((~p)&(y==0)).sum()); fn=int(((~p)&(y==1)).sum())
    tpr=tp/max(tp+fn,1); fpr=fp/max(fp+tn,1); prec=tp/max(tp+fp,1)
    return {'tpr':tpr,'fpr':fpr,'precision':prec,'f1':2*prec*tpr/max(prec+tpr,1e-12),'tp':tp,'fp':fp,'tn':tn,'fn':fn,'fp_per_1000':1000*fpr}

def curves(y,s):
    y=np.asarray(y,int); s=np.asarray(s,float)
    ap=float(average_precision_score(y,s)) if len(np.unique(y))>1 else np.nan
    if (y==1).sum()==0: return {'AP':ap,'P_at_R90':np.nan}
    p,r,_=precision_recall_curve(y,s)
    ok=np.where(r>=.90)[0]
    pr=float(np.max(p[ok])) if len(ok) else np.nan
    return {'AP':ap,'P_at_R90':pr}

def exact_signflip_p(diff):
    d=np.asarray(diff,float); d=d[np.isfinite(d)]
    if len(d)==0: return np.nan
    obs=abs(d.mean()); vals=[]
    for signs in itertools.product([-1,1],repeat=len(d)):
        vals.append(abs(np.mean(d*np.asarray(signs))))
    return float(np.mean(np.asarray(vals)>=obs-1e-15))

def holm_adjust(p_values):
    """Holm step-down FWER adjustment; NaN values are excluded from the family."""
    p=np.asarray(p_values,dtype=float)
    out=np.full(len(p),np.nan,dtype=float)
    finite=np.flatnonzero(np.isfinite(p))
    if len(finite)==0:
        return out

    vals=p[finite]
    order=np.argsort(vals,kind='mergesort')
    m=len(vals)
    adjusted_sorted=np.empty(m,dtype=float)
    running=0.0

    for rank,local_pos in enumerate(order):
        candidate=(m-rank)*vals[local_pos]
        running=max(running,candidate)
        adjusted_sorted[rank]=min(1.0,running)

    for rank,local_pos in enumerate(order):
        out[finite[local_pos]]=adjusted_sorted[rank]
    return out

def apply_holm_family(df,mask,family_name,p_col='exact_signflip_p'):
    idx=df.index[mask].to_numpy()
    if len(idx)==0:
        return
    df.loc[idx,'holm_family']=family_name
    df.loc[idx,'holm_p']=holm_adjust(df.loc[idx,p_col].to_numpy(float))
    df.loc[idx,'raw_p_lt_0p05']=df.loc[idx,p_col].to_numpy(float)<.05
    df.loc[idx,'holm_p_lt_0p05']=df.loc[idx,'holm_p'].to_numpy(float)<.05

def mean_ci(diff):
    d=np.asarray(diff,float); d=d[np.isfinite(d)]
    if len(d)<2: return (float(np.mean(d)) if len(d) else np.nan,np.nan,np.nan)
    m=float(d.mean()); se=float(d.std(ddof=1)/math.sqrt(len(d))); q=float(student_t.ppf(.975,len(d)-1))
    return m,m-q*se,m+q*se

atomic_json(AUDIT/'LABEL_BUDGET_PROTOCOL.json',{
    'budgets':BUDGETS,'model_seeds':MODEL_SEEDS,'label_rank_seeds':LABEL_RANK_SEEDS,
    'nested_within_replicate':True,'paired_R0_R1_same_rows':True,'anchor_20k_rank_seed':20260813,
    'full_200k_class_counts':SSL_META.y.value_counts().sort_index().to_dict()
})
print('LABEL BUDGET PROTOCOL PASS')

# CELL 8
# 05 — Probe models and feature-cache access
URL_H=int(json.loads((EMB/'URL_SSL.complete.json').read_text())['hidden_size'])
TEXT_H=int(json.loads((EMB/'TEXT_R0_SSL.complete.json').read_text())['hidden_size'])
FEATURE_DIM=URL_H+TEXT_H

def emb_mm(stream,role): return np.load(EMB/f'{stream}_{role}.npy',mmap_mode='r')

def take_features(rep,role,idx=None):
    u=emb_mm('URL',role); t=emb_mm('TEXT_R0' if rep=='R0' else 'TEXT_R1',role)
    if idx is None: idx=np.arange(len(u))
    return np.concatenate([np.asarray(u[idx],dtype=np.float32),np.asarray(t[idx],dtype=np.float32)],axis=1)

class MLPProbe(nn.Module):
    def __init__(self,d):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,256),nn.GELU(),nn.Dropout(.15),nn.Linear(256,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def fit_mlp(X,y,seed):
    seed_all(seed); m=MLPProbe(X.shape[1]).to(DEVICE)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-3)
    xx=torch.tensor(np.asarray(X,dtype=np.float32)); yy=torch.tensor(np.asarray(y),dtype=torch.float32)
    dl=DataLoader(TensorDataset(xx,yy),batch_size=MLP_BATCH,shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0)
    for _ in range(MLP_EPOCHS):
        m.train()
        for a,b in dl:
            a=a.to(DEVICE); b=b.to(DEVICE); opt.zero_grad(set_to_none=True)
            loss=F.binary_cross_entropy_with_logits(m(a),b); loss.backward(); opt.step()
    return m.eval()

def score_linear(rep,role,coef,intercept,chunk=10000):
    u=emb_mm('URL',role); t=emb_mm('TEXT_R0' if rep=='R0' else 'TEXT_R1',role); out=np.empty(len(u),np.float32)
    for st in range(0,len(u),chunk):
        en=min(st+chunk,len(u)); X=np.concatenate([np.asarray(u[st:en],np.float32),np.asarray(t[st:en],np.float32)],1)
        out[st:en]=(X@coef.T).reshape(-1)+float(intercept.reshape(-1)[0])
    return out

@torch.no_grad()
def score_mlp(rep,role,m,chunk=4096):
    u=emb_mm('URL',role); t=emb_mm('TEXT_R0' if rep=='R0' else 'TEXT_R1',role); out=np.empty(len(u),np.float32)
    m.eval()
    for st in range(0,len(u),chunk):
        en=min(st+chunk,len(u)); X=np.concatenate([np.asarray(u[st:en],np.float32),np.asarray(t[st:en],np.float32)],1)
        out[st:en]=m(torch.tensor(X,device=DEVICE)).float().cpu().numpy()
    return out

# CELL 9
# 06 — N5 label-efficiency experiment (R0 vs R1 only)
PARTS=RESULTS/'curve_parts'; PARTS.mkdir(exist_ok=True)

for run_i,(model_seed,rank_seed) in enumerate(zip(MODEL_SEEDS,LABEL_RANK_SEEDS)):
    for B in BUDGETS:
        train_idx=BUDGET_INDEX[rank_seed][B]
        ytr=SSL_META.y.iloc[train_idx].to_numpy(int)
        for rep in ['R0','R1']:
            Xtr=take_features(rep,'SSL',train_idx)
            for probe in ['LINEAR','MLP']:
                part=PARTS/f'run{run_i}_{rep}_{probe}_{B}.csv'
                if part.exists():
                    continue
                cdir=CKPT/'CURVE'/f'run{run_i}'/rep/f'B{B}'; cdir.mkdir(parents=True,exist_ok=True)
                if probe=='LINEAR':
                    mp=cdir/'linear.npz'
                    if mp.exists():
                        z=np.load(mp); coef=z['coef']; intercept=z['intercept']
                    else:
                        clf=LogisticRegression(C=1,max_iter=2000,solver='liblinear',class_weight='balanced',random_state=model_seed).fit(Xtr,ytr)
                        coef=clf.coef_.astype(np.float32); intercept=clf.intercept_.astype(np.float32); np.savez(mp,coef=coef,intercept=intercept)
                    dev_score=score_linear(rep,'DEV',coef,intercept); cal_score=score_linear(rep,'CAL',coef,intercept); fin_score=score_linear(rep,'FINAL',coef,intercept)
                else:
                    mp=cdir/'mlp.pt'; m=MLPProbe(FEATURE_DIM).to(DEVICE)
                    if mp.exists(): m.load_state_dict(torch.load(mp,map_location=DEVICE)); m.eval()
                    else:
                        del m; m=fit_mlp(Xtr,ytr,model_seed); atomic_torch(mp,m.state_dict())
                    dev_score=score_mlp(rep,'DEV',m); cal_score=score_mlp(rep,'CAL',m); fin_score=score_mlp(rep,'FINAL',m)
                    del m; torch.cuda.empty_cache()

                rows=[]
                # DEV-only curve for technical sweet-spot selection.
                yc=DEVY[REP_CAL]; ye=DEVY[REP_EVAL]; sc=dev_score[REP_CAL]; se=dev_score[REP_EVAL]
                for f in TARGET_FPRS:
                    th=thr_fpr(sc[yc==0],f)
                    rows.append({'run':run_i,'model_seed':model_seed,'label_rank_seed':rank_seed,'budget':B,'rep':rep,'probe':probe,
                                 'dataset':'DEV_EVAL','scenario':'DEV_EVAL','target_fpr':f,'threshold':th,**op(ye,se,th),**curves(ye,se)})
                # FINAL/OOD: threshold comes only from benign CAL.
                for f in TARGET_FPRS:
                    th=thr_fpr(cal_score,f)
                    for sn,mask in FINAL_MASKS.items():
                        y=FINAL_META.y.to_numpy(int)[mask]; s=fin_score[mask]
                        rows.append({'run':run_i,'model_seed':model_seed,'label_rank_seed':rank_seed,'budget':B,'rep':rep,'probe':probe,
                                     'dataset':'FINAL','scenario':sn,'target_fpr':f,'threshold':th,**op(y,s,th),**curves(y,s)})
                pd.DataFrame(rows).to_csv(part,index=False)
                if len(list(PARTS.glob('*.csv'))) % 10 == 0:
                    checkpoint_status()
                print({'CURVE_DONE':True,'run':run_i,'rep':rep,'probe':probe,'budget':B})
            del Xtr; gc.collect()

parts=[pd.read_csv(p) for p in sorted(PARTS.glob('*.csv'))]
CURVE=pd.concat(parts,ignore_index=True)
CURVE.to_csv(RESULTS/'LABEL_EFFICIENCY_R0_R1_N5_ALL.csv',index=False)
print({'LABEL_CURVE_ROWS':len(CURVE),'parts':len(parts)})
checkpoint_status()

# CELL 10
# 07 — Aggregate curve, paired SSL deltas, marginal label utility and DEV-only sweet-spot candidates
CURVE=pd.read_csv(RESULTS/'LABEL_EFFICIENCY_R0_R1_N5_ALL.csv')

AGG=CURVE.groupby(['budget','rep','probe','dataset','scenario','target_fpr'],dropna=False).agg(
    n=('tpr','size'),tpr_mean=('tpr','mean'),tpr_std=('tpr','std'),fpr_mean=('fpr','mean'),fpr_std=('fpr','std'),
    precision_mean=('precision','mean'),AP_mean=('AP','mean'),P_at_R90_mean=('P_at_R90','mean')
).reset_index()
AGG.to_csv(RESULTS/'LABEL_EFFICIENCY_R0_R1_N5_AGG.csv',index=False)

# Paired R1-R0 effects.
deltas=[]
for keys,g in CURVE.groupby(['budget','probe','dataset','scenario','target_fpr'],dropna=False):
    piv=g.pivot_table(index='run',columns='rep',values='tpr',aggfunc='first')
    if not {'R0','R1'}.issubset(piv.columns): continue
    d=(piv.R1-piv.R0).dropna().to_numpy(float)
    m,lo,hi=mean_ci(d)
    deltas.append({'budget':keys[0],'probe':keys[1],'dataset':keys[2],'scenario':keys[3],'target_fpr':keys[4],
                   'n':len(d),'mean_delta_tpr_pp':100*m,'ci95_lo_pp':100*lo,'ci95_hi_pp':100*hi,
                   'positive_runs':int((d>0).sum()),'exact_signflip_p':exact_signflip_p(d)})
DELTA=pd.DataFrame(deltas)
DELTA['holm_family']='DESCRIPTIVE_NOT_INFERENTIAL_FAMILY'
DELTA['holm_p']=np.nan
DELTA['raw_p_lt_0p05']=False
DELTA['holm_p_lt_0p05']=False

# Family 1 (FF2): Official FINAL at the primary 0.5% FPR operating point,
# seven label budgets × two probes = 14 paired hypotheses.
ff2_mask=(
    (DELTA.dataset=='FINAL') &
    (DELTA.scenario=='OFFICIAL_TEST') &
    np.isclose(DELTA.target_fpr,PRIMARY_FPR)
)
apply_holm_family(
    DELTA,ff2_mask,
    'FF2_OFFICIAL_PRIMARY_7BUDGETS_X_2PROBES'
)

# Family 2 (FF3): primary 20k condition under the four non-Official
# FINAL shift scenarios × two probes = 8 paired hypotheses.
# Official is intentionally not duplicated in this second family.
ff3_mask=(
    (DELTA.dataset=='FINAL') &
    (DELTA.budget==20000) &
    (DELTA.scenario!='OFFICIAL_TEST') &
    np.isclose(DELTA.target_fpr,PRIMARY_FPR)
)
apply_holm_family(
    DELTA,ff3_mask,
    'FF3_20K_SHIFT_PRIMARY_4SCENARIOS_X_2PROBES'
)

DELTA.to_csv(RESULTS/'SSL_DELTA_R1_MINUS_R0_N5.csv',index=False)
STAT_N5=DELTA[DELTA.holm_family!='DESCRIPTIVE_NOT_INFERENTIAL_FAMILY'].copy()
STAT_N5.to_csv(RESULTS/'STATISTICAL_N5_HOLM_FAMILIES.csv',index=False)

atomic_json(AUDIT/'STATISTICAL_INFERENCE_PROTOCOL.json',{
    'status':'PASS',
    'alpha':0.05,
    'paired_n5':{
        'effect':'R1-R0 TPR under same run/budget/probe/scenario',
        'ci':'two-sided 95% t interval of paired mean effect',
        'raw_test':'exact two-sided sign-flip test',
        'minimum_attainable_two_sided_p_N5':0.0625,
        'holm_families':{
            'FF2_OFFICIAL_PRIMARY_7BUDGETS_X_2PROBES':
                'Official FINAL, target FPR=0.5%, 7 budgets x Linear/MLP',
            'FF3_20K_SHIFT_PRIMARY_4SCENARIOS_X_2PROBES':
                '20k FINAL, target FPR=0.5%, four non-Official shift scenarios x Linear/MLP'
        },
        'full_low_fpr_grid_outside_predefined_families':'descriptive',
        'dev_sweetspot_analysis':'descriptive selection analysis'
    },
    'deep_n3':{
        'role':'optimization-seed stability check',
        'minimum_attainable_two_sided_p_N3':0.25
    },
    'xgb':{
        'role':'classical system reference',
        'formal_hypothesis_test_against_deep':False
    }
})

# DEV-only 95%-retention and descriptive knee candidate, separately by representation/probe.
primary=AGG[(AGG.dataset=='DEV_EVAL')&(AGG.scenario=='DEV_EVAL')&np.isclose(AGG.target_fpr,PRIMARY_FPR)].copy()

def knee_candidate(df):
    q=df.sort_values('budget'); x=np.log10(q.budget.to_numpy(float)); y=q.tpr_mean.to_numpy(float)
    if len(q)<3 or np.ptp(y)<=0: return np.nan
    xn=(x-x.min())/max(np.ptp(x),1e-12); yn=(y-y.min())/max(np.ptp(y),1e-12)
    # distance above chord from first to last, descriptive only
    chord=yn[0]+(yn[-1]-yn[0])*xn
    return int(q.iloc[int(np.argmax(yn-chord))].budget)

sweet=[]; marginal=[]
for (rep,probe),g in primary.groupby(['rep','probe']):
    g=g.sort_values('budget'); ref=float(g.loc[g.budget==200000,'tpr_mean'].iloc[0])
    ok=g[g.tpr_mean>=RETENTION_TARGET*ref]
    b95=int(ok.budget.min()) if len(ok) else np.nan
    sweet.append({'rep':rep,'probe':probe,'reference_budget':200000,'reference_tpr_dev':ref,
                  'retention_target':RETENTION_TARGET,'B95_dev':b95,'knee_candidate_dev':knee_candidate(g)})
    prev=None
    for _,r in g.iterrows():
        if prev is not None:
            db=r.budget-prev.budget; dt=r.tpr_mean-prev.tpr_mean
            marginal.append({'rep':rep,'probe':probe,'from_budget':int(prev.budget),'to_budget':int(r.budget),
                             'delta_tpr_pp':100*dt,'pp_per_10k_labels':100*dt/db*10000})
        prev=r
pd.DataFrame(sweet).to_csv(RESULTS/'TECHNICAL_SWEETSPOT_DEV_CANDIDATES.csv',index=False)
pd.DataFrame(marginal).to_csv(RESULTS/'MARGINAL_LABEL_UTILITY_DEV.csv',index=False)

# Minimal plots. Selection logic is DEV-only; FINAL plots are descriptive validation.
import matplotlib.pyplot as plt
for probe in ['LINEAR','MLP']:
    fig,ax=plt.subplots(figsize=(7,4))
    for rep in ['R0','R1']:
        g=primary[(primary.probe==probe)&(primary.rep==rep)].sort_values('budget')
        ax.errorbar(g.budget,g.tpr_mean,yerr=g.tpr_std,marker='o',label=rep)
    ax.set_xscale('log'); ax.set_xlabel('Labelbudget'); ax.set_ylabel('TPR bei 0,5 % FPR (DEV)'); ax.set_title(f'Label-Effizienz — {probe}'); ax.grid(True,alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(FIG/f'label_curve_DEV_{probe}.png',dpi=180); plt.close(fig)

fig,ax=plt.subplots(figsize=(7,4))
for probe in ['LINEAR','MLP']:
    g=DELTA[(DELTA.dataset=='DEV_EVAL')&(DELTA.scenario=='DEV_EVAL')&np.isclose(DELTA.target_fpr,PRIMARY_FPR)&(DELTA.probe==probe)].sort_values('budget')
    ax.plot(g.budget,g.mean_delta_tpr_pp,marker='o',label=probe)
ax.axhline(0,linewidth=1); ax.set_xscale('log'); ax.set_xlabel('Labelbudget'); ax.set_ylabel('Δ TPR R1−R0 [pp]'); ax.set_title('Marginaler DAPT-Zusatznutzen (DEV)'); ax.grid(True,alpha=.25); ax.legend(); fig.tight_layout()
fig.savefig(FIG/'ssl_delta_DEV.png',dpi=180); plt.close(fig)

print(pd.read_csv(RESULTS/'TECHNICAL_SWEETSPOT_DEV_CANDIDATES.csv').to_string(index=False))

# CELL 12
# 08 — Classical TF-IDF + XGBoost baseline (CPU, raw-margin low-FPR evaluation)
XGB_ROOT=ROOT/'xgb_baseline'
XGB_CKPT=XGB_ROOT/'models'
XGB_CACHE=XGB_ROOT/'feature_cache'
for p in [XGB_ROOT,XGB_CKPT,XGB_CACHE]: p.mkdir(parents=True,exist_ok=True)

# Fixed CPU implementation so this baseline does not consume VRAM and has an explicit hardware profile.
def xgb_params(seed):
    p=dict(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        min_child_weight=XGB_MIN_CHILD_WEIGHT,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE,
        reg_lambda=XGB_REG_LAMBDA,
        reg_alpha=0.0,
        max_bin=XGB_MAX_BIN,
        random_state=int(seed),
        n_jobs=-1,
        tree_method='hist',
    )
    try:
        major=int(str(xgb.__version__).split('.')[0])
    except Exception:
        major=2
    if major>=2:
        p['device']='cpu'
    return p

def sparse_cache_paths(role):
    return XGB_CACHE/f'X_{role}.npz'

# Load only static strings needed by this baseline.
SSL_XGB=read_role('SSL',['sha256','url','text'])
DEV_XGB=read_role('DEV',['sha256','url','text'])
CAL_XGB=read_role('CAL',['sha256','url','text'])
FINAL_XGB=read_role('FINAL',['sha256','url','text'])
for q in [SSL_XGB,DEV_XGB,CAL_XGB,FINAL_XGB]:
    q['sha256']=q.sha256.astype(str).str.lower()
    q['url']=q.url.fillna('').astype(str)
    q['text']=q.text.fillna('').astype(str)

if not np.array_equal(SSL_XGB.sha256.to_numpy(),SSL_META.sha256.astype(str).str.lower().to_numpy()):
    raise RuntimeError('XGB SSL alignment mismatch')
if not np.array_equal(DEV_XGB.sha256.to_numpy(),DEV_META.sha256.astype(str).str.lower().to_numpy()):
    raise RuntimeError('XGB DEV alignment mismatch')
if not np.array_equal(CAL_XGB.sha256.to_numpy(),CAL_META.sha256.astype(str).str.lower().to_numpy()):
    raise RuntimeError('XGB CAL alignment mismatch')
if not np.array_equal(FINAL_XGB.sha256.to_numpy(),FINAL_META.sha256.astype(str).str.lower().to_numpy()):
    raise RuntimeError('XGB FINAL alignment mismatch')

vec_url_path=XGB_CACHE/'url_tfidf.joblib'
vec_text_path=XGB_CACHE/'text_tfidf.joblib'
sparse_paths={r:XGB_CACHE/f'X_{r}.npz' for r in ['SSL','DEV','CAL','FINAL']}

tfidf_timing={}
if vec_url_path.exists() and vec_text_path.exists() and all(p.exists() for p in sparse_paths.values()):
    from scipy.sparse import load_npz
    url_vec=joblib.load(vec_url_path); text_vec=joblib.load(vec_text_path)
    X_SSL=load_npz(sparse_paths['SSL']); X_DEV=load_npz(sparse_paths['DEV'])
    X_CAL=load_npz(sparse_paths['CAL']); X_FINAL=load_npz(sparse_paths['FINAL'])
    tfidf_timing={'cache_reuse':True}
    print('XGB TF-IDF CACHE REUSED')
else:
    from scipy.sparse import save_npz
    url_vec=TfidfVectorizer(
        analyzer='char',ngram_range=(3,5),min_df=2,max_features=XGB_URL_MAX_FEATURES,
        sublinear_tf=True,norm='l2',dtype=np.float32
    )
    text_vec=TfidfVectorizer(
        analyzer='word',ngram_range=(1,2),min_df=3,max_df=.995,max_features=XGB_TEXT_MAX_FEATURES,
        sublinear_tf=True,strip_accents='unicode',norm='l2',dtype=np.float32
    )

    t0=time.perf_counter()
    Xu_ssl=url_vec.fit_transform(SSL_XGB.url)
    url_fit_s=time.perf_counter()-t0

    t0=time.perf_counter()
    Xt_ssl=text_vec.fit_transform(SSL_XGB.text)
    text_fit_s=time.perf_counter()-t0

    def tfidf_transform_pair(frame):
        t0=time.perf_counter()
        xu=url_vec.transform(frame.url)
        xt=text_vec.transform(frame.text)
        x=csr_matrix(hstack([xu,xt],format='csr'),dtype=np.float32)
        return x,time.perf_counter()-t0

    X_SSL=csr_matrix(hstack([Xu_ssl,Xt_ssl],format='csr'),dtype=np.float32)
    X_DEV,dev_transform_s=tfidf_transform_pair(DEV_XGB)
    X_CAL,cal_transform_s=tfidf_transform_pair(CAL_XGB)
    X_FINAL,final_transform_s=tfidf_transform_pair(FINAL_XGB)
    del Xu_ssl,Xt_ssl

    joblib.dump(url_vec,vec_url_path,compress=3)
    joblib.dump(text_vec,vec_text_path,compress=3)
    save_npz(sparse_paths['SSL'],X_SSL,compressed=True)
    save_npz(sparse_paths['DEV'],X_DEV,compressed=True)
    save_npz(sparse_paths['CAL'],X_CAL,compressed=True)
    save_npz(sparse_paths['FINAL'],X_FINAL,compressed=True)
    tfidf_timing={
        'cache_reuse':False,'url_fit_s':url_fit_s,'text_fit_s':text_fit_s,
        'dev_transform_s':dev_transform_s,'cal_transform_s':cal_transform_s,'final_transform_s':final_transform_s,
    }

atomic_json(AUDIT/'XGB_TFIDF_FEATURE_AUDIT.json',{
    'status':'PASS',
    'role':'classical system baseline; NOT SSL control',
    'fit_domain':'SSL 200k inputs only; labels unused for TF-IDF fitting',
    'url_features':int(len(url_vec.vocabulary_)),
    'text_features':int(len(text_vec.vocabulary_)),
    'total_features':int(X_SSL.shape[1]),
    'timing':tfidf_timing,
    'sparse_shapes':{k:list(v.shape) for k,v in {'SSL':X_SSL,'DEV':X_DEV,'CAL':X_CAL,'FINAL':X_FINAL}.items()},
})

# Full XGB N3 curve: three seeds at every budget.
# The label rows are fixed by XGB_CURVE_RANK_SEED so the three seeds isolate XGB training randomness.
specs=[]
for B in BUDGETS:
    for s in XGB_N3_SEEDS:
        specs.append((int(s),int(B),'N3_FULL_CURVE'))
specs=sorted(specs,key=lambda x:(x[1],x[0]))

# All XGB conditions use the exact v7.2/v8.1 anchor ordering for nested labels.
XGB_LABEL_INDEX=BUDGET_INDEX[XGB_CURVE_RANK_SEED]
y_ssl=SSL_META.y.to_numpy(int)
y_dev=DEV_META.y.to_numpy(int)
y_final=FINAL_META.y.to_numpy(int)

XGB_SCORE_DIR=XGB_ROOT/'scores'
XGB_PARTS=XGB_ROOT/'parts'
for p in [XGB_SCORE_DIR,XGB_PARTS]: p.mkdir(exist_ok=True)
xgb_train=[]

def model_margin(model,X):
    return np.asarray(model.predict(X,output_margin=True),dtype=np.float64)

for seed,B,role_tag in specs:
    part=XGB_PARTS/f'seed{seed}_B{B}.csv'
    audit_part=XGB_PARTS/f'seed{seed}_B{B}_audit.json'
    if part.exists() and audit_part.exists():
        print({'XGB_REUSE_PART':part.name})
        continue

    idx=XGB_LABEL_INDEX[B] if B<200000 else np.arange(len(SSL_META),dtype=int)
    model_dir=XGB_CKPT/f'seed{seed}_B{B}'
    model_dir.mkdir(parents=True,exist_ok=True)
    model_path=model_dir/'model.ubj'
    done_path=model_dir/'COMPLETE.json'

    model=xgb.XGBClassifier(**xgb_params(seed))
    train_origin='TRAINED'
    if done_path.exists() and model_path.exists():
        model.load_model(model_path)
        train_origin='CACHE_REUSE'
        train_s=0.0
    else:
        t0=time.perf_counter()
        model.fit(X_SSL[idx],y_ssl[idx],verbose=False)
        train_s=time.perf_counter()-t0
        model.save_model(model_path)
        atomic_json(done_path,{
            'status':'COMPLETE','seed':int(seed),'budget':int(B),'role_tag':role_tag,
            'rows':int(len(idx)),'positive_rows':int(y_ssl[idx].sum()),
            'rank_seed':int(XGB_CURVE_RANK_SEED),'params':xgb_params(seed),
        })

    t0=time.perf_counter()
    dev_s=model_margin(model,X_DEV)
    cal_s=model_margin(model,X_CAL)
    fin_s=model_margin(model,X_FINAL)
    scoring_s=time.perf_counter()-t0

    np.save(XGB_SCORE_DIR/f'seed{seed}_B{B}_DEV_margin.npy',dev_s.astype(np.float32))
    np.save(XGB_SCORE_DIR/f'seed{seed}_B{B}_CAL_margin.npy',cal_s.astype(np.float32))
    np.save(XGB_SCORE_DIR/f'seed{seed}_B{B}_FINAL_margin.npy',fin_s.astype(np.float32))

    unique_cal=int(np.unique(cal_s).size)
    _,counts_cal=np.unique(cal_s,return_counts=True)
    max_tie=int(counts_cal.max()) if len(counts_cal) else 0

    local_rows=[]
    local_resolution=[]
    for f in TARGET_FPRS:
        th=thr_fpr(cal_s,f)
        local_rows.append({
            'seed':seed,'budget':B,'role_tag':role_tag,'dataset':'DEV','scenario':'DEV',
            'target_fpr':f,'threshold':th,**op(y_dev,dev_s,th),**curves(y_dev,dev_s)
        })
        for scen,mask in FINAL_MASKS.items():
            yy=y_final[mask]; ss=fin_s[mask]
            met=op(yy,ss,th); cur=curves(yy,ss)
            local_rows.append({
                'seed':seed,'budget':B,'role_tag':role_tag,'dataset':'FINAL','scenario':scen,
                'target_fpr':f,'threshold':th,**met,**cur
            })
            if scen=='OFFICIAL_TEST':
                ratio=float(met['fpr']/f) if f>0 else np.nan
                local_resolution.append({
                    'seed':seed,'budget':B,'target_fpr':f,'realized_fpr':met['fpr'],
                    'realized_to_target_ratio':ratio,
                    'valid_for_matched_low_fpr':bool(ratio>=XGB_LOW_FPR_MIN_REALIZED_RATIO),
                    'cal_unique_scores':unique_cal,'cal_max_tie_count':max_tie,
                })

    pd.DataFrame(local_rows).to_csv(part,index=False)
    atomic_json(audit_part,{
        'status':'COMPLETE','seed':int(seed),'budget':int(B),'train_origin':train_origin,
        'train_s':float(train_s),'score_dev_cal_final_s':float(scoring_s),
        'model_bytes':int(model_path.stat().st_size),
        'cal_unique_scores':unique_cal,'cal_max_tie_count':max_tie,
        'low_fpr_resolution':local_resolution,
    })
    print({'XGB_DONE':True,'seed':seed,'budget':B,'train_s':round(train_s,2),
           'score_s':round(scoring_s,2),'cal_unique':unique_cal,'max_tie':max_tie})
    del model
    gc.collect()
    checkpoint_status()

# Reconstruct global XGB tables exclusively from condition parts. Safe after any resumed session.
xgb_parts=sorted(XGB_PARTS.glob('seed*_B*.csv'))
if len(xgb_parts) != len(XGB_N3_SEEDS)*len(BUDGETS):
    raise RuntimeError(f'XGB full curve incomplete: {len(xgb_parts)}/{len(XGB_N3_SEEDS)*len(BUDGETS)} condition parts')
XGB_ALL=pd.concat([pd.read_csv(p) for p in xgb_parts],ignore_index=True)

audit_objs=[]
for p in sorted(XGB_PARTS.glob('seed*_B*_audit.json')):
    audit_objs.append(json.loads(p.read_text()))
XGB_TRAIN=pd.DataFrame([{
    'seed':o['seed'],'budget':o['budget'],'origin':o['train_origin'],
    'train_s':o['train_s'],'score_dev_cal_final_s':o['score_dev_cal_final_s'],
    'model_bytes':o['model_bytes'],'cal_unique_scores':o['cal_unique_scores'],
    'cal_max_tie_count':o['cal_max_tie_count'],
} for o in audit_objs])
XGB_RESOLUTION=pd.DataFrame([
    {'seed':o['seed'],'budget':o['budget'],**r}
    for o in audit_objs for r in o['low_fpr_resolution']
])

XGB_ALL.to_csv(RESULTS/'XGB_CLASSICAL_ALL.csv',index=False)
XGB_TRAIN.to_csv(RESULTS/'XGB_CLASSICAL_TRAINING_TIMING.csv',index=False)
XGB_RESOLUTION.to_csv(AUDIT/'XGB_LOW_FPR_RESOLUTION_AUDIT.csv',index=False)

# Descriptive seed-82 label curve.
XGB_CURVE=XGB_ALL[
    (XGB_ALL.seed==XGB_CURVE_SEED)&
    (XGB_ALL.dataset=='FINAL')&
    (XGB_ALL.scenario=='OFFICIAL_TEST')&
    np.isclose(XGB_ALL.target_fpr,PRIMARY_FPR)
].sort_values('budget').copy()
XGB_CURVE.to_csv(RESULTS/'XGB_CLASSICAL_LABEL_CURVE_SEED82_PRIMARY.csv',index=False)

# N3 full-budget aggregation.
XGB_N3=XGB_ALL[XGB_ALL.seed.isin(XGB_N3_SEEDS)&XGB_ALL.budget.isin(BUDGETS)].copy()
XGB_N3_AGG=XGB_N3.groupby(['budget','dataset','scenario','target_fpr']).agg(
    n=('tpr','size'),tpr_mean=('tpr','mean'),tpr_std=('tpr','std'),
    fpr_mean=('fpr','mean'),fpr_std=('fpr','std'),
    precision_mean=('precision','mean'),AP_mean=('AP','mean'),P_at_R90_mean=('P_at_R90','mean')
).reset_index()
XGB_N3_AGG.to_csv(RESULTS/'XGB_CLASSICAL_N3_ALL_BUDGETS_AGG.csv',index=False)

primary_res=XGB_RESOLUTION[np.isclose(XGB_RESOLUTION.target_fpr,PRIMARY_FPR)].copy()
checkpoint_status()
print('XGB primary low-FPR resolution audit:')
print(primary_res.to_string(index=False))

# CELL 13
# 09 — Deep model classes and DOM utilities for operational benchmark / risk bands
# These definitions reproduce the v7.2 selected R1 TRI/DUAL architecture for checkpoint loading.
from transformers import get_linear_schedule_with_warmup

PROJ_DIM=256; DOM_MAX_NODES=512; DOM_DIM=128; DOM_LAYERS=3; DEEP_LAST_N=4
DOM_VOCAB=json.loads(DOM_VOCAB_PATH.read_text())
URL_H=AutoModel.from_pretrained(URL_SRC,token=HF_TOKEN).config.hidden_size
TEXT_H=AutoModel.from_pretrained(TEXT_DAPT_SRC,token=HF_TOKEN).config.hidden_size

def freeze_last(model,n):
    for p in model.parameters(): p.requires_grad=False
    blocks=None
    for attr in ['encoder.layer','transformer.layer']:
        cur=model
        ok=True
        for a in attr.split('.'):
            if not hasattr(cur,a): ok=False; break
            cur=getattr(cur,a)
        if ok: blocks=cur; break
    if blocks is not None:
        for b in blocks[-n:]:
            for p in b.parameters(): p.requires_grad=True
    for name in ['pooler']:
        if hasattr(model,name) and getattr(model,name) is not None:
            for p in getattr(model,name).parameters(): p.requires_grad=True

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
        x=_rv(r,k)
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,list): return x
        try:return list(x)
        except:return []
    tags=arr('dom_tag'); par=arr('dom_parent_idx'); dep=arr('dom_depth'); att=arr('dom_attr_count'); chi=arr('dom_child_count')
    n=min(len(tags),DOM_MAX_NODES)
    tag=np.array([DOM_VOCAB.get(str(x),UNK) for x in tags[:n]],np.int64)
    depth=np.array([bucket(x) for x in dep[:n]],np.int64)
    attr=np.array([bucket(x) for x in att[:n]],np.int64)
    child=np.array([bucket(x) for x in chi[:n]],np.int64)
    edges=[]
    for ii,pp in enumerate(par[:n]):
        try:pp=int(pp)
        except:continue
        if ii>0 and 0<=pp<n:edges.extend([(pp,ii),(ii,pp)])
    return tag,depth,attr,child,np.asarray(edges,np.int64)

def graph_batch(rows):
    T=[];D=[];A=[];C=[];E=[];B=[];off=0
    row_iter=rows.itertuples(index=False) if isinstance(rows,pd.DataFrame) else rows
    n_graphs=len(rows)
    for bi,r in enumerate(row_iter):
        t,d,a,c,e=graph_row(r);n=len(t)
        # Defensive fallback for an empty DOM: one PAD node keeps graph indexing valid.
        if n==0:
            t=np.array([PAD],np.int64); d=np.array([0],np.int64); a=np.array([0],np.int64); c=np.array([0],np.int64); e=np.zeros((0,2),np.int64); n=1
        T.append(t);D.append(d);A.append(a);C.append(c);B.append(np.full(n,bi,np.int64))
        if len(e):E.append(e+off)
        off+=n
    def cat(xs):return torch.tensor(np.concatenate(xs) if xs else np.array([],np.int64),dtype=torch.long,device=DEVICE)
    edge=torch.tensor(np.concatenate(E,0).T if E else np.zeros((2,0),np.int64),dtype=torch.long,device=DEVICE)
    return {'tag':cat(T),'depth':cat(D),'attr':cat(A),'child':cat(C),'edge':edge,'batch':cat(B),'n_graphs':n_graphs}

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

# Hard preflight for the exact v7.2 DOM checkpoint architecture.
_dom_state=torch.load(DOM_ENCODER_PATH,map_location='cpu')
_dom_probe=DOMEncoder()
_dom_probe.load_state_dict(_dom_state,strict=True)
print({'DOM_CHECKPOINT_COMPATIBILITY':'PASS','tag_shape':tuple(_dom_probe.tag.weight.shape),'state_keys':len(_dom_state)})
del _dom_state,_dom_probe

class DeepFusion(nn.Module):
    def __init__(self,use_dom,text_src=None):
        super().__init__(); self.use_dom=use_dom
        self.text_src=str(text_src or TEXT_DAPT_SRC)
        self.u=AutoModel.from_pretrained(URL_SRC,token=HF_TOKEN); self.t=AutoModel.from_pretrained(self.text_src,token=HF_TOKEN)
        freeze_last(self.u,DEEP_LAST_N); freeze_last(self.t,DEEP_LAST_N)
        self.ua=nn.Linear(int(self.u.config.hidden_size),PROJ_DIM); self.ta=nn.Linear(int(self.t.config.hidden_size),PROJ_DIM)
        self.dom=DOMEncoder() if use_dom else None
        if use_dom:self.dom.load_state_dict(torch.load(DOM_ENCODER_PATH,map_location='cpu'))
        self.da=nn.Linear(DOM_DIM,PROJ_DIM) if use_dom else None
        self.gate=nn.Sequential(nn.Linear(PROJ_DIM,64),nn.GELU(),nn.Linear(64,1))
        nm=3 if use_dom else 2
        self.head=nn.Sequential(nn.Linear(PROJ_DIM*(nm+1),512),nn.GELU(),nn.Dropout(.2),nn.Linear(512,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,1))
        self.uaux=nn.Linear(PROJ_DIM,1); self.taux=nn.Linear(PROJ_DIM,1); self.daux=nn.Linear(PROJ_DIM,1) if use_dom else None
    def forward(self,u,t,g=None):
        zu=F.normalize(self.ua(pooled(self.u(**u),u['attention_mask'])),dim=-1); zt=F.normalize(self.ta(pooled(self.t(**t),t['attention_mask'])),dim=-1); zs=[zu,zt]
        if self.use_dom: zs.append(F.normalize(self.da(self.dom(g)),dim=-1))
        st=torch.stack(zs,1); gw=torch.softmax(self.gate(st).squeeze(-1),1); weighted=(st*gw[:,:,None]).sum(1)
        main=self.head(torch.cat(zs+[weighted],1)).squeeze(-1)
        return main,self.uaux(zu).squeeze(-1)

def tok_url(rows):
    b=url_tok(
        rows.url.fillna('').astype(str).tolist(),
        padding=True,truncation=True,max_length=URL_MAX_LEN,return_tensors='pt'
    )
    return {k:v.to(DEVICE,non_blocking=True) for k,v in b.items()}

def tok_text(rows):
    b=text_tok(
        rows.text.fillna('').astype(str).tolist(),
        padding=True,truncation=True,max_length=TEXT_MAX_LEN,return_tensors='pt'
    )
    return {k:v.to(DEVICE,non_blocking=True) for k,v in b.items()}

def load_deep(d,use_dom,text_src=None):
    m=DeepFusion(use_dom,text_src=text_src).to(DEVICE); m.load_state_dict(torch.load(Path(d)/'model_state.pt',map_location=DEVICE)); m.eval(); return m

SYSTEM=V7_FREEZE
URL_LOW=float(SYSTEM['cascade']['url_low']); URL_HIGH=float(SYSTEM['cascade']['url_high'])
print({'V7_SYSTEM':'READY','url_low':URL_LOW,'url_high':URL_HIGH})

# CELL 14
# 10 — Deep-N3 matched end-to-end robustness check: training
# Scientific role: additional optimization-seed robustness check for FF2.
# The causal comparison remains R1-R0 at the SAME label budget and SAME architecture.
# The 20k labeled subset is held fixed across seeds; only training/initialization randomness changes.

from transformers import get_linear_schedule_with_warmup

DEEP_ROOT=ROOT/'deep_n3'
DEEP_CKPT=CKPT/'DEEP_N3'
DEEP_SCORE_ROOT=SCORES/'DEEP_N3'
for p in [DEEP_ROOT,DEEP_CKPT,DEEP_SCORE_ROOT]: p.mkdir(parents=True,exist_ok=True)

# Exact v7.2 20k anchor: same deterministic labeled rows for all Deep-N3 seeds.
DEEP_20K_IDX=BUDGET_INDEX[20260813][20000]
if len(DEEP_20K_IDX)!=20000 or int(SSL_META.y.iloc[DEEP_20K_IDX].sum())!=10000:
    raise RuntimeError('Deep-N3 20k anchor reconstruction failed.')

SSL_DEEP=read_role('SSL',['sha256','url','text'])
SSL_DEEP['sha256']=SSL_DEEP.sha256.astype(str).str.lower()
if not np.array_equal(SSL_DEEP.sha256.to_numpy(),SSL_META.sha256.astype(str).str.lower().to_numpy()):
    raise RuntimeError('Deep-N3 physical SSL rows are not aligned with SSL_META.')
SSL_DEEP['y']=SSL_META.y.to_numpy(int)

class WebTrainDataset(Dataset):
    def __init__(self,frame,indices):
        self.frame=frame.iloc[np.asarray(indices,dtype=int)].reset_index(drop=True)
    def __len__(self): return len(self.frame)
    def __getitem__(self,i):
        r=self.frame.iloc[i]
        return str(r.url) if pd.notna(r.url) else '', str(r.text) if pd.notna(r.text) else '', int(r.y)

def collate_web(batch):
    urls=[x[0] for x in batch]; texts=[x[1] for x in batch]
    y=torch.tensor([x[2] for x in batch],dtype=torch.float32,device=DEVICE)
    u=url_tok(urls,padding=True,truncation=True,max_length=URL_MAX_LEN,return_tensors='pt')
    t=text_tok(texts,padding=True,truncation=True,max_length=TEXT_MAX_LEN,return_tensors='pt')
    u={k:v.to(DEVICE,non_blocking=True) for k,v in u.items()}
    t={k:v.to(DEVICE,non_blocking=True) for k,v in t.items()}
    return u,t,y

def dual_forward_all(m,u,t):
    zu=F.normalize(m.ua(pooled(m.u(**u),u['attention_mask'])),dim=-1)
    zt=F.normalize(m.ta(pooled(m.t(**t),t['attention_mask'])),dim=-1)
    zs=[zu,zt]
    st=torch.stack(zs,1)
    gw=torch.softmax(m.gate(st).squeeze(-1),1)
    weighted=(st*gw[:,:,None]).sum(1)
    main=m.head(torch.cat(zs+[weighted],1)).squeeze(-1)
    aux=[m.uaux(zu).squeeze(-1),m.taux(zt).squeeze(-1)]
    return main,aux

def v7_condition_dir(rep,budget):
    return V7_ROOT/'checkpoints/DEEP'/f'DUAL_{rep}_{int(budget)//1000}K'

def deep_condition_dir(seed,rep,budget):
    # Seed 82 reuses the exact v7.2 checkpoint whenever present.
    v7=v7_condition_dir(rep,budget)
    if int(seed)==82 and (v7/'model_state.pt').exists():
        return v7,'V7_2_REUSE'
    return DEEP_CKPT/f'seed{seed}'/f'DUAL_{rep}_{int(budget)//1000}K','V8_3'

def train_dual(seed,rep,budget):
    out,origin=deep_condition_dir(seed,rep,budget)
    if origin=='V7_2_REUSE':
        print({'DEEP_N3_REUSE':str(out),'seed':seed,'rep':rep,'budget':budget})
        return out,origin

    out.mkdir(parents=True,exist_ok=True)
    done=out/'COMPLETE.json'
    state=out/'model_state.pt'
    if done.exists() and state.exists():
        return out,'V8_3_RESUME'

    idx=DEEP_20K_IDX if int(budget)==20000 else np.arange(len(SSL_META),dtype=int)
    text_src=TEXT_BASE_SRC if rep=='R0' else TEXT_DAPT_SRC
    seed_all(seed)
    m=DeepFusion(False,text_src=text_src).to(DEVICE)
    ds=WebTrainDataset(SSL_DEEP,idx)
    dl=DataLoader(ds,batch_size=DEEP_BATCH,shuffle=True,num_workers=0,collate_fn=collate_web,
                  generator=torch.Generator().manual_seed(int(seed)),drop_last=False)

    enc=[]
    for module in [m.u,m.t]:
        enc.extend([p for p in module.parameters() if p.requires_grad])
    enc_ids={id(p) for p in enc}
    heads=[p for p in m.parameters() if p.requires_grad and id(p) not in enc_ids]
    opt=torch.optim.AdamW([
        {'params':enc,'lr':DEEP_ENCODER_LR},
        {'params':heads,'lr':DEEP_HEAD_LR}
    ],weight_decay=DEEP_WEIGHT_DECAY)
    total_steps=max(1,len(dl)*DEEP_EPOCHS)
    warmup=int(round(DEEP_WARMUP_FRAC*total_steps))
    sched=get_linear_schedule_with_warmup(opt,num_warmup_steps=warmup,num_training_steps=total_steps)

    losses=[]
    m.train()
    step=0
    for epoch in range(DEEP_EPOCHS):
        for u,t,y in dl:
            opt.zero_grad(set_to_none=True)
            with amp_ctx():
                main,aux=dual_forward_all(m,u,t)
                main_loss=F.binary_cross_entropy_with_logits(main,y)
                aux_loss=torch.stack([F.binary_cross_entropy_with_logits(a,y) for a in aux]).mean()
                loss=main_loss + DEEP_AUX_TOTAL*aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.0)
            opt.step(); sched.step()
            step+=1; losses.append(float(loss.detach().cpu()))
            if step%1000==0 or step==total_steps:
                print({f'DEEP_N3_{rep}_{budget}_seed{seed}_batch':step,'of':total_steps,
                       'loss100':float(np.mean(losses[-100:]))})
    m.eval()
    atomic_torch(state,m.state_dict())
    atomic_json(done,{
        'status':'COMPLETE','seed':int(seed),'rep':rep,'budget':int(budget),'architecture':'DUAL',
        'epochs':DEEP_EPOCHS,'batch':DEEP_BATCH,'encoder_lr':DEEP_ENCODER_LR,'head_lr':DEEP_HEAD_LR,
        'weight_decay':DEEP_WEIGHT_DECAY,'warmup_frac':DEEP_WARMUP_FRAC,'aux_total':DEEP_AUX_TOTAL,
        'training_rows':int(len(idx)),'twenty_k_anchor_rank_seed':20260813 if int(budget)==20000 else None
    })
    del m,ds,dl,opt,sched
    gc.collect(); torch.cuda.empty_cache()
    return out,'V8_3_TRAINED'



# 11 — Deep-N3 scoring on CAL / FINAL / OOD and paired R1-R0 aggregation

def load_deep_rep(model_dir,rep):
    src=TEXT_BASE_SRC if rep=='R0' else TEXT_DAPT_SRC
    return load_deep(model_dir,False,text_src=src)

@torch.no_grad()
def score_dual_frame(m,df,batch=DEEP_SCORE_BATCH):
    out=np.empty(len(df),dtype=np.float32)
    m.eval()
    for st in range(0,len(df),batch):
        en=min(st+batch,len(df))
        rows=df.iloc[st:en]
        u=tok_url(rows); t=tok_text(rows)
        with amp_ctx():
            main,_=dual_forward_all(m,u,t)
        out[st:en]=main.float().detach().cpu().numpy()
    return out

def v7_score_candidate(rep,budget,role):
    # Prefer exact cached v7.2 raw FP32 logits for seed 82.
    d=V7_ROOT/'scores/DEEP'/f'DUAL_{rep}_{int(budget)//1000}K'
    candidates=[
        d/f'{role}_gated_logit.npy',
        d/f'{role}_main_logit.npy',
        d/f'{role}_logit.npy',
    ]
    for p in candidates:
        if p.exists(): return p
    return None

CAL_DF=read_role('CAL',['sha256','url','text'])
FINAL_DF=read_role('FINAL',['sha256','url','text'])
DEV_DF=read_role('DEV',['sha256','url','text'])
for q in [CAL_DF,FINAL_DF,DEV_DF]: q['sha256']=q.sha256.astype(str).str.lower()

def get_deep_scores(seed,rep,budget,role):
    cache=DEEP_SCORE_ROOT/f'seed{seed}'/f'DUAL_{rep}_{int(budget)//1000}K'
    cache.mkdir(parents=True,exist_ok=True)
    out=cache/f'{role}_logit.npy'
    if out.exists(): return np.asarray(np.load(out,mmap_mode='r'),dtype=np.float32),'V8_CACHE'

    if int(seed)==82:
        p=v7_score_candidate(rep,budget,role)
        if p is not None:
            arr=np.asarray(np.load(p,mmap_mode='r'),dtype=np.float32)
            np.save(out,arr)
            return arr,'V7_2_CACHE_REUSE'

    model_dir,_=deep_condition_dir(seed,rep,budget)
    model_dir=Path(model_dir)
    m=load_deep_rep(model_dir,rep)
    frame={'CAL':CAL_DF,'FINAL':FINAL_DF,'DEV':DEV_DF}[role]
    arr=score_dual_frame(m,frame)
    np.save(out,arr)
    del m; gc.collect(); torch.cuda.empty_cache()
    print({'DEEP_N3_SCORE':'COMPLETE','seed':seed,'rep':rep,'budget':budget,'role':role,'rows':len(arr)})
    return arr,'V8_3_RECOMPUTED'



DEEP_PARTS=RESULTS/'deep_n3_parts'
DEEP_PARTS.mkdir(parents=True,exist_ok=True)

def deep_condition_part(seed,rep,budget):
    return DEEP_PARTS/f'seed{seed}_{rep}_B{budget}.csv'

def deep_condition_scored_marker(seed,rep,budget):
    return DEEP_PARTS/f'seed{seed}_{rep}_B{budget}_COMPLETE.json'

def persist_deep_condition(seed,rep,budget):
    part=deep_condition_part(seed,rep,budget)
    marker=deep_condition_scored_marker(seed,rep,budget)
    if part.exists() and marker.exists():
        print({'DEEP_N3_REUSE_SCORED_PART':part.name})
        return

    model_dir,origin=train_dual(seed,rep,budget)

    dev_s,src_dev=get_deep_scores(seed,rep,budget,'DEV')
    cal_s,src_cal=get_deep_scores(seed,rep,budget,'CAL')
    fin_s,src_fin=get_deep_scores(seed,rep,budget,'FINAL')

    local=[]
    for fpr_target in TARGET_FPRS:
        th=thr_fpr(cal_s,fpr_target)
        dmet=op(DEV_META.y.to_numpy(int),dev_s,th)
        dcur=curves(DEV_META.y.to_numpy(int),dev_s)
        local.append({
            'seed':seed,'budget':budget,'rep':rep,'dataset':'DEV','scenario':'DEV',
            'target_fpr':fpr_target,'threshold':th,**dmet,**dcur
        })
        for scen,mask in FINAL_MASKS.items():
            yy=FINAL_META.y.to_numpy(int)[mask]; ss=fin_s[mask]
            met=op(yy,ss,th); cur=curves(yy,ss)
            local.append({
                'seed':seed,'budget':budget,'rep':rep,'dataset':'FINAL','scenario':scen,
                'target_fpr':fpr_target,'threshold':th,**met,**cur
            })

    pd.DataFrame(local).to_csv(part,index=False)
    atomic_json(marker,{
        'status':'COMPLETE_SCORED','seed':int(seed),'rep':rep,'budget':int(budget),
        'model_origin':origin,'score_sources':{'DEV':src_dev,'CAL':src_cal,'FINAL':src_fin},
        'score_files':[
            str(DEEP_SCORE_ROOT/f'seed{seed}'/f'DUAL_{rep}_{int(budget)//1000}K'/'DEV_logit.npy'),
            str(DEEP_SCORE_ROOT/f'seed{seed}'/f'DUAL_{rep}_{int(budget)//1000}K'/'CAL_logit.npy'),
            str(DEEP_SCORE_ROOT/f'seed{seed}'/f'DUAL_{rep}_{int(budget)//1000}K'/'FINAL_logit.npy'),
        ]
    })

    # After all logits/metrics are safely persisted, non-v7 model states are optional.
    # Deleting them keeps a saved Kaggle output substantially smaller while preserving exact scored evidence.
    model_dir=Path(model_dir)
    if KEEP_NEW_DEEP_MODEL_STATES_AFTER_SCORING is False and origin!='V7_2_REUSE':
        state=model_dir/'model_state.pt'
        if state.exists():
            state.unlink()
            atomic_json(model_dir/'MODEL_STATE_REMOVED_AFTER_SCORING.json',{
                'status':'REMOVED_AFTER_SCORE_PERSISTENCE',
                'reason':'compact multi-session resume; exact DEV/CAL/FINAL raw score caches and condition metrics retained'
            })

    gc.collect(); torch.cuda.empty_cache()
    checkpoint_status()

# SESSION1 intentionally stops before new Deep training.
if RUN_DEEP_NEW:
    for seed in DEEP_N3_SEEDS:
        for budget in DEEP_N3_BUDGETS:
            for rep in ['R0','R1']:
                persist_deep_condition(seed,rep,budget)
else:
    print({'DEEP_N3_NEW_TRAINING':'SKIPPED','session_part':SESSION_PART})

# Seed82 may be entirely reusable from v7.2. In SESSION3 we still reconstruct its compact parts if absent.
if SESSION_PART=='SESSION3_FINALIZE':
    for budget in DEEP_N3_BUDGETS:
        for rep in ['R0','R1']:
            part=deep_condition_part(82,rep,budget)
            if not part.exists():
                persist_deep_condition(82,rep,budget)

# CELL 15
# 11 — Deep-N3 aggregation from condition-level scored parts
expected_deep_parts=len(DEEP_N3_SEEDS)*len(DEEP_N3_BUDGETS)*2
parts=sorted(DEEP_PARTS.glob('seed*_B*.csv'))

if len(parts) < expected_deep_parts:
    if SESSION_PART=='SESSION1_CURVE_XGB':
        print({'DEEP_N3_AGGREGATION':'DEFERRED','parts':len(parts),'expected':expected_deep_parts})
        DEEP_N3=pd.DataFrame()
        DEEP_N3_AGG=pd.DataFrame()
        DEEP_N3_DELTA=pd.DataFrame()
    else:
        raise RuntimeError(
            f'Deep-N3 incomplete: {len(parts)}/{expected_deep_parts} scored condition parts. '
            'Attach the previous v8.3 output and run SESSION2_DEEP to resume.'
        )
else:
    DEEP_N3=pd.concat([pd.read_csv(p) for p in parts],ignore_index=True)
    DEEP_N3.to_csv(RESULTS/'DEEP_N3_R0_R1_20K_200K_ALL.csv',index=False)

    src_rows=[]
    for p in sorted(DEEP_PARTS.glob('*_COMPLETE.json')):
        o=json.loads(p.read_text())
        src_rows.append({
            'seed':o['seed'],'rep':o['rep'],'budget':o['budget'],
            'model_origin':o['model_origin'],
            'dev':o['score_sources']['DEV'],'cal':o['score_sources']['CAL'],'final':o['score_sources']['FINAL']
        })
    pd.DataFrame(src_rows).to_csv(AUDIT/'DEEP_N3_SCORE_SOURCES.csv',index=False)

    DEEP_N3_AGG=DEEP_N3.groupby(['budget','rep','dataset','scenario','target_fpr']).agg(
        n=('tpr','size'),tpr_mean=('tpr','mean'),tpr_std=('tpr','std'),
        fpr_mean=('fpr','mean'),fpr_std=('fpr','std'),
        precision_mean=('precision','mean'),AP_mean=('AP','mean'),P_at_R90_mean=('P_at_R90','mean')
    ).reset_index()
    DEEP_N3_AGG.to_csv(RESULTS/'DEEP_N3_R0_R1_20K_200K_AGG.csv',index=False)

    deep_delta=[]
    for keys,g in DEEP_N3.groupby(['budget','dataset','scenario','target_fpr']):
        piv=g.pivot_table(index='seed',columns='rep',values='tpr',aggfunc='first')
        if not {'R0','R1'}.issubset(piv.columns): continue
        d=(piv.R1-piv.R0).dropna().to_numpy(float)
        m,lo,hi=mean_ci(d)
        deep_delta.append({
            'budget':keys[0],'dataset':keys[1],'scenario':keys[2],'target_fpr':keys[3],
            'n':len(d),'mean_delta_tpr_pp':100*m,'ci95_lo_pp':100*lo,'ci95_hi_pp':100*hi,
            'positive_seeds':int((d>0).sum()),'exact_signflip_p':exact_signflip_p(d)
        })
    DEEP_N3_DELTA=pd.DataFrame(deep_delta)
    DEEP_N3_DELTA['holm_family']='DESCRIPTIVE_NOT_INFERENTIAL_FAMILY'
    DEEP_N3_DELTA['holm_p']=np.nan
    DEEP_N3_DELTA['raw_p_lt_0p05']=False
    DEEP_N3_DELTA['holm_p_lt_0p05']=False

    # Stability family only: two budgets × all FINAL scenarios at primary FPR.
    deep_stability_mask=(
        (DEEP_N3_DELTA.dataset=='FINAL') &
        np.isclose(DEEP_N3_DELTA.target_fpr,PRIMARY_FPR)
    )
    apply_holm_family(
        DEEP_N3_DELTA,deep_stability_mask,
        'DEEP_N3_STABILITY_2BUDGETS_X_5SCENARIOS'
    )
    DEEP_N3_DELTA.to_csv(RESULTS/'DEEP_N3_SSL_DELTA_R1_MINUS_R0.csv',index=False)
    DEEP_N3_DELTA[
        DEEP_N3_DELTA.holm_family=='DEEP_N3_STABILITY_2BUDGETS_X_5SCENARIOS'
    ].to_csv(RESULTS/'STATISTICAL_DEEP_N3_HOLM_STABILITY.csv',index=False)

    primary_deep=DEEP_N3_AGG[
        (DEEP_N3_AGG.dataset=='FINAL')&
        (DEEP_N3_AGG.scenario=='OFFICIAL_TEST')&
        np.isclose(DEEP_N3_AGG.target_fpr,PRIMARY_FPR)
    ].copy()
    primary_deep.to_csv(RESULTS/'DEEP_N3_PRIMARY_OFFICIAL_0p5FPR.csv',index=False)

    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for rep in ['R0','R1']:
        g=primary_deep[primary_deep.rep==rep].sort_values('budget')
        ax.errorbar(g.budget,g.tpr_mean,yerr=g.tpr_std,marker='o',label=rep)
    ax.set_xscale('log')
    ax.set_xticks(DEEP_N3_BUDGETS,labels=[str(x) for x in DEEP_N3_BUDGETS])
    ax.set_xlabel('Labelbudget'); ax.set_ylabel('TPR bei CAL-0,5 % FPR (FINAL)')
    ax.set_title('Deep-N3: R0 vs. DAPT-R1')
    ax.grid(True,alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(FIG/'deep_n3_R0_R1_20K_200K.png',dpi=180); plt.close(fig)

    print(primary_deep.to_string(index=False))
    print(DEEP_N3_DELTA[
        (DEEP_N3_DELTA.dataset=='FINAL')&
        (DEEP_N3_DELTA.scenario=='OFFICIAL_TEST')&
        np.isclose(DEEP_N3_DELTA.target_fpr,PRIMARY_FPR)
    ].to_string(index=False))
    checkpoint_status()

# CELL 17
if DEEP_N3_AGG.empty:
    print({'CROSS_SYSTEM_REFERENCE':'DEFERRED_UNTIL_DEEP_COMPLETE'})
else:
    # 12 — Cross-system primary comparison + CPU XGB operational benchmark

    # -------------------------
    # A) Primary matched table
    # -------------------------
    deep_primary=DEEP_N3_AGG[
        (DEEP_N3_AGG.dataset=='FINAL')&
        (DEEP_N3_AGG.scenario=='OFFICIAL_TEST')&
        np.isclose(DEEP_N3_AGG.target_fpr,PRIMARY_FPR)
    ].copy()
    deep_primary['system']='DEEP_DUAL_'+deep_primary['rep']
    deep_primary=deep_primary[['budget','system','n','tpr_mean','tpr_std','fpr_mean','fpr_std',
                               'precision_mean','AP_mean','P_at_R90_mean']]

    xgb_primary=XGB_N3_AGG[
        (XGB_N3_AGG.dataset=='FINAL')&
        (XGB_N3_AGG.scenario=='OFFICIAL_TEST')&
        np.isclose(XGB_N3_AGG.target_fpr,PRIMARY_FPR)
    ].copy()
    xgb_primary['system']='TFIDF_XGB'
    xgb_primary=xgb_primary[['budget','system','n','tpr_mean','tpr_std','fpr_mean','fpr_std',
                             'precision_mean','AP_mean','P_at_R90_mean']]

    SYSTEM_PRIMARY=pd.concat([xgb_primary,deep_primary],ignore_index=True).sort_values(['budget','system'])
    SYSTEM_PRIMARY.to_csv(RESULTS/'SYSTEM_REFERENCE_XGB_VS_DEEP_N3_PRIMARY.csv',index=False)

    # Attach low-FPR validity flag for XGB endpoint means.
    valid_endpoint=(XGB_RESOLUTION[
        XGB_RESOLUTION.seed.isin(XGB_N3_SEEDS)&
        XGB_RESOLUTION.budget.isin(DEEP_N3_BUDGETS)&
        np.isclose(XGB_RESOLUTION.target_fpr,PRIMARY_FPR)
    ].groupby('budget')['valid_for_matched_low_fpr'].all().to_dict())
    SYSTEM_PRIMARY['low_fpr_comparison_valid']=SYSTEM_PRIMARY.apply(
        lambda r: bool(valid_endpoint.get(int(r.budget),False)) if r.system=='TFIDF_XGB' else True,axis=1
    )
    SYSTEM_PRIMARY.to_csv(RESULTS/'SYSTEM_REFERENCE_XGB_VS_DEEP_N3_PRIMARY.csv',index=False)
    print(SYSTEM_PRIMARY.to_string(index=False))

    # -------------------------
    # B) Figure
    # -------------------------
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.5,4.3))
    for system,g in SYSTEM_PRIMARY.groupby('system'):
        g=g.sort_values('budget')
        ax.errorbar(g.budget,100*g.tpr_mean,yerr=100*g.tpr_std,marker='o',capsize=3,label=system)
    ax.set_xscale('log')
    ax.set_xticks([20000,200000],labels=['20k','200k'])
    ax.set_xlabel('Labelbudget')
    ax.set_ylabel('TPR at CAL 0.5% FPR [%]')
    ax.set_title('Classical system reference vs. matched Deep R0/R1')
    ax.grid(True,alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG/'system_reference_XGB_vs_DEEP_N3.png',dpi=180)
    plt.close(fig)

    # -------------------------
    # C) CPU end-to-end serving benchmark for XGB
    # Scope: in-memory URL/text -> TF-IDF transform -> XGB raw-margin prediction.
    # Disk I/O / network / browser rendering are excluded.
    # -------------------------
    def xgb_load_model(seed=82,budget=200000):
        p=XGB_CKPT/f'seed{seed}_B{budget}'/'model.ubj'
        if not p.exists(): raise RuntimeError(f'Missing XGB checkpoint {p}')
        m=xgb.XGBClassifier(**xgb_params(seed))
        m.load_model(p)
        return m,p

    def xgb_transform_frame(frame):
        xu=url_vec.transform(frame.url.fillna('').astype(str))
        xt=text_vec.transform(frame.text.fillna('').astype(str))
        return csr_matrix(hstack([xu,xt],format='csr'),dtype=np.float32)

    # Deterministic sample; reuse the same SHA ranking principle as deep operational benchmark.
    xgb_bench_hash=np.array([int(hashlib.sha256(x.encode()).hexdigest(),16) for x in DEV_XGB.sha256],dtype=object).argsort()
    XGB_B1=DEV_XGB.iloc[xgb_bench_hash[:min(XGB_BENCH_B1_N,len(DEV_XGB))]].reset_index(drop=True)
    XGB_TPUT=DEV_XGB.iloc[xgb_bench_hash[:min(XGB_BENCH_TPUT_N,len(DEV_XGB))]].reset_index(drop=True)

    xgb_model,xgb_model_path=xgb_load_model(82,200000)

    # warm-up
    for _ in range(3):
        xx=xgb_transform_frame(XGB_B1.iloc[:8])
        _=xgb_model.predict(xx,output_margin=True)

    lat=[]
    for i in range(len(XGB_B1)):
        row=XGB_B1.iloc[i:i+1]
        t0=time.perf_counter()
        xx=xgb_transform_frame(row)
        _=xgb_model.predict(xx,output_margin=True)
        lat.append((time.perf_counter()-t0)*1000)

    times=[]
    for _ in range(XGB_BENCH_REPEATS):
        t0=time.perf_counter()
        for st in range(0,len(XGB_TPUT),XGB_BENCH_BATCH):
            q=XGB_TPUT.iloc[st:st+XGB_BENCH_BATCH]
            xx=xgb_transform_frame(q)
            _=xgb_model.predict(xx,output_margin=True)
        times.append(time.perf_counter()-t0)

    XGB_OP=pd.DataFrame([{
        'system':'TFIDF_XGB_200K',
        'hardware':'CPU',
        'measurement_scope':'in-memory URL/text -> TF-IDF -> XGB margin; excludes disk/network/browser/raw HTML parsing',
        'batch_throughput':XGB_BENCH_BATCH,
        'n_b1':len(XGB_B1),
        'n_throughput':len(XGB_TPUT),
        'repeats':XGB_BENCH_REPEATS,
        'latency_b1_median_ms':float(np.median(lat)),
        'latency_b1_p95_ms':float(np.percentile(lat,95)),
        'throughput_pages_s_mean':float(np.mean([len(XGB_TPUT)/t for t in times])),
        'throughput_pages_s_std':float(np.std([len(XGB_TPUT)/t for t in times],ddof=1)) if len(times)>1 else 0.0,
        'model_state_bytes':int(xgb_model_path.stat().st_size),
        'url_vocab_size':int(len(url_vec.vocabulary_)),
        'text_vocab_size':int(len(text_vec.vocabulary_)),
    }])
    XGB_OP.to_csv(RESULTS/'XGB_OPERATIONAL_BENCHMARK.csv',index=False)
    print(XGB_OP.to_string(index=False))
    del xgb_model
    gc.collect()

# CELL 18
# 13 — Operational benchmark: actual URL-first cascade vs full model
if not RUN_FINALIZE:
    print({'OPERATIONAL_BENCHMARK':'DEFERRED','session_part':SESSION_PART})
else:
    # Measurement includes tokenization + stored-DOM graph assembly + model forward, but excludes disk I/O and raw HTML acquisition/parsing.
    try:
        import psutil
    except Exception:
        psutil=None

    # Deterministic DEV sample, loaded fully into memory before timing (disk I/O excluded).
    DEV_OP=read_role('DEV',['sha256','url','text','label']+DOM_COLS)
    DEV_OP['sha256']=DEV_OP.sha256.astype(str).str.lower()
    bench_idx=np.array([int(hashlib.sha256(x.encode()).hexdigest(),16) for x in DEV_OP.sha256],dtype=object).argsort()
    DEV_B1=DEV_OP.iloc[bench_idx[:min(BENCH_B1_N,len(DEV_OP))]].reset_index(drop=True)
    DEV_TPUT=DEV_OP.iloc[bench_idx[:min(BENCH_TPUT_N,len(DEV_OP))]].reset_index(drop=True)


    @torch.no_grad()
    def forward_url_branch(m,rows):
        u=tok_url(rows)
        with amp_ctx(): zu=F.normalize(m.ua(pooled(m.u(**u),u['attention_mask'])),dim=-1); s=m.uaux(zu).squeeze(-1)
        return s.float()

    @torch.no_grad()
    def forward_full(m,rows,use_dom):
        u=tok_url(rows); t=tok_text(rows); g=graph_batch(rows) if use_dom else None
        with amp_ctx(): s,_=m(u,t,g)
        return s.float()

    @torch.no_grad()
    def forward_cascade(m,rows):
        # Genuine compute routing: text + DOM + fusion are executed only for uncertain rows.
        u=tok_url(rows)
        with amp_ctx():
            zu=F.normalize(m.ua(pooled(m.u(**u),u['attention_mask'])),dim=-1); us=m.uaux(zu).squeeze(-1)
        out=torch.empty_like(us,dtype=torch.float32); hi=us>=URL_HIGH; lo=us<=URL_LOW; esc=~(hi|lo)
        out[hi]=1e9; out[lo]=-1e9
        if esc.any():
            ids=torch.where(esc)[0].detach().cpu().numpy(); sub=rows.iloc[ids].reset_index(drop=True); zue=zu[esc]
            t=tok_text(sub); g=graph_batch(sub)
            with amp_ctx():
                zt=F.normalize(m.ta(pooled(m.t(**t),t['attention_mask'])),dim=-1); zd=F.normalize(m.da(m.dom(g)),dim=-1)
                zs=[zue,zt,zd]; st=torch.stack(zs,1); gw=torch.softmax(m.gate(st).squeeze(-1),1); weighted=(st*gw[:,:,None]).sum(1)
                main=m.head(torch.cat(zs+[weighted],1)).squeeze(-1)
            out[esc]=main.float()
        return out,float(esc.float().mean().item())

    class GPUMonitor:
        def __init__(self,interval=.1): self.interval=interval; self.rows=[]; self.stop_evt=threading.Event(); self.th=None
        def _run(self):
            while not self.stop_evt.is_set():
                try:
                    raw=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]
                    u,m=[float(x.strip()) for x in raw.split(',')[:2]]; self.rows.append((time.time(),u,m))
                except Exception: pass
                time.sleep(self.interval)
        def __enter__(self): self.th=threading.Thread(target=self._run,daemon=True); self.th.start(); return self
        def __exit__(self,*a): self.stop_evt.set(); self.th.join(timeout=2)

    def state_bytes(d): return sum(p.stat().st_size for p in Path(d).rglob('*') if p.is_file())

    def bench_system(name,m,mode,rows_b1,rows_tput,batch,use_dom=True):
        # Warm-up
        for _ in range(5):
            x=rows_b1.iloc[:min(8,len(rows_b1))]
            if mode=='url': forward_url_branch(m,x)
            elif mode=='cascade': forward_cascade(m,x)
            else: forward_full(m,x,use_dom)
        torch.cuda.synchronize()
        # B=1 latency distribution
        lat=[]; esc=[]; torch.cuda.reset_peak_memory_stats()
        for i in range(len(rows_b1)):
            x=rows_b1.iloc[i:i+1]
            t0=time.perf_counter()
            if mode=='url': forward_url_branch(m,x)
            elif mode=='cascade': _,e=forward_cascade(m,x); esc.append(e)
            else: forward_full(m,x,use_dom)
            torch.cuda.synchronize(); lat.append((time.perf_counter()-t0)*1000)
        peak_vram=torch.cuda.max_memory_allocated()/1024**2
        # Batched throughput (repeat)
        times=[]; esc2=[]; util=[]; mem=[]; peak_vram_tput=[]
        for r in range(BENCH_REPEATS):
            torch.cuda.reset_peak_memory_stats(); t0=time.perf_counter()
            with GPUMonitor(GPU_POLL_S) as mon:
                for st in range(0,len(rows_tput),batch):
                    x=rows_tput.iloc[st:st+batch]
                    if mode=='url': forward_url_branch(m,x)
                    elif mode=='cascade': _,e=forward_cascade(m,x); esc2.append(e*len(x))
                    else: forward_full(m,x,use_dom)
                torch.cuda.synchronize()
            times.append(time.perf_counter()-t0)
            peak_vram_tput.append(torch.cuda.max_memory_allocated()/1024**2)
            if mon.rows:
                util.extend([x[1] for x in mon.rows]); mem.extend([x[2] for x in mon.rows])
        return {
            'system':name,'mode':mode,'batch_throughput':batch,'n_b1':len(rows_b1),'n_throughput':len(rows_tput),'repeats':BENCH_REPEATS,
            'latency_b1_median_ms':float(np.median(lat)),'latency_b1_p95_ms':float(np.percentile(lat,95)),
            'throughput_pages_s_mean':float(np.mean([len(rows_tput)/t for t in times])),
            'throughput_pages_s_std':float(np.std([len(rows_tput)/t for t in times],ddof=1)) if len(times)>1 else 0.0,
            'peak_vram_b1_allocated_mb':float(peak_vram),
            'peak_vram_throughput_allocated_mb':float(max(peak_vram_tput)) if peak_vram_tput else np.nan,
            'gpu_util_mean_pct':float(np.mean(util)) if util else np.nan,
            'nvidia_memory_used_mean_mb':float(np.mean(mem)) if mem else np.nan,
            'cascade_escalation_rate_sample':float(sum(esc2)/(len(rows_tput)*BENCH_REPEATS)) if mode=='cascade' else np.nan,
        }

    TRI=load_deep(TRI_DIR,True)
    ops=[]
    ops.append(bench_system('URL_ONLY_FROM_TRI',TRI,'url',DEV_B1,DEV_TPUT,BENCH_BATCH,True))
    ops.append(bench_system('TRI_R1_200K_FULL',TRI,'full',DEV_B1,DEV_TPUT,BENCH_BATCH,True))
    ops.append(bench_system('TRI_R1_200K_CASCADE',TRI,'cascade',DEV_B1,DEV_TPUT,BENCH_BATCH,True))
    for x in ops: x['model_state_bytes']=state_bytes(TRI_DIR)
    del TRI; gc.collect(); torch.cuda.empty_cache()

    DUAL=load_deep(DUAL_DIR,False)
    x=bench_system('DUAL_R1_200K_FULL',DUAL,'full',DEV_B1,DEV_TPUT,BENCH_BATCH,False); x['model_state_bytes']=state_bytes(DUAL_DIR); ops.append(x)
    del DUAL; gc.collect(); torch.cuda.empty_cache()

    OPS=pd.DataFrame(ops)
    full_tput=float(OPS.loc[OPS.system=='TRI_R1_200K_FULL','throughput_pages_s_mean'].iloc[0]); casc_tput=float(OPS.loc[OPS.system=='TRI_R1_200K_CASCADE','throughput_pages_s_mean'].iloc[0])
    OPS['vs_tri_full_throughput_speedup']=OPS.throughput_pages_s_mean/full_tput
    OPS.to_csv(RESULTS/'OPERATIONAL_BENCHMARK.csv',index=False)
    print(OPS.to_string(index=False))

    # Combined reporting table. Hardware/scope remain explicit; values are not treated as same-device microbenchmarks.
    OPS_REPORT=OPS.copy()
    OPS_REPORT['hardware']='GPU'
    OPS_REPORT['measurement_scope']='in-memory URL/text/stored-DOM -> model forward; excludes disk/network/browser/raw HTML parsing'
    XGB_REPORT=XGB_OP.copy()

    # Normalize selected common columns without hiding architecture-specific metrics.
    OPS_REPORT.to_csv(RESULTS/'DEEP_OPERATIONAL_BENCHMARK.csv',index=False)
    with open(AUDIT/'OPERATIONAL_SCOPE.txt','w',encoding='utf-8') as f:
        f.write(
            'Deep benchmark: GPU, in-memory URL/text/stored DOM through model serving path.\\n'
            'XGB benchmark: CPU, in-memory URL/text through TF-IDF and XGB margin prediction.\\n'
            'Both exclude disk I/O, network acquisition, browser rendering and raw HTML parsing.\\n'
            'Do not equate full-path reduction with latency reduction; use measured throughput/latency.\\n'
        )


# CELL 19
# 14 — Reuse v7.2 deep score caches when available; otherwise recompute selected TRI scores
if not RUN_FINALIZE:
    print({'RISK_SCORE_PREPARATION':'DEFERRED','session_part':SESSION_PART})
else:
    # This keeps risk-band analysis cheap when the full v7.2 output dataset is attached.

    def _score_tri_microbatches(m,rows,batch_size=RISK_SCORE_BATCH):
        """OOM-safe TRI inference for already aligned in-memory rows."""
        full_parts=[]; url_parts=[]
        for bst in range(0,len(rows),batch_size):
            xb=rows.iloc[bst:bst+batch_size].reset_index(drop=True)
            with torch.inference_mode():
                u=tok_url(xb); t=tok_text(xb); g=graph_batch(xb)
                with amp_ctx(): main,url=m(u,t,g)
            full_parts.append(main.float().cpu().numpy())
            url_parts.append(url.float().cpu().numpy())
            del u,t,g,main,url,xb
        return np.concatenate(full_parts),np.concatenate(url_parts)

    def load_or_score_selected(role):
        score_dir=V7_ROOT/'scores/DEEP/TRI_R1_200K'
        gp=score_dir/f'{role}_gated_logit.npy'; up=score_dir/f'{role}_url_logit.npy'
        if gp.exists() and up.exists():
            return np.asarray(np.load(gp,mmap_mode='r'),dtype=np.float32),np.asarray(np.load(up,mmap_mode='r'),dtype=np.float32),'V7_CACHE'
        # Fallback scoring (slower). DOM is read in aligned 5k slices, but GPU inference is micro-batched.
        print(f'WARNING: v7 score cache missing for {role}; recomputing selected TRI logits with batch={RISK_SCORE_BATCH}.')
        gc.collect(); torch.cuda.empty_cache()
        m=load_deep(TRI_DIR,True); n=COUNTS[role]; gscore=np.empty(n,np.float32); uscore=np.empty(n,np.float32)
        base=read_role(role,['sha256','url','text','label'] + (['date'] if role=='FINAL' else []))
        base['sha256']=base.sha256.astype(str).str.lower()
        for st in range(0,n,5000):
            en=min(st+5000,n)
            x=base.iloc[st:en].copy().reset_index(drop=True)
            ddom=read_role_slice(role,st,en,['sha256']+DOM_COLS); ddom['sha256']=ddom.sha256.astype(str).str.lower()
            if not np.array_equal(x.sha256.to_numpy(),ddom.sha256.to_numpy()):
                raise RuntimeError(f'{role} DOM SHA alignment failed {st}:{en}')
            for c in DOM_COLS: x[c]=ddom[c].tolist()
            fs,us=_score_tri_microbatches(m,x,RISK_SCORE_BATCH)
            gscore[st:en]=fs; uscore[st:en]=us
            del fs,us,x,ddom
            print({'RISK_SCORE':role,'rows':en,'total':n,'gpu_batch':RISK_SCORE_BATCH})
        del m,base; gc.collect(); torch.cuda.empty_cache()
        return gscore,uscore,'RECOMPUTED'

    # Clear allocator state left by the operational microbenchmark before risk-score inference.
    gc.collect(); torch.cuda.empty_cache()
    CAL_FULL,CAL_URL,cal_src=load_or_score_selected('CAL')
    FINAL_FULL,FINAL_URL,fin_src=load_or_score_selected('FINAL')

    # DEV is not part of the v7 CAL/FINAL score cache. Score it once with the same OOM-safe micro-batch.
    gc.collect(); torch.cuda.empty_cache()
    TRI=load_deep(TRI_DIR,True)
    DEV_FULL,DEV_URL=_score_tri_microbatches(TRI,DEV_OP,RISK_SCORE_BATCH)
    print({'RISK_SCORE':'DEV','rows':len(DEV_OP),'total':len(DEV_OP),'gpu_batch':RISK_SCORE_BATCH})
    del TRI; gc.collect(); torch.cuda.empty_cache()

    # Reproduce v7 cascade raw scores: confident URL decisions become +/- infinity-like logits.
    def cascade_scores(full,url):
        s=np.asarray(full,dtype=np.float32).copy(); u=np.asarray(url,dtype=np.float32); s[u>=URL_HIGH]=1e9; s[u<=URL_LOW]=-1e9; return s
    CAL_CASCADE=cascade_scores(CAL_FULL,CAL_URL); FINAL_CASCADE=cascade_scores(FINAL_FULL,FINAL_URL); DEV_CASCADE=cascade_scores(DEV_FULL,DEV_URL)
    print({'CAL_SCORE_SOURCE':cal_src,'FINAL_SCORE_SOURCE':fin_src})

# CELL 20
# 15 — Operational risk bands + workload projections
if not RUN_FINALIZE:
    print({'RISK_BANDS':'DEFERRED','session_part':SESSION_PART})
else:
    # Raw logits are ordinal risk scores, not calibrated probabilities.

    def thr_tpr(pos_scores,target):
        s=np.asarray(pos_scores,float)
        if len(s)==0:return -np.inf
        try:return float(np.quantile(s,1-float(target),method='lower'))
        except TypeError:return float(np.quantile(s,1-float(target),interpolation='lower'))

    def band_metrics(y,s,low_th,high_th):
        y=np.asarray(y,int); s=np.asarray(s,float)
        band=np.where(s<low_th,'LOW',np.where(s>=high_th,'HIGH','REVIEW'))
        out={}
        for b in ['LOW','REVIEW','HIGH']:
            m=band==b
            out[f'{b.lower()}_rate']=float(m.mean())
            out[f'{b.lower()}_benign_rate']=float(m[y==0].mean()) if (y==0).any() else np.nan
            out[f'{b.lower()}_phish_rate']=float(m[y==1].mean()) if (y==1).any() else np.nan
        out['phish_auto_clear_rate']=out['low_phish_rate']
        out['benign_high_risk_rate']=out['high_benign_rate']
        return out

    def projected_workload(class_cond,prevalence):
        p=float(prevalence); b=1-p
        return {
            'prevalence_assumption':p,
            'projected_low_rate':b*class_cond['low_benign_rate']+p*class_cond['low_phish_rate'],
            'projected_review_rate':b*class_cond['review_benign_rate']+p*class_cond['review_phish_rate'],
            'projected_high_rate':b*class_cond['high_benign_rate']+p*class_cond['high_phish_rate'],
            'projected_false_high_per_100k':100000*b*class_cond['high_benign_rate'],
            'projected_true_high_per_100k':100000*p*class_cond['high_phish_rate'],
            'projected_phish_auto_clear_per_100k':100000*p*class_cond['low_phish_rate'],
        }

    RISK_ROWS=[]; PROJ=[]; THRESH={}
    for system,dev_s,cal_s,fin_s in [('FULL_TRI',DEV_FULL,CAL_FULL,FINAL_FULL),('CASCADE',DEV_CASCADE,CAL_CASCADE,FINAL_CASCADE)]:
        low=thr_tpr(dev_s[DEVY==1],LOW_RISK_TPR_TARGET)
        high=thr_fpr(cal_s,HIGH_RISK_FPR_TARGET)
        if not low<high:
            raise RuntimeError(f'Risk-band policy invalid for {system}: low threshold {low} >= high threshold {high}. Do not silently retune on FINAL.')
        THRESH[system]={'low_threshold_from_DEV_TPR99':low,'high_threshold_from_CAL_FPR005':high}
        for sn,mask in FINAL_MASKS.items():
            y=FINAL_META.y.to_numpy(int)[mask]; s=fin_s[mask]; bm=band_metrics(y,s,low,high)
            row={'system':system,'scenario':sn,'low_threshold':low,'high_threshold':high,**bm}; RISK_ROWS.append(row)
            for p in PREVALENCE_SCENARIOS:
                PROJ.append({'system':system,'scenario':sn,**projected_workload(bm,p)})

    RISK=pd.DataFrame(RISK_ROWS); RISK.to_csv(RESULTS/'RISK_BANDS_FINAL_OOD.csv',index=False)
    PROJ=pd.DataFrame(PROJ); PROJ.to_csv(RESULTS/'BUSINESS_WORKLOAD_PREVALENCE_SCENARIOS.csv',index=False)
    atomic_json(AUDIT/'RISK_BAND_THRESHOLDS.json',{
        'status':'POST_HOC_OPERATIONAL_POLICY','score_type':'raw_fp32_logit_ordinal_not_probability',
        'low_policy':f'DEV threshold retaining >= {LOW_RISK_TPR_TARGET:.1%} phishing','high_policy':f'CAL threshold targeting <= {HIGH_RISK_FPR_TARGET:.2%} benign FPR',
        'thresholds':THRESH,'prevalence_scenarios_are_assumptions':PREVALENCE_SCENARIOS
    })
    print(RISK[RISK.scenario=='OFFICIAL_TEST'].to_string(index=False))

# CELL 22
# 16 — Final-system stratified weighted bootstrap: AP and P@R90, 1000 replicates
if not RUN_FINALIZE:
    print({'FINAL_BOOTSTRAP':'DEFERRED','session_part':SESSION_PART})
else:
    y_boot=FINAL_META.y.to_numpy(int)
    score_map={
        'FULL_TRI':np.asarray(FINAL_FULL,dtype=np.float64),
        'CASCADE':np.asarray(FINAL_CASCADE,dtype=np.float64),
    }

    for k,s in score_map.items():
        if len(s)!=len(y_boot):
            raise RuntimeError(f'Bootstrap alignment mismatch for {k}: {len(s)} != {len(y_boot)}')

    def prepare_score_groups(y,s):
        order=np.argsort(-np.asarray(s,float),kind='mergesort')
        ys=np.asarray(y,int)[order]
        ss=np.asarray(s,float)[order]
        starts=np.r_[0,np.flatnonzero(ss[1:]!=ss[:-1])+1]
        return {
            'order':order,
            'starts':starts,
            'pos_sorted':(ys==1).astype(np.int8),
            'neg_sorted':(ys==0).astype(np.int8),
        }

    def weighted_pr_metrics(counts,prep,n_pos):
        c=counts[:,prep['order']]
        grouped_pos=np.add.reduceat(
            c*prep['pos_sorted'][None,:],prep['starts'],axis=1
        )
        grouped_neg=np.add.reduceat(
            c*prep['neg_sorted'][None,:],prep['starts'],axis=1
        )

        cum_tp=np.cumsum(grouped_pos,axis=1,dtype=np.float64)
        cum_fp=np.cumsum(grouped_neg,axis=1,dtype=np.float64)
        precision=cum_tp/np.maximum(cum_tp+cum_fp,1.0)
        recall=cum_tp/float(n_pos)

        # Non-interpolated AP with weighted score groups.
        ap=np.sum((grouped_pos/float(n_pos))*precision,axis=1)

        # Same definition used elsewhere: best precision among thresholds reaching >=90% recall.
        pr90=np.max(np.where(recall>=.90,precision,-np.inf),axis=1)
        return ap,pr90

    prep={k:prepare_score_groups(y_boot,s) for k,s in score_map.items()}

    pos_idx=np.flatnonzero(y_boot==1)
    neg_idx=np.flatnonzero(y_boot==0)
    n_pos=len(pos_idx); n_neg=len(neg_idx); n=len(y_boot)

    rng=np.random.default_rng(BOOTSTRAP_SEED)
    p_pos=np.full(n_pos,1.0/n_pos,dtype=np.float64)
    p_neg=np.full(n_neg,1.0/n_neg,dtype=np.float64)

    values={k:{'AP':[],'P_at_R90':[]} for k in BOOTSTRAP_SYSTEMS}

    completed=0
    for st in range(0,BOOTSTRAP_REPS,BOOTSTRAP_BATCH):
        b=min(BOOTSTRAP_BATCH,BOOTSTRAP_REPS-st)

        # Exact class-stratified nonparametric bootstrap represented as observation counts.
        pos_counts=rng.multinomial(n_pos,p_pos,size=b).astype(np.int16,copy=False)
        neg_counts=rng.multinomial(n_neg,p_neg,size=b).astype(np.int16,copy=False)

        counts=np.zeros((b,n),dtype=np.int16)
        counts[:,pos_idx]=pos_counts
        counts[:,neg_idx]=neg_counts
        del pos_counts,neg_counts

        for system in BOOTSTRAP_SYSTEMS:
            ap,pr90=weighted_pr_metrics(counts,prep[system],n_pos)
            values[system]['AP'].extend(ap.tolist())
            values[system]['P_at_R90'].extend(pr90.tolist())

        completed+=b
        if completed%100==0 or completed==BOOTSTRAP_REPS:
            print({'BOOTSTRAP_DONE':completed,'of':BOOTSTRAP_REPS})

    rows=[]
    for system in BOOTSTRAP_SYSTEMS:
        point=curves(y_boot,score_map[system])
        for metric in ['AP','P_at_R90']:
            vals=np.asarray(values[system][metric],float)
            rows.append({
                'system':system,
                'dataset':'FINAL',
                'scenario':'OFFICIAL_TEST',
                'metric':metric,
                'point_estimate':float(point[metric]),
                'bootstrap_reps':int(BOOTSTRAP_REPS),
                'bootstrap_seed':int(BOOTSTRAP_SEED),
                'ci95_lo':float(np.quantile(vals,.025)),
                'ci95_hi':float(np.quantile(vals,.975)),
                'bootstrap_mean':float(vals.mean()),
                'bootstrap_std':float(vals.std(ddof=1)),
                'method':'class-stratified nonparametric weighted bootstrap on fixed raw scores'
            })

    BOOT=pd.DataFrame(rows)
    BOOT.to_csv(RESULTS/'FINAL_DEEP_BOOTSTRAP_AP_PR90_1000.csv',index=False)

    atomic_json(AUDIT/'FINAL_BOOTSTRAP_PROTOCOL.json',{
        'status':'PASS',
        'reps':BOOTSTRAP_REPS,
        'seed':BOOTSTRAP_SEED,
        'systems':BOOTSTRAP_SYSTEMS,
        'dataset':'FINAL/OFFICIAL_TEST',
        'metrics':['AP','P_at_R90'],
        'stratified_by_class':True,
        'same_resamples_across_systems':True,
        'additional_training':False,
        'additional_model_inference':False,
        'tie_handling':'equal score values aggregated before weighted PR computation'
    })
    print(BOOT.to_string(index=False))

# CELL 23
# 16b — Statistical decision summary
stat_file=RESULTS/'STATISTICAL_N5_HOLM_FAMILIES.csv'
if stat_file.exists():
    st=pd.read_csv(stat_file)
    by_family=st.groupby('holm_family').agg(
        tests=('exact_signflip_p','size'),
        min_raw_p=('exact_signflip_p','min'),
        min_holm_p=('holm_p','min'),
        raw_p_lt_0p05=('raw_p_lt_0p05','sum'),
        holm_p_lt_0p05=('holm_p_lt_0p05','sum'),
        positive_pairs_min=('positive_runs','min'),
        positive_pairs_max=('positive_runs','max'),
    ).reset_index()

    by_family.to_csv(RESULTS/'STATISTICAL_DECISION_SUMMARY.csv',index=False)
    atomic_json(AUDIT/'STATISTICAL_DECISION_INTERPRETATION.json',{
        'status':'PASS',
        'alpha':0.05,
        'significance_is_required_for_valid_result':False,
        'n5_minimum_exact_two_sided_signflip_p':0.0625,
        'deep_n3_minimum_exact_two_sided_signflip_p':0.25,
        'interpretation_rule':
            'Report paired effect size, CI, positive-run count, raw exact p and Holm p. '
            'Do not claim statistical significance unless adjusted p<0.05 actually occurs.',
        'preferred_non_significance_language':
            'consistent/richtungsstabiler gepaarter Effekt with quantified uncertainty'
    })
    print(by_family.to_string(index=False))

# CELL 24
# 17 — Final evidence package + completion marker
# FF2 primary evidence: N5 R0/R1 label curve. Deep-N3 is a secondary end-to-end robustness check.
# XGBoost is a separate classical system reference and never replaces R0 as the SSL control.
# Mandatory completion checks
required_results=[
    'LABEL_EFFICIENCY_R0_R1_N5_ALL.csv',
    'SSL_DELTA_R1_MINUS_R0_N5.csv',
    'STATISTICAL_N5_HOLM_FAMILIES.csv',
    'STATISTICAL_DECISION_SUMMARY.csv',
    'XGB_CLASSICAL_ALL.csv',
    'XGB_CLASSICAL_N3_ALL_BUDGETS_AGG.csv',
    'DEEP_N3_R0_R1_20K_200K_ALL.csv',
    'DEEP_N3_SSL_DELTA_R1_MINUS_R0.csv',
    'STATISTICAL_DEEP_N3_HOLM_STABILITY.csv',
    'OPERATIONAL_BENCHMARK.csv',
    'XGB_OPERATIONAL_BENCHMARK.csv',
    'RISK_BANDS_FINAL_OOD.csv',
    'BUSINESS_WORKLOAD_PREVALENCE_SCENARIOS.csv',
    'FINAL_DEEP_BOOTSTRAP_AP_PR90_1000.csv',
]
missing=[x for x in required_results if not (RESULTS/x).exists()]

if missing and SESSION_PART!='ALL' and not RUN_FINALIZE:
    atomic_json(ROOT/'V8_4_SESSION_CHECKPOINT_COMPLETE.json',{
        'status':'PARTIAL_SESSION_COMPLETE',
        'session_part':SESSION_PART,
        'missing_final_outputs':missing,
        'instruction':'Save this Kaggle notebook output, attach it to the next session, and continue with the next SESSION_PART.'
    })
    checkpoint_status()
    print({
        'V8_4_SESSION':'PARTIAL_COMPLETE',
        'session_part':SESSION_PART,
        'missing_final_outputs':missing,
        'NEXT_STEP':'Save Version / output, attach it to the next session and resume.'
    })
else:
    if missing:
        raise RuntimeError(f'V8.3 finalization incomplete; missing outputs: {missing}')

if not missing:
    xgb_primary_audit=pd.read_csv(AUDIT/'XGB_LOW_FPR_RESOLUTION_AUDIT.csv')
    xgb_primary_audit=xgb_primary_audit[
        xgb_primary_audit.seed.isin(XGB_N3_SEEDS)&
        xgb_primary_audit.budget.isin(XGB_N3_BUDGETS)&
        np.isclose(xgb_primary_audit.target_fpr,PRIMARY_FPR)
    ]
    xgb_low_fpr_all_valid=bool(xgb_primary_audit.valid_for_matched_low_fpr.all())

    atomic_json(AUDIT/'V8_4_COMPLETION_AUDIT.json',{
        'status':'PASS',
        'required_results_present':True,
        'xgb_primary_low_fpr_all_valid':xgb_low_fpr_all_valid,
        'xgb_primary_resolution_rows':xgb_primary_audit.to_dict(orient='records'),
        'tok_helpers_defined_before_deep_scoring':True,
        'r0_remains_ssl_control':True,
        'xgb_role':'classical system reference',
    })

    summary={
        'status':'COMPLETE',
        'version':VERSION,
        'protocol_status':'POST_HOC_EXTENSION_AFTER_V7_2',
        'label_curve':{
            'budgets':BUDGETS,'paired_replicates':5,'representations':['R0','R1'],'probes':['LINEAR','MLP'],
            'same_rows_R0_R1_within_run':True,'nested_budgets_within_run':True,
            'primary_metric':'TPR@0.5% FPR','sweetspot_selection_domain':'DEV_ONLY',
            'files':['LABEL_EFFICIENCY_R0_R1_N5_ALL.csv','LABEL_EFFICIENCY_R0_R1_N5_AGG.csv','SSL_DELTA_R1_MINUS_R0_N5.csv',
    'STATISTICAL_N5_HOLM_FAMILIES.csv',
    'STATISTICAL_DECISION_SUMMARY.csv','TECHNICAL_SWEETSPOT_DEV_CANDIDATES.csv','MARGINAL_LABEL_UTILITY_DEV.csv']
        },

    'classical_xgb_baseline':{
        'scientific_role':'classical system reference; NOT SSL control',
        'feature_pipeline':'URL char TF-IDF 3-5 + HTML-text word TF-IDF 1-2 -> XGBoost',
        'tfidf_fit_domain':'SSL 200k inputs only; labels unused for vectorizer fitting',
        'curve_seed':XGB_CURVE_SEED,
        'curve_budgets':BUDGETS,
        'n3_seeds':XGB_N3_SEEDS,
        'n3_budgets':BUDGETS,
        'raw_margin_thresholding':True,
        'files':['XGB_CLASSICAL_ALL.csv','XGB_CLASSICAL_LABEL_CURVE_SEED82_PRIMARY.csv',
                 'XGB_CLASSICAL_N3_ALL_BUDGETS_AGG.csv','XGB_CLASSICAL_TRAINING_TIMING.csv',
                 'SYSTEM_REFERENCE_XGB_VS_DEEP_N3_PRIMARY.csv','XGB_OPERATIONAL_BENCHMARK.csv'],
        'audit_files':['XGB_LOW_FPR_RESOLUTION_AUDIT.csv','XGB_TFIDF_FEATURE_AUDIT.json']
    },

'statistical_robustness':{
    'n5_file':'STATISTICAL_N5_HOLM_FAMILIES.csv',
    'decision_summary':'STATISTICAL_DECISION_SUMMARY.csv',
    'deep_n3_file':'STATISTICAL_DEEP_N3_HOLM_STABILITY.csv',
    'n5_test':'exact two-sided sign-flip with Holm correction in predefined families',
    'n5_min_two_sided_p':0.0625,
    'deep_n3_min_two_sided_p':0.25,
    'interpretation':'N5 effect/CI/directional stability; Deep-N3 optimization-seed stability',
    'final_system_bootstrap_file':'FINAL_DEEP_BOOTSTRAP_AP_PR90_1000.csv',
    'final_system_bootstrap_reps':BOOTSTRAP_REPS
},
        'deep_n3':{
            'seeds':DEEP_N3_SEEDS,'budgets':DEEP_N3_BUDGETS,'architecture':'DUAL','fixed_20k_rows_across_seeds':True,
            'seed82_reused_from_v7_2_when_available':True,
            'files':['DEEP_N3_R0_R1_20K_200K_ALL.csv','DEEP_N3_R0_R1_20K_200K_AGG.csv','DEEP_N3_SSL_DELTA_R1_MINUS_R0.csv',
    'STATISTICAL_DEEP_N3_HOLM_STABILITY.csv','DEEP_N3_PRIMARY_OFFICIAL_0p5FPR.csv']
        },
        'operational':{
            'deep_file':'OPERATIONAL_BENCHMARK.csv','xgb_file':'XGB_OPERATIONAL_BENCHMARK.csv',
            'benchmark_excludes':['disk_io','network_acquisition','browser_rendering','raw_html_parsing'],
            'hardware_note':'Deep benchmark uses GPU; XGB benchmark uses CPU. Keep hardware and measurement scope explicit.'
        },
        'risk_scoring':{'score_interpretation':'ordinal raw logit, not calibrated probability','file':'RISK_BANDS_FINAL_OOD.csv','workload_file':'BUSINESS_WORKLOAD_PREVALENCE_SCENARIOS.csv'},
        'guardrails':[
            'Do not call the 95% retention point an economic optimum without measured annotation/infrastructure costs.',
            'Do not infer universal SSL superiority; report the measured R1-R0 curve and its dependence on label budget.',
            'N5 exact two-sided sign-flip cannot reach p<0.05; report effects/CI/directional stability plus raw and Holm-adjusted p-values transparently.',
            'Deep-N3 exact two-sided sign-flip has minimum p=0.25; use it as optimization-seed stability evidence, not as a significance claim.',
            'XGB is an additional classical system reference. Never interpret XGB-vs-R1 as the causal SSL effect; that remains R1-vs-R0.',
            'If XGB low-FPR realized/target ratio fails the resolution audit, do not present that operating point as a matched 0.5%-FPR comparison.',
            'Earlier FINAL results were historically known; this is post-hoc robustness/operational evidence.',
            'Prevalence projections are scenarios, not measured production prevalence.',
            'Full-path reduction is not equivalent to latency reduction; use the measured operational benchmark.'
        ]
    }
    atomic_json(RESULTS/'V8_EVIDENCE_SUMMARY.json',summary)

    # ZIP only compact results / figures / audits. Large embeddings and model checkpoints stay outside the package.
    import zipfile
    pkg=ROOT/'phreshphish_v8_4_FINAL_RESULTS.zip'
    with zipfile.ZipFile(pkg,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in [RESULTS,AUDIT,FIG]:
            for p in folder.rglob('*'):
                if p.is_file(): z.write(p,p.relative_to(ROOT))
    atomic_json(ROOT/'V8_4_FINAL_COMPLETE.json',{'status':'COMPLETE','package':str(pkg),'version':VERSION})
    print(json.dumps(summary,indent=2))
    print({'RESULT_PACKAGE':str(pkg)})