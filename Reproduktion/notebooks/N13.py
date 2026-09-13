# KI-Unterstuetzung: siehe Erklaerung der Arbeit.
# Ursprungsdatei: PUTRA_TARGET_ADAPTATION_LABEL_EFFICIENCY_N10_v1.ipynb
# CELL 1
# 00 — Imports, Protokoll und feste Parameter
import os, gc, re, json, math, time, random, hashlib, itertools, zipfile, shutil, warnings, html as html_std
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import t as student_t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, roc_auc_score
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings('ignore')

VERSION = 'PUTRA_TARGET_ADAPTATION_LABEL_EFFICIENCY_N10_v1'
MASTER_SEED = 20260911
SSL_SEED = 20260812

# Temporal external design
EARLY_FRACTION = 0.75
LABEL_BUDGETS = [200, 500, 1000, 2000]
N_REPS = 10
MODEL_SEEDS = [142, 162, 182, 202, 222, 242, 262, 282, 302, 322]
LABEL_RANK_SEEDS = [9101, 9102, 9103, 9104, 9105, 9106, 9107, 9108, 9109, 9110]

# Target-domain MLM: intentionally mirrors the existing DAPT hyperparameters
TARGET_DAPT_EPOCHS = 1
TARGET_DAPT_BATCH_CANDIDATES = [16, 8, 4]
TARGET_DAPT_LR = 5e-5
TARGET_DAPT_WEIGHT_DECAY = 0.01
TARGET_DAPT_WARMUP_FRAC = 0.06
TARGET_MLM_PROB = 0.15

URL_MAX_LEN = 128
TEXT_MAX_LEN = 256
TEXT_MAX_CHARS = 50_000
BERT_REVISION = '86b5e0934494bd15c9632b12f734a8a67f723594'

LINEAR_SPEC = {
    'C': 1.0,
    'solver': 'liblinear',
    'class_weight': 'balanced',
    'max_iter': 2000,
}

PUTRA_DOI = '10.5281/zenodo.8041387'
PUTRA_FILES = {
    'phishing.csv': {
        'url': 'https://zenodo.org/records/8041387/files/phishing.csv?download=1',
        'md5': '513962464c413fc30b2030547a12868a',
        'label': 1,
    },
    'not-phishing.csv': {
        'url': 'https://zenodo.org/records/8041387/files/not-phishing.csv?download=1',
        'md5': 'f5d218eb67f5d7bd0571e8089a8fc392',
        'label': 0,
    },
}

SEARCH_ROOT = Path('/kaggle/input') if Path('/kaggle/input').exists() else Path('/mnt/data')
ROOT = Path('/kaggle/working/putra_target_adaptation_v1') if Path('/kaggle/working').exists() else Path('/mnt/data/putra_target_adaptation_v1')
DATA_DIR = ROOT / 'data'
CKPT_DIR = ROOT / 'target_adapted_checkpoints'
EMB_DIR = ROOT / 'embeddings'
RESULTS_DIR = ROOT / 'results'
AUDIT_DIR = ROOT / 'audit'
FIG_DIR = ROOT / 'figures'
CACHE_DIR = ROOT / '_cache'
for p in [ROOT, DATA_DIR, CKPT_DIR, EMB_DIR, RESULTS_DIR, AUDIT_DIR, FIG_DIR, CACHE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError('GPU erforderlich. In Kaggle unter Settings -> Accelerator eine GPU aktivieren.')
DEVICE = torch.device('cuda')
AMP = True


def seed_all(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    os.replace(tmp, path)


def md5_file(path, chunk=8*1024*1024):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


PROTOCOL = {
    'version': VERSION,
    'status': 'POST_HOC_EXTERNAL_TARGET_ADAPTATION_ANALYSIS',
    'prior_direct_putra_screen_seen': True,
    'no_optional_stopping': True,
    'all_planned_replicates_reported': True,
    'temporal_design': f'global scan_date cutoff; early <= empirical {EARLY_FRACTION:.2f} quantile, late > cutoff',
    'variants': {
        'E0_SOURCE_RUT': 'existing PhreshPhish RUT URL+Text encoders, frozen',
        'E1_TARGET_RUT': 'same RUT encoders plus one MLM epoch on unlabeled early Putra target pool',
    },
    'target_ssl': {
        'epochs': TARGET_DAPT_EPOCHS,
        'batch_candidates': TARGET_DAPT_BATCH_CANDIDATES,
        'lr': TARGET_DAPT_LR,
        'weight_decay': TARGET_DAPT_WEIGHT_DECAY,
        'warmup_fraction': TARGET_DAPT_WARMUP_FRAC,
        'mlm_probability': TARGET_MLM_PROB,
        'labels_used': False,
    },
    'label_budgets': LABEL_BUDGETS,
    'model_seeds': MODEL_SEEDS,
    'label_rank_seeds': LABEL_RANK_SEEDS,
    'paired_same_labeled_rows_E0_E1': True,
    'nested_budgets_within_seed': True,
    'primary_endpoint': 'normalized area under AP learning curve over log10(label budget)',
    'primary_test': 'two-sided exact paired sign-flip test over N10 AULC differences',
    'secondary_AP_family': 'four paired AP tests, Holm-corrected across label budgets',
    'test_used_for_training_or_adaptation': False,
}
atomic_json(AUDIT_DIR / 'TARGET_ADAPTATION_PROTOCOL_LOCK.json', PROTOCOL)
seed_all(MASTER_SEED)

print({
    'version': VERSION,
    'device': torch.cuda.get_device_name(0),
    'search_root': str(SEARCH_ROOT),
    'budgets': LABEL_BUDGETS,
    'n_reps': N_REPS,
})

# CELL 3
# 01 — Optional frühere Putra-CSVs aus angehängten Ergebnis-ZIPs wiederverwenden
CACHE_NAMES = {'phishing.csv', 'not-phishing.csv'}
cache_hits = []

for zpath in SEARCH_ROOT.rglob('*.zip'):
    try:
        with zipfile.ZipFile(zpath) as zf:
            members = [n for n in zf.namelist() if Path(n).name in CACHE_NAMES]
            if not members:
                continue
            d = CACHE_DIR / hashlib.sha1(str(zpath).encode()).hexdigest()[:10]
            d.mkdir(parents=True, exist_ok=True)
            for n in members:
                target = d / Path(n).name
                if not target.exists():
                    with zf.open(n) as src, open(target, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
            cache_hits.append({'zip': str(zpath), 'files': [Path(n).name for n in members]})
    except zipfile.BadZipFile:
        pass

atomic_json(AUDIT_DIR / 'PUTRA_INPUT_CACHE_AUDIT.json', cache_hits)
print({'putra_cache_archives_found': len(cache_hits)})

# CELL 4
# 02 — CSVs laden, Text normalisieren, scan_date parsen, temporal splitten

def resolve_putra_file(name, spec):
    # 1) direkt angehängte Kaggle Inputs
    hits = [p for p in SEARCH_ROOT.rglob(name) if p.is_file()]
    # 2) extrahierter Cache
    hits += [p for p in CACHE_DIR.rglob(name) if p.is_file()]
    if hits:
        p = sorted(hits, key=lambda x: (len(str(x)), str(x)))[0]
    else:
        p = DATA_DIR / name
        if not p.exists():
            print('Download', name)
            urlretrieve(spec['url'], p)
    digest = md5_file(p)
    if digest != spec['md5']:
        raise RuntimeError(f'MD5 mismatch für {name}: {digest}')
    return p

frames = []
input_audit = {}
for name, spec in PUTRA_FILES.items():
    p = resolve_putra_file(name, spec)
    q = pd.read_csv(p, usecols=['url', 'features.text', 'scan_date'], low_memory=False)
    q['url'] = q['url'].fillna('').astype(str).str.strip()
    q['text'] = (
        q['features.text'].fillna('').astype(str)
        .map(html_std.unescape)
        .map(lambda s: re.sub(r'\s+', ' ', s).strip()[:TEXT_MAX_CHARS])
    )
    q['scan_dt'] = pd.to_datetime(q['scan_date'], errors='coerce', utc=True)
    q['label'] = int(spec['label'])
    q['source_file'] = name
    frames.append(q[['url', 'text', 'scan_dt', 'label', 'source_file']])
    input_audit[name] = {'path': str(p), 'rows': len(q), 'md5': md5_file(p)}

ALL = pd.concat(frames, ignore_index=True)
ALL['complete'] = ALL['url'].str.len().gt(0) & ALL['text'].str.len().gt(0) & ALL['scan_dt'].notna()
ALL = ALL[ALL['complete']].copy().reset_index(drop=True)

if len(ALL) < 5000 or ALL['label'].nunique() != 2:
    raise RuntimeError('Putra-Daten nach Complete-Case-Filter unerwartet klein oder einklassig.')

# Ein globaler Zeitcutoff: keine identischen Zeitstempel werden künstlich auf beide Seiten verteilt.
cutoff = ALL['scan_dt'].quantile(EARLY_FRACTION, interpolation='nearest')
EARLY = ALL[ALL['scan_dt'] <= cutoff].copy().reset_index(drop=True)
LATE_RAW = ALL[ALL['scan_dt'] > cutoff].copy().reset_index(drop=True)

# Konservativer Cross-split-Leakage-Schutz: exakte URLs, die bereits im frühen Pool vorkommen, aus Late entfernen.
early_urls = set(EARLY['url'].astype(str))
late_url_overlap_mask = LATE_RAW['url'].astype(str).isin(early_urls)
LATE = LATE_RAW[~late_url_overlap_mask].copy().reset_index(drop=True)

if min(EARLY['label'].value_counts()) < max(LABEL_BUDGETS)//2:
    raise RuntimeError('Früher Putra-Pool enthält für die geplanten balancierten Labelbudgets zu wenige Fälle einer Klasse.')
if min(LATE['label'].value_counts()) < 200:
    raise RuntimeError('Später Putra-Test ist für robuste Auswertung einer Klasse zu klein.')

SPLIT_AUDIT = {
    'putra_doi': PUTRA_DOI,
    'complete_rows': int(len(ALL)),
    'cutoff_utc': str(cutoff),
    'early_rows': int(len(EARLY)),
    'late_rows_before_overlap_filter': int(len(LATE_RAW)),
    'late_rows_after_overlap_filter': int(len(LATE)),
    'late_exact_url_overlaps_removed': int(late_url_overlap_mask.sum()),
    'early_date_range': [str(EARLY.scan_dt.min()), str(EARLY.scan_dt.max())],
    'late_date_range': [str(LATE.scan_dt.min()), str(LATE.scan_dt.max())],
    'early_class_counts': {str(k): int(v) for k,v in EARLY.label.value_counts().sort_index().items()},
    'late_class_counts': {str(k): int(v) for k,v in LATE.label.value_counts().sort_index().items()},
    'features_text_used': True,
    'features_html_used_as_raw_html': False,
    'inputs': input_audit,
}
atomic_json(AUDIT_DIR / 'PUTRA_TEMPORAL_SPLIT_AUDIT.json', SPLIT_AUDIT)

EARLY.to_parquet(DATA_DIR / 'putra_early_pool.parquet', index=False)
LATE.to_parquet(DATA_DIR / 'putra_late_test.parquet', index=False)
print(json.dumps(SPLIT_AUDIT, indent=2, ensure_ascii=False))

# CELL 6
# 03 — RUT-Ausgangscheckpoint robust auflösen

def resolve_checkpoint(preferred, needle):
    p = Path(preferred)
    if p.exists():
        return str(p)
    hits = []
    for cfg in SEARCH_ROOT.rglob('config.json'):
        d = cfg.parent
        s = str(d).lower()
        if needle.lower() in s and ((d/'model.safetensors').exists() or (d/'pytorch_model.bin').exists()):
            hits.append(d)
    if not hits:
        raise RuntimeError(f'Checkpoint nicht gefunden: {needle}')
    hits.sort(key=lambda x: (len(str(x)), str(x)))
    return str(hits[0])

URL_SOURCE = resolve_checkpoint(
    '/kaggle/input/newdataset/final_url_ssl_integrated_v2/checkpoints/URL_DAPT_BERT_200K',
    'URL_DAPT_BERT_200K'
)
TEXT_SOURCE = resolve_checkpoint(
    '/kaggle/input/datasets/cristinakaufalt/bigresults/phreshphish_FINAL_DEEP_TRIMODAL_SSL_v4_3_LABELSET_ALIGNMENT_FIX/checkpoints/R1_DAPT_TEXT',
    'R1_DAPT_TEXT'
)

# Lokalen RoBERTa-Tokenizer bevorzugen, da der Text-DAPT-Checkpoint denselben Tokenizer verwendet.
TEXT_TOKENIZER_SOURCE = 'roberta-base'
for cfg in SEARCH_ROOT.rglob('config.json'):
    d = cfg.parent
    if d.name == 'roberta-base' and ((d/'vocab.json').exists() or (d/'tokenizer.json').exists()):
        TEXT_TOKENIZER_SOURCE = str(d)
        break

HF_TOKEN = os.environ.get('HF_TOKEN')
if not HF_TOKEN and Path('/kaggle/working').exists():
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret('HF_TOKEN')
    except Exception:
        HF_TOKEN = None

SOURCE_AUDIT = {
    'URL_SOURCE_RUT': URL_SOURCE,
    'TEXT_SOURCE_RUT': TEXT_SOURCE,
    'TEXT_TOKENIZER_SOURCE': TEXT_TOKENIZER_SOURCE,
}
atomic_json(AUDIT_DIR / 'SOURCE_RUT_CHECKPOINTS.json', SOURCE_AUDIT)
print(SOURCE_AUDIT)

# CELL 8
# 04 — Kleine Dataset-Hülle und manuelle MLM-Schleife mit OOM-Fallback
class EncodedListDataset(Dataset):
    def __init__(self, enc):
        self.enc = enc
        self.n = len(next(iter(enc.values())))
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        return {k: torch.tensor(v[i], dtype=torch.long) for k,v in self.enc.items()}


def run_target_mlm(name, source_model, tokenizer_source, values, max_len, out_dir):
    out_dir = Path(out_dir)
    done = out_dir / 'COMPLETE.json'
    if done.exists() and (out_dir/'config.json').exists() and ((out_dir/'model.safetensors').exists() or (out_dir/'pytorch_model.bin').exists()):
        print({'REUSE_TARGET_MLM': name, 'path': str(out_dir)})
        return str(out_dir)

    tok = AutoTokenizer.from_pretrained(tokenizer_source, token=HF_TOKEN)
    vals = pd.Series(values).fillna('').astype(str).tolist()
    enc = tok(vals, truncation=True, max_length=max_len, padding=False, add_special_tokens=True)
    ds = EncodedListDataset(enc)
    del enc
    gc.collect()

    last_error = None
    for bs in TARGET_DAPT_BATCH_CANDIDATES:
        try:
            seed_all(SSL_SEED)
            model = AutoModelForMaskedLM.from_pretrained(source_model, token=HF_TOKEN).to(DEVICE)
            model.train()
            coll = DataCollatorForLanguageModeling(tokenizer=tok, mlm_probability=TARGET_MLM_PROB)
            dl = DataLoader(
                ds,
                batch_size=bs,
                shuffle=True,
                generator=torch.Generator().manual_seed(SSL_SEED),
                collate_fn=coll,
                num_workers=0,
            )
            opt = torch.optim.AdamW(model.parameters(), lr=TARGET_DAPT_LR, weight_decay=TARGET_DAPT_WEIGHT_DECAY)
            total_steps = len(dl) * TARGET_DAPT_EPOCHS
            sched = get_linear_schedule_with_warmup(opt, int(TARGET_DAPT_WARMUP_FRAC*total_steps), total_steps)
            scaler = torch.cuda.amp.GradScaler(enabled=AMP)
            losses = []
            t0 = time.time()

            for ep in range(TARGET_DAPT_EPOCHS):
                for step, batch in enumerate(dl, 1):
                    batch = {k:v.to(DEVICE) for k,v in batch.items()}
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.float16):
                        loss = model(**batch).loss
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    losses.append(float(loss.detach().cpu()))
                    if step % 100 == 0 or step == len(dl):
                        print({
                            'target_mlm': name,
                            'batch': bs,
                            'step': step,
                            'of': len(dl),
                            'loss100': round(float(np.mean(losses[-100:])), 4),
                        })

            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_dir, safe_serialization=True)
            tok.save_pretrained(out_dir)
            atomic_json(done, {
                'status': 'COMPLETE',
                'name': name,
                'source_model': str(source_model),
                'tokenizer_source': str(tokenizer_source),
                'target_rows': len(vals),
                'epochs': TARGET_DAPT_EPOCHS,
                'batch_size': bs,
                'lr': TARGET_DAPT_LR,
                'weight_decay': TARGET_DAPT_WEIGHT_DECAY,
                'warmup_fraction': TARGET_DAPT_WARMUP_FRAC,
                'mlm_probability': TARGET_MLM_PROB,
                'labels_used': False,
                'loss_mean': float(np.mean(losses)),
                'loss_last100': float(np.mean(losses[-100:])),
                'elapsed_minutes': (time.time()-t0)/60,
            })
            del model, dl, opt, sched, scaler
            gc.collect(); torch.cuda.empty_cache()
            return str(out_dir)

        except torch.cuda.OutOfMemoryError as e:
            last_error = e
            print({'OOM_RETRY': name, 'batch_size': bs})
            try:
                del model
            except Exception:
                pass
            gc.collect(); torch.cuda.empty_cache()

    raise RuntimeError(f'Target MLM {name} scheiterte für alle Batchgrößen.') from last_error

URL_TARGET = run_target_mlm(
    'URL_TARGET_MLM', URL_SOURCE, URL_SOURCE, EARLY['url'], URL_MAX_LEN,
    CKPT_DIR / 'URL_RUT_PLUS_PUTRA_MLM'
)
TEXT_TARGET = run_target_mlm(
    'TEXT_TARGET_MLM', TEXT_SOURCE, TEXT_TOKENIZER_SOURCE, EARLY['text'], TEXT_MAX_LEN,
    CKPT_DIR / 'TEXT_RUT_PLUS_PUTRA_MLM'
)

print({'URL_TARGET': URL_TARGET, 'TEXT_TARGET': TEXT_TARGET})

# CELL 10
# 05 — Mean-pooled Embeddings für EARLY und LATE

def masked_mean(h, m):
    mm = m.unsqueeze(-1).to(h.dtype)
    return (h*mm).sum(1) / mm.sum(1).clamp_min(1)

@torch.no_grad()
def embed_values(values, model_src, tokenizer_src, max_len, batch_size, out_path):
    out_path = Path(out_path)
    if out_path.exists():
        a = np.load(out_path, mmap_mode='r')
        if a.shape == (len(values), 768):
            print({'REUSE_EMBED': out_path.name, 'shape': a.shape})
            return out_path

    tok = AutoTokenizer.from_pretrained(tokenizer_src, token=HF_TOKEN)
    model = AutoModel.from_pretrained(model_src, token=HF_TOKEN).to(DEVICE).eval()
    if int(model.config.hidden_size) != 768:
        raise RuntimeError(f'Unerwartete Hidden-Dimension für {model_src}: {model.config.hidden_size}')

    arr = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float32, shape=(len(values), 768))
    vals = pd.Series(values).fillna('').astype(str).tolist()

    for st in range(0, len(vals), batch_size):
        en = min(st+batch_size, len(vals))
        b = tok(vals[st:en], padding=True, truncation=True, max_length=max_len, return_tensors='pt')
        b = {k:v.to(DEVICE) for k,v in b.items()}
        with torch.autocast('cuda', dtype=torch.float16):
            o = model(**b)
            z = masked_mean(o.last_hidden_state, b['attention_mask'])
        arr[st:en] = z.float().cpu().numpy()
        if en % 2000 < batch_size or en == len(vals):
            arr.flush()
            print({'embed': out_path.name, 'rows': en, 'total': len(vals)})

    arr.flush()
    del arr, model, tok
    gc.collect(); torch.cuda.empty_cache()
    return out_path

VARIANTS = {
    'E0_SOURCE_RUT': {'url_model': URL_SOURCE, 'url_tok': URL_SOURCE, 'text_model': TEXT_SOURCE, 'text_tok': TEXT_TOKENIZER_SOURCE},
    'E1_TARGET_RUT': {'url_model': URL_TARGET, 'url_tok': URL_TARGET, 'text_model': TEXT_TARGET, 'text_tok': TEXT_TARGET},
}

for split_name, frame in [('EARLY', EARLY), ('LATE', LATE)]:
    for variant, cfg in VARIANTS.items():
        embed_values(frame['url'], cfg['url_model'], cfg['url_tok'], URL_MAX_LEN, 128,
                     EMB_DIR/f'{variant}_URL_{split_name}.npy')
        embed_values(frame['text'], cfg['text_model'], cfg['text_tok'], TEXT_MAX_LEN, 64,
                     EMB_DIR/f'{variant}_TEXT_{split_name}.npy')


def X_of(variant, split):
    u = np.load(EMB_DIR/f'{variant}_URL_{split}.npy', mmap_mode='r')
    t = np.load(EMB_DIR/f'{variant}_TEXT_{split}.npy', mmap_mode='r')
    return np.concatenate([np.asarray(u,np.float32), np.asarray(t,np.float32)], axis=1)

XEARLY = {v:X_of(v,'EARLY') for v in VARIANTS}
XLATE  = {v:X_of(v,'LATE') for v in VARIANTS}
y_early = EARLY['label'].to_numpy(int)
y_late = LATE['label'].to_numpy(int)

print({'early': {k:v.shape for k,v in XEARLY.items()}, 'late': {k:v.shape for k,v in XLATE.items()}})

# CELL 12
# 06 — Labelstichproben, Probe-Training und Late-Test-Auswertung

def stable_u64(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)

# stabile Instanz-ID aus URL + Zeit + Textpräfix; nur für Sampling/Audit
EARLY_IDS = (
    EARLY['url'].astype(str) + '|' + EARLY['scan_dt'].astype(str) + '|' + EARLY['text'].astype(str).str.slice(0,256)
).tolist()


def ranked_indices(seed):
    by_class = {}
    for lab in [0,1]:
        idx = np.where(y_early == lab)[0]
        ranked = sorted(idx.tolist(), key=lambda i: stable_u64(f'PUTRA_TARGET|{seed}|{lab}|{EARLY_IDS[i]}'))
        by_class[lab] = ranked
    return by_class


def indices_for_budget(ranked, budget):
    if budget % 2 != 0:
        raise ValueError('Budgets müssen für balancierte Auswahl gerade sein.')
    n = budget//2
    idx = np.asarray(ranked[0][:n] + ranked[1][:n], dtype=np.int32)
    return np.sort(idx)


def sample_hash(idx):
    h = hashlib.sha256()
    for i in sorted(np.asarray(idx,dtype=int).tolist()):
        h.update(EARLY_IDS[i].encode()); h.update(b'\n')
    return h.hexdigest()


def precision_at_recall(y,s,target=.90):
    p,r,_ = precision_recall_curve(y,s)
    ok = np.where(r >= target)[0]
    return float(np.max(p[ok])) if len(ok) else np.nan


def ranking_metrics(y,s):
    y=np.asarray(y,int); s=np.asarray(s,float)
    fpr,tpr,_=roc_curve(y,s)
    ii=np.where(tpr>=.90)[0]
    f90=float(fpr[ii[0]]) if len(ii) else np.nan
    return {
        'AP': float(average_precision_score(y,s)),
        'AUC': float(roc_auc_score(y,s)),
        'P_at_R90': precision_at_recall(y,s,.90),
        'FPR_at_TPR90': f90,
    }

rows=[]
selection_audit=[]

for run,(rank_seed,model_seed) in enumerate(zip(LABEL_RANK_SEEDS,MODEL_SEEDS)):
    ranked=ranked_indices(rank_seed)
    prev=set()

    for budget in LABEL_BUDGETS:
        idx=indices_for_budget(ranked,budget)
        idxset=set(idx.tolist())
        if prev and not prev.issubset(idxset):
            raise RuntimeError(f'Nicht verschachtelte Budgets in Run {run}.')
        prev=idxset

        ytr=y_early[idx]
        if int((ytr==0).sum()) != budget//2 or int((ytr==1).sum()) != budget//2:
            raise RuntimeError(f'Klassenbalance verletzt run={run} budget={budget}')
        sh=sample_hash(idx)

        for variant in ['E0_SOURCE_RUT','E1_TARGET_RUT']:
            clf=LogisticRegression(
                C=LINEAR_SPEC['C'], solver=LINEAR_SPEC['solver'],
                class_weight=LINEAR_SPEC['class_weight'], max_iter=LINEAR_SPEC['max_iter'],
                random_state=model_seed,
            )
            clf.fit(XEARLY[variant][idx], ytr)
            score=clf.decision_function(XLATE[variant]).astype(np.float32)
            rows.append({
                'run':run,'label_rank_seed':rank_seed,'model_seed':model_seed,
                'budget':budget,'variant':variant,'sample_hash':sh,
                'n_train':len(idx),'n_test':len(y_late),
                **ranking_metrics(y_late,score),
            })
            del clf

        selection_audit.append({'run':run,'budget':budget,'sample_hash':sh,'rows':len(idx)})

    print({'run_complete':run,'max_budget_hash':selection_audit[-1]['sample_hash'][:12]})

RES=pd.DataFrame(rows).sort_values(['run','budget','variant']).reset_index(drop=True)
RES.to_csv(RESULTS_DIR/'PUTRA_TARGET_ADAPTATION_N10_RUN_METRICS.csv',index=False)
atomic_json(AUDIT_DIR/'LABEL_SELECTION_AUDIT.json',selection_audit)

# Paarungscheck E0/E1
pair=RES.pivot_table(index=['run','budget'],columns='variant',values='sample_hash',aggfunc='first')
if not (pair['E0_SOURCE_RUT']==pair['E1_TARGET_RUT']).all():
    raise RuntimeError('E0/E1-Paarung verletzt.')

display(RES[['run','budget','variant','AP','AUC','P_at_R90','FPR_at_TPR90']])

# CELL 14
# 07 — AULC, Sign-Flip, 95%-KI und Holm

def exact_signflip_p(diff):
    d=np.asarray(diff,float)
    d=d[np.isfinite(d)]
    obs=abs(d.mean())
    vals=[abs(np.mean(d*np.asarray(signs))) for signs in itertools.product([-1,1],repeat=len(d))]
    return float(np.mean(np.asarray(vals)>=obs-1e-15))


def mean_ci95(diff):
    d=np.asarray(diff,float)
    m=float(d.mean())
    se=float(d.std(ddof=1)/math.sqrt(len(d)))
    q=float(student_t.ppf(.975,len(d)-1))
    return m,m-q*se,m+q*se


def holm_adjust(pvals):
    p=np.asarray(pvals,float)
    m=len(p)
    order=np.argsort(p)
    adj=np.empty(m,float)
    running=0.0
    for rank,idx in enumerate(order):
        val=(m-rank)*p[idx]
        running=max(running,val)
        adj[idx]=min(1.0,running)
    return adj

# Primärer Endpunkt: normierte AP-AULC über log10(Budget)
x=np.log10(np.asarray(LABEL_BUDGETS,float))
x=(x-x.min())/(x.max()-x.min())

aulc_rows=[]
for run in range(N_REPS):
    for variant in ['E0_SOURCE_RUT','E1_TARGET_RUT']:
        q=RES[(RES.run==run)&(RES.variant==variant)].set_index('budget').loc[LABEL_BUDGETS]
        aulc=float(np.trapz(q['AP'].to_numpy(float),x))
        aulc_rows.append({'run':run,'variant':variant,'AP_AULC':aulc})
AULC=pd.DataFrame(aulc_rows)

piv=AULC.pivot(index='run',columns='variant',values='AP_AULC').sort_index()
d=(piv['E1_TARGET_RUT']-piv['E0_SOURCE_RUT']).to_numpy(float)
m,lo,hi=mean_ci95(d)
PRIMARY=pd.DataFrame([{
    'endpoint':'AP_AULC_log_budget',
    'n_pairs':N_REPS,
    'E0_mean':float(piv['E0_SOURCE_RUT'].mean()),
    'E1_mean':float(piv['E1_TARGET_RUT'].mean()),
    'mean_delta':m,
    'ci95_lo':lo,'ci95_hi':hi,
    'positive_pairs':int((d>0).sum()),
    'negative_pairs':int((d<0).sum()),
    'exact_signflip_p':exact_signflip_p(d),
}])
PRIMARY.to_csv(RESULTS_DIR/'PUTRA_TARGET_ADAPTATION_PRIMARY_AULC.csv',index=False)

# Sekundäre AP-Familie je Budget
sec=[]
for budget in LABEL_BUDGETS:
    q=RES[RES.budget.eq(budget)].pivot(index='run',columns='variant',values='AP').sort_index()
    dd=(q['E1_TARGET_RUT']-q['E0_SOURCE_RUT']).to_numpy(float)
    mm,ll,hh=mean_ci95(dd)
    sec.append({
        'budget':budget,
        'E0_AP_mean':float(q['E0_SOURCE_RUT'].mean()),
        'E1_AP_mean':float(q['E1_TARGET_RUT'].mean()),
        'delta_AP':mm,'delta_AP_pp':100*mm,
        'ci95_lo':ll,'ci95_hi':hh,
        'ci95_lo_pp':100*ll,'ci95_hi_pp':100*hh,
        'positive_pairs':int((dd>0).sum()),
        'negative_pairs':int((dd<0).sum()),
        'p_raw':exact_signflip_p(dd),
    })
SECONDARY=pd.DataFrame(sec)
SECONDARY['p_holm']=holm_adjust(SECONDARY['p_raw'].to_numpy(float))
SECONDARY.to_csv(RESULTS_DIR/'PUTRA_TARGET_ADAPTATION_AP_BY_BUDGET_STATS.csv',index=False)

# Deskriptive Lernkurven für weitere Metriken
AGG=RES.groupby(['budget','variant'],as_index=False).agg(
    AP_mean=('AP','mean'),AP_sd=('AP','std'),
    AUC_mean=('AUC','mean'),
    P_at_R90_mean=('P_at_R90','mean'),
    FPR_at_TPR90_mean=('FPR_at_TPR90','mean'),
)
AGG.to_csv(RESULTS_DIR/'PUTRA_TARGET_ADAPTATION_LEARNING_CURVES.csv',index=False)

display(PRIMARY)
display(SECONDARY)
display(AGG)

# CELL 15
# 08 — Deskriptive Label-Äquivalenz und Plots
# Kleinster E1-Budgetpunkt, dessen mittlere AP mindestens E0@2000 erreicht.
base_target=float(AGG[(AGG.budget==max(LABEL_BUDGETS))&(AGG.variant=='E0_SOURCE_RUT')]['AP_mean'].iloc[0])
elig=AGG[(AGG.variant=='E1_TARGET_RUT') & (AGG.AP_mean>=base_target)].sort_values('budget')
smallest_equiv=int(elig.iloc[0].budget) if len(elig) else None

LABEL_EQUIV={
    'reference':'E0_SOURCE_RUT at 2000 labels',
    'reference_mean_AP':base_target,
    'smallest_E1_budget_with_mean_AP_at_least_reference':smallest_equiv,
    'inferential_status':'DESCRIPTIVE_ONLY_NO_NONINFERIORITY_CLAIM',
}
atomic_json(RESULTS_DIR/'PUTRA_TARGET_ADAPTATION_LABEL_EQUIVALENCE_DESCRIPTIVE.json',LABEL_EQUIV)
print(LABEL_EQUIV)

import matplotlib.pyplot as plt

fig,ax=plt.subplots(figsize=(8,5))
for variant in ['E0_SOURCE_RUT','E1_TARGET_RUT']:
    q=AGG[AGG.variant.eq(variant)].sort_values('budget')
    ax.plot(q['budget'],q['AP_mean'],marker='o',label=variant)
ax.set_xscale('log')
ax.set_xlabel('Labeled Putra target instances')
ax.set_ylabel('Average Precision on later Putra test')
ax.set_title('Target-domain unlabeled adaptation and label efficiency')
ax.grid(alpha=.2)
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR/'PUTRA_TARGET_ADAPTATION_AP_LEARNING_CURVE.png',dpi=180)
plt.show()

fig,ax=plt.subplots(figsize=(8,5))
for variant in ['E0_SOURCE_RUT','E1_TARGET_RUT']:
    q=AGG[AGG.variant.eq(variant)].sort_values('budget')
    ax.plot(q['budget'],100*q['FPR_at_TPR90_mean'],marker='o',label=variant)
ax.set_xscale('log')
ax.set_xlabel('Labeled Putra target instances')
ax.set_ylabel('FPR required for TPR=90% (%)')
ax.set_title('Low-FPR ranking on later Putra test')
ax.grid(alpha=.2)
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR/'PUTRA_TARGET_ADAPTATION_FPR_AT_TPR90.png',dpi=180)
plt.show()

# CELL 17
# 09 — Finales Manifest und Ergebnispaket
primary=PRIMARY.iloc[0].to_dict()
secondary=SECONDARY.to_dict(orient='records')

SUMMARY={
    'status':'COMPLETE',
    'version':VERSION,
    'scientific_status':'POST_HOC_EXTERNAL_TARGET_ADAPTATION_ANALYSIS',
    'putra_split':SPLIT_AUDIT,
    'variants':PROTOCOL['variants'],
    'target_ssl':PROTOCOL['target_ssl'],
    'label_budgets':LABEL_BUDGETS,
    'n_pairs':N_REPS,
    'primary':primary,
    'secondary_AP_by_budget':secondary,
    'label_equivalence_descriptive':LABEL_EQUIV,
    'guardrails':[
        'A prior direct Putra transfer screen was known before this experiment.',
        'The temporal late Putra test is not used for target MLM or probe fitting.',
        'Target MLM uses early Putra URL/text without labels.',
        'E0 and E1 receive identical labeled target rows within each run/budget.',
        'N10 quantifies labeled-sample/probe variation at fixed target-adapted checkpoints.',
        'Putra features.text is provider-preprocessed text, not raw HTML processed by the PhreshPhish pipeline.',
        'The label-equivalence calculation is descriptive and not a formal noninferiority result.',
    ],
}
atomic_json(RESULTS_DIR/'PUTRA_TARGET_ADAPTATION_SUMMARY.json',SUMMARY)
print(json.dumps(SUMMARY,indent=2,ensure_ascii=False))

zip_path=ROOT/'PUTRA_TARGET_ADAPTATION_LABEL_EFFICIENCY_N10_RESULTS_v1.zip'
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as zf:
    for folder in [RESULTS_DIR,AUDIT_DIR,FIG_DIR]:
        for p in folder.rglob('*'):
            if p.is_file():
                zf.write(p,p.relative_to(ROOT))
    for p in CKPT_DIR.rglob('COMPLETE.json'):
        zf.write(p,p.relative_to(ROOT))

print({'RESULTS_ZIP':str(zip_path)})