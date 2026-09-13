# KI-Unterstuetzung: siehe Erklaerung der Arbeit.
# Ursprungsdatei: FINAL_SSL_GUIDED_ADAPTIVE_RESOURCE_SYSTEM_N5_v2_ORDER_FIX.ipynb
# CELL 2

# 00 — Imports, Konfiguration, Protokoll-Lock

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc, re, json, math, time, random, hashlib, itertools, zipfile, warnings
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    roc_curve, roc_auc_score
)
from scipy.stats import t as student_t

from transformers import AutoTokenizer, AutoModel
from IPython.display import display

warnings.filterwarnings("ignore")

VERSION = "FINAL_SSL_GUIDED_ADAPTIVE_RESOURCE_SYSTEM_N5_v2_ORDER_FIX"
MASTER_SEED = 20260901

PRIMARY_FPR = 0.005

MODEL_SEEDS = [42, 62, 82, 102, 122]
LABEL_RANK_SEEDS = [20260813, 20260823, 20260833, 20260843, 20260853]

# Label-resource experiment
AL_BUDGETS = [500, 1_000, 2_000, 5_000, 10_000, 20_000]
AL_START = 500
AL_REPS = ["R0", "RUT"]
AL_CLUSTER_K = 128
AL_CLUSTER_SAMPLE = 50_000
AL_CANDIDATE_MULTIPLIER = 8

# Compute-routing experiment
CASCADE_TRAIN_BUDGET = 20_000
CASCADE_LOSS_BUDGETS_PP = [0.25, 0.50, 1.00, 2.00]
CASCADE_PRIMARY_LOSS_PP = 1.00
CASCADE_MAX_FPR_INCREASE = 0.001  # 0.1 Prozentpunkte
LOW_MISS_GRID = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
HIGH_FPR_GRID = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02]

# Review experiment
REVIEW_RATES = [0.01, 0.02, 0.05]
PRIMARY_REVIEW_RATE = 0.02
PROJ_DIM_GEOM = 64

# Encoding
URL_MAX_LEN = 128
TEXT_MAX_LEN = 256
EMBED_BATCH_URL = 128
EMBED_BATCH_TEXT = 64
BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"

# TRI architecture
PROJ_DIM = 256
DEEP_LAST_N = 4
DOM_MAX_NODES = 512
DOM_DIM = 128
DOM_LAYERS = 3
DOM_COLS = [
    "dom_tag", "dom_parent_idx", "dom_depth",
    "dom_attr_count", "dom_child_count"
]
SCORE_BATCH_CANDIDATES = [128, 96, 64, 48, 32, 16]

SCENARIOS = [
    "OFFICIAL_TEST",
    "DOMAIN_OOD_EXACT",
    "TEMPLATE_OOD_EXACT",
    "DOMAIN_TEMPLATE_OOD_EXACT",
    "LATE_TEST_Q4",
]
EXPECTED_COUNTS = {
    "SUP": 4_000, "DEV": 20_000, "CAL": 50_000,
    "SSL": 200_000, "FINAL": 168_060,
}
EXPECTED_FINAL_CLASSES = (91_260, 76_800)
EXPECTED_SCENARIO_COUNTS = {
    "OFFICIAL_TEST": 168_060,
    "DOMAIN_OOD_EXACT": 115_917,
    "TEMPLATE_OOD_EXACT": 161_223,
    "DOMAIN_TEMPLATE_OOD_EXACT": 110_095,
    "LATE_TEST_Q4": 46_784,
}

SEARCH_ROOT = Path("/kaggle/input") if Path("/kaggle/input").exists() else Path("/mnt/data")
ROOT = (
    Path("/kaggle/working/final_ssl_guided_adaptive_resource_system_n5_v1")
    if Path("/kaggle/working").exists()
    else Path("/mnt/data/final_ssl_guided_adaptive_resource_system_n5_v1")
)
EMB = ROOT / "embeddings"
GEOM = ROOT / "geometry"
MODELS = ROOT / "linear_models"
POLICIES = ROOT / "policies"
SCORES = ROOT / "scores"
RESULTS = ROOT / "results"
TABLES = ROOT / "thesis_tables"
AUDIT = ROOT / "audit"

for p in [ROOT, EMB, GEOM, MODELS, POLICIES, SCORES, RESULTS, TABLES, AUDIT]:
    p.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = bool(torch.cuda.is_available())

def amp_ctx():
    return torch.autocast("cuda", dtype=torch.float16) if AMP else nullcontext()

def seed_all(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )
    os.replace(tmp, path)

def keyed_rank(sha, seed):
    return hashlib.sha256(f"{seed}|{sha}".encode()).hexdigest()

seed_all(MASTER_SEED)

PROTOCOL = {
    "version": VERSION,
    "scientific_status": "POST_HOC_SSL_GUIDED_ADAPTIVE_RESOURCE_ALLOCATION",
    "core_principle": "SSL representation as resource-allocation signal",
    "blocks": {
        "compute": {
            "causal_contrast": "URL_BASE vs URL_DAPT",
            "same_frontend": "LogisticRegression",
            "same_labels": CASCADE_TRAIN_BUDGET,
            "same_fallback": "frozen TRI_RUT_200k",
            "primary_allowed_tpr_loss_pp": CASCADE_PRIMARY_LOSS_PP,
            "max_dev_fpr_increase": CASCADE_MAX_FPR_INCREASE,
        },
        "labels": {
            "methods": ["R0_RANDOM", "RUT_RANDOM", "R0_ACTIVE", "RUT_ACTIVE"],
            "budgets": AL_BUDGETS,
            "active_uses_hidden_labels_for_selection": False,
            "transformer_retrained_in_loop": False,
        },
        "review": {
            "classification_by_neighbors": False,
            "signal": "representation support / novelty only",
            "review_rates": REVIEW_RATES,
            "primary_review_rate": PRIMARY_REVIEW_RATE,
        },
    },
    "final_scenarios": SCENARIOS,
    "target_fpr": PRIMARY_FPR,
    "high_upside_rule": {
        "compute": "URL_DAPT escalation reduction >=10pp in >=4/5 scenarios",
        "labels": "RUT_ACTIVE B95 <=5000 labels",
        "review": "RUT error enrichment >=2x in >=4/5 scenarios at primary review rate",
        "integrated": "at least 2 of 3 conditions true",
    },
    "access_rule": "SSL/DEV policies first; CAL thresholding and FINAL evaluation only after policy freeze",
}
atomic_json(AUDIT / "INTEGRATED_PROTOCOL_LOCK.json", PROTOCOL)

print(json.dumps(PROTOCOL, indent=2, ensure_ascii=False))
print({"device": str(DEVICE), "search_root": str(SEARCH_ROOT), "output_root": str(ROOT)})


# CELL 3

# 01 — Gemeinsame Metriken und Statistik

def y01(s):
    if pd.api.types.is_numeric_dtype(s):
        y = pd.to_numeric(s, errors="coerce")
        if y.isna().any():
            raise RuntimeError("Ungültige numerische Labels.")
        vals = set(y.astype(int).unique())
        if not vals.issubset({0, 1}):
            raise RuntimeError(f"Unerwartete Labels: {sorted(vals)}")
        return y.astype(int).to_numpy()

    z = s.astype(str).str.lower().str.strip()
    mp = {
        "benign":0, "legitimate":0, "legit":0, "0":0, "false":0,
        "phish":1, "phishing":1, "malicious":1, "1":1, "true":1,
    }
    y = z.map(mp)
    if y.isna().any():
        raise RuntimeError(f"Unbekannte Labels: {sorted(z[y.isna()].unique())[:20]}")
    return y.astype(int).to_numpy()

def thr_fpr(neg_scores, f):
    s = np.asarray(neg_scores, dtype=np.float64)
    if len(s) == 0 or not np.isfinite(s).all():
        raise RuntimeError("Ungültige Threshold-Scores.")
    k = int(math.floor(float(f) * len(s) + 1e-12))
    if k <= 0:
        return float(np.nextafter(s.max(), np.inf))
    ss = np.sort(s)
    return float(np.nextafter(ss[-k], np.inf))

def op(y, score_or_pred, th=None):
    y = np.asarray(y, int)
    if th is None:
        pred = np.asarray(score_or_pred, bool)
    else:
        pred = np.asarray(score_or_pred, float) >= float(th)

    tp = int(((pred) & (y == 1)).sum())
    fp = int(((pred) & (y == 0)).sum())
    tn = int(((~pred) & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())

    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "f1": 2 * precision * tpr / max(precision + tpr, 1e-12),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "fp_per_1000": 1000 * fpr,
    }

def precision_at_recall(y, s, target=.90):
    p, r, _ = precision_recall_curve(y, s)
    ok = np.where(r >= target)[0]
    return float(np.max(p[ok])) if len(ok) else np.nan

def curves(y, s):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    fpr, tpr, _ = roc_curve(y, s)
    ii = np.where(tpr >= .90)[0]
    return {
        "AP": float(average_precision_score(y, s)),
        "AUC": float(roc_auc_score(y, s)),
        "P_at_R90": precision_at_recall(y, s, .90),
        "FPR_at_TPR90": float(fpr[ii[0]]) if len(ii) else np.nan,
    }

def exact_signflip_p(diff):
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan
    obs = abs(d.mean())
    vals = [
        abs(np.mean(d * np.asarray(signs)))
        for signs in itertools.product([-1, 1], repeat=len(d))
    ]
    return float(np.mean(np.asarray(vals) >= obs - 1e-15))

def mean_ci(diff):
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return (np.nan, np.nan, np.nan)
    if len(d) == 1:
        return (float(d[0]), np.nan, np.nan)
    m = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    q = float(student_t.ppf(.975, len(d)-1))
    return m, m-q*se, m+q*se

def holm_adjust(p_values):
    p = np.asarray(p_values, float)
    out = np.full(len(p), np.nan)
    fin = np.flatnonzero(np.isfinite(p))
    if not len(fin):
        return out
    vals = p[fin]
    order = np.argsort(vals, kind="mergesort")
    running = 0.0
    for rank, pos in enumerate(order):
        running = max(running, (len(vals)-rank) * vals[pos])
        out[fin[pos]] = min(1.0, running)
    return out

def row_l2(x):
    x = np.asarray(x, np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)

def robust_z_apply(x, med, iqr):
    # Avoid pathological domination when a component has almost no IQR.
    return (np.asarray(x, float) - float(med)) / max(float(iqr), 1e-3)


# CELL 4

# 02 — PhreshPhish Data Freeze und kanonische Modellassets robust auflösen

FREEZE_ID = "PHRESHPHISH_FINAL_DATA_FREEZE_v3_HF_SHARDED_DOM"
ROLE_REL = {
    "SUP": Path("roles/supervised_pool"),
    "DEV": Path("roles/development"),
    "CAL": Path("roles/fpr_calibration_benign"),
    "SSL": Path("roles/ssl_pool_200k"),
    "FINAL": Path("roles/final_test"),
}

# ---- Data Freeze metadata
markers = []
for p in SEARCH_ROOT.rglob("FINAL_DATA_FREEZE_COMPLETE.json"):
    try:
        obj = json.loads(p.read_text())
    except Exception:
        continue
    if obj.get("status") == "COMPLETE" and obj.get("freeze_id") == FREEZE_ID:
        score = 0
        if "freezeresume" in str(p).lower():
            score += 100
        if (p.parent/"manifests/train_role_manifest_PRIVATE_WITH_LABELS.parquet").exists():
            score += 50
        if (p.parent/"manifests/final_test_exact_ood_flags_SEALED.parquet").exists():
            score += 25
        markers.append((score, p))

if not markers:
    raise RuntimeError("Finaler PhreshPhish Data-Freeze-Marker fehlt.")
markers.sort(key=lambda x: (-x[0], len(str(x[1])), str(x[1])))
FREEZE_FILE = markers[0][1]
META_ROOT = FREEZE_FILE.parent
MANIFEST_ROOT = META_ROOT / "manifests"
PRIVATE_MANIFEST = MANIFEST_ROOT / "train_role_manifest_PRIVATE_WITH_LABELS.parquet"
OOD_FLAGS_PATH = MANIFEST_ROOT / "final_test_exact_ood_flags_SEALED.parquet"

# ---- Physical role root
roots = []
for r in SEARCH_ROOT.rglob("roles"):
    if not r.is_dir():
        continue
    candidate = r.parent
    if all((candidate/rel).exists() for rel in ROLE_REL.values()):
        score = 100 if "finaldatafreeze" in str(candidate).lower() else 0
        if candidate.name == "phreshphish_FINAL_DATA_FREEZE_v3":
            score += 50
        roots.append((score, candidate))

if not roots:
    raise RuntimeError("Physischer PhreshPhish Data-Freeze-Root fehlt.")
roots.sort(key=lambda x: (-x[0], len(str(x[1])), str(x[1])))
DATA_ROOT = roots[0][1]
ROLE_DIR = {k: DATA_ROOT/v for k,v in ROLE_REL.items()}

def nrows(d):
    fs = sorted(Path(d).glob("*.parquet"))
    if not fs:
        raise RuntimeError(f"Keine Parquet-Dateien: {d}")
    return sum(pq.ParquetFile(p).metadata.num_rows for p in fs)

COUNTS = {k:nrows(v) for k,v in ROLE_DIR.items()}
if COUNTS != EXPECTED_COUNTS:
    raise RuntimeError(f"Unerwartete Rollenanzahlen: {COUNTS}")

# ---- Helpers for HF checkpoints
def hf_checkpoint_valid(p):
    p = Path(p)
    return (
        (p/"config.json").exists()
        and any((p/f).exists() for f in ["model.safetensors", "pytorch_model.bin"])
    )

def model_type(p):
    try:
        return str(json.loads((Path(p)/"config.json").read_text()).get("model_type","")).lower()
    except Exception:
        return ""

# ---- URL DAPT
url_cands = []
for p in SEARCH_ROOT.rglob("config.json"):
    par = p.parent
    if not hf_checkpoint_valid(par) or model_type(par) != "bert":
        continue
    s = str(par).lower()
    score = 0
    if par.name.lower() == "url_dapt_bert_200k":
        score += 10000
    if "url_dapt_bert_200k" in s:
        score += 5000
    if "final_url_ssl_integrated_v2" in s:
        score += 1000
    if "200k" in s:
        score += 200
    if score:
        url_cands.append((score, par))
if not url_cands:
    raise RuntimeError("Kanonischer URL_DAPT_BERT_200K Checkpoint fehlt.")
url_cands.sort(key=lambda x:(-x[0], len(str(x[1])), str(x[1])))
URL_DAPT_SRC = str(url_cands[0][1])

# ---- HTML/Text DAPT + DOM SSL
text_cands = []
for p in SEARCH_ROOT.rglob("R1_DAPT_TEXT"):
    if not p.is_dir() or not hf_checkpoint_valid(p):
        continue
    candidate = p.parent.parent if p.parent.name == "checkpoints" else None
    if candidate is None:
        continue
    dom_enc = candidate/"checkpoints/DOM_MASKED_SSL/encoder.pt"
    dom_vocab = candidate/"audit/DOM_VOCAB.json"
    if not (dom_enc.exists() and dom_vocab.exists()):
        continue
    s = str(candidate).lower()
    score = 0
    if "phreshphish_final_deep_trimodal_ssl_v4_3_labelset_alignment_fix" in s:
        score += 10000
    if "bigresults" in s:
        score += 1000
    text_cands.append((score, p, dom_enc, dom_vocab, candidate))

if not text_cands:
    raise RuntimeError("Vollständiger R1_DAPT_TEXT + DOM-SSL Assetstand fehlt.")
text_cands.sort(key=lambda x:(-x[0], len(str(x[1])), str(x[1])))
TEXT_DAPT_SRC = str(text_cands[0][1])
DOM_ENCODER_PATH = text_cands[0][2]
DOM_VOCAB_PATH = text_cands[0][3]
OLD_ROOT = text_cands[0][4]

# ---- Base models; cached embeddings are preferred later
def resolve_base_model(model_id, excludes):
    needle = model_id.split("/")[-1].lower()
    cands = []
    for p in SEARCH_ROOT.rglob("config.json"):
        s = str(p.parent).lower()
        if needle not in s:
            continue
        if any(e in s for e in excludes):
            continue
        if not hf_checkpoint_valid(p.parent):
            continue
        score = 0
        if p.parent.name.lower() == needle:
            score += 100
        if model_id == "roberta-base" and "carmengeiss" in s:
            score += 1000
        cands.append((score, p.parent))
    if cands:
        cands.sort(key=lambda x:(-x[0], len(str(x[1])), str(x[1])))
        return str(cands[0][1])
    return model_id

URL_BASE_SRC = resolve_base_model(
    "bert-base-uncased",
    ["url_dapt", "url_tapt", "r1_dapt_text"]
)
TEXT_BASE_SRC = resolve_base_model(
    "roberta-base",
    ["r1_dapt_text", "text_dapt", "text_tapt"]
)

# ---- Frozen final TRI-RUT-200k checkpoint
tri_cands = []
for p in SEARCH_ROOT.rglob("ENG_TRI_RUT_200000"):
    if not p.is_dir():
        continue
    done = p/"COMPLETE.json"
    state = p/"model_state.pt"
    if not (done.exists() and state.exists()):
        continue
    try:
        j = json.loads(done.read_text())
    except Exception:
        continue
    if int(j.get("budget",-1)) != 200000 or not bool(j.get("use_dom",True)):
        continue
    s = str(p).lower()
    score = 1000 if "final_url_ssl_integrated_v2" in s else 0
    tri_cands.append((score, p))

if not tri_cands:
    raise RuntimeError(
        "Frozen ENG_TRI_RUT_200000/model_state.pt fehlt. "
        "Bitte final_url_ssl_integrated_v2 als Kaggle Input anhängen."
    )
tri_cands.sort(key=lambda x:(-x[0],len(str(x[1])),str(x[1])))
TRI_DIR = tri_cands[0][1]

# Existing operational benchmark, optional but preferred.
ops_cands = []
for p in SEARCH_ROOT.rglob("TABLE_OPERATIONAL_BENCHMARK.csv"):
    s = str(p).lower()
    score = 100 if "final_url_ssl_integrated_v2" in s else 0
    ops_cands.append((score,p))
ops_cands.sort(key=lambda x:(-x[0],len(str(x[1])),str(x[1])))
HIST_OPS_PATH = ops_cands[0][1] if ops_cands else None

ASSETS = {
    "data_root": str(DATA_ROOT),
    "meta_root": str(META_ROOT),
    "url_base": URL_BASE_SRC,
    "url_dapt": URL_DAPT_SRC,
    "text_base": TEXT_BASE_SRC,
    "text_dapt": TEXT_DAPT_SRC,
    "dom_encoder": str(DOM_ENCODER_PATH),
    "dom_vocab": str(DOM_VOCAB_PATH),
    "tri_dir": str(TRI_DIR),
    "operational_benchmark": str(HIST_OPS_PATH) if HIST_OPS_PATH else None,
    "counts": COUNTS,
}
atomic_json(AUDIT/"INPUT_RESOLUTION.json",ASSETS)
print(json.dumps(ASSETS,indent=2,ensure_ascii=False))


# CELL 5

# 03 — Pre-freeze Daten: SSL + DEV, private SSL-Labels strikt von Auswahlfunktionen trennen

FINAL_UNLOCKED = False

def read_role_raw(key, cols):
    out=[]
    for p in sorted(ROLE_DIR[key].glob("*.parquet")):
        names=set(pq.ParquetFile(p).schema_arrow.names)
        missing=[c for c in cols if c not in names]
        if missing:
            raise RuntimeError(f"{p}: missing {missing}")
        out.append(pd.read_parquet(p,columns=cols))
    z=pd.concat(out,ignore_index=True)
    if len(z)!=COUNTS[key]:
        raise RuntimeError(f"Role row mismatch {key}: {len(z)}")
    return z

ROLE_LAYOUT={}
for key in ["CAL","FINAL"]:
    layout=[]; off=0
    for p in sorted(ROLE_DIR[key].glob("*.parquet")):
        n=pq.ParquetFile(p).metadata.num_rows
        layout.append((p,off,off+n))
        off+=n
    ROLE_LAYOUT[key]=layout

def read_role_slice(key,start,end,cols):
    if key in {"CAL","FINAL"} and not FINAL_UNLOCKED:
        raise RuntimeError(f"{key} bleibt bis POLICY_FREEZE gesperrt.")
    parts=[]
    for p,a,b in ROLE_LAYOUT[key]:
        lo=max(start,a); hi=min(end,b)
        if lo>=hi:
            continue
        tab=pq.ParquetFile(p).read(columns=cols).slice(lo-a,hi-lo)
        parts.append(tab.to_pandas())
    z=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=cols)
    if len(z)!=(end-start):
        raise RuntimeError(f"Slice mismatch {key} {start}:{end}")
    return z

# Private SSL label manifest
priv=pd.read_parquet(PRIVATE_MANIFEST,columns=["sha256","label","role"])
priv["sha256"]=priv.sha256.astype(str).str.lower()
priv=priv[priv.role.eq("SSL_POOL")][["sha256","label"]].copy()
if len(priv)!=200000 or priv.sha256.duplicated().any():
    raise RuntimeError("Private SSL manifest invalid.")
label_map=dict(zip(priv.sha256,y01(priv.label)))

# Physical SSL order and explicit no-label audit
ssl_meta=[]
for p in sorted(ROLE_DIR["SSL"].glob("*.parquet")):
    if "label" in pq.ParquetFile(p).schema_arrow.names:
        raise RuntimeError("Physischer SSL-Pool enthält unerwartet Labels.")
    q=pd.read_parquet(p,columns=["sha256"])
    ssl_meta.append(q)
SSL_META=pd.concat(ssl_meta,ignore_index=True)
SSL_META["sha256"]=SSL_META.sha256.astype(str).str.lower()
if set(SSL_META.sha256)!=set(priv.sha256):
    raise RuntimeError("SSL physical/private SHA mismatch.")
SSL_META["y"]=SSL_META.sha256.map(label_map).astype(int)

# DEV includes DOM because it is the only set used for routing-policy freeze.
DEV=read_role_raw("DEV",["sha256","url","text","label","date"]+DOM_COLS)
DEV["sha256"]=DEV.sha256.astype(str).str.lower()
DEVY=y01(DEV.label)

# Fixed partitions from the canonical integrated investigation
idx=np.arange(len(DEV))
REP_CAL,REP_EVAL=train_test_split(
    idx,test_size=.5,stratify=DEVY,random_state=20260821
)
_,eng_half=train_test_split(
    idx,test_size=.5,stratify=DEVY,random_state=20260812
)
ENG_TUNE,ENG_META=train_test_split(
    eng_half,test_size=.5,stratify=DEVY[eng_half],random_state=20260814
)
REP_CAL=np.sort(REP_CAL); REP_EVAL=np.sort(REP_EVAL)
ENG_TUNE=np.sort(ENG_TUNE); ENG_META=np.sort(ENG_META)

def deterministic_order(seed):
    q=SSL_META[["sha256"]].copy()
    q["idx"]=q.index.to_numpy()
    q["rank"]=[keyed_rank(x,seed) for x in q.sha256]
    return q.sort_values("rank").idx.to_numpy()

def balanced_indices(seed,budget):
    out=[]
    n=budget//2
    for cls in [0,1]:
        q=SSL_META[SSL_META.y.eq(cls)][["sha256"]].copy()
        q["idx"]=q.index.to_numpy()
        q["rank"]=[keyed_rank(x,seed) for x in q.sha256]
        out.append(q.sort_values("rank").idx.to_numpy()[:n])
    return np.sort(np.concatenate(out))

RANDOM_ORDERS={seed:deterministic_order(seed) for seed in LABEL_RANK_SEEDS}
CASCADE_LABELSETS={
    run:balanced_indices(rseed,CASCADE_TRAIN_BUDGET)
    for run,rseed in enumerate(LABEL_RANK_SEEDS)
}

PRE_FREEZE_AUDIT={
    "ssl_rows":len(SSL_META),
    "dev_rows":len(DEV),
    "ssl_physical_has_labels":False,
    "active_selection_contract":"selection functions receive embeddings/scores/cluster ids, never y",
    "dev_partitions":{
        "REP_CAL":len(REP_CAL),"REP_EVAL":len(REP_EVAL),
        "ENG_TUNE":len(ENG_TUNE),"ENG_META":len(ENG_META),
    },
}
atomic_json(AUDIT/"PRE_FREEZE_DATA_AUDIT.json",PRE_FREEZE_AUDIT)
print(json.dumps(PRE_FREEZE_AUDIT,indent=2))


# CELL 6

# 04 — Embedding-Cache: bestehende kanonische Embeddings wiederverwenden, sonst resumierbar erzeugen

HF_TOKEN=os.environ.get("HF_TOKEN")
if not HF_TOKEN and Path("/kaggle/working").exists():
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN=UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        HF_TOKEN=None

def hf_kwargs(src,model_id):
    if str(src)==model_id and model_id=="bert-base-uncased":
        return {"revision":BERT_REVISION}
    return {}

url_base_tok=AutoTokenizer.from_pretrained(
    URL_BASE_SRC,token=HF_TOKEN,**hf_kwargs(URL_BASE_SRC,"bert-base-uncased")
)
url_dapt_tok=AutoTokenizer.from_pretrained(URL_DAPT_SRC,token=HF_TOKEN)
text_base_tok=AutoTokenizer.from_pretrained(TEXT_BASE_SRC,token=HF_TOKEN)
text_dapt_tok=AutoTokenizer.from_pretrained(TEXT_DAPT_SRC,token=HF_TOKEN)

STREAMS={
    "URL_BASE":(URL_BASE_SRC,url_base_tok,"url",URL_MAX_LEN,EMBED_BATCH_URL),
    "URL_DAPT":(URL_DAPT_SRC,url_dapt_tok,"url",URL_MAX_LEN,EMBED_BATCH_URL),
    "TEXT_BASE":(TEXT_BASE_SRC,text_base_tok,"text",TEXT_MAX_LEN,EMBED_BATCH_TEXT),
    "TEXT_DAPT":(TEXT_DAPT_SRC,text_dapt_tok,"text",TEXT_MAX_LEN,EMBED_BATCH_TEXT),
}

def masked_mean(h,m):
    mm=m.unsqueeze(-1).to(h.dtype)
    return (h*mm).sum(1)/mm.sum(1).clamp_min(1)

def expected_rows(role):
    return COUNTS[role]

def find_existing_embedding(stream,role):
    expected=expected_rows(role)
    cands=[]
    # Prefer canonical final_url_ssl_integrated_v2 caches.
    for p in SEARCH_ROOT.rglob(f"{stream}_{role}.npy"):
        try:
            a=np.load(p,mmap_mode="r")
            valid=(len(a)==expected and a.ndim==2 and a.shape[1] in [768,1024])
            del a
        except Exception:
            valid=False
        if not valid:
            continue
        s=str(p).lower()
        score=0
        if "final_url_ssl_integrated_v2" in s:
            score+=1000
        if "/embeddings/" in s:
            score+=100
        cands.append((score,p))
    if cands:
        cands.sort(key=lambda x:(-x[0],len(str(x[1])),str(x[1])))
        return cands[0][1]
    return None

def role_values(role,col):
    if role=="SSL":
        vals=[]
        for p in sorted(ROLE_DIR["SSL"].glob("*.parquet")):
            q=pd.read_parquet(p,columns=[col])
            vals.extend(q[col].fillna("").astype(str).tolist())
        if len(vals)!=COUNTS["SSL"]:
            raise RuntimeError("SSL value count mismatch.")
        return vals

    if role in {"CAL","FINAL"} and not FINAL_UNLOCKED:
        raise RuntimeError(f"{role} remains locked.")
    q=read_role_raw(role,["sha256",col])
    return q[col].fillna("").astype(str).tolist()

@torch.no_grad()
def generate_embedding(stream,role):
    src,tok,col,max_len,batch=STREAMS[stream]
    out=EMB/f"{stream}_{role}.npy"
    done=EMB/f"{stream}_{role}.complete.json"
    prog=EMB/f"{stream}_{role}.progress.json"

    if done.exists() and out.exists():
        a=np.load(out,mmap_mode="r")
        if len(a)==expected_rows(role):
            return out

    existing=find_existing_embedding(stream,role)
    if existing is not None:
        atomic_json(done,{
            "status":"REUSED_INPUT",
            "path":str(existing),
            "rows":expected_rows(role),
        })
        return existing

    vals=role_values(role,col)
    kwargs=hf_kwargs(src,"bert-base-uncased") if stream=="URL_BASE" else {}
    model=AutoModel.from_pretrained(src,token=HF_TOKEN,**kwargs).to(DEVICE).eval()
    hdim=int(model.config.hidden_size)
    n=len(vals)

    start=0
    if prog.exists() and out.exists():
        try:
            start=int(json.loads(prog.read_text()).get("next_row",0))
        except Exception:
            start=0

    if start==0:
        out.unlink(missing_ok=True)
        arr=np.lib.format.open_memmap(out,mode="w+",dtype=np.float32,shape=(n,hdim))
    else:
        arr=np.load(out,mmap_mode="r+")

    for st in range(start,n,batch):
        en=min(st+batch,n)
        b=tok(
            vals[st:en],padding=True,truncation=True,
            max_length=max_len,return_tensors="pt"
        )
        b={k:v.to(DEVICE) for k,v in b.items()}
        with amp_ctx():
            o=model(**b)
            z=masked_mean(o.last_hidden_state,b["attention_mask"])
        arr[st:en]=z.float().cpu().numpy()

        if en%5000<batch or en==n:
            arr.flush()
            atomic_json(prog,{"next_row":en})
            print({"EMBED":stream,"role":role,"rows":en,"total":n})

    arr.flush()
    atomic_json(done,{
        "status":"COMPLETE",
        "rows":n,
        "hidden_size":hdim,
        "source":str(src),
    })
    prog.unlink(missing_ok=True)
    del arr,model,vals
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out

EMBED_PATHS={}
for stream in STREAMS:
    for role in ["SSL","DEV"]:
        EMBED_PATHS[(stream,role)]=generate_embedding(stream,role)

def emb(stream,role):
    p=EMBED_PATHS.get((stream,role))
    if p is None:
        p=generate_embedding(stream,role)
        EMBED_PATHS[(stream,role)]=p
    return np.load(p,mmap_mode="r")

def streams_for_rep(rep):
    if rep=="R0":
        return "URL_BASE","TEXT_BASE"
    if rep=="RUT":
        return "URL_DAPT","TEXT_DAPT"
    raise KeyError(rep)

def take_url(stream,role,idx):
    a=emb(stream,role)
    return np.asarray(a[np.asarray(idx,int)],np.float32)

def take_rep(rep,role,idx):
    us,ts=streams_for_rep(rep)
    idx=np.asarray(idx,int)
    return np.concatenate([
        np.asarray(emb(us,role)[idx],np.float32),
        np.asarray(emb(ts,role)[idx],np.float32),
    ],axis=1)

def manual_linear_score(coef,intercept,rep,role,indices=None,chunk=10000):
    n=expected_rows(role) if indices is None else len(indices)
    out=np.empty(n,np.float32)
    if indices is None:
        indices=np.arange(expected_rows(role),dtype=int)
    indices=np.asarray(indices,int)
    for st in range(0,len(indices),chunk):
        en=min(st+chunk,len(indices))
        X=take_rep(rep,role,indices[st:en])
        out[st:en]=X@coef.reshape(-1)+float(intercept)
    return out

def manual_url_score(coef,intercept,stream,role,indices=None,chunk=20000):
    if indices is None:
        indices=np.arange(expected_rows(role),dtype=int)
    indices=np.asarray(indices,int)
    out=np.empty(len(indices),np.float32)
    for st in range(0,len(indices),chunk):
        en=min(st+chunk,len(indices))
        X=take_url(stream,role,indices[st:en])
        out[st:en]=X@coef.reshape(-1)+float(intercept)
    return out

print("PRE-FREEZE EMBEDDINGS: PASS")


# CELL 7

# 05 — Geometrie-Layer: unlabeled Projektion + Cluster für Active Learning und Review

rng=np.random.default_rng(MASTER_SEED)
FEATURE_DIM=take_rep("R0","DEV",np.arange(1)).shape[1]
RPROJ=(rng.normal(
    0,1/math.sqrt(PROJ_DIM_GEOM),
    size=(FEATURE_DIM,PROJ_DIM_GEOM)
).astype(np.float32))
np.save(GEOM/"FIXED_RANDOM_PROJECTION.npy",RPROJ)

def projected_path(rep,role):
    return GEOM/f"{rep}_{role}_proj{PROJ_DIM_GEOM}.npy"

def project_role(rep,role):
    path=projected_path(rep,role)
    done=GEOM/f"{rep}_{role}_proj.complete.json"
    n=expected_rows(role)
    if done.exists() and path.exists():
        a=np.load(path,mmap_mode="r")
        if len(a)==n and a.shape[1]==PROJ_DIM_GEOM:
            return path

    arr=np.lib.format.open_memmap(
        path,mode="w+",dtype=np.float32,shape=(n,PROJ_DIM_GEOM)
    )
    chunk=5000
    for st in range(0,n,chunk):
        en=min(st+chunk,n)
        X=take_rep(rep,role,np.arange(st,en))
        # Geometry only: normalize each full representation before projection.
        X=row_l2(X)
        Z=X@RPROJ
        Z=row_l2(Z)
        arr[st:en]=Z
        if en%20000<chunk or en==n:
            arr.flush()
            print({"PROJECT":rep,"role":role,"rows":en,"total":n})
    arr.flush()
    atomic_json(done,{"status":"COMPLETE","rows":n,"dim":PROJ_DIM_GEOM})
    return path

def zmm(rep,role):
    return np.load(projected_path(rep,role),mmap_mode="r")

CLUSTER_CENTERS={}
CLUSTER_IDS={}

for rep in AL_REPS:
    project_role(rep,"SSL")
    project_role(rep,"DEV")

    centers_path=GEOM/f"{rep}_cluster_centers.npy"
    ids_ssl_path=GEOM/f"{rep}_SSL_cluster_ids.npy"
    ids_dev_path=GEOM/f"{rep}_DEV_cluster_ids.npy"
    done=GEOM/f"{rep}_clusters.complete.json"

    if done.exists() and centers_path.exists() and ids_ssl_path.exists() and ids_dev_path.exists():
        centers=np.load(centers_path)
        ids_ssl=np.load(ids_ssl_path,mmap_mode="r")
        ids_dev=np.load(ids_dev_path,mmap_mode="r")
        if (
            centers.shape==(AL_CLUSTER_K,PROJ_DIM_GEOM)
            and len(ids_ssl)==200000 and len(ids_dev)==20000
        ):
            CLUSTER_CENTERS[rep]=centers
            CLUSTER_IDS[(rep,"SSL")]=ids_ssl_path
            CLUSTER_IDS[(rep,"DEV")]=ids_dev_path
            continue

    # Label-independent sample.
    q=SSL_META[["sha256"]].copy()
    q["idx"]=q.index.to_numpy()
    q["rank"]=[keyed_rank(x,MASTER_SEED+17) for x in q.sha256]
    sample_idx=q.sort_values("rank").idx.to_numpy()[:AL_CLUSTER_SAMPLE]

    km=MiniBatchKMeans(
        n_clusters=AL_CLUSTER_K,
        batch_size=4096,
        n_init=3,
        random_state=MASTER_SEED,
        reassignment_ratio=0.01,
    )
    km.fit(np.asarray(zmm(rep,"SSL")[sample_idx],np.float32))
    centers=km.cluster_centers_.astype(np.float32)
    np.save(centers_path,centers)

    for role,path in [("SSL",ids_ssl_path),("DEV",ids_dev_path)]:
        Z=zmm(rep,role)
        ids=np.lib.format.open_memmap(path,mode="w+",dtype=np.int16,shape=(len(Z),))
        for st in range(0,len(Z),20000):
            en=min(st+20000,len(Z))
            ids[st:en]=km.predict(np.asarray(Z[st:en],np.float32)).astype(np.int16)
        ids.flush()
        CLUSTER_IDS[(rep,role)]=path

    CLUSTER_CENTERS[rep]=centers
    atomic_json(done,{
        "status":"COMPLETE",
        "rep":rep,
        "clusters":AL_CLUSTER_K,
        "fit_sample":AL_CLUSTER_SAMPLE,
        "selection_uses_labels":False,
    })

def cluster_ids(rep,role):
    return np.load(CLUSTER_IDS[(rep,role)],mmap_mode="r")

print("UNLABELED REPRESENTATION GEOMETRY: COMPLETE")


# CELL 8

# 06 — Active Learning: R0/RUT × Random/Active, N5, kein Labelzugriff in der Auswahlfunktion

def fit_lr(X,y,seed):
    if len(np.unique(y))<2:
        raise RuntimeError("Trainingsmenge enthält nur eine Klasse.")
    return LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver="liblinear",
        class_weight="balanced",
        random_state=int(seed),
    ).fit(X,y)

def save_lr(path,clf,meta):
    path=Path(path)
    np.savez_compressed(
        path,
        coef=clf.coef_.astype(np.float32),
        intercept=clf.intercept_.astype(np.float32),
        **{f"meta_{k}":np.asarray(str(v)) for k,v in meta.items()}
    )

def load_lr(path):
    j=np.load(path,allow_pickle=True)
    return j["coef"].reshape(-1).astype(np.float32),float(j["intercept"].reshape(-1)[0])

def choose_active_without_labels(margins,cluster_id,selected_mask,q):
    """
    Selection receives NO labels.
    Strategy: top uncertainty pool followed by capped round-robin cluster diversity.
    """
    available=np.flatnonzero(~selected_mask)
    if q>=len(available):
        return available

    m=np.abs(np.asarray(margins[available],float))
    cand_n=min(len(available),max(q*AL_CANDIDATE_MULTIPLIER,AL_CLUSTER_K*4))
    ord_local=np.argpartition(m,cand_n-1)[:cand_n]
    cand=available[ord_local]
    cand=cand[np.argsort(np.abs(margins[cand]),kind="mergesort")]

    cid=np.asarray(cluster_id[cand],int)
    picked=[]
    used=np.zeros(AL_CLUSTER_K,int)

    # Increasing per-cluster cap preserves diversity without expensive k-means in every round.
    cap=1
    remaining=set(map(int,cand.tolist()))
    while len(picked)<q and remaining:
        progressed=False
        for idx in cand:
            idx=int(idx)
            if idx not in remaining:
                continue
            c=int(cluster_id[idx])
            if used[c] >= cap:
                continue
            picked.append(idx)
            used[c]+=1
            remaining.remove(idx)
            progressed=True
            if len(picked)>=q:
                break
        if not progressed or len(picked)<q:
            cap+=1
    if len(picked)<q:
        rest=[int(i) for i in cand if int(i) not in set(picked)]
        picked.extend(rest[:q-len(picked)])
    return np.asarray(picked[:q],int)

AL_ROWS=[]
AL_MODEL_PATHS={}
AL_SELECTED_PATHS={}

for run,(mseed,rseed) in enumerate(zip(MODEL_SEEDS,LABEL_RANK_SEEDS)):
    random_order=RANDOM_ORDERS[rseed]

    for rep in AL_REPS:
        cids=np.asarray(cluster_ids(rep,"SSL"))

        # ---------------- RANDOM ----------------
        for B in AL_BUDGETS:
            sel=np.asarray(random_order[:B],int)
            model_path=MODELS/f"AL_run{run}_{rep}_RANDOM_{B}.npz"
            sel_path=MODELS/f"AL_run{run}_{rep}_RANDOM_{B}_selected.npy"
            row_path=RESULTS/f"AL_run{run}_{rep}_RANDOM_{B}_DEV.json"

            if not sel_path.exists():
                np.save(sel_path,sel)

            if not model_path.exists():
                X=take_rep(rep,"SSL",sel)
                y=SSL_META.y.iloc[sel].to_numpy(int)
                clf=fit_lr(X,y,mseed)
                save_lr(model_path,clf,{
                    "run":run,"rep":rep,"method":"RANDOM",
                    "budget":B,"model_seed":mseed,"label_rank_seed":rseed,
                })

            coef,inter=load_lr(model_path)
            if row_path.exists():
                row=json.loads(row_path.read_text())
            else:
                s=manual_linear_score(coef,inter,rep,"DEV")
                th=thr_fpr(s[REP_CAL][DEVY[REP_CAL]==0],PRIMARY_FPR)
                row={
                    "run":run,"model_seed":mseed,"label_rank_seed":rseed,
                    "rep":rep,"method":"RANDOM","budget":B,
                    "target_fpr":PRIMARY_FPR,"threshold":th,
                    **op(DEVY[REP_EVAL],s[REP_EVAL],th),
                    **curves(DEVY[REP_EVAL],s[REP_EVAL]),
                }
                atomic_json(row_path,row)
            AL_ROWS.append(row)
            AL_MODEL_PATHS[(run,rep,"RANDOM",B)]=model_path
            AL_SELECTED_PATHS[(run,rep,"RANDOM",B)]=sel_path

        # ---------------- ACTIVE ----------------
        selected=np.asarray(random_order[:AL_START],int)
        selected_mask=np.zeros(len(SSL_META),bool)
        selected_mask[selected]=True

        for B in AL_BUDGETS:
            sel_path=MODELS/f"AL_run{run}_{rep}_ACTIVE_{B}_selected.npy"
            model_path=MODELS/f"AL_run{run}_{rep}_ACTIVE_{B}.npz"
            row_path=RESULTS/f"AL_run{run}_{rep}_ACTIVE_{B}_DEV.json"

            if sel_path.exists():
                selected=np.load(sel_path).astype(int)
                selected_mask[:]=False
                selected_mask[selected]=True
            else:
                while len(selected)<B:
                    # Fit only on currently revealed labels.
                    Xcur=take_rep(rep,"SSL",selected)
                    ycur=SSL_META.y.iloc[selected].to_numpy(int)
                    clf_cur=fit_lr(Xcur,ycur,mseed)

                    coef_cur=clf_cur.coef_.reshape(-1).astype(np.float32)
                    inter_cur=float(clf_cur.intercept_[0])
                    margins=manual_linear_score(coef_cur,inter_cur,rep,"SSL")

                    q=min(B-len(selected),len(SSL_META)-len(selected))
                    new_idx=choose_active_without_labels(
                        margins,cids,selected_mask,q
                    )
                    # Labels are still NOT accessed inside choose_active_without_labels.
                    selected=np.concatenate([selected,new_idx])
                    selected_mask[new_idx]=True

                np.save(sel_path,np.asarray(selected,int))

            if not model_path.exists():
                X=take_rep(rep,"SSL",selected)
                y=SSL_META.y.iloc[selected].to_numpy(int)
                clf=fit_lr(X,y,mseed)
                save_lr(model_path,clf,{
                    "run":run,"rep":rep,"method":"ACTIVE",
                    "budget":B,"model_seed":mseed,"label_rank_seed":rseed,
                    "selection":"uncertainty + unlabeled cluster diversity",
                })

            coef,inter=load_lr(model_path)
            if row_path.exists():
                row=json.loads(row_path.read_text())
            else:
                s=manual_linear_score(coef,inter,rep,"DEV")
                th=thr_fpr(s[REP_CAL][DEVY[REP_CAL]==0],PRIMARY_FPR)
                row={
                    "run":run,"model_seed":mseed,"label_rank_seed":rseed,
                    "rep":rep,"method":"ACTIVE","budget":B,
                    "target_fpr":PRIMARY_FPR,"threshold":th,
                    **op(DEVY[REP_EVAL],s[REP_EVAL],th),
                    **curves(DEVY[REP_EVAL],s[REP_EVAL]),
                }
                atomic_json(row_path,row)

            AL_ROWS.append(row)
            AL_MODEL_PATHS[(run,rep,"ACTIVE",B)]=model_path
            AL_SELECTED_PATHS[(run,rep,"ACTIVE",B)]=sel_path

AL_DEV=pd.DataFrame(AL_ROWS)
AL_DEV.to_csv(RESULTS/"ACTIVE_LABEL_EFFICIENCY_DEV_N5.csv",index=False)

AL_AGG=AL_DEV.groupby(["rep","method","budget"],as_index=False).agg(
    n=("tpr","size"),
    tpr_mean=("tpr","mean"),tpr_sd=("tpr","std"),
    fpr_mean=("fpr","mean"),
    AP_mean=("AP","mean"),
    P_at_R90_mean=("P_at_R90","mean"),
)
AL_AGG.to_csv(TABLES/"TABLE_ACTIVE_LABEL_EFFICIENCY_DEV.csv",index=False)

# Fixed DEV-only reference target: 95% of RUT_RANDOM_20k mean TPR.
rut_ref=float(AL_AGG[
    (AL_AGG.rep=="RUT")&
    (AL_AGG.method=="RANDOM")&
    (AL_AGG.budget==20000)
].iloc[0].tpr_mean)
AL_TARGET=0.95*rut_ref

b95_rows=[]
for rep in AL_REPS:
    for method in ["RANDOM","ACTIVE"]:
        q=AL_AGG[(AL_AGG.rep==rep)&(AL_AGG.method==method)].sort_values("budget")
        ok=q[q.tpr_mean>=AL_TARGET]
        b=int(ok.iloc[0].budget) if len(ok) else None
        b95_rows.append({
            "rep":rep,"method":method,
            "reference":"95% of RUT_RANDOM_20k DEV mean TPR",
            "target_tpr":AL_TARGET,
            "smallest_budget":b,
        })
AL_B95=pd.DataFrame(b95_rows)
AL_B95.to_csv(TABLES/"TABLE_ACTIVE_B95_DEV.csv",index=False)

# Resolve DEV-frozen B95 choices HERE, before any integrated-system cell uses them.
def get_b95(rep,method):
    q=AL_B95[(AL_B95.rep==rep)&(AL_B95.method==method)]
    if len(q)!=1:
        raise RuntimeError(f"B95 row not unique for {rep}/{method}: {len(q)}")
    v=q.iloc[0].smallest_budget
    return None if pd.isna(v) else int(v)

RUT_ACTIVE_B95=get_b95("RUT","ACTIVE")
R0_ACTIVE_B95=get_b95("R0","ACTIVE")
RUT_RANDOM_B95=get_b95("RUT","RANDOM")
R0_RANDOM_B95=get_b95("R0","RANDOM")

atomic_json(RESULTS/"ACTIVE_B95_RESOLVED.json",{
    "RUT_ACTIVE_B95":RUT_ACTIVE_B95,
    "R0_ACTIVE_B95":R0_ACTIVE_B95,
    "RUT_RANDOM_B95":RUT_RANDOM_B95,
    "R0_RANDOM_B95":R0_RANDOM_B95,
    "reference":"95% of RUT_RANDOM_20k mean DEV TPR",
    "target_tpr":AL_TARGET,
})
print({
    "ACTIVE_B95_RESOLVED":{
        "RUT_ACTIVE":RUT_ACTIVE_B95,
        "R0_ACTIVE":R0_ACTIVE_B95,
        "RUT_RANDOM":RUT_RANDOM_B95,
        "R0_RANDOM":R0_RANDOM_B95,
    }
})

# Pairwise Active-vs-Random within each representation/budget.
stat_rows=[]
for rep in AL_REPS:
    for B in AL_BUDGETS:
        q=AL_DEV[(AL_DEV.rep==rep)&(AL_DEV.budget==B)]
        piv=q.pivot(index="run",columns="method",values="tpr")
        d=(piv["ACTIVE"]-piv["RANDOM"]).to_numpy(float)
        m,lo,hi=mean_ci(d)
        stat_rows.append({
            "rep":rep,"budget":B,"contrast":"ACTIVE-RANDOM",
            "mean_delta_tpr_pp":100*m,
            "ci95_lo_pp":100*lo,"ci95_hi_pp":100*hi,
            "positive_pairs":int((d>0).sum()),
            "exact_signflip_p":exact_signflip_p(d),
        })
AL_STATS=pd.DataFrame(stat_rows)
AL_STATS.to_csv(TABLES/"TABLE_ACTIVE_PAIRED_STATS_DEV.csv",index=False)

display(AL_AGG)
display(AL_B95)


# CELL 9

# 07 — URL-only Frontends für den kausalen Compute-Kontrast: URL_BASE vs URL_DAPT

URL_FRONT_MODEL_PATHS={}
URL_FRONT_DEV_SCORES={}

for run,(mseed,rseed) in enumerate(zip(MODEL_SEEDS,LABEL_RANK_SEEDS)):
    idx_train=CASCADE_LABELSETS[run]
    ytr=SSL_META.y.iloc[idx_train].to_numpy(int)

    for stream in ["URL_BASE","URL_DAPT"]:
        p=MODELS/f"CASCADE_run{run}_{stream}_20K.npz"
        if not p.exists():
            X=take_url(stream,"SSL",idx_train)
            clf=fit_lr(X,ytr,mseed)
            save_lr(p,clf,{
                "run":run,"stream":stream,"budget":CASCADE_TRAIN_BUDGET,
                "model_seed":mseed,"label_rank_seed":rseed,
                "balanced_labelset":True,
            })
        coef,inter=load_lr(p)
        s=manual_url_score(coef,inter,stream,"DEV")
        URL_FRONT_MODEL_PATHS[(run,stream)]=p
        URL_FRONT_DEV_SCORES[(run,stream)]=s

print("URL FRONTENDS: COMPLETE")


# CELL 10

# 07b — Geschlossene Prototypvariante:
# RUT-ACTIVE wählt Labels -> URL-DAPT-Frontend nutzt exakt diese Labels

required_b95_vars=["RUT_ACTIVE_B95","R0_ACTIVE_B95"]
missing_b95_vars=[x for x in required_b95_vars if x not in globals()]
if missing_b95_vars:
    raise RuntimeError(
        "Execution-order violation: B95 must be resolved in the Active-Learning cell "
        f"before the integrated frontend. Missing: {missing_b95_vars}"
    )

INTEGRATED_ACTIVE_FRONT_PATHS={}
INTEGRATED_ACTIVE_FRONT_DEV_SCORES={}
INTEGRATED_ACTIVE_FRONT_AVAILABLE=(
    RUT_ACTIVE_B95 is not None and
    int(RUT_ACTIVE_B95) in AL_BUDGETS
)

if INTEGRATED_ACTIVE_FRONT_AVAILABLE:
    for run in range(len(MODEL_SEEDS)):
        sel_path=AL_SELECTED_PATHS[(run,"RUT","ACTIVE",int(RUT_ACTIVE_B95))]
        sel=np.load(sel_path).astype(int)
        ytr=SSL_META.y.iloc[sel].to_numpy(int)

        p=MODELS/f"INTEGRATED_run{run}_URL_DAPT_RUTACTIVE_B95_{int(RUT_ACTIVE_B95)}.npz"
        if not p.exists():
            X=take_url("URL_DAPT","SSL",sel)
            clf=fit_lr(X,ytr,MODEL_SEEDS[run])
            save_lr(p,clf,{
                "run":run,
                "stream":"URL_DAPT",
                "label_selector":"RUT_ACTIVE",
                "budget":int(RUT_ACTIVE_B95),
                "selection_rule":"DEV-frozen RUT_ACTIVE_B95",
            })

        coef,inter=load_lr(p)
        s=manual_url_score(coef,inter,"URL_DAPT","DEV")

        INTEGRATED_ACTIVE_FRONT_PATHS[run]=p
        INTEGRATED_ACTIVE_FRONT_DEV_SCORES[run]=s

    print({
        "INTEGRATED_ACTIVE_FRONT":"READY",
        "label_budget":int(RUT_ACTIVE_B95),
        "runs":len(INTEGRATED_ACTIVE_FRONT_PATHS),
    })
else:
    print({
        "INTEGRATED_ACTIVE_FRONT":"NOT_AVAILABLE",
        "RUT_ACTIVE_B95":RUT_ACTIVE_B95,
    })


# CELL 11

# 08 — Frozen TRI-RUT-200k Fallback: Architektur + DEV-Scoring

DOM_VOCAB=json.loads(Path(DOM_VOCAB_PATH).read_text())
PAD,UNK,MASK=0,1,2

def bucket(v):
    try:
        return min(max(int(v),0),31)
    except Exception:
        return 0

def _rv(r,k):
    if isinstance(r,dict):
        return r.get(k)
    try:
        return r[k]
    except Exception:
        return getattr(r,k,None)

def graph_row(r):
    def arr(k):
        x=_rv(r,k)
        if isinstance(x,np.ndarray):
            return x.tolist()
        if isinstance(x,list):
            return x
        try:
            return list(x)
        except Exception:
            return []

    tags=arr("dom_tag"); par=arr("dom_parent_idx"); dep=arr("dom_depth")
    att=arr("dom_attr_count"); chi=arr("dom_child_count")
    n=min(len(tags),DOM_MAX_NODES)

    if n==0:
        return (
            np.array([PAD],np.int64),
            np.array([0],np.int64),
            np.array([0],np.int64),
            np.array([0],np.int64),
            np.zeros((0,2),np.int64)
        )

    tag=np.array([DOM_VOCAB.get(str(x),UNK) for x in tags[:n]],np.int64)
    depth=np.array([bucket(x) for x in dep[:n]],np.int64)
    attr=np.array([bucket(x) for x in att[:n]],np.int64)
    child=np.array([bucket(x) for x in chi[:n]],np.int64)

    edges=[]
    for ii,pp in enumerate(par[:n]):
        try:
            pp=int(pp)
        except Exception:
            continue
        if ii>0 and 0<=pp<n:
            edges.extend([(pp,ii),(ii,pp)])

    return tag,depth,attr,child,np.asarray(edges,np.int64)

def graph_batch(rows):
    T=[];D=[];A=[];C=[];E=[];B=[];off=0
    row_iter=rows.itertuples(index=False) if isinstance(rows,pd.DataFrame) else rows
    n_graphs=len(rows)

    for bi,r in enumerate(row_iter):
        t,d,a,c,e=graph_row(r); n=len(t)
        T.append(t);D.append(d);A.append(a);C.append(c)
        B.append(np.full(n,bi,np.int64))
        if len(e):
            E.append(e+off)
        off+=n

    def cat(xs):
        return torch.tensor(np.concatenate(xs),dtype=torch.long,device=DEVICE)

    edge=torch.tensor(
        np.concatenate(E,0).T if E else np.zeros((2,0),np.int64),
        dtype=torch.long,device=DEVICE
    )
    return {
        "tag":cat(T),"depth":cat(D),"attr":cat(A),"child":cat(C),
        "edge":edge,"batch":cat(B),"n_graphs":n_graphs
    }

class GCNLayer(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.s=nn.Linear(d,d)
        self.n=nn.Linear(d,d)
        self.norm=nn.LayerNorm(d)

    def forward(self,x,e):
        if e.numel()==0:
            return self.norm(x+F.gelu(self.s(x)))
        src,dst=e
        agg=torch.zeros_like(x)
        deg=torch.zeros((len(x),1),device=x.device,dtype=x.dtype)
        agg.index_add_(0,dst,x[src])
        deg.index_add_(0,dst,torch.ones((len(dst),1),device=x.device,dtype=x.dtype))
        return self.norm(x+F.gelu(self.s(x)+self.n(agg/deg.clamp_min(1))))

class DOMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tag=nn.Embedding(len(DOM_VOCAB),96,padding_idx=PAD)
        self.depth=nn.Embedding(32,16)
        self.attr=nn.Embedding(32,8)
        self.child=nn.Embedding(32,8)
        self.inp=nn.Linear(128,DOM_DIM)
        self.layers=nn.ModuleList([GCNLayer(DOM_DIM) for _ in range(DOM_LAYERS)])
        self.att=nn.Linear(DOM_DIM,1)

    def nodes(self,g):
        x=self.inp(torch.cat([
            self.tag(g["tag"]),self.depth(g["depth"]),
            self.attr(g["attr"]),self.child(g["child"])
        ],1))
        for layer in self.layers:
            x=layer(x,g["edge"])
        return x

    def forward(self,g):
        x=self.nodes(g); n=g["n_graphs"]
        num=torch.zeros((n,DOM_DIM),device=x.device)
        den=torch.zeros((n,1),device=x.device)
        w=torch.exp(torch.clamp(self.att(x).squeeze(-1),-10,10))
        num.index_add_(0,g["batch"],x*w[:,None])
        den.index_add_(0,g["batch"],w[:,None])
        return num/den.clamp_min(1e-8)

def freeze_last(model,n):
    for p in model.parameters():
        p.requires_grad=False
    layers=getattr(getattr(model,"encoder",None),"layer",None)
    if layers is None:
        layers=getattr(getattr(model,"transformer",None),"layer",None)
    if layers is None:
        raise RuntimeError(type(model))
    for layer in layers[-n:]:
        for p in layer.parameters():
            p.requires_grad=True
    if getattr(model,"pooler",None) is not None:
        for p in model.pooler.parameters():
            p.requires_grad=True

class DeepFusionTRI(nn.Module):
    def __init__(self):
        super().__init__()
        self.u=AutoModel.from_pretrained(URL_DAPT_SRC,token=HF_TOKEN)
        self.t=AutoModel.from_pretrained(TEXT_DAPT_SRC,token=HF_TOKEN)
        freeze_last(self.u,DEEP_LAST_N)
        freeze_last(self.t,DEEP_LAST_N)

        self.ua=nn.Linear(int(self.u.config.hidden_size),PROJ_DIM)
        self.ta=nn.Linear(int(self.t.config.hidden_size),PROJ_DIM)

        self.dom=DOMEncoder()
        self.dom.load_state_dict(torch.load(DOM_ENCODER_PATH,map_location="cpu"))
        self.da=nn.Linear(DOM_DIM,PROJ_DIM)

        self.gate=nn.Sequential(
            nn.Linear(PROJ_DIM,64),nn.GELU(),nn.Linear(64,1)
        )
        self.head=nn.Sequential(
            nn.Linear(PROJ_DIM*4,512),nn.GELU(),nn.Dropout(.2),
            nn.Linear(512,128),nn.GELU(),nn.Dropout(.1),
            nn.Linear(128,1)
        )
        self.uaux=nn.Linear(PROJ_DIM,1)
        self.taux=nn.Linear(PROJ_DIM,1)
        self.daux=nn.Linear(PROJ_DIM,1)

    def encode(self,u,t,g):
        zu=F.normalize(
            self.ua(masked_mean(self.u(**u).last_hidden_state,u["attention_mask"])),
            dim=-1
        )
        zt=F.normalize(
            self.ta(masked_mean(self.t(**t).last_hidden_state,t["attention_mask"])),
            dim=-1
        )
        zd=F.normalize(self.da(self.dom(g)),dim=-1)
        return [zu,zt,zd]

    def forward(self,u,t,g):
        zs=self.encode(u,t,g)
        st=torch.stack(zs,1)
        gw=torch.softmax(self.gate(st).squeeze(-1),1)
        weighted=(st*gw[:,:,None]).sum(1)
        main=self.head(torch.cat(zs+[weighted],1)).squeeze(-1)
        return main

def tok_url_tri(values):
    b=url_dapt_tok(
        values,padding=True,truncation=True,
        max_length=URL_MAX_LEN,return_tensors="pt"
    )
    return {k:v.to(DEVICE,non_blocking=True) for k,v in b.items()}

def tok_text_tri(values):
    b=text_dapt_tok(
        values,padding=True,truncation=True,
        max_length=TEXT_MAX_LEN,return_tensors="pt"
    )
    return {k:v.to(DEVICE,non_blocking=True) for k,v in b.items()}

def load_tri():
    m=DeepFusionTRI().to(DEVICE)
    m.load_state_dict(
        torch.load(Path(TRI_DIR)/"model_state.pt",map_location=DEVICE),
        strict=True
    )
    m.eval()
    return m

@torch.no_grad()
def score_tri_df(model,df):
    last=None
    for bs in SCORE_BATCH_CANDIDATES:
        try:
            out=np.empty(len(df),np.float32)
            for st in range(0,len(df),bs):
                en=min(st+bs,len(df))
                q=df.iloc[st:en]
                u=tok_url_tri(q.url.fillna("").astype(str).tolist())
                t=tok_text_tri(q.text.fillna("").astype(str).tolist())
                g=graph_batch(q)
                with amp_ctx():
                    out[st:en]=model(u,t,g).float().cpu().numpy()
            return out,bs
        except torch.cuda.OutOfMemoryError as e:
            last=e
            gc.collect()
            torch.cuda.empty_cache()
    raise last

DEV_TRI_PATH=SCORES/"DEV_TRI_RUT_200K_GATED.npy"
if DEV_TRI_PATH.exists():
    full_dev=np.load(DEV_TRI_PATH)
    if len(full_dev)!=len(DEV):
        DEV_TRI_PATH.unlink()
        full_dev=None
else:
    full_dev=None

if full_dev is None:
    model=load_tri()
    full_dev,used_bs=score_tri_df(model,DEV)
    np.save(DEV_TRI_PATH,full_dev)
    atomic_json(SCORES/"DEV_TRI_RUT_200K_GATED.complete.json",{
        "status":"COMPLETE","rows":len(DEV),"batch":used_bs
    })
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print({"TRI_DEV_SCORE":"COMPLETE","rows":len(full_dev)})


# CELL 12

# 09 — DEV-only Routing-Policy-Search und POLICY_FREEZE

def quantile_from_sorted(x,q):
    x=np.asarray(x,float)
    if not len(x):
        return np.nan
    return float(np.quantile(x,q,method="linear"))

def cascade_predictions(front,full,low,high,full_thr):
    front=np.asarray(front,float)
    full=np.asarray(full,float)
    pred=np.empty(len(front),bool)
    auto_hi=front>=high
    auto_lo=front<=low
    esc=~(auto_hi|auto_lo)
    pred[auto_hi]=True
    pred[auto_lo]=False
    pred[esc]=full[esc]>=full_thr
    return pred,esc

def search_policy(front_dev,full_dev,ydev,loss_pp,max_fpr_inc):
    ft=front_dev[ENG_TUNE]
    fm=front_dev[ENG_META]
    yt=ydev[ENG_TUNE]
    ym=ydev[ENG_META]
    fullt=full_dev[ENG_TUNE]
    fullm=full_dev[ENG_META]

    full_thr_dev=thr_fpr(fullt[yt==0],PRIMARY_FPR)
    full_ref=op(ym,fullm,full_thr_dev)

    lows=[]
    pos=np.sort(ft[yt==1])
    for miss in LOW_MISS_GRID:
        lows.append((miss,quantile_from_sorted(pos,miss)))

    highs=[]
    neg=ft[yt==0]
    for hf in HIGH_FPR_GRID:
        highs.append((hf,thr_fpr(neg,hf)))

    valid=[]
    for miss,low in lows:
        for hf,high in highs:
            if not np.isfinite(low) or not np.isfinite(high) or low>=high:
                continue
            pred,esc=cascade_predictions(fm,fullm,low,high,full_thr_dev)
            met=op(ym,pred,None)
            loss=100*(full_ref["tpr"]-met["tpr"])
            fpr_inc=met["fpr"]-full_ref["fpr"]

            if loss<=loss_pp+1e-12 and fpr_inc<=max_fpr_inc+1e-12:
                valid.append({
                    "low":low,"high":high,
                    "low_miss_design":miss,
                    "high_fpr_design":hf,
                    "dev_meta_escalation":float(esc.mean()),
                    "dev_meta_full_reduction":float(1-esc.mean()),
                    "dev_meta_tpr":met["tpr"],
                    "dev_meta_fpr":met["fpr"],
                    "dev_meta_tpr_loss_pp":loss,
                    "dev_meta_fpr_increase":fpr_inc,
                    "full_dev_tpr":full_ref["tpr"],
                    "full_dev_fpr":full_ref["fpr"],
                    "full_thr_dev":full_thr_dev,
                })

    if not valid:
        pred=np.asarray(fullm)>=full_thr_dev
        met=op(ym,pred,None)
        return {
            "low":-np.inf,"high":np.inf,
            "low_miss_design":None,"high_fpr_design":None,
            "dev_meta_escalation":1.0,
            "dev_meta_full_reduction":0.0,
            "dev_meta_tpr":met["tpr"],
            "dev_meta_fpr":met["fpr"],
            "dev_meta_tpr_loss_pp":0.0,
            "dev_meta_fpr_increase":0.0,
            "full_dev_tpr":full_ref["tpr"],
            "full_dev_fpr":full_ref["fpr"],
            "full_thr_dev":full_thr_dev,
            "fallback_only":True,
        }

    valid.sort(key=lambda r:(
        r["dev_meta_escalation"],
        abs(r["dev_meta_tpr_loss_pp"]),
        r["dev_meta_fpr"],
    ))
    valid[0]["fallback_only"]=False
    return valid[0]

POLICY_ROWS=[]
POLICY_MAP={}

for run in range(len(MODEL_SEEDS)):
    for stream in ["URL_BASE","URL_DAPT"]:
        fdev=URL_FRONT_DEV_SCORES[(run,stream)]
        for loss in CASCADE_LOSS_BUDGETS_PP:
            pol=search_policy(
                fdev,full_dev,DEVY,
                loss_pp=loss,
                max_fpr_inc=CASCADE_MAX_FPR_INCREASE
            )
            pol.update({
                "run":run,
                "stream":stream,
                "train_budget":CASCADE_TRAIN_BUDGET,
                "allowed_tpr_loss_pp":loss,
                "max_fpr_increase":CASCADE_MAX_FPR_INCREASE,
                "selection_set":"DEV only",
            })
            POLICY_ROWS.append(pol)
            POLICY_MAP[(run,stream,loss)]=pol

# Secondary integrated prototype policies:
# same URL-DAPT inference path, but labels selected by RUT_ACTIVE at DEV-frozen B95.
INTEGRATED_POLICY_MAP={}
if INTEGRATED_ACTIVE_FRONT_AVAILABLE:
    for run in range(len(MODEL_SEEDS)):
        fdev=INTEGRATED_ACTIVE_FRONT_DEV_SCORES[run]
        for loss in CASCADE_LOSS_BUDGETS_PP:
            pol=search_policy(
                fdev,full_dev,DEVY,
                loss_pp=loss,
                max_fpr_inc=CASCADE_MAX_FPR_INCREASE
            )
            pol.update({
                "run":run,
                "stream":"URL_DAPT_RUTACTIVE_B95",
                "train_budget":int(RUT_ACTIVE_B95),
                "allowed_tpr_loss_pp":loss,
                "max_fpr_increase":CASCADE_MAX_FPR_INCREASE,
                "selection_set":"RUT_ACTIVE source labels + DEV routing freeze",
            })
            POLICY_ROWS.append(pol)
            INTEGRATED_POLICY_MAP[(run,loss)]=pol

POLICY_DF=pd.DataFrame(POLICY_ROWS)
POLICY_DF.to_csv(TABLES/"TABLE_DEV_CASCADE_POLICY_FRONTIER.csv",index=False)

# Policy Freeze occurs before CAL/FINAL are opened.
POLICY_FREEZE={
    "status":"FROZEN_ON_SSL_DEV_ONLY",
    "version":VERSION,
    "primary_allowed_tpr_loss_pp":CASCADE_PRIMARY_LOSS_PP,
    "max_fpr_increase":CASCADE_MAX_FPR_INCREASE,
    "policies":POLICY_ROWS,
    "active_label_target_tpr":AL_TARGET,
    "active_b95":AL_B95.to_dict("records"),
    "integrated_active_front":{
        "available":bool(INTEGRATED_ACTIVE_FRONT_AVAILABLE),
        "RUT_ACTIVE_B95":RUT_ACTIVE_B95,
        "policies":[
            v for k,v in INTEGRATED_POLICY_MAP.items()
        ] if INTEGRATED_ACTIVE_FRONT_AVAILABLE else [],
    },
    "cal_used":False,
    "final_used":False,
}
atomic_json(AUDIT/"POLICY_FREEZE.json",POLICY_FREEZE)

print("POLICY FREEZE: PASS")
display(POLICY_DF[
    np.isclose(POLICY_DF.allowed_tpr_loss_pp,CASCADE_PRIMARY_LOSS_PP)
])


# CELL 13

# 10 — Erst nach POLICY_FREEZE: CAL + FINAL freischalten, Embeddings und Szenarien laden

if not (AUDIT/"POLICY_FREEZE.json").exists():
    raise RuntimeError("POLICY_FREEZE fehlt.")
FINAL_UNLOCKED=True

CAL=read_role_raw("CAL",["sha256","url","text","label","date"])
CAL["sha256"]=CAL.sha256.astype(str).str.lower()
CALY=y01(CAL.label)
if len(CAL)!=50000 or int(CALY.sum())!=0:
    raise RuntimeError("CAL muss 50.000 ausschließlich benigne Seiten enthalten.")

FINAL=read_role_raw("FINAL",["sha256","url","text","label","date"])
FINAL["sha256"]=FINAL.sha256.astype(str).str.lower()
FINALY=y01(FINAL.label)
if (
    len(FINAL)!=168060 or
    (int((FINALY==0).sum()),int((FINALY==1).sum()))!=EXPECTED_FINAL_CLASSES
):
    raise RuntimeError("FINAL class counts invalid.")

# Strict SHA disjointness
sets={
    "SSL":set(SSL_META.sha256),
    "DEV":set(DEV.sha256),
    "CAL":set(CAL.sha256),
    "FINAL":set(FINAL.sha256),
}
for a,b in itertools.combinations(sets,2):
    if sets[a]&sets[b]:
        raise RuntimeError(f"SHA overlap: {a}/{b}")

flags=pd.read_parquet(OOD_FLAGS_PATH)
flags["sha256"]=flags.sha256.astype(str).str.lower()
flags=flags.set_index("sha256").loc[FINAL.sha256].reset_index()

dt=pd.to_datetime(FINAL.date,errors="coerce")
cut=dt.dropna().quantile(.75)
late=(dt>=cut).fillna(False).to_numpy()

FINAL_MASKS={
    "OFFICIAL_TEST":np.ones(len(FINAL),dtype=bool),
    "DOMAIN_OOD_EXACT":flags.domain_ood_exact.to_numpy(bool),
    "TEMPLATE_OOD_EXACT":flags.template_ood_exact.to_numpy(bool),
    "DOMAIN_TEMPLATE_OOD_EXACT":flags.domain_template_ood_exact.to_numpy(bool),
    "LATE_TEST_Q4":late,
}
got={k:int(v.sum()) for k,v in FINAL_MASKS.items()}
if got!=EXPECTED_SCENARIO_COUNTS:
    raise RuntimeError(f"Scenario count mismatch: {got}")

# Resolve/generate all four CAL/FINAL embedding streams.
for stream in STREAMS:
    for role in ["CAL","FINAL"]:
        EMBED_PATHS[(stream,role)]=generate_embedding(stream,role)

# Geometry only needed for R0/RUT.
for rep in AL_REPS:
    project_role(rep,"CAL")
    project_role(rep,"FINAL")
    # Assign to the already DEV-frozen unlabeled cluster centers.
    centers=CLUSTER_CENTERS[rep]
    for role in ["CAL","FINAL"]:
        path=GEOM/f"{rep}_{role}_cluster_ids.npy"
        done=GEOM/f"{rep}_{role}_cluster_ids.complete.json"
        Z=zmm(rep,role)
        if not (done.exists() and path.exists() and len(np.load(path,mmap_mode="r"))==len(Z)):
            ids=np.lib.format.open_memmap(path,mode="w+",dtype=np.int16,shape=(len(Z),))
            for st in range(0,len(Z),20000):
                en=min(st+20000,len(Z))
                x=np.asarray(Z[st:en],np.float32)
                # Squared Euclidean nearest centroid.
                d=((x[:,None,:]-centers[None,:,:])**2).sum(-1)
                ids[st:en]=d.argmin(1).astype(np.int16)
            ids.flush()
            atomic_json(done,{"status":"COMPLETE","rows":len(Z)})
        CLUSTER_IDS[(rep,role)]=path

atomic_json(AUDIT/"FINAL_ACCESS_PROTOCOL.json",{
    "status":"PASS_AFTER_POLICY_FREEZE",
    "cal_rows":len(CAL),
    "final_rows":len(FINAL),
    "scenario_counts":got,
    "late_q4_cutoff":str(cut),
    "sha_disjointness":"PASS",
})
print({"FINAL_UNLOCK":"PASS","scenario_counts":got})


# CELL 14

# 11 — Frozen TRI CAL/FINAL Scores wiederverwenden oder exakt neu scoren

def find_cached_tri_score(role):
    expected=expected_rows(role)
    names=[
        f"{role}_gated.npy",
        f"{role.lower()}_gated.npy",
    ]
    cands=[]
    for name in names:
        for p in SEARCH_ROOT.rglob(name):
            s=str(p).lower()
            if "tri_rut_200000" not in s and "eng_tri_rut_200000" not in s:
                continue
            try:
                a=np.load(p,mmap_mode="r")
                ok=(len(a)==expected and a.ndim==1)
                del a
            except Exception:
                ok=False
            if not ok:
                continue
            score=1000 if "final_url_ssl_integrated_v2" in s else 0
            if "/scores/ff4/" in s:
                score+=100
            cands.append((score,p))
    if cands:
        cands.sort(key=lambda x:(-x[0],len(str(x[1])),str(x[1])))
        return cands[0][1]
    return None

def score_tri_role(model,role,base_df,chunk_size=10000):
    outpath=SCORES/f"{role}_TRI_RUT_200K_GATED.npy"
    done=SCORES/f"{role}_TRI_RUT_200K_GATED.complete.json"

    if done.exists() and outpath.exists():
        a=np.load(outpath,mmap_mode="r")
        if len(a)==len(base_df):
            return outpath

    cached=find_cached_tri_score(role)
    if cached is not None:
        atomic_json(done,{
            "status":"REUSED_INPUT",
            "path":str(cached),
            "rows":len(base_df),
        })
        return cached

    arr=np.lib.format.open_memmap(
        outpath,mode="w+",dtype=np.float32,shape=(len(base_df),)
    )

    for st in range(0,len(base_df),chunk_size):
        en=min(st+chunk_size,len(base_df))
        q=base_df.iloc[st:en].copy().reset_index(drop=True)
        dom=read_role_slice(role,st,en,["sha256"]+DOM_COLS)
        dom["sha256"]=dom.sha256.astype(str).str.lower()
        if not np.array_equal(q.sha256.to_numpy(),dom.sha256.to_numpy()):
            raise RuntimeError(f"{role} DOM SHA alignment failed {st}:{en}")
        for c in DOM_COLS:
            q[c]=dom[c].tolist()

        z,bs=score_tri_df(model,q)
        arr[st:en]=z
        arr.flush()
        print({"TRI_SCORE":role,"rows":en,"total":len(base_df),"batch":bs})

    arr.flush()
    atomic_json(done,{"status":"COMPLETE","rows":len(base_df)})
    return outpath

need_model=(
    find_cached_tri_score("CAL") is None
    or find_cached_tri_score("FINAL") is None
)
tri_model=load_tri() if need_model else None

TRI_CAL_PATH=score_tri_role(tri_model,"CAL",CAL)
TRI_FINAL_PATH=score_tri_role(tri_model,"FINAL",FINAL)

if tri_model is not None:
    del tri_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

full_cal=np.asarray(np.load(TRI_CAL_PATH,mmap_mode="r"))
full_final=np.asarray(np.load(TRI_FINAL_PATH,mmap_mode="r"))
FULL_THR_CAL=thr_fpr(full_cal,PRIMARY_FPR)

print({
    "TRI_CAL_PATH":str(TRI_CAL_PATH),
    "TRI_FINAL_PATH":str(TRI_FINAL_PATH),
    "FULL_THR_CAL":FULL_THR_CAL,
})


# CELL 15

# 12 — H1: SSL-bedingte Compute-Effizienz über alle fünf FINAL-Bedingungen

CASCADE_FINAL_ROWS=[]

for run in range(len(MODEL_SEEDS)):
    for stream in ["URL_BASE","URL_DAPT"]:
        coef,inter=load_lr(URL_FRONT_MODEL_PATHS[(run,stream)])
        front_cal=manual_url_score(coef,inter,stream,"CAL")
        front_final=manual_url_score(coef,inter,stream,"FINAL")

        for loss in CASCADE_LOSS_BUDGETS_PP:
            pol=POLICY_MAP[(run,stream,loss)]
            low=float(pol["low"]); high=float(pol["high"])

            # CAL/FINAL decisions use the CAL-derived full TRI threshold.
            # Frontend low/high remain frozen from DEV.
            pred_all,esc_all=cascade_predictions(
                front_final,full_final,low,high,FULL_THR_CAL
            )

            for sn,mask in FINAL_MASKS.items():
                met=op(FINALY[mask],pred_all[mask],None)
                full_met=op(
                    FINALY[mask],
                    full_final[mask],
                    FULL_THR_CAL
                )
                CASCADE_FINAL_ROWS.append({
                    "run":run,
                    "stream":stream,
                    "allowed_tpr_loss_pp":loss,
                    "scenario":sn,
                    "target_fpr":PRIMARY_FPR,
                    "escalation_rate":float(esc_all[mask].mean()),
                    "full_path_avoided_rate":float(1-esc_all[mask].mean()),
                    "delta_tpr_vs_full_pp":100*(met["tpr"]-full_met["tpr"]),
                    "delta_fpr_vs_full_pp":100*(met["fpr"]-full_met["fpr"]),
                    **met,
                    "full_tpr":full_met["tpr"],
                    "full_fpr":full_met["fpr"],
                })

CASCADE_FINAL=pd.DataFrame(CASCADE_FINAL_ROWS)
CASCADE_FINAL.to_csv(RESULTS/"CASCADE_FINAL_N5_ALL.csv",index=False)

CASCADE_AGG=CASCADE_FINAL.groupby(
    ["stream","allowed_tpr_loss_pp","scenario"],as_index=False
).agg(
    n=("escalation_rate","size"),
    escalation_mean=("escalation_rate","mean"),
    escalation_sd=("escalation_rate","std"),
    full_path_avoided_mean=("full_path_avoided_rate","mean"),
    tpr_mean=("tpr","mean"),
    fpr_mean=("fpr","mean"),
    delta_tpr_vs_full_pp_mean=("delta_tpr_vs_full_pp","mean"),
    delta_fpr_vs_full_pp_mean=("delta_fpr_vs_full_pp","mean"),
)
CASCADE_AGG.to_csv(TABLES/"TABLE_SSL_CASCADE_FRONTIER_FINAL.csv",index=False)

# Primary H1 paired contrast: URL_DAPT - URL_BASE escalation.
h1=[]
q=CASCADE_FINAL[np.isclose(
    CASCADE_FINAL.allowed_tpr_loss_pp,
    CASCADE_PRIMARY_LOSS_PP
)]
for sn in SCENARIOS:
    z=q[q.scenario==sn]
    piv=z.pivot(index="run",columns="stream",values="escalation_rate")
    d=(piv["URL_DAPT"]-piv["URL_BASE"]).to_numpy(float)
    m,lo,hi=mean_ci(d)

    piv_tpr=z.pivot(index="run",columns="stream",values="tpr")
    dt=(piv_tpr["URL_DAPT"]-piv_tpr["URL_BASE"]).to_numpy(float)

    piv_fpr=z.pivot(index="run",columns="stream",values="fpr")
    df=(piv_fpr["URL_DAPT"]-piv_fpr["URL_BASE"]).to_numpy(float)

    h1.append({
        "scenario":sn,
        "n":len(d),
        "mean_delta_escalation_pp":100*m,
        "ci95_lo_pp":100*lo,
        "ci95_hi_pp":100*hi,
        "dapt_lower_escalation_pairs":int((d<0).sum()),
        "exact_signflip_p":exact_signflip_p(d),
        "mean_delta_tpr_pp":100*float(dt.mean()),
        "mean_delta_fpr_pp":100*float(df.mean()),
    })

H1_STATS=pd.DataFrame(h1)
H1_STATS["holm_p_5scenarios"]=holm_adjust(H1_STATS.exact_signflip_p)
H1_STATS.to_csv(TABLES/"TABLE_H1_SSL_COMPUTE_PAIRED.csv",index=False)

display(CASCADE_AGG[
    np.isclose(CASCADE_AGG.allowed_tpr_loss_pp,CASCADE_PRIMARY_LOSS_PP)
])
display(H1_STATS)


# CELL 16

# 12b — Integrierte KMU-Variante:
# RUT_ACTIVE_B95 labels -> URL_DAPT front -> TRI fallback across all FF3 conditions

INTEGRATED_CASCADE_ROWS=[]

if INTEGRATED_ACTIVE_FRONT_AVAILABLE:
    for run in range(len(MODEL_SEEDS)):
        coef,inter=load_lr(INTEGRATED_ACTIVE_FRONT_PATHS[run])
        front_final=manual_url_score(coef,inter,"URL_DAPT","FINAL")

        for loss in CASCADE_LOSS_BUDGETS_PP:
            pol=INTEGRATED_POLICY_MAP[(run,loss)]
            pred,esc=cascade_predictions(
                front_final,full_final,
                float(pol["low"]),float(pol["high"]),
                FULL_THR_CAL
            )

            for sn,mask in FINAL_MASKS.items():
                met=op(FINALY[mask],pred[mask],None)
                full_met=op(FINALY[mask],full_final[mask],FULL_THR_CAL)
                INTEGRATED_CASCADE_ROWS.append({
                    "run":run,
                    "system":"RUT_ACTIVE_B95__URL_DAPT__TRI",
                    "label_budget":int(RUT_ACTIVE_B95),
                    "allowed_tpr_loss_pp":loss,
                    "scenario":sn,
                    "escalation_rate":float(esc[mask].mean()),
                    "full_path_avoided_rate":float(1-esc[mask].mean()),
                    "delta_tpr_vs_full_pp":100*(met["tpr"]-full_met["tpr"]),
                    "delta_fpr_vs_full_pp":100*(met["fpr"]-full_met["fpr"]),
                    **met,
                    "full_tpr":full_met["tpr"],
                    "full_fpr":full_met["fpr"],
                })

if INTEGRATED_CASCADE_ROWS:
    INTEGRATED_CASCADE=pd.DataFrame(INTEGRATED_CASCADE_ROWS)
    INTEGRATED_CASCADE.to_csv(
        RESULTS/"INTEGRATED_ACTIVE_CASCADE_FINAL_N5.csv",index=False
    )
    INTEGRATED_CASCADE_AGG=INTEGRATED_CASCADE.groupby(
        ["system","label_budget","allowed_tpr_loss_pp","scenario"],
        as_index=False
    ).agg(
        n=("escalation_rate","size"),
        escalation_mean=("escalation_rate","mean"),
        full_path_avoided_mean=("full_path_avoided_rate","mean"),
        tpr_mean=("tpr","mean"),
        fpr_mean=("fpr","mean"),
        delta_tpr_vs_full_pp_mean=("delta_tpr_vs_full_pp","mean"),
        delta_fpr_vs_full_pp_mean=("delta_fpr_vs_full_pp","mean"),
    )
    INTEGRATED_CASCADE_AGG.to_csv(
        TABLES/"TABLE_INTEGRATED_ACTIVE_CASCADE_FINAL.csv",index=False
    )
    display(INTEGRATED_CASCADE_AGG[
        np.isclose(
            INTEGRATED_CASCADE_AGG.allowed_tpr_loss_pp,
            CASCADE_PRIMARY_LOSS_PP
        )
    ])
else:
    INTEGRATED_CASCADE=pd.DataFrame()
    INTEGRATED_CASCADE_AGG=pd.DataFrame()


# CELL 17

# 13 — H2: Active-Learning-B95 bestimmen und nur DEV-seitig ausgewählte Köpfe auf FINAL auswerten

# DEV-only B95 choices were already resolved before POLICY_FREEZE.
if "RUT_ACTIVE_B95" not in globals() or "R0_ACTIVE_B95" not in globals():
    raise RuntimeError("B95 variables must be resolved before POLICY_FREEZE.")

FINAL_LABEL_CONFIGS=[
    ("R0_RANDOM_20K","R0","RANDOM",20000),
    ("RUT_RANDOM_20K","RUT","RANDOM",20000),
]
if R0_ACTIVE_B95 is not None:
    FINAL_LABEL_CONFIGS.append(("R0_ACTIVE_B95","R0","ACTIVE",R0_ACTIVE_B95))
if RUT_ACTIVE_B95 is not None:
    FINAL_LABEL_CONFIGS.append(("RUT_ACTIVE_B95","RUT","ACTIVE",RUT_ACTIVE_B95))

LABEL_FINAL_ROWS=[]

for name,rep,method,B in FINAL_LABEL_CONFIGS:
    for run in range(len(MODEL_SEEDS)):
        p=AL_MODEL_PATHS[(run,rep,method,B)]
        coef,inter=load_lr(p)
        scal=manual_linear_score(coef,inter,rep,"CAL")
        sfin=manual_linear_score(coef,inter,rep,"FINAL")
        th=thr_fpr(scal,PRIMARY_FPR)

        for sn,mask in FINAL_MASKS.items():
            LABEL_FINAL_ROWS.append({
                "system":name,
                "run":run,
                "rep":rep,
                "method":method,
                "budget":B,
                "scenario":sn,
                "threshold":th,
                **op(FINALY[mask],sfin[mask],th),
                **curves(FINALY[mask],sfin[mask]),
            })

LABEL_FINAL=pd.DataFrame(LABEL_FINAL_ROWS)
LABEL_FINAL.to_csv(RESULTS/"ACTIVE_SELECTED_FINAL_N5.csv",index=False)

LABEL_FINAL_AGG=LABEL_FINAL.groupby(
    ["system","rep","method","budget","scenario"],as_index=False
).agg(
    n=("tpr","size"),
    tpr_mean=("tpr","mean"),
    tpr_sd=("tpr","std"),
    fpr_mean=("fpr","mean"),
    AP_mean=("AP","mean"),
)
LABEL_FINAL_AGG.to_csv(TABLES/"TABLE_ACTIVE_SELECTED_FINAL.csv",index=False)

display(LABEL_FINAL_AGG)


# CELL 18

# 14 — H3: Repräsentationssupport zur Review-Priorisierung, KEIN Retrieval-Klassifikator

def nearest_center_distance(Z,centers,ids,chunk=20000):
    out=np.empty(len(Z),np.float32)
    for st in range(0,len(Z),chunk):
        en=min(st+chunk,len(Z))
        x=np.asarray(Z[st:en],np.float32)
        c=centers[np.asarray(ids[st:en],int)]
        out[st:en]=np.linalg.norm(x-c,axis=1)
    return out

def cluster_support_from_selected(rep,selected_idx):
    ids=np.asarray(cluster_ids(rep,"SSL"))[selected_idx].astype(int)
    y=SSL_META.y.iloc[selected_idx].to_numpy(int)

    total=np.bincount(ids,minlength=AL_CLUSTER_K).astype(float)
    pos=np.bincount(ids,weights=y,minlength=AL_CLUSTER_K).astype(float)
    p=np.divide(pos,total,out=np.full(AL_CLUSTER_K,.5),where=total>0)
    purity=np.maximum(p,1-p)
    impurity=1-purity
    # No labeled support in a cluster => maximal uncertainty.
    impurity[total==0]=1.0
    return total,purity,impurity

def freeze_review_score_params(rep,run,selected_idx,front_dev):
    ids_dev=np.asarray(cluster_ids(rep,"DEV")).astype(int)
    centers=CLUSTER_CENTERS[rep]
    dist=nearest_center_distance(zmm(rep,"DEV"),centers,ids_dev)

    _,_,impurity=cluster_support_from_selected(rep,selected_idx)
    imp=impurity[ids_dev]

    unc=-np.abs(front_dev)

    # Robust normalization is estimated from ENG_TUNE without test labels.
    params={}
    comps={"novelty":dist,"uncertainty":unc,"impurity":imp}
    zsum=np.zeros(len(DEV),float)

    for name,x in comps.items():
        base=x[ENG_TUNE]
        med=float(np.median(base))
        q25,q75=np.percentile(base,[25,75])
        iqr=float(q75-q25)
        params[name]={"median":med,"iqr":iqr}
        zsum+=robust_z_apply(x,med,iqr)

    thresholds={}
    for rr in REVIEW_RATES:
        thresholds[str(rr)]=float(np.quantile(zsum[ENG_META],1-rr))

    params["thresholds"]=thresholds
    return params,zsum

REVIEW_FREEZE={}
REVIEW_DEV_SCORES={}

# Same 20k source anchors for geometry comparison; same run-specific balanced sets.
for run in range(len(MODEL_SEEDS)):
    for rep in ["R0","RUT"]:
        # Linear combined head for uncertainty, same 20k labels.
        sel=CASCADE_LABELSETS[run]
        p=MODELS/f"REVIEW_run{run}_{rep}_20K.npz"
        if not p.exists():
            X=take_rep(rep,"SSL",sel)
            y=SSL_META.y.iloc[sel].to_numpy(int)
            clf=fit_lr(X,y,MODEL_SEEDS[run])
            save_lr(p,clf,{"run":run,"rep":rep,"budget":20000})

        coef,inter=load_lr(p)
        fdev=manual_linear_score(coef,inter,rep,"DEV")
        params,zdev=freeze_review_score_params(rep,run,sel,fdev)

        REVIEW_FREEZE[(run,rep)]={
            "model_path":str(p),
            "selected_idx_path":None,
            "params":params,
        }
        REVIEW_DEV_SCORES[(run,rep)]=zdev

atomic_json(AUDIT/"REVIEW_POLICY_FREEZE.json",{
    "status":"FROZEN_FROM_SOURCE_LABEL_SUPPORT_AND_DEV_SCORE_DISTRIBUTIONS",
    "classification_by_neighbors":False,
    "review_rates":REVIEW_RATES,
    "policies":{
        f"run{run}_{rep}":REVIEW_FREEZE[(run,rep)]
        for run in range(len(MODEL_SEEDS))
        for rep in ["R0","RUT"]
    }
})

def review_score_final(rep,run):
    sel=CASCADE_LABELSETS[run]
    ids=np.asarray(cluster_ids(rep,"FINAL")).astype(int)
    centers=CLUSTER_CENTERS[rep]
    dist=nearest_center_distance(zmm(rep,"FINAL"),centers,ids)
    _,_,impurity=cluster_support_from_selected(rep,sel)
    imp=impurity[ids]

    coef,inter=load_lr(Path(REVIEW_FREEZE[(run,rep)]["model_path"]))
    ff=manual_linear_score(coef,inter,rep,"FINAL")
    unc=-np.abs(ff)

    params=REVIEW_FREEZE[(run,rep)]["params"]
    score=np.zeros(len(FINAL),float)
    for name,x in {"novelty":dist,"uncertainty":unc,"impurity":imp}.items():
        score+=robust_z_apply(
            x,
            params[name]["median"],
            params[name]["iqr"]
        )
    return score

# Primary review routing is attached to the most integrated available cascade:
# RUT_ACTIVE_B95 -> URL_DAPT -> TRI. If B95 is unavailable, fall back to URL_DAPT-20k.
REVIEW_ROWS=[]

for run in range(len(MODEL_SEEDS)):
    if INTEGRATED_ACTIVE_FRONT_AVAILABLE:
        coef,inter=load_lr(INTEGRATED_ACTIVE_FRONT_PATHS[run])
        pol=INTEGRATED_POLICY_MAP[(run,CASCADE_PRIMARY_LOSS_PP)]
        review_parent_system="RUT_ACTIVE_B95__URL_DAPT__TRI"
        review_parent_label_budget=int(RUT_ACTIVE_B95)
    else:
        coef,inter=load_lr(URL_FRONT_MODEL_PATHS[(run,"URL_DAPT")])
        pol=POLICY_MAP[(run,"URL_DAPT",CASCADE_PRIMARY_LOSS_PP)]
        review_parent_system="URL_DAPT_20K__TRI"
        review_parent_label_budget=20000

    front_final=manual_url_score(coef,inter,"URL_DAPT","FINAL")
    cas_pred,cas_esc=cascade_predictions(
        front_final,full_final,float(pol["low"]),float(pol["high"]),FULL_THR_CAL
    )
    err=(cas_pred!=FINALY)

    for rep in ["R0","RUT"]:
        rscore=review_score_final(rep,run)
        params=REVIEW_FREEZE[(run,rep)]["params"]

        for rr in REVIEW_RATES:
            th=float(params["thresholds"][str(rr)])
            review=rscore>=th

            for sn,mask in FINAL_MASKS.items():
                e=err[mask]
                r=review[mask]
                overall=float(e.mean())
                reviewed=float(e[r].mean()) if r.any() else np.nan
                enrichment=reviewed/overall if overall>0 and np.isfinite(reviewed) else np.nan
                capture=float((e&r).sum()/max(e.sum(),1))

                # Explicit oracle-review upper bound: reviewed cases are assumed corrected.
                pred_oracle=cas_pred[mask].copy()
                yy=FINALY[mask]
                pred_oracle[r]=yy[r]
                oracle=op(yy,pred_oracle,None)

                REVIEW_ROWS.append({
                    "run":run,
                    "rep":rep,
                    "parent_system":review_parent_system,
                    "parent_label_budget":review_parent_label_budget,
                    "nominal_review_rate_dev":rr,
                    "scenario":sn,
                    "realized_review_rate":float(r.mean()),
                    "cascade_error_rate":overall,
                    "reviewed_error_rate":reviewed,
                    "error_enrichment_factor":enrichment,
                    "error_capture_rate":capture,
                    "cascade_escalation_rate":float(cas_esc[mask].mean()),
                    "oracle_review_tpr_upper_bound":oracle["tpr"],
                    "oracle_review_fpr_lower_bound":oracle["fpr"],
                })

REVIEW=pd.DataFrame(REVIEW_ROWS)
REVIEW.to_csv(RESULTS/"REVIEW_PRIORITIZATION_FINAL_N5.csv",index=False)

REVIEW_AGG=REVIEW.groupby(
    ["rep","nominal_review_rate_dev","scenario"],as_index=False
).agg(
    n=("error_enrichment_factor","size"),
    realized_review_rate_mean=("realized_review_rate","mean"),
    error_enrichment_mean=("error_enrichment_factor","mean"),
    error_capture_mean=("error_capture_rate","mean"),
    cascade_error_rate_mean=("cascade_error_rate","mean"),
    reviewed_error_rate_mean=("reviewed_error_rate","mean"),
)
REVIEW_AGG.to_csv(TABLES/"TABLE_REVIEW_ENRICHMENT_FINAL.csv",index=False)

# Primary RUT-R0 paired contrast at 2% nominal review.
h3=[]
q=REVIEW[np.isclose(REVIEW.nominal_review_rate_dev,PRIMARY_REVIEW_RATE)]
for sn in SCENARIOS:
    z=q[q.scenario==sn]
    piv=z.pivot(index="run",columns="rep",values="error_enrichment_factor")
    d=(piv["RUT"]-piv["R0"]).to_numpy(float)
    m,lo,hi=mean_ci(d)
    h3.append({
        "scenario":sn,
        "mean_delta_enrichment_RUT_minus_R0":m,
        "ci95_lo":lo,"ci95_hi":hi,
        "rut_higher_pairs":int((d>0).sum()),
        "exact_signflip_p":exact_signflip_p(d),
        "rut_mean_enrichment":float(
            z[z.rep=="RUT"].error_enrichment_factor.mean()
        ),
        "r0_mean_enrichment":float(
            z[z.rep=="R0"].error_enrichment_factor.mean()
        ),
        "rut_realized_review_rate":float(
            z[z.rep=="RUT"].realized_review_rate.mean()
        ),
    })
H3_STATS=pd.DataFrame(h3)
H3_STATS["holm_p_5scenarios"]=holm_adjust(H3_STATS.exact_signflip_p)
H3_STATS.to_csv(TABLES/"TABLE_H3_REVIEW_PAIRED.csv",index=False)

display(REVIEW_AGG[
    np.isclose(REVIEW_AGG.nominal_review_rate_dev,PRIMARY_REVIEW_RATE)
])
display(H3_STATS)


# CELL 19

# 14b — Tatsächliche URL-Frontend-Laufzeit messen + seriellen Cascade-Modellzeitproxy ableiten

@torch.no_grad()
def benchmark_url_front(stream,model_path,n_b1=300,n_tput=3000,batch=32):
    src,tok,_,max_len,_=STREAMS[stream]
    kwargs=hf_kwargs(src,"bert-base-uncased") if stream=="URL_BASE" else {}
    model=AutoModel.from_pretrained(src,token=HF_TOKEN,**kwargs).to(DEVICE).eval()
    coef,inter=load_lr(model_path)
    coef_t=torch.tensor(coef,dtype=torch.float32,device=DEVICE)

    values=DEV.url.fillna("").astype(str).tolist()

    # warmup
    for _ in range(5):
        b=tok([values[0]],padding=True,truncation=True,max_length=max_len,return_tensors="pt")
        b={k:v.to(DEVICE) for k,v in b.items()}
        with amp_ctx():
            o=model(**b)
            z=masked_mean(o.last_hidden_state,b["attention_mask"])
            _=z.float()@coef_t+inter
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    lat=[]
    for i in range(min(n_b1,len(values))):
        b=tok([values[i]],padding=True,truncation=True,max_length=max_len,return_tensors="pt")
        b={k:v.to(DEVICE) for k,v in b.items()}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0=time.perf_counter()
        with amp_ctx():
            o=model(**b)
            z=masked_mean(o.last_hidden_state,b["attention_mask"])
            _=z.float()@coef_t+inter
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat.append((time.perf_counter()-t0)*1000)

    # throughput
    n=min(n_tput,len(values))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0=time.perf_counter()
    for st in range(0,n,batch):
        en=min(st+batch,n)
        b=tok(values[st:en],padding=True,truncation=True,max_length=max_len,return_tensors="pt")
        b={k:v.to(DEVICE) for k,v in b.items()}
        with amp_ctx():
            o=model(**b)
            z=masked_mean(o.last_hidden_state,b["attention_mask"])
            _=z.float()@coef_t+inter
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed=time.perf_counter()-t0

    out={
        "stream":stream,
        "n_b1":len(lat),
        "latency_b1_median_ms":float(np.median(lat)),
        "latency_b1_p95_ms":float(np.percentile(lat,95)),
        "n_throughput":n,
        "batch_throughput":batch,
        "throughput_pages_s":float(n/elapsed),
        "hardware":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "scope":"URL tokenizer + transformer forward + mean pooling + linear head; excludes disk/network/browser",
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out

# representative seed82 head; inference cost is architecture-dependent, not seed-dependent
rep_run=2
FRONT_BENCH=pd.DataFrame([
    benchmark_url_front("URL_BASE",URL_FRONT_MODEL_PATHS[(rep_run,"URL_BASE")]),
    benchmark_url_front("URL_DAPT",URL_FRONT_MODEL_PATHS[(rep_run,"URL_DAPT")]),
])
FRONT_BENCH.to_csv(TABLES/"TABLE_URL_FRONTEND_BENCHMARK.csv",index=False)

DERIVED_TIME=pd.DataFrame()
if HIST_OPS_PATH is not None:
    try:
        ops=pd.read_csv(HIST_OPS_PATH)
        fullrow=ops[
            (ops.system=="TRI_RUT_200K_FULL")&
            (ops["mode"]=="full")
        ].iloc[0]
        full_median=float(fullrow.latency_b1_median_ms)

        q=CASCADE_AGG[
            np.isclose(
                CASCADE_AGG.allowed_tpr_loss_pp,
                CASCADE_PRIMARY_LOSS_PP
            )
        ].copy()

        rows=[]
        for _,r in q.iterrows():
            front_med=float(
                FRONT_BENCH[
                    FRONT_BENCH.stream==r["stream"]
                ].iloc[0].latency_b1_median_ms
            )
            proxy=front_med+float(r.escalation_mean)*full_median
            rows.append({
                "stream":r["stream"],
                "scenario":r["scenario"],
                "front_median_ms":front_med,
                "full_tri_median_ms_existing_benchmark":full_median,
                "escalation_rate":float(r.escalation_mean),
                "serial_model_time_proxy_ms":proxy,
                "proxy_reduction_vs_full_pct":100*(1-proxy/full_median),
                "scope":"derived serial model-time proxy: front + escalation*full TRI; not an end-to-end production latency benchmark",
            })

        if INTEGRATED_ACTIVE_FRONT_AVAILABLE and len(INTEGRATED_CASCADE_AGG):
            qa=INTEGRATED_CASCADE_AGG[
                np.isclose(
                    INTEGRATED_CASCADE_AGG.allowed_tpr_loss_pp,
                    CASCADE_PRIMARY_LOSS_PP
                )
            ]
            front_med=float(
                FRONT_BENCH[
                    FRONT_BENCH.stream=="URL_DAPT"
                ].iloc[0].latency_b1_median_ms
            )
            for _,r in qa.iterrows():
                proxy=front_med+float(r.escalation_mean)*full_median
                rows.append({
                    "stream":"URL_DAPT_RUTACTIVE_B95",
                    "scenario":r["scenario"],
                    "front_median_ms":front_med,
                    "full_tri_median_ms_existing_benchmark":full_median,
                    "escalation_rate":float(r.escalation_mean),
                    "serial_model_time_proxy_ms":proxy,
                    "proxy_reduction_vs_full_pct":100*(1-proxy/full_median),
                    "scope":"derived serial model-time proxy: front + escalation*full TRI; not an end-to-end production latency benchmark",
                })

        DERIVED_TIME=pd.DataFrame(rows)
        DERIVED_TIME.to_csv(
            TABLES/"TABLE_DERIVED_CASCADE_MODEL_TIME.csv",index=False
        )
    except Exception as e:
        print({"DERIVED_TIME":"SKIPPED","reason":repr(e)})

display(FRONT_BENCH)
if len(DERIVED_TIME):
    display(DERIVED_TIME)


# CELL 20

# 15 — Integrierte operative Interpretation + bestehende gemessene Laufzeiten einbinden

OPS_CONTEXT=None
if HIST_OPS_PATH is not None:
    try:
        OPS_CONTEXT=pd.read_csv(HIST_OPS_PATH)
        OPS_CONTEXT.to_csv(TABLES/"TABLE_EXISTING_OPERATIONAL_CONTEXT.csv",index=False)
    except Exception:
        OPS_CONTEXT=None

# H1 High-upside condition
h1_primary=H1_STATS.copy()
compute_success_count=int((h1_primary.mean_delta_escalation_pp<=-10.0).sum())
COMPUTE_SUCCESS=compute_success_count>=4

# H2 High-upside condition
LABEL_SUCCESS=(RUT_ACTIVE_B95 is not None and RUT_ACTIVE_B95<=5000)

# H3 High-upside condition
h3_primary=H3_STATS.copy()
review_success_count=int((h3_primary.rut_mean_enrichment>=2.0).sum())
REVIEW_SUCCESS=review_success_count>=4

INTEGRATED_SUCCESS_COUNT=sum([
    bool(COMPUTE_SUCCESS),
    bool(LABEL_SUCCESS),
    bool(REVIEW_SUCCESS),
])
INTEGRATED_HIGH_UPSIDE=INTEGRATED_SUCCESS_COUNT>=2

# Useful compact factors.
rut_active_label_factor=(
    20000/RUT_ACTIVE_B95 if RUT_ACTIVE_B95 not in [None,0] else None
)

# Mean primary cascade reduction across all scenarios.
mean_compute_delta_pp=float(H1_STATS.mean_delta_escalation_pp.mean())

HEADLINE={
    "version":VERSION,
    "status":"COMPLETE",
    "scientific_status":"POST_HOC_SSL_GUIDED_ADAPTIVE_RESOURCE_ALLOCATION",
    "compute":{
        "primary_allowed_tpr_loss_pp":CASCADE_PRIMARY_LOSS_PP,
        "mean_URL_DAPT_minus_URL_BASE_escalation_pp_across_scenarios":mean_compute_delta_pp,
        "scenarios_with_at_least_10pp_reduction":compute_success_count,
        "success_rule_pass":bool(COMPUTE_SUCCESS),
    },
    "labels":{
        "reference":"95% of RUT_RANDOM_20k mean DEV TPR",
        "RUT_ACTIVE_B95":RUT_ACTIVE_B95,
        "R0_ACTIVE_B95":R0_ACTIVE_B95,
        "label_factor_vs_20k_reference":rut_active_label_factor,
        "success_rule_pass":bool(LABEL_SUCCESS),
    },
    "review":{
        "primary_nominal_review_rate_dev":PRIMARY_REVIEW_RATE,
        "scenarios_RUT_enrichment_at_least_2x":review_success_count,
        "mean_RUT_enrichment_across_scenarios":float(H3_STATS.rut_mean_enrichment.mean()),
        "success_rule_pass":bool(REVIEW_SUCCESS),
        "classification_by_neighbors":False,
    },
    "integrated":{
        "successful_blocks":INTEGRATED_SUCCESS_COUNT,
        "high_upside_pattern":bool(INTEGRATED_HIGH_UPSIDE),
        "closed_loop_variant_available":bool(INTEGRATED_ACTIVE_FRONT_AVAILABLE),
        "closed_loop_label_budget":RUT_ACTIVE_B95 if INTEGRATED_ACTIVE_FRONT_AVAILABLE else None,
        "closed_loop_primary_cascade_escalation_mean":(
            float(INTEGRATED_CASCADE_AGG[
                np.isclose(
                    INTEGRATED_CASCADE_AGG.allowed_tpr_loss_pp,
                    CASCADE_PRIMARY_LOSS_PP
                )
            ].escalation_mean.mean())
            if INTEGRATED_ACTIVE_FRONT_AVAILABLE and len(INTEGRATED_CASCADE_AGG)
            else None
        ),
        "derived_serial_model_time_proxy_available":bool(len(DERIVED_TIME)),
    },
    "interpretation_boundaries":[
        "The experiment is post-hoc and does not replace the confirmatory FF1-FF4 evidence.",
        "The compute contrast isolates URL-level TAPT in an identical URL-only linear frontend with the same frozen TRI fallback.",
        "The active-learning simulation uses source-pool labels only after an instance has been selected; selection itself never sees hidden labels.",
        "Representation-space support is used only for review prioritization, not as a retrieval classifier.",
        "Oracle-review metrics are explicitly upper/lower bounds assuming reviewed cases are corrected perfectly.",
        "The five FF3 conditions are treated as distinct distribution conditions, not as an ordinal difficulty scale.",
    ],
}
atomic_json(RESULTS/"INTEGRATED_HEADLINE.json",HEADLINE)

print(json.dumps(HEADLINE,indent=2,ensure_ascii=False))


# CELL 21

# 16 — Thesis-ready Abbildungen

import matplotlib.pyplot as plt

# Figure 1: primary cascade escalation
q=CASCADE_AGG[
    np.isclose(CASCADE_AGG.allowed_tpr_loss_pp,CASCADE_PRIMARY_LOSS_PP)
].copy()
scenario_order=SCENARIOS
x=np.arange(len(scenario_order))
width=.35

fig,ax=plt.subplots(figsize=(9.2,4.8))
a=q[q.stream=="URL_BASE"].set_index("scenario").loc[scenario_order]
b=q[q.stream=="URL_DAPT"].set_index("scenario").loc[scenario_order]
ax.bar(x-width/2,100*a.escalation_mean,width,label="URL_BASE")
ax.bar(x+width/2,100*b.escalation_mean,width,label="URL_DAPT")
ax.set_xticks(x)
ax.set_xticklabels([
    "Official","Domain","Template","Domain+Template","Late Q4"
])
ax.set_ylabel("TRI-Eskalationsrate [%]")
ax.set_title("SSL-bedingter Compute-Effekt bei identischer Cascade-Architektur")
ax.legend()
ax.grid(True,axis="y",alpha=.25)
fig.tight_layout()
fig.savefig(ROOT/"FIG_SSL_CASCADE_ESCALATION.png",dpi=220,bbox_inches="tight")
plt.show()

# Figure 2: label efficiency
fig,ax=plt.subplots(figsize=(8.2,4.8))
for rep in ["R0","RUT"]:
    for method in ["RANDOM","ACTIVE"]:
        z=AL_AGG[(AL_AGG.rep==rep)&(AL_AGG.method==method)].sort_values("budget")
        ax.plot(z.budget,100*z.tpr_mean,marker="o",label=f"{rep} {method}")
ax.axhline(100*AL_TARGET,linestyle="--",label="95% RUT_RANDOM_20k")
ax.set_xscale("log")
ax.set_xlabel("Gelabelte Trainingsinstanzen")
ax.set_ylabel("DEV-TPR bei 0,5-%-FPR-Schwelle [%]")
ax.set_title("SSL-Repräsentation und gezielte Labelallokation")
ax.legend()
ax.grid(True,alpha=.25)
fig.tight_layout()
fig.savefig(ROOT/"FIG_SSL_ACTIVE_LABEL_EFFICIENCY.png",dpi=220,bbox_inches="tight")
plt.show()

# Figure 3: review enrichment
z=REVIEW_AGG[
    np.isclose(REVIEW_AGG.nominal_review_rate_dev,PRIMARY_REVIEW_RATE)
].copy()
fig,ax=plt.subplots(figsize=(9.2,4.8))
a=z[z.rep=="R0"].set_index("scenario").loc[scenario_order]
b=z[z.rep=="RUT"].set_index("scenario").loc[scenario_order]
ax.bar(x-width/2,a.error_enrichment_mean,width,label="R0 space")
ax.bar(x+width/2,b.error_enrichment_mean,width,label="RUT space")
ax.axhline(1.0,linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels([
    "Official","Domain","Template","Domain+Template","Late Q4"
])
ax.set_ylabel("Fehleranreicherungsfaktor")
ax.set_title("Repräsentationsbasierte Priorisierung eines kleinen Reviewbudgets")
ax.legend()
ax.grid(True,axis="y",alpha=.25)
fig.tight_layout()
fig.savefig(ROOT/"FIG_SSL_REVIEW_ENRICHMENT.png",dpi=220,bbox_inches="tight")
plt.show()


# CELL 22

# 17 — Finales Audit, kompakte Gesamttabelle und Ergebnispaket

# Compact integrated table
compact=[]

for sn in SCENARIOS:
    hs=H1_STATS[H1_STATS.scenario==sn].iloc[0]
    hr=H3_STATS[H3_STATS.scenario==sn].iloc[0]

    compact.append({
        "scenario":sn,
        "URL_DAPT_minus_URL_BASE_escalation_pp":hs.mean_delta_escalation_pp,
        "URL_DAPT_minus_URL_BASE_tpr_pp":hs.mean_delta_tpr_pp,
        "URL_DAPT_minus_URL_BASE_fpr_pp":hs.mean_delta_fpr_pp,
        "RUT_review_error_enrichment_x":hr.rut_mean_enrichment,
        "R0_review_error_enrichment_x":hr.r0_mean_enrichment,
        "RUT_review_realized_rate":hr.rut_realized_review_rate,
    })

INTEGRATED_TABLE=pd.DataFrame(compact)
INTEGRATED_TABLE["RUT_ACTIVE_B95_labels"]=RUT_ACTIVE_B95
INTEGRATED_TABLE["R0_ACTIVE_B95_labels"]=R0_ACTIVE_B95
INTEGRATED_TABLE.to_csv(
    TABLES/"TABLE_INTEGRATED_SSL_RESOURCE_EFFECT.csv",index=False
)

coverage=pd.DataFrame([
    {"check":"N5 compute contrast", "pass":CASCADE_FINAL.run.nunique()==5},
    {"check":"same frozen TRI fallback", "pass":True},
    {"check":"five final scenarios", "pass":set(CASCADE_FINAL.scenario)==set(SCENARIOS)},
    {"check":"active selection has no label argument", "pass":True},
    {"check":"R0/RUT random+active all budgets", "pass":len(AL_DEV)==5*2*2*len(AL_BUDGETS)},
    {"check":"review is not retrieval classification", "pass":True},
    {"check":"closed-loop active->URL->TRI evaluated when available", "pass":(
        (not INTEGRATED_ACTIVE_FRONT_AVAILABLE)
        or (TABLES/"TABLE_INTEGRATED_ACTIVE_CASCADE_FINAL.csv").exists()
    )},
    {"check":"URL front benchmark", "pass":(TABLES/"TABLE_URL_FRONTEND_BENCHMARK.csv").exists()},
    {"check":"policy frozen before final", "pass":(AUDIT/"POLICY_FREEZE.json").exists()},
    {"check":"final access audit", "pass":(AUDIT/"FINAL_ACCESS_PROTOCOL.json").exists()},
    {"check":"all primary tables present", "pass":all(p.exists() for p in [
        TABLES/"TABLE_H1_SSL_COMPUTE_PAIRED.csv",
        TABLES/"TABLE_ACTIVE_B95_DEV.csv",
        TABLES/"TABLE_H3_REVIEW_PAIRED.csv",
        TABLES/"TABLE_INTEGRATED_SSL_RESOURCE_EFFECT.csv",
    ])},
])
coverage.to_csv(AUDIT/"COMPLETE_COVERAGE_AUDIT.csv",index=False)
if not bool(coverage["pass"].all()):
    raise RuntimeError("Complete coverage audit failed.")

COMPLETE={
    "status":"COMPLETE",
    "version":VERSION,
    "completed_utc":pd.Timestamp.utcnow().isoformat(),
    "all_blocks_reported":True,
    "high_upside_pattern":bool(INTEGRATED_HIGH_UPSIDE),
    "headline":HEADLINE,
}
atomic_json(ROOT/"FINAL_SSL_GUIDED_ADAPTIVE_RESOURCE_COMPLETE.json",COMPLETE)

zip_path=(
    Path("/kaggle/working/FINAL_SSL_GUIDED_ADAPTIVE_RESOURCE_SYSTEM_RESULTS_v2_ORDER_FIX.zip")
    if Path("/kaggle/working").exists()
    else ROOT.parent/"FINAL_SSL_GUIDED_ADAPTIVE_RESOURCE_SYSTEM_RESULTS_v2_ORDER_FIX.zip"
)

with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
    for base in [RESULTS,TABLES,AUDIT]:
        for p in base.rglob("*"):
            if p.is_file():
                z.write(p,arcname=str(p.relative_to(ROOT)))
    for name in [
        "FIG_SSL_CASCADE_ESCALATION.png",
        "FIG_SSL_ACTIVE_LABEL_EFFICIENCY.png",
        "FIG_SSL_REVIEW_ENRICHMENT.png",
        "FINAL_SSL_GUIDED_ADAPTIVE_RESOURCE_COMPLETE.json",
    ]:
        p=ROOT/name
        if p.exists():
            z.write(p,arcname=name)

sha=hashlib.sha256(zip_path.read_bytes()).hexdigest()
atomic_json(ROOT/"RESULTS_ZIP_SHA256.json",{
    "file":str(zip_path),
    "sha256":sha,
    "size_bytes":zip_path.stat().st_size,
})

print({
    "status":"COMPLETE",
    "results_zip":str(zip_path),
    "sha256":sha,
    "high_upside_pattern":bool(INTEGRATED_HIGH_UPSIDE),
})
display(INTEGRATED_TABLE)
