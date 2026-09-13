# KI-Unterstuetzung: siehe Erklaerung der Arbeit.
# Ursprungsdatei: FINAL_FF4_SEQUENTIAL_ARCHITECTURE_ABLATION_N10_v1.ipynb
# CELL 1
# 00 — Imports, feste Konfiguration und prospektiv gelockter Closure-Plan
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gc, json, math, time, random, hashlib, itertools, shutil, warnings, zipfile
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)
from scipy.stats import t as student_t, beta as beta_dist

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")

VERSION = "FINAL_FF4_SEQUENTIAL_ARCHITECTURE_ABLATION_N10_v1"
SCIENTIFIC_STATUS = "POST_HOC_FF4_CLOSURE; PLAN_AND_SEEDS_FIXED_BEFORE_EXECUTION"

# Exactly 10 fresh paired downstream replicates for this closure analysis.
MODEL_SEEDS = [442, 462, 482, 502, 522, 542, 562, 582, 602, 622]
LABEL_RANK_SEEDS = [9401, 9402, 9403, 9404, 9405, 9406, 9407, 9408, 9409, 9410]

BUDGET = 20_000
PRIMARY_FPR = 0.005

# Final deep-system hyperparameters retained unchanged.
URL_MAX_LEN = 128
TEXT_MAX_LEN = 256
PROJ_DIM = 256
DEEP_LAST_N = 4
DEEP_EPOCHS = 1
ENCODER_LR = 1e-5
HEAD_LR = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
AUX_TOTAL_WEIGHT = 0.30
GRAD_CLIP = 1.0

# A single batch size is deliberately held fixed across ALL four variants
# to avoid a batch-size / number-of-updates confound in the component ablation.
ABLATION_BATCH = 8
SCORE_BATCH_CANDIDATES = [128, 96, 64, 48, 32, 16, 8]

# DOM settings from the final integrated system.
DOM_MAX_NODES = 512
DOM_DIM = 128
DOM_LAYERS = 3
DOM_COLS = [
    "dom_tag", "dom_parent_idx", "dom_depth",
    "dom_attr_count", "dom_child_count",
]

VARIANTS = {
    "P0_URL": {
        "modalities": ["url"],
        "fusion": "none",
        "use_text": False,
        "use_dom": False,
        "head_input_dim": PROJ_DIM,
    },
    "P1_DUAL_EQUAL": {
        "modalities": ["url", "text"],
        "fusion": "equal",
        "use_text": True,
        "use_dom": False,
        "head_input_dim": PROJ_DIM * 3,
    },
    "P2_TRI_EQUAL": {
        "modalities": ["url", "text", "dom"],
        "fusion": "equal",
        "use_text": True,
        "use_dom": True,
        "head_input_dim": PROJ_DIM * 4,
    },
    "P3_TRI_GATED": {
        "modalities": ["url", "text", "dom"],
        "fusion": "gated",
        "use_text": True,
        "use_dom": True,
        "head_input_dim": PROJ_DIM * 4,
    },
}

PRIMARY_CONTRASTS = [
    ("TEXT_ADD", "P0_URL", "P1_DUAL_EQUAL"),
    ("DOM_ADD", "P1_DUAL_EQUAL", "P2_TRI_EQUAL"),
    ("GATING_ADD", "P2_TRI_EQUAL", "P3_TRI_GATED"),
]

if len(MODEL_SEEDS) != 10 or len(LABEL_RANK_SEEDS) != 10:
    raise RuntimeError("N10 seed plan invalid.")
if len(set(MODEL_SEEDS)) != 10 or len(set(LABEL_RANK_SEEDS)) != 10:
    raise RuntimeError("Seeds must be unique.")
if list(VARIANTS) != ["P0_URL","P1_DUAL_EQUAL","P2_TRI_EQUAL","P3_TRI_GATED"]:
    raise RuntimeError("Variant order changed.")
if len(PRIMARY_CONTRASTS) != 3:
    raise RuntimeError("Exactly three primary adjacent contrasts are required.")

if not torch.cuda.is_available():
    raise RuntimeError("GPU required. Target environment: Kaggle T4 or better.")
DEVICE = torch.device("cuda")
AMP = True

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

ROOT = Path("/kaggle/working/final_ff4_sequential_architecture_ablation_n10_v1")
if not Path("/kaggle/working").exists():
    ROOT = Path("/mnt/data/final_ff4_sequential_architecture_ablation_n10_v1")

RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
AUDIT = ROOT / "audit"
TABLES = ROOT / "tables"
for p in [ROOT, RUNS, RESULTS, AUDIT, TABLES]:
    p.mkdir(parents=True, exist_ok=True)

def seed_all(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, path)

def atomic_torch(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)

def amp_ctx():
    return torch.autocast("cuda", dtype=torch.float16) if AMP else nullcontext()

PROTOCOL = {
    "version": VERSION,
    "scientific_status": SCIENTIFIC_STATUS,
    "historical_final_known": True,
    "used_for_retroactive_model_selection": False,
    "no_optional_stopping": True,
    "all_40_planned_fits_must_be_reported": True,
    "representation": "RUT = URL_DAPT + TEXT_DAPT; fixed SSL checkpoints",
    "dom_ssl_checkpoint_fixed": True,
    "transformer_ssl_retraining": False,
    "budget": BUDGET,
    "model_seeds": MODEL_SEEDS,
    "label_rank_seeds": LABEL_RANK_SEEDS,
    "paired_same_labelset_within_replicate": True,
    "variants": VARIANTS,
    "primary_contrasts": [
        {"name": n, "baseline": a, "candidate": b}
        for n, a, b in PRIMARY_CONTRASTS
    ],
    "primary_endpoint": {
        "dataset": "DEV / ENG_META",
        "metric": "TPR",
        "threshold_source": "disjoint DEV / ENG_TUNE benign subset",
        "target_fpr": PRIMARY_FPR,
        "test": "two-sided exact paired sign-flip permutation test",
        "multiplicity": "Holm across exactly 3 adjacent architecture contrasts",
    },
    "secondary_metrics": ["FPR", "AP", "AUC", "P_at_R90", "FPR_at_TPR90"],
    "deep_hyperparameters": {
        "last_trainable_transformer_layers": DEEP_LAST_N,
        "epochs": DEEP_EPOCHS,
        "encoder_lr": ENCODER_LR,
        "head_lr": HEAD_LR,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "aux_total_weight": AUX_TOTAL_WEIGHT,
        "gradient_clip": GRAD_CLIP,
        "ablation_batch_fixed_all_variants": ABLATION_BATCH,
    },
    "access_policy": {
        "readable_roles": ["SSL", "DEV"],
        "CAL": "FORBIDDEN",
        "FINAL": "FORBIDDEN",
    },
    "cascade": "NOT_RECOMPUTED; existing frozen Full-TRI vs Cascade evidence remains separate",
}
atomic_json(AUDIT / "FF4_N10_PROTOCOL_LOCK.json", PROTOCOL)

print(torch.cuda.get_device_name(0))
print(json.dumps(PROTOCOL, indent=2, ensure_ascii=False))

# CELL 2
# 01 — Small-result resume: import already completed conditions from prior Kaggle outputs
SEARCH_ROOT = Path("/kaggle/input") if Path("/kaggle/input").exists() else Path("/mnt/data")

def import_previous_condition_results():
    imported = 0
    conflicts = []
    for p in SEARCH_ROOT.rglob("FF4_N10_CONDITION_RESULT.json"):
        try:
            obj = json.loads(p.read_text())
        except Exception:
            continue
        if obj.get("version") != VERSION or obj.get("status") != "COMPLETE":
            continue

        rep = int(obj["replicate"])
        variant = str(obj["variant"])
        if variant not in VARIANTS or not (0 <= rep < 10):
            continue

        dst = RUNS / f"rep{rep:02d}" / variant / "FF4_N10_CONDITION_RESULT.json"
        if dst.exists():
            old = json.loads(dst.read_text())
            # Completed duplicates must be scientifically identical in key design fields.
            keys = ["replicate","model_seed","label_rank_seed","variant","selection_sha256"]
            if any(str(old.get(k)) != str(obj.get(k)) for k in keys):
                conflicts.append((str(p), str(dst)))
                continue
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(dst, obj)
            imported += 1

    if conflicts:
        raise RuntimeError(f"Conflicting previous condition results: {conflicts[:3]}")
    print({"previous_completed_conditions_imported": imported})

import_previous_condition_results()

# CELL 3
# 02 — Robust input resolver; ONLY SSL and DEV data roles are addressable

ROLE_REL = {
    "SSL": Path("roles/ssl_pool_200k"),
    "DEV": Path("roles/development"),
}
EXPECTED_COUNTS = {"SSL": 200_000, "DEV": 20_000}

def parquet_rows(d):
    fs = sorted(Path(d).glob("*.parquet"))
    if not fs:
        raise RuntimeError(f"No parquet files found: {d}")
    return sum(pq.ParquetFile(p).metadata.num_rows for p in fs)

# 1) Physical canonical Data Freeze.
data_candidates = []
exact_data = Path("/kaggle/input/datasets/cristinakaufalt/finaldatafreeze/phreshphish_FINAL_DATA_FREEZE_v3")
if exact_data.exists():
    data_candidates.append((1000, exact_data))

for r in SEARCH_ROOT.rglob("roles"):
    if not r.is_dir():
        continue
    root = r.parent
    if not all((root / rel).exists() for rel in ROLE_REL.values()):
        continue
    score = 0
    s = str(root).lower()
    if "finaldatafreeze" in s:
        score += 100
    if root.name == "phreshphish_FINAL_DATA_FREEZE_v3":
        score += 50
    data_candidates.append((score, root))

if not data_candidates:
    raise RuntimeError(
        "Canonical physical Data Freeze not found. Attach the finaldatafreeze Kaggle dataset."
    )

# Deduplicate candidates.
uniq = {}
for score, p in data_candidates:
    uniq[str(p)] = max(score, uniq.get(str(p), -1))
data_candidates = sorted(
    [(score, Path(p)) for p, score in uniq.items()],
    key=lambda x: (-x[0], len(str(x[1])), str(x[1])),
)
DATA_ROOT = data_candidates[0][1]
ROLE_DIR = {k: DATA_ROOT / v for k, v in ROLE_REL.items()}
COUNTS = {k: parquet_rows(v) for k, v in ROLE_DIR.items()}
if COUNTS != EXPECTED_COUNTS:
    raise RuntimeError(f"Unexpected SSL/DEV role counts: {COUNTS}")

# 2) Private SSL labels. Search is independent of physical-role root.
manifest_candidates = []
exact_manifest = DATA_ROOT / "manifests" / "train_role_manifest_PRIVATE_WITH_LABELS.parquet"
if exact_manifest.exists():
    manifest_candidates.append((1000, exact_manifest))

for p in SEARCH_ROOT.rglob("train_role_manifest_PRIVATE_WITH_LABELS.parquet"):
    score = 0
    s = str(p).lower()
    if "finaldatafreeze" in s:
        score += 100
    if "freezeresume" in s:
        score += 20
    manifest_candidates.append((score, p))

if not manifest_candidates:
    raise RuntimeError(
        "Private SSL label manifest not found: train_role_manifest_PRIVATE_WITH_LABELS.parquet"
    )

manifest_candidates.sort(key=lambda x: (-x[0], len(str(x[1])), str(x[1])))
PRIVATE_MANIFEST = manifest_candidates[0][1]

def hf_model_dir_ok(p):
    p = Path(p)
    if not (p / "config.json").exists():
        return False
    return any((p / n).exists() for n in [
        "model.safetensors", "pytorch_model.bin",
        "model.safetensors.index.json", "pytorch_model.bin.index.json",
    ])

# 3) URL-DAPT checkpoint: MUST be existing; no retraining fallback.
url_dapt_candidates = []
for p in SEARCH_ROOT.rglob("URL_DAPT_BERT_200K"):
    if not p.is_dir() or not hf_model_dir_ok(p):
        continue
    score = 0
    s = str(p).lower()
    if "final_url_ssl_integrated_v2" in s:
        score += 200
    if "newdataset" in s:
        score += 100
    if "checkpoints" in s:
        score += 20
    url_dapt_candidates.append((score, p))

if not url_dapt_candidates:
    raise RuntimeError(
        "Frozen URL-DAPT checkpoint URL_DAPT_BERT_200K not found. "
        "Attach the final integrated Kaggle output dataset; this notebook will not retrain DAPT."
    )
url_dapt_candidates.sort(key=lambda x: (-x[0], len(str(x[1])), str(x[1])))
URL_DAPT_SRC = url_dapt_candidates[0][1]

# 4) Historical Text-DAPT + DOM-SSL assets must come from one coherent root.
old_roots = []
for p in SEARCH_ROOT.rglob("phreshphish_FINAL_DEEP_TRIMODAL_SSL_v4_3_LABELSET_ALIGNMENT_FIX"):
    if not p.is_dir():
        continue
    req = [
        p / "checkpoints" / "R1_DAPT_TEXT",
        p / "checkpoints" / "DOM_MASKED_SSL" / "encoder.pt",
        p / "audit" / "DOM_VOCAB.json",
    ]
    score = sum([
        hf_model_dir_ok(req[0]),
        req[1].exists(),
        req[2].exists(),
    ])
    if score == 3:
        old_roots.append((300, p))

# Fallback: infer common root from R1_DAPT_TEXT.
for p in SEARCH_ROOT.rglob("R1_DAPT_TEXT"):
    if not p.is_dir() or not hf_model_dir_ok(p):
        continue
    root = p.parent.parent
    enc = root / "checkpoints" / "DOM_MASKED_SSL" / "encoder.pt"
    vocab = root / "audit" / "DOM_VOCAB.json"
    if enc.exists() and vocab.exists():
        score = 200
        if "v4_3" in str(root).lower():
            score += 50
        old_roots.append((score, root))

if not old_roots:
    raise RuntimeError(
        "Coherent historical Text-DAPT + DOM-SSL asset root not found. "
        "Required together: checkpoints/R1_DAPT_TEXT, "
        "checkpoints/DOM_MASKED_SSL/encoder.pt, audit/DOM_VOCAB.json."
    )

# Deduplicate.
old_uniq = {}
for score, p in old_roots:
    old_uniq[str(p)] = max(score, old_uniq.get(str(p), -1))
old_roots = sorted(
    [(score, Path(p)) for p, score in old_uniq.items()],
    key=lambda x: (-x[0], len(str(x[1])), str(x[1])),
)
OLD_ROOT = old_roots[0][1]
TEXT_DAPT_SRC = OLD_ROOT / "checkpoints" / "R1_DAPT_TEXT"
DOM_ENCODER_PATH = OLD_ROOT / "checkpoints" / "DOM_MASKED_SSL" / "encoder.pt"
DOM_VOCAB_PATH = OLD_ROOT / "audit" / "DOM_VOCAB.json"

def sha256_file(path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def weight_files(model_dir):
    p = Path(model_dir)
    out = []
    for pat in ["*.safetensors", "*.bin"]:
        out.extend(p.glob(pat))
    return sorted(out)

# Preflight local-only tokenizers/models. No network is permitted.
url_tok = AutoTokenizer.from_pretrained(URL_DAPT_SRC, local_files_only=True)
text_tok = AutoTokenizer.from_pretrained(TEXT_DAPT_SRC, local_files_only=True)

_u = AutoModel.from_pretrained(URL_DAPT_SRC, local_files_only=True)
_t = AutoModel.from_pretrained(TEXT_DAPT_SRC, local_files_only=True)
if int(_u.config.hidden_size) != 768 or int(_t.config.hidden_size) != 768:
    raise RuntimeError(
        f"Unexpected hidden sizes: URL={_u.config.hidden_size}, TEXT={_t.config.hidden_size}"
    )
del _u, _t
gc.collect()

url_weights = weight_files(URL_DAPT_SRC)
text_weights = weight_files(TEXT_DAPT_SRC)
if not url_weights or not text_weights:
    raise RuntimeError("Model weight files missing after resolver validation.")

INPUT_RESOLUTION = {
    "status": "PASS",
    "data_root": str(DATA_ROOT),
    "role_counts": COUNTS,
    "private_manifest": str(PRIVATE_MANIFEST),
    "url_dapt_src": str(URL_DAPT_SRC),
    "text_dapt_src": str(TEXT_DAPT_SRC),
    "dom_encoder": str(DOM_ENCODER_PATH),
    "dom_vocab": str(DOM_VOCAB_PATH),
    "url_dapt_weight_file": str(url_weights[0]),
    "url_dapt_weight_sha256": sha256_file(url_weights[0]),
    "text_dapt_weight_file": str(text_weights[0]),
    "text_dapt_weight_sha256": sha256_file(text_weights[0]),
    "dom_encoder_sha256": sha256_file(DOM_ENCODER_PATH),
    "dom_vocab_sha256": sha256_file(DOM_VOCAB_PATH),
    "network_fallback": False,
    "cal_resolved": False,
    "final_resolved": False,
}
atomic_json(AUDIT / "INPUT_RESOLUTION.json", INPUT_RESOLUTION)
print(json.dumps(INPUT_RESOLUTION, indent=2, ensure_ascii=False))

# CELL 4
# 03 — SSL labels, DEV-only engineering split and paired 20k labelsets

def y01(s):
    if pd.api.types.is_numeric_dtype(s):
        y = pd.to_numeric(s).astype(int).to_numpy()
        if not np.isin(y, [0,1]).all():
            raise RuntimeError("Numeric labels outside {0,1}.")
        return y

    z = s.astype(str).str.lower().str.strip()
    mapping = {
        "benign":0, "legitimate":0, "legit":0, "0":0, "false":0,
        "phish":1, "phishing":1, "malicious":1, "1":1, "true":1,
    }
    y = z.map(mapping)
    if y.isna().any():
        raise RuntimeError(f"Unknown labels: {sorted(z[y.isna()].unique())[:10]}")
    return y.astype(int).to_numpy()

def read_role(key, cols):
    # Hard access policy: function knows only SSL and DEV.
    if key not in {"SSL", "DEV"}:
        raise RuntimeError(f"Forbidden role access attempted: {key}")
    out = []
    for p in sorted(ROLE_DIR[key].glob("*.parquet")):
        names = set(pq.ParquetFile(p).schema_arrow.names)
        missing = [c for c in cols if c not in names]
        if missing:
            raise RuntimeError(f"{p}: missing columns {missing}")
        out.append(pd.read_parquet(p, columns=cols))
    z = pd.concat(out, ignore_index=True)
    if len(z) != EXPECTED_COUNTS[key]:
        raise RuntimeError(f"{key} row mismatch: {len(z)}")
    return z

# Private SSL labels.
priv = pd.read_parquet(
    PRIVATE_MANIFEST,
    columns=["sha256", "label", "role"],
)
priv["sha256"] = priv.sha256.astype(str).str.lower()
SSL_PRIVATE = (
    priv[priv.role.eq("SSL_POOL")][["sha256","label"]]
    .copy()
    .reset_index(drop=True)
)
if len(SSL_PRIVATE) != 200_000 or SSL_PRIVATE.sha256.duplicated().any():
    raise RuntimeError("SSL private manifest invalid.")
SSL_PRIVATE["y"] = y01(SSL_PRIVATE.label)

# Physical SSL SHA order; explicitly prove that no physical SSL parquet contains labels.
ssl_meta_parts = []
for p in sorted(ROLE_DIR["SSL"].glob("*.parquet")):
    schema = set(pq.ParquetFile(p).schema_arrow.names)
    if "label" in schema:
        raise RuntimeError(f"Physical SSL file unexpectedly contains label: {p}")
    q = pd.read_parquet(p, columns=["sha256"])
    ssl_meta_parts.append(q)

SSL_META = pd.concat(ssl_meta_parts, ignore_index=True)
SSL_META["sha256"] = SSL_META.sha256.astype(str).str.lower()

if set(SSL_META.sha256) != set(SSL_PRIVATE.sha256):
    raise RuntimeError("Physical/private SSL SHA mismatch.")

label_map = dict(zip(SSL_PRIVATE.sha256, SSL_PRIVATE.y.astype(int)))
SSL_META["y"] = SSL_META.sha256.map(label_map)
if SSL_META.y.isna().any():
    raise RuntimeError("Could not attach private SSL labels by SHA.")
SSL_META["y"] = SSL_META.y.astype(int)

# DEV is the only evaluation role in this notebook.
DEV = read_role(
    "DEV",
    ["sha256","url","text","label"] + DOM_COLS,
)
DEV["sha256"] = DEV.sha256.astype(str).str.lower()
DEVY = y01(DEV.label)

if len(DEV) != 20_000:
    raise RuntimeError("DEV must contain exactly 20,000 rows.")
if set(SSL_META.sha256) & set(DEV.sha256):
    raise RuntimeError("SSL/DEV SHA overlap.")

# Reconstruct EXACT engineering split used by the final integrated system:
# first a stratified half, then split that half 50/50 into ENG_TUNE and ENG_META.
idx = np.arange(len(DEV))
_, eng_half = train_test_split(
    idx,
    test_size=.5,
    stratify=DEVY,
    random_state=20260812,
)
ENG_TUNE, ENG_META = train_test_split(
    eng_half,
    test_size=.5,
    stratify=DEVY[eng_half],
    random_state=20260814,
)
ENG_TUNE = np.sort(ENG_TUNE)
ENG_META = np.sort(ENG_META)

if len(set(ENG_TUNE) & set(ENG_META)):
    raise RuntimeError("ENG_TUNE / ENG_META overlap.")
if len(ENG_TUNE) != 5_000 or len(ENG_META) != 5_000:
    raise RuntimeError(
        f"Engineering split sizes changed: tune={len(ENG_TUNE)}, meta={len(ENG_META)}"
    )

def keyed_rank(sha, seed):
    return hashlib.sha256(f"{seed}|{sha}".encode()).hexdigest()

def balanced_20k_indices(rank_seed):
    out = []
    for cls in [0,1]:
        q = SSL_META[SSL_META.y.eq(cls)][["sha256"]].copy()
        q["idx"] = q.index.to_numpy()
        q["rank"] = [keyed_rank(x, rank_seed) for x in q.sha256]
        out.append(q.sort_values("rank").idx.to_numpy()[: BUDGET // 2])
    ii = np.sort(np.concatenate(out))
    yy = SSL_META.y.iloc[ii].to_numpy()
    if len(ii) != BUDGET or int((yy == 0).sum()) != 10_000 or int((yy == 1).sum()) != 10_000:
        raise RuntimeError(f"Balanced 20k construction failed for rank seed {rank_seed}")
    return ii

REPLICATES = []
selection_sets = []
for rep, (model_seed, rank_seed) in enumerate(zip(MODEL_SEEDS, LABEL_RANK_SEEDS)):
    ii = balanced_20k_indices(rank_seed)
    sha = SSL_META.sha256.iloc[ii].astype(str).tolist()
    sel_hash = hashlib.sha256("\n".join(sorted(sha)).encode()).hexdigest()
    selection_sets.append(set(sha))
    REPLICATES.append({
        "replicate": rep,
        "model_seed": model_seed,
        "label_rank_seed": rank_seed,
        "rows": len(ii),
        "benign": int((SSL_META.y.iloc[ii] == 0).sum()),
        "phish": int((SSL_META.y.iloc[ii] == 1).sum()),
        "selection_sha256": sel_hash,
        "indices": ii,
    })

rep_plan = pd.DataFrame([
    {k:v for k,v in r.items() if k != "indices"}
    for r in REPLICATES
])
rep_plan.to_csv(AUDIT / "FF4_N10_REPLICATE_PLAN.csv", index=False)

overlap = []
for i, j in itertools.combinations(range(10), 2):
    a, b = selection_sets[i], selection_sets[j]
    overlap.append({
        "rep_i": i,
        "rep_j": j,
        "intersection": len(a & b),
        "union": len(a | b),
        "jaccard": len(a & b) / len(a | b),
    })
pd.DataFrame(overlap).to_csv(
    AUDIT / "FF4_N10_LABEL_SAMPLE_OVERLAP.csv",
    index=False,
)

PRE_FREEZE = {
    "status": "PASS",
    "physical_ssl_rows": len(SSL_META),
    "physical_ssl_contains_label": False,
    "private_labels_joined_by_sha": True,
    "ssl_dev_sha_disjoint": True,
    "dev_rows": len(DEV),
    "eng_tune_rows": len(ENG_TUNE),
    "eng_meta_rows": len(ENG_META),
    "eng_tune_class_counts": {
        "benign": int((DEVY[ENG_TUNE] == 0).sum()),
        "phish": int((DEVY[ENG_TUNE] == 1).sum()),
    },
    "eng_meta_class_counts": {
        "benign": int((DEVY[ENG_META] == 0).sum()),
        "phish": int((DEVY[ENG_META] == 1).sum()),
    },
    "cal_content_accessed": False,
    "final_content_accessed": False,
}
atomic_json(AUDIT / "PRE_FREEZE_DATA_PROTOCOL.json", PRE_FREEZE)

print(rep_plan)
print(json.dumps(PRE_FREEZE, indent=2))

# CELL 5
# 04 — Metrics and exact paired statistics

def thr_fpr(neg_scores, f):
    s = np.asarray(neg_scores, dtype=np.float64)
    if len(s) == 0 or not np.isfinite(s).all():
        raise RuntimeError("Invalid negative scores for thresholding.")
    k = int(math.floor(float(f) * len(s) + 1e-12))
    if k <= 0:
        return float(np.nextafter(s.max(), np.inf))
    ss = np.sort(s)
    return float(np.nextafter(ss[-k], np.inf))

def ci_fpr(fp, n):
    if n == 0:
        return (np.nan, np.nan)
    return (
        0.0 if fp == 0 else float(beta_dist.ppf(.025, fp, n-fp+1)),
        1.0 if fp == n else float(beta_dist.ppf(.975, fp+1, n-fp)),
    )

def op(y, score, th):
    y = np.asarray(y, int)
    s = np.asarray(score, float)
    pred = s >= th

    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    tn = int(((~pred) & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())

    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)
    lo, hi = ci_fpr(fp, fp + tn)

    return {
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "f1": 2 * precision * tpr / max(precision + tpr, 1e-12),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "fpr_ci_lo": lo,
        "fpr_ci_hi": hi,
    }

def precision_at_recall(y, s, target=.90):
    p, r, _ = precision_recall_curve(y, s)
    ok = np.where(r >= target)[0]
    return float(np.max(p[ok])) if len(ok) else np.nan

def curves(y, s):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    fpr, tpr, _ = roc_curve(y, s)

    def fpr_at_tpr(target):
        ii = np.where(tpr >= target)[0]
        return float(fpr[ii[0]]) if len(ii) else np.nan

    return {
        "AP": float(average_precision_score(y, s)),
        "AUC": float(roc_auc_score(y, s)),
        "P_at_R90": precision_at_recall(y, s, .90),
        "FPR_at_TPR90": fpr_at_tpr(.90),
        "FPR_at_TPR95": fpr_at_tpr(.95),
    }

def exact_signflip_p(diff):
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan
    obs = abs(d.mean())
    vals = []
    for signs in itertools.product([-1,1], repeat=len(d)):
        vals.append(abs(np.mean(d * np.asarray(signs))))
    return float(np.mean(np.asarray(vals) >= obs - 1e-15))

def mean_ci(diff):
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return (
            float(d.mean()) if len(d) else np.nan,
            np.nan,
            np.nan,
        )
    m = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    q = float(student_t.ppf(.975, len(d)-1))
    return m, m - q*se, m + q*se

def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p))
    if len(finite) == 0:
        return out

    vals = p[finite]
    order = np.argsort(vals, kind="mergesort")
    m = len(vals)
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0

    for rank, local_pos in enumerate(order):
        candidate = (m - rank) * vals[local_pos]
        running = max(running, candidate)
        adjusted_sorted[rank] = min(1.0, running)

    for rank, local_pos in enumerate(order):
        out[finite[local_pos]] = adjusted_sorted[rank]
    return out

# Unit checks for the statistical helper logic before expensive training.
assert abs(exact_signflip_p(np.ones(10)) - 2/1024) < 1e-12
assert len(holm_adjust([.01,.02,.5])) == 3
print("METRIC/STATISTICS PREFLIGHT: PASS")

# CELL 6
# 05 — DOM graph construction and fixed pretrained encoder compatibility

DOM_VOCAB = json.loads(DOM_VOCAB_PATH.read_text())
if not isinstance(DOM_VOCAB, dict):
    raise RuntimeError("DOM_VOCAB.json must contain a dictionary.")
PAD, UNK, MASK = 0, 1, 2

def bucket(v):
    try:
        return min(max(int(v), 0), 31)
    except Exception:
        return 0

def _rv(r, k):
    if isinstance(r, dict):
        return r.get(k)
    try:
        return r[k]
    except Exception:
        return getattr(r, k, None)

def graph_row(r):
    def arr(k):
        x = _rv(r, k)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, list):
            return x
        try:
            return list(x)
        except Exception:
            return []

    tags = arr("dom_tag")
    par = arr("dom_parent_idx")
    dep = arr("dom_depth")
    att = arr("dom_attr_count")
    chi = arr("dom_child_count")

    n = min(len(tags), DOM_MAX_NODES)
    if n == 0:
        return (
            np.array([PAD], np.int64),
            np.array([0], np.int64),
            np.array([0], np.int64),
            np.array([0], np.int64),
            np.zeros((0,2), np.int64),
        )

    tag = np.array([DOM_VOCAB.get(str(x), UNK) for x in tags[:n]], np.int64)
    depth = np.array([bucket(x) for x in dep[:n]], np.int64)
    attr = np.array([bucket(x) for x in att[:n]], np.int64)
    child = np.array([bucket(x) for x in chi[:n]], np.int64)

    edges = []
    for ii, pp in enumerate(par[:n]):
        try:
            pp = int(pp)
        except Exception:
            continue
        if ii > 0 and 0 <= pp < n:
            edges.extend([(pp,ii), (ii,pp)])

    return tag, depth, attr, child, np.asarray(edges, np.int64)

def graph_batch(rows):
    T, D, A, C, E, B = [], [], [], [], [], []
    off = 0
    row_iter = rows.itertuples(index=False) if isinstance(rows, pd.DataFrame) else rows
    n_graphs = len(rows)

    for bi, r in enumerate(row_iter):
        t, d, a, c, e = graph_row(r)
        n = len(t)
        T.append(t); D.append(d); A.append(a); C.append(c)
        B.append(np.full(n, bi, np.int64))
        if len(e):
            E.append(e + off)
        off += n

    def cat(xs):
        return torch.tensor(
            np.concatenate(xs),
            dtype=torch.long,
            device=DEVICE,
        )

    edge = torch.tensor(
        np.concatenate(E, 0).T if E else np.zeros((2,0), np.int64),
        dtype=torch.long,
        device=DEVICE,
    )

    return {
        "tag": cat(T),
        "depth": cat(D),
        "attr": cat(A),
        "child": cat(C),
        "edge": edge,
        "batch": cat(B),
        "n_graphs": n_graphs,
    }

class GCNLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.s = nn.Linear(d,d)
        self.n = nn.Linear(d,d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, e):
        if e.numel() == 0:
            return self.norm(x + F.gelu(self.s(x)))
        src, dst = e
        agg = torch.zeros_like(x)
        deg = torch.zeros((len(x),1), device=x.device, dtype=x.dtype)
        agg.index_add_(0, dst, x[src])
        deg.index_add_(
            0, dst,
            torch.ones((len(dst),1), device=x.device, dtype=x.dtype),
        )
        return self.norm(
            x + F.gelu(self.s(x) + self.n(agg / deg.clamp_min(1)))
        )

class DOMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tag = nn.Embedding(len(DOM_VOCAB), 96, padding_idx=PAD)
        self.depth = nn.Embedding(32,16)
        self.attr = nn.Embedding(32,8)
        self.child = nn.Embedding(32,8)
        self.inp = nn.Linear(128, DOM_DIM)
        self.layers = nn.ModuleList([GCNLayer(DOM_DIM) for _ in range(DOM_LAYERS)])
        self.att = nn.Linear(DOM_DIM,1)

    def nodes(self, g):
        x = self.inp(torch.cat([
            self.tag(g["tag"]),
            self.depth(g["depth"]),
            self.attr(g["attr"]),
            self.child(g["child"]),
        ], 1))
        for layer in self.layers:
            x = layer(x, g["edge"])
        return x

    def forward(self, g):
        x = self.nodes(g)
        n = g["n_graphs"]
        num = torch.zeros((n,DOM_DIM), device=x.device)
        den = torch.zeros((n,1), device=x.device)
        w = torch.exp(torch.clamp(self.att(x).squeeze(-1), -10, 10))
        num.index_add_(0, g["batch"], x * w[:,None])
        den.index_add_(0, g["batch"], w[:,None])
        return num / den.clamp_min(1e-8)

# Hard compatibility check before any downstream fit.
_dom = DOMEncoder()
_dom.load_state_dict(
    torch.load(DOM_ENCODER_PATH, map_location="cpu"),
    strict=True,
)
del _dom
gc.collect()
print({"DOM_CHECKPOINT_COMPATIBILITY":"PASS", "vocab_size":len(DOM_VOCAB)})

# CELL 7
# 06 — Sequential FF4 model: separately trained URL, DUAL-Equal, TRI-Equal and TRI-Gated

def freeze_last(model, n):
    for p in model.parameters():
        p.requires_grad = False

    layers = getattr(getattr(model, "encoder", None), "layer", None)
    if layers is None:
        layers = getattr(getattr(model, "transformer", None), "layer", None)
    if layers is None:
        raise RuntimeError(f"Unsupported transformer structure: {type(model)}")

    for layer in layers[-n:]:
        for p in layer.parameters():
            p.requires_grad = True

    if getattr(model, "pooler", None) is not None:
        for p in model.pooler.parameters():
            p.requires_grad = True

def masked_mean(h, m):
    mm = m.unsqueeze(-1).to(h.dtype)
    return (h * mm).sum(1) / mm.sum(1).clamp_min(1)

def pooled(out, mask):
    return masked_mean(out.last_hidden_state, mask)

class FF4SequentialModel(nn.Module):
    def __init__(self, variant):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)

        cfg = VARIANTS[variant]
        self.variant = variant
        self.use_text = bool(cfg["use_text"])
        self.use_dom = bool(cfg["use_dom"])
        self.fusion = str(cfg["fusion"])

        # Existing fixed RUT transformer checkpoints.
        self.url_encoder = AutoModel.from_pretrained(
            URL_DAPT_SRC,
            local_files_only=True,
        )
        freeze_last(self.url_encoder, DEEP_LAST_N)
        self.url_proj = nn.Linear(
            int(self.url_encoder.config.hidden_size),
            PROJ_DIM,
        )

        if self.use_text:
            self.text_encoder = AutoModel.from_pretrained(
                TEXT_DAPT_SRC,
                local_files_only=True,
            )
            freeze_last(self.text_encoder, DEEP_LAST_N)
            self.text_proj = nn.Linear(
                int(self.text_encoder.config.hidden_size),
                PROJ_DIM,
            )
        else:
            self.text_encoder = None
            self.text_proj = None

        if self.use_dom:
            self.dom_encoder = DOMEncoder()
            self.dom_encoder.load_state_dict(
                torch.load(DOM_ENCODER_PATH, map_location="cpu"),
                strict=True,
            )
            self.dom_proj = nn.Linear(DOM_DIM, PROJ_DIM)
        else:
            self.dom_encoder = None
            self.dom_proj = None

        # Only P3 has a trainable gate. Equal variants have no dormant gate parameters.
        if self.fusion == "gated":
            self.gate = nn.Sequential(
                nn.Linear(PROJ_DIM,64),
                nn.GELU(),
                nn.Linear(64,1),
            )
        else:
            self.gate = None

        self.head = nn.Sequential(
            nn.Linear(int(cfg["head_input_dim"]), 512),
            nn.GELU(),
            nn.Dropout(.2),
            nn.Linear(512,128),
            nn.GELU(),
            nn.Dropout(.1),
            nn.Linear(128,1),
        )

        aux = {"url": nn.Linear(PROJ_DIM,1)}
        if self.use_text:
            aux["text"] = nn.Linear(PROJ_DIM,1)
        if self.use_dom:
            aux["dom"] = nn.Linear(PROJ_DIM,1)
        self.aux_heads = nn.ModuleDict(aux)

    def encode(self, u, t=None, g=None):
        zu = F.normalize(
            self.url_proj(
                pooled(
                    self.url_encoder(**u),
                    u["attention_mask"],
                )
            ),
            dim=-1,
        )
        zs = [zu]
        names = ["url"]

        if self.use_text:
            zt = F.normalize(
                self.text_proj(
                    pooled(
                        self.text_encoder(**t),
                        t["attention_mask"],
                    )
                ),
                dim=-1,
            )
            zs.append(zt)
            names.append("text")

        if self.use_dom:
            zd = F.normalize(
                self.dom_proj(self.dom_encoder(g)),
                dim=-1,
            )
            zs.append(zd)
            names.append("dom")

        return names, zs

    def forward(self, u, t=None, g=None):
        names, zs = self.encode(u, t, g)

        gate_weights = None
        if self.fusion == "none":
            features = zs
        else:
            st = torch.stack(zs, 1)
            if self.fusion == "equal":
                fused = st.mean(1)
            elif self.fusion == "gated":
                gate_weights = torch.softmax(
                    self.gate(st).squeeze(-1),
                    1,
                )
                fused = (st * gate_weights[:,:,None]).sum(1)
            else:
                raise RuntimeError(self.fusion)
            features = zs + [fused]

        main = self.head(torch.cat(features, 1)).squeeze(-1)
        aux = {
            name: self.aux_heads[name](z).squeeze(-1)
            for name, z in zip(names, zs)
        }
        return main, aux, gate_weights

def _wb(logits, y):
    return F.binary_cross_entropy_with_logits(logits, y)

def deep_loss(main, aux, y):
    aux_loss = torch.stack([_wb(a, y) for a in aux.values()]).mean()
    return _wb(main, y) + AUX_TOTAL_WEIGHT * aux_loss

def parameter_audit(variant):
    seed_all(123456)
    m = FF4SequentialModel(variant)
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    row = {
        "variant": variant,
        "modalities": "+".join(VARIANTS[variant]["modalities"]),
        "fusion": VARIANTS[variant]["fusion"],
        "head_input_dim": VARIANTS[variant]["head_input_dim"],
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
    }
    del m
    gc.collect()
    return row

MODEL_AUDIT = pd.DataFrame([parameter_audit(v) for v in VARIANTS])
MODEL_AUDIT.to_csv(AUDIT / "FF4_MODEL_PARAMETER_AUDIT.csv", index=False)
print(MODEL_AUDIT)

# CELL 8
# 07 — Variant-aware SSL streaming, tokenization and scoring

def tok_url_values(values):
    b = url_tok(
        values,
        padding=True,
        truncation=True,
        max_length=URL_MAX_LEN,
        return_tensors="pt",
    )
    return {k:v.to(DEVICE, non_blocking=True) for k,v in b.items()}

def tok_text_values(values):
    b = text_tok(
        values,
        padding=True,
        truncation=True,
        max_length=TEXT_MAX_LEN,
        return_tensors="pt",
    )
    return {k:v.to(DEVICE, non_blocking=True) for k,v in b.items()}

class SelectedSSLIter(IterableDataset):
    def __init__(self, allowed_sha, seed, variant):
        self.allowed_sha = set(allowed_sha)
        self.seed = int(seed)
        self.variant = variant
        self.files = sorted(ROLE_DIR["SSL"].glob("*.parquet"))

    def __iter__(self):
        cfg = VARIANTS[self.variant]
        rng = np.random.default_rng(self.seed)
        files = list(self.files)
        rng.shuffle(files)

        yielded = 0
        for p in files:
            cols = ["sha256","url"]
            if cfg["use_text"]:
                cols.append("text")
            if cfg["use_dom"]:
                cols += DOM_COLS

            q = pd.read_parquet(p, columns=cols)
            q["sha256"] = q.sha256.astype(str).str.lower()
            q = q[q.sha256.isin(self.allowed_sha)]

            ii = np.arange(len(q))
            rng.shuffle(ii)

            for j in ii:
                r = q.iloc[j]
                sha = str(r.sha256)
                y = label_map.get(sha)
                if y is None:
                    raise RuntimeError(f"Missing SSL label for SHA {sha}")
                yielded += 1
                yield r, int(y)

        if yielded != BUDGET:
            raise RuntimeError(
                f"SelectedSSLIter yielded {yielded}, expected {BUDGET} "
                f"for variant={self.variant}, seed={self.seed}"
            )

def deep_collate(rows, variant):
    cfg = VARIANTS[variant]
    rs = [x[0] for x in rows]
    y = torch.tensor(
        [x[1] for x in rows],
        dtype=torch.float32,
        device=DEVICE,
    )

    u = tok_url_values([
        str(r.url) if pd.notna(r.url) else ""
        for r in rs
    ])

    t = None
    if cfg["use_text"]:
        t = tok_text_values([
            str(r.text) if pd.notna(r.text) else ""
            for r in rs
        ])

    g = graph_batch(rs) if cfg["use_dom"] else None
    return u, t, g, y

@torch.no_grad()
def score_rows(model, df, variant, batch):
    cfg = VARIANTS[variant]
    out = np.empty(len(df), dtype=np.float32)
    gate_sum = None
    gate_n = 0

    for st in range(0, len(df), batch):
        en = min(st + batch, len(df))
        rows = df.iloc[st:en]

        u = tok_url_values(
            rows.url.fillna("").astype(str).tolist()
        )
        t = None
        if cfg["use_text"]:
            t = tok_text_values(
                rows.text.fillna("").astype(str).tolist()
            )
        g = graph_batch(rows) if cfg["use_dom"] else None

        with amp_ctx():
            main, _, gw = model(u, t, g)

        out[st:en] = main.float().cpu().numpy()

        if gw is not None:
            x = gw.float().cpu().numpy()
            gate_sum = x.sum(0) if gate_sum is None else gate_sum + x.sum(0)
            gate_n += len(x)

    gate_means = None
    if gate_sum is not None:
        m = gate_sum / max(gate_n, 1)
        gate_means = {
            "gate_url_mean": float(m[0]),
            "gate_text_mean": float(m[1]),
            "gate_dom_mean": float(m[2]),
        }

    return out, gate_means

def score_adaptive(model, df, variant):
    last = None
    for bs in SCORE_BATCH_CANDIDATES:
        try:
            s, gates = score_rows(model, df, variant, bs)
            return s, gates, bs
        except torch.cuda.OutOfMemoryError as e:
            last = e
            gc.collect()
            torch.cuda.empty_cache()
            print({"SCORE_OOM":variant, "batch":bs})
    raise last

print("DATASET/COLLATE/SCORING DEFINITIONS: PASS")

# CELL 9
# 08 — One condition = train separately, tune threshold on ENG_TUNE, evaluate on ENG_META, then delete model state

def condition_paths(rep, variant):
    d = RUNS / f"rep{rep:02d}" / variant
    d.mkdir(parents=True, exist_ok=True)
    return {
        "dir": d,
        "result": d / "FF4_N10_CONDITION_RESULT.json",
        "state": d / "model_state.pt",
        "train_meta": d / "TRAINING_META.json",
    }

def completed_condition(rep, variant):
    p = condition_paths(rep, variant)["result"]
    if not p.exists():
        return None
    obj = json.loads(p.read_text())
    if obj.get("status") != "COMPLETE" or obj.get("version") != VERSION:
        return None
    return obj

def train_and_score_condition(rep_obj, variant):
    rep = int(rep_obj["replicate"])
    model_seed = int(rep_obj["model_seed"])
    rank_seed = int(rep_obj["label_rank_seed"])
    selection_sha256 = str(rep_obj["selection_sha256"])
    allowed_idx = np.asarray(rep_obj["indices"], dtype=int)
    allowed_sha = SSL_META.sha256.iloc[allowed_idx].astype(str).tolist()

    prev = completed_condition(rep, variant)
    if prev is not None:
        # Protect against accidental mismatch when resuming.
        expected = {
            "model_seed": model_seed,
            "label_rank_seed": rank_seed,
            "selection_sha256": selection_sha256,
        }
        if any(str(prev.get(k)) != str(v) for k,v in expected.items()):
            raise RuntimeError(
                f"Resume mismatch rep={rep} variant={variant}: {expected} vs previous result"
            )
        print({"SKIP_COMPLETE": f"rep{rep:02d}/{variant}"})
        return prev

    p = condition_paths(rep, variant)

    seed_all(model_seed)
    gc.collect()
    torch.cuda.empty_cache()

    model = FF4SequentialModel(variant).to(DEVICE)

    enc_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("url_encoder.") or name.startswith("text_encoder."):
            enc_params.append(param)
        else:
            head_params.append(param)

    if not enc_params or not head_params:
        raise RuntimeError(
            f"Parameter grouping failed for {variant}: enc={len(enc_params)}, head={len(head_params)}"
        )

    optimizer = torch.optim.AdamW([
        {
            "params": enc_params,
            "lr": ENCODER_LR,
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": head_params,
            "lr": HEAD_LR,
            "weight_decay": WEIGHT_DECAY,
        },
    ])

    steps = math.ceil(BUDGET / ABLATION_BATCH) * DEEP_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(WARMUP_RATIO * steps),
        steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)

    ds = SelectedSSLIter(allowed_sha, model_seed, variant)
    dl = DataLoader(
        ds,
        batch_size=ABLATION_BATCH,
        collate_fn=lambda x: deep_collate(x, variant),
        num_workers=0,
    )

    losses = []
    seen = 0
    t0 = time.time()
    model.train()

    for u, t, g, y in dl:
        if seen >= BUDGET:
            break

        optimizer.zero_grad(set_to_none=True)
        with amp_ctx():
            main, aux, _ = model(u, t, g)
            loss = deep_loss(main, aux, y)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss rep={rep}, variant={variant}, seen={seen}"
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        seen += len(y)
        losses.append(float(loss.detach().cpu()))

        if len(losses) % 250 == 0:
            print({
                "TRAIN": f"rep{rep:02d}/{variant}",
                "seen": seen,
                "of": BUDGET,
                "loss100": round(float(np.mean(losses[-100:])), 5),
                "elapsed_min": round((time.time()-t0)/60, 2),
            })

    if seen != BUDGET:
        raise RuntimeError(
            f"Training row count mismatch rep={rep}, variant={variant}: {seen}"
        )

    train_minutes = float((time.time() - t0) / 60)

    # Save one temporary state before evaluation. It is deleted after the condition result is durable.
    atomic_torch(p["state"], model.state_dict())
    atomic_json(p["train_meta"], {
        "status": "TRAINED_AWAITING_OR_FINISHED_SCORING",
        "version": VERSION,
        "replicate": rep,
        "variant": variant,
        "model_seed": model_seed,
        "label_rank_seed": rank_seed,
        "selection_sha256": selection_sha256,
        "budget": BUDGET,
        "batch_size": ABLATION_BATCH,
        "optimizer_steps": len(losses),
        "mean_train_loss": float(np.mean(losses)),
        "last100_train_loss": float(np.mean(losses[-100:])),
        "train_minutes": train_minutes,
    })

    # DEV-only scoring, with disjoint threshold-tune and meta-evaluation subsets.
    model.eval()
    tune_df = DEV.iloc[ENG_TUNE].reset_index(drop=True)
    meta_df = DEV.iloc[ENG_META].reset_index(drop=True)
    y_tune = DEVY[ENG_TUNE]
    y_meta = DEVY[ENG_META]

    tune_scores, _, tune_bs = score_adaptive(model, tune_df, variant)
    meta_scores, gate_means, meta_bs = score_adaptive(model, meta_df, variant)

    threshold = thr_fpr(
        tune_scores[y_tune == 0],
        PRIMARY_FPR,
    )

    row = {
        "status": "COMPLETE",
        "version": VERSION,
        "scientific_status": SCIENTIFIC_STATUS,
        "replicate": rep,
        "model_seed": model_seed,
        "label_rank_seed": rank_seed,
        "selection_sha256": selection_sha256,
        "variant": variant,
        "modalities": "+".join(VARIANTS[variant]["modalities"]),
        "fusion": VARIANTS[variant]["fusion"],
        "budget": BUDGET,
        "train_batch": ABLATION_BATCH,
        "train_steps": len(losses),
        "train_minutes": train_minutes,
        "mean_train_loss": float(np.mean(losses)),
        "last100_train_loss": float(np.mean(losses[-100:])),
        "threshold_source": "DEV_ENG_TUNE_NEGATIVES",
        "target_fpr": PRIMARY_FPR,
        "threshold": float(threshold),
        "tune_score_batch": int(tune_bs),
        "meta_score_batch": int(meta_bs),
        **op(y_meta, meta_scores, threshold),
        **curves(y_meta, meta_scores),
        "gate_url_mean": (
            np.nan if gate_means is None else gate_means["gate_url_mean"]
        ),
        "gate_text_mean": (
            np.nan if gate_means is None else gate_means["gate_text_mean"]
        ),
        "gate_dom_mean": (
            np.nan if gate_means is None else gate_means["gate_dom_mean"]
        ),
        "cal_accessed": False,
        "final_accessed": False,
    }

    atomic_json(p["result"], row)

    # Completed conditions retain only small audit/results, not ~GB model states.
    p["state"].unlink(missing_ok=True)

    del model, ds, dl, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    print({
        "COMPLETE": f"rep{rep:02d}/{variant}",
        "TPR_pct": round(100*row["tpr"], 4),
        "FPR_pct": round(100*row["fpr"], 4),
        "AP": round(row["AP"], 6),
        "train_min": round(train_minutes, 2),
    })
    return row

print("CONDITION RUNNER: PASS")

# CELL 10
# 09 — Execute the locked N10 × 4 sequential architecture matrix = exactly 40 fits

all_rows = []

for rep_obj in REPLICATES:
    rep = int(rep_obj["replicate"])
    print("\n" + "="*90)
    print({
        "REPLICATE": rep,
        "model_seed": rep_obj["model_seed"],
        "label_rank_seed": rep_obj["label_rank_seed"],
        "selection_sha256": rep_obj["selection_sha256"],
    })

    # Fixed order is part of the plan; results are reported regardless of direction.
    for variant in VARIANTS:
        row = train_and_score_condition(rep_obj, variant)
        all_rows.append(row)

RUNS_DF = pd.DataFrame(all_rows).sort_values(
    ["replicate","variant"]
).reset_index(drop=True)

expected_pairs = {
    (rep, variant)
    for rep in range(10)
    for variant in VARIANTS
}
got_pairs = set(zip(RUNS_DF.replicate.astype(int), RUNS_DF.variant.astype(str)))

if got_pairs != expected_pairs or len(RUNS_DF) != 40:
    missing = sorted(expected_pairs - got_pairs)
    extra = sorted(got_pairs - expected_pairs)
    raise RuntimeError(
        f"40-condition matrix incomplete. missing={missing[:5]}, extra={extra[:5]}"
    )

if RUNS_DF[["replicate","variant"]].duplicated().any():
    raise RuntimeError("Duplicate condition rows detected.")

RUNS_DF.to_csv(RESULTS / "FF4_N10_ALL_RUNS.csv", index=False)
print(RUNS_DF[[
    "replicate","variant","tpr","fpr","AP","AUC","FPR_at_TPR90","train_minutes"
]])
print("40-CONDITION MATRIX: COMPLETE")

# CELL 11
# 10 — Aggregate variants and compute the 3 pre-specified exact paired primary contrasts

def agg_metric(df, metric):
    x = df[metric].to_numpy(float)
    return {
        f"{metric}_mean": float(np.mean(x)),
        f"{metric}_sd": float(np.std(x, ddof=1)),
        f"{metric}_min": float(np.min(x)),
        f"{metric}_max": float(np.max(x)),
    }

agg_rows = []
for variant in VARIANTS:
    q = RUNS_DF[RUNS_DF.variant.eq(variant)].copy()
    row = {
        "variant": variant,
        "n": len(q),
        "modalities": "+".join(VARIANTS[variant]["modalities"]),
        "fusion": VARIANTS[variant]["fusion"],
    }
    for metric in ["tpr","fpr","AP","AUC","P_at_R90","FPR_at_TPR90","train_minutes"]:
        row.update(agg_metric(q, metric))
    for g in ["gate_url_mean","gate_text_mean","gate_dom_mean"]:
        vals = q[g].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        row[f"{g}_mean"] = float(vals.mean()) if len(vals) else np.nan
    agg_rows.append(row)

VARIANT_AGG = pd.DataFrame(agg_rows)
VARIANT_AGG.to_csv(TABLES / "TABLE_FF4_N10_VARIANT_AGG.csv", index=False)

contrast_rows = []
paired_detail = []

for name, baseline, candidate in PRIMARY_CONTRASTS:
    a = RUNS_DF[RUNS_DF.variant.eq(baseline)].sort_values("replicate")
    b = RUNS_DF[RUNS_DF.variant.eq(candidate)].sort_values("replicate")

    if not np.array_equal(a.replicate.to_numpy(), b.replicate.to_numpy()):
        raise RuntimeError(f"Pairing mismatch for {name}")
    if not np.array_equal(a.model_seed.to_numpy(), b.model_seed.to_numpy()):
        raise RuntimeError(f"Model-seed pairing mismatch for {name}")
    if not np.array_equal(a.label_rank_seed.to_numpy(), b.label_rank_seed.to_numpy()):
        raise RuntimeError(f"Label-rank pairing mismatch for {name}")
    if not np.array_equal(a.selection_sha256.to_numpy(), b.selection_sha256.to_numpy()):
        raise RuntimeError(f"Labelset pairing mismatch for {name}")

    d_tpr = b.tpr.to_numpy(float) - a.tpr.to_numpy(float)
    d_fpr = b.fpr.to_numpy(float) - a.fpr.to_numpy(float)
    d_ap = b.AP.to_numpy(float) - a.AP.to_numpy(float)
    d_auc = b.AUC.to_numpy(float) - a.AUC.to_numpy(float)
    d_fpr90 = b.FPR_at_TPR90.to_numpy(float) - a.FPR_at_TPR90.to_numpy(float)

    m, lo, hi = mean_ci(d_tpr)

    contrast_rows.append({
        "contrast": name,
        "baseline": baseline,
        "candidate": candidate,
        "n_pairs": len(d_tpr),
        "mean_delta_tpr": m,
        "mean_delta_tpr_pp": 100*m,
        "ci95_delta_tpr_lo": lo,
        "ci95_delta_tpr_hi": hi,
        "ci95_delta_tpr_lo_pp": 100*lo,
        "ci95_delta_tpr_hi_pp": 100*hi,
        "positive_tpr_pairs": int((d_tpr > 0).sum()),
        "zero_tpr_pairs": int((d_tpr == 0).sum()),
        "negative_tpr_pairs": int((d_tpr < 0).sum()),
        "exact_two_sided_signflip_p": exact_signflip_p(d_tpr),
        "mean_delta_fpr": float(np.mean(d_fpr)),
        "mean_delta_fpr_pp": 100*float(np.mean(d_fpr)),
        "mean_delta_AP": float(np.mean(d_ap)),
        "mean_delta_AUC": float(np.mean(d_auc)),
        "mean_delta_FPR_at_TPR90": float(np.mean(d_fpr90)),
    })

    for rep, dt, dfp, dap, dauc, df90 in zip(
        a.replicate.astype(int),
        d_tpr, d_fpr, d_ap, d_auc, d_fpr90,
    ):
        paired_detail.append({
            "contrast": name,
            "replicate": int(rep),
            "model_seed": int(a[a.replicate.eq(rep)].iloc[0].model_seed),
            "label_rank_seed": int(a[a.replicate.eq(rep)].iloc[0].label_rank_seed),
            "delta_tpr": float(dt),
            "delta_tpr_pp": 100*float(dt),
            "delta_fpr": float(dfp),
            "delta_fpr_pp": 100*float(dfp),
            "delta_AP": float(dap),
            "delta_AUC": float(dauc),
            "delta_FPR_at_TPR90": float(df90),
        })

CONTRASTS = pd.DataFrame(contrast_rows)
CONTRASTS["holm_p"] = holm_adjust(
    CONTRASTS["exact_two_sided_signflip_p"].to_numpy(float)
)
CONTRASTS["holm_significant_0p05"] = CONTRASTS.holm_p < .05
CONTRASTS["mean_tpr_direction_positive"] = CONTRASTS.mean_delta_tpr > 0

PAIRED_DETAIL = pd.DataFrame(paired_detail)

CONTRASTS.to_csv(
    TABLES / "TABLE_FF4_N10_ADJACENT_PRIMARY_CONTRASTS.csv",
    index=False,
)
PAIRED_DETAIL.to_csv(
    RESULTS / "FF4_N10_PAIRED_CONTRAST_DETAIL.csv",
    index=False,
)

print("\nVARIANT AGGREGATES")
print(VARIANT_AGG[[
    "variant","tpr_mean","tpr_sd","fpr_mean","AP_mean","FPR_at_TPR90_mean"
]])

print("\nPRIMARY ADJACENT CONTRASTS")
print(CONTRASTS[[
    "contrast","baseline","candidate",
    "mean_delta_tpr_pp","ci95_delta_tpr_lo_pp","ci95_delta_tpr_hi_pp",
    "positive_tpr_pairs","exact_two_sided_signflip_p","holm_p",
    "mean_delta_fpr_pp","mean_delta_AP","mean_delta_FPR_at_TPR90",
]])

# CELL 12
# 11 — Gate audit and scientific decision table (without rewriting the historical architecture choice)

GATE_RUNS = RUNS_DF[
    RUNS_DF.variant.eq("P3_TRI_GATED")
][[
    "replicate","model_seed","label_rank_seed",
    "gate_url_mean","gate_text_mean","gate_dom_mean",
]].copy()
GATE_RUNS.to_csv(TABLES / "TABLE_FF4_N10_GATE_WEIGHTS.csv", index=False)

gate_summary = {
    "url_mean": float(GATE_RUNS.gate_url_mean.mean()),
    "text_mean": float(GATE_RUNS.gate_text_mean.mean()),
    "dom_mean": float(GATE_RUNS.gate_dom_mean.mean()),
}

decision_rows = []
for r in CONTRASTS.itertuples(index=False):
    decision_rows.append({
        "component_step": r.contrast,
        "baseline": r.baseline,
        "candidate": r.candidate,
        "mean_delta_tpr_pp": float(r.mean_delta_tpr_pp),
        "ci95_lo_pp": float(r.ci95_delta_tpr_lo_pp),
        "ci95_hi_pp": float(r.ci95_delta_tpr_hi_pp),
        "exact_p": float(r.exact_two_sided_signflip_p),
        "holm_p": float(r.holm_p),
        "holm_significant_0p05": bool(r.holm_significant_0p05),
        "mean_direction_positive": bool(r.mean_tpr_direction_positive),
        "interpretation_rule": (
            "Component receives confirmatory closure support only if "
            "the mean TPR direction is positive and Holm-adjusted p < 0.05. "
            "Otherwise report effect/uncertainty without claiming confirmed improvement."
        ),
    })

DECISION = pd.DataFrame(decision_rows)
DECISION.to_csv(
    TABLES / "TABLE_FF4_N10_COMPONENT_DECISIONS.csv",
    index=False,
)

all_three_supported = bool(
    (DECISION.mean_direction_positive & DECISION.holm_significant_0p05).all()
)

print({
    "gate_weight_means": gate_summary,
    "all_three_adjacent_steps_confirmed": all_three_supported,
    "note": "This does not retroactively change the historical architecture freeze."
})
print(DECISION)

# CELL 13
# 12 — Reproducibility audit, machine-readable completion marker, Markdown summary and result ZIP

# Hard completeness/integrity assertions.
if len(RUNS_DF) != 40:
    raise RuntimeError("Expected exactly 40 completed condition rows.")
if RUNS_DF.cal_accessed.astype(bool).any():
    raise RuntimeError("Unexpected CAL access flag.")
if RUNS_DF.final_accessed.astype(bool).any():
    raise RuntimeError("Unexpected FINAL access flag.")
if len(CONTRASTS) != 3:
    raise RuntimeError("Expected exactly three primary contrasts.")
if not (AUDIT / "FF4_N10_PROTOCOL_LOCK.json").exists():
    raise RuntimeError("Protocol lock missing.")
if any((RUNS.rglob("model_state.pt"))):
    raise RuntimeError("Completed run still contains large temporary model state(s).")

total_train_min = float(RUNS_DF.train_minutes.sum())

headline = {}
for r in CONTRASTS.itertuples(index=False):
    headline[r.contrast] = {
        "baseline": r.baseline,
        "candidate": r.candidate,
        "mean_delta_tpr_pp": float(r.mean_delta_tpr_pp),
        "ci95_pp": [
            float(r.ci95_delta_tpr_lo_pp),
            float(r.ci95_delta_tpr_hi_pp),
        ],
        "positive_pairs": int(r.positive_tpr_pairs),
        "exact_two_sided_p": float(r.exact_two_sided_signflip_p),
        "holm_p": float(r.holm_p),
        "holm_significant_0p05": bool(r.holm_significant_0p05),
        "mean_delta_fpr_pp": float(r.mean_delta_fpr_pp),
    }

COMPLETE = {
    "status": "COMPLETE",
    "version": VERSION,
    "completed_utc": pd.Timestamp.utcnow().isoformat(),
    "scientific_status": SCIENTIFIC_STATUS,
    "historical_final_known": True,
    "used_for_retroactive_model_selection": False,
    "planned_fits": 40,
    "completed_fits": 40,
    "n_replicates": 10,
    "budget_per_fit": BUDGET,
    "representation": "RUT",
    "variants": list(VARIANTS),
    "primary_contrasts": headline,
    "all_three_adjacent_steps_confirmed": bool(
        (DECISION.mean_direction_positive & DECISION.holm_significant_0p05).all()
    ),
    "gate_weight_means": gate_summary,
    "total_training_gpu_minutes_sum": total_train_min,
    "cal_accessed": False,
    "final_accessed": False,
    "cascade_recomputed": False,
    "outputs": {
        "runs": "results/FF4_N10_ALL_RUNS.csv",
        "paired_detail": "results/FF4_N10_PAIRED_CONTRAST_DETAIL.csv",
        "variant_aggregates": "tables/TABLE_FF4_N10_VARIANT_AGG.csv",
        "primary_contrasts": "tables/TABLE_FF4_N10_ADJACENT_PRIMARY_CONTRASTS.csv",
        "component_decisions": "tables/TABLE_FF4_N10_COMPONENT_DECISIONS.csv",
        "gate_weights": "tables/TABLE_FF4_N10_GATE_WEIGHTS.csv",
        "protocol": "audit/FF4_N10_PROTOCOL_LOCK.json",
        "input_resolution": "audit/INPUT_RESOLUTION.json",
        "replicate_plan": "audit/FF4_N10_REPLICATE_PLAN.csv",
        "label_overlap": "audit/FF4_N10_LABEL_SAMPLE_OVERLAP.csv",
    },
}
atomic_json(
    ROOT / "FINAL_FF4_SEQUENTIAL_ARCHITECTURE_ABLATION_N10_COMPLETE.json",
    COMPLETE,
)

# Human-readable compact transfer summary.
def pct(x, digits=3):
    return f"{100*x:.{digits}f}%"

lines = []
lines.append("# FINAL FF4 Sequential Architecture Ablation N10 — Summary")
lines.append("")
lines.append(f"- Status: **{COMPLETE['status']}**")
lines.append(f"- Version: `{VERSION}`")
lines.append(f"- Scientific status: `{SCIENTIFIC_STATUS}`")
lines.append("- CAL/FINAL accessed: **No / No**")
lines.append("- Cascade recomputed: **No**")
lines.append("- Representation: **RUT**, fixed URL-/Text-DAPT checkpoints; fixed DOM-SSL encoder.")
lines.append("- Training: 20,000 balanced SSL labels per replicate, N10 paired, 1 epoch, fixed batch 8.")
lines.append("- DEV protocol: threshold on ENG_TUNE negatives at target FPR 0.5%; endpoint on disjoint ENG_META.")
lines.append("")
lines.append("## Variant aggregates")
lines.append("")
lines.append("| Variant | Mean TPR | Mean FPR | Mean AP | Mean FPR@TPR90 |")
lines.append("|---|---:|---:|---:|---:|")
for r in VARIANT_AGG.itertuples(index=False):
    lines.append(
        f"| {r.variant} | {pct(r.tpr_mean)} | {pct(r.fpr_mean)} | "
        f"{r.AP_mean:.6f} | {pct(r.FPR_at_TPR90_mean)} |"
    )

lines.append("")
lines.append("## Pre-specified adjacent primary contrasts")
lines.append("")
lines.append("| Contrast | ΔTPR mean | 95% CI | Positive pairs | exact p | Holm p | ΔFPR mean |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")
for r in CONTRASTS.itertuples(index=False):
    lines.append(
        f"| {r.contrast}: {r.candidate} − {r.baseline} | "
        f"{r.mean_delta_tpr_pp:+.3f} pp | "
        f"[{r.ci95_delta_tpr_lo_pp:+.3f}; {r.ci95_delta_tpr_hi_pp:+.3f}] pp | "
        f"{int(r.positive_tpr_pairs)}/10 | "
        f"{r.exact_two_sided_signflip_p:.8f} | "
        f"{r.holm_p:.8f} | "
        f"{r.mean_delta_fpr_pp:+.3f} pp |"
    )

lines.append("")
lines.append("## Interpretation boundary")
lines.append("")
lines.append(
    "This is a post-hoc FF4 closure analysis after the final architecture was already known. "
    "The plan and fresh seeds were fixed before this execution. The results support or fail to support "
    "the incremental component claims; they are not presented as a retroactive architecture-selection procedure."
)
lines.append(
    "The separate frozen Full-TRI → URL-first Cascade evaluation remains the evidence for the final inference-resource trade-off."
)

summary_path = ROOT / "FINAL_FF4_SEQUENTIAL_ARCHITECTURE_ABLATION_N10_summary.md"
summary_path.write_text("\n".join(lines), encoding="utf-8")

# Remove any transient .pt before packaging.
for p in ROOT.rglob("*.pt"):
    p.unlink(missing_ok=True)

zip_base = Path("/kaggle/working/FINAL_FF4_SEQUENTIAL_ARCHITECTURE_ABLATION_N10_RESULTS_v1")
if not Path("/kaggle/working").exists():
    zip_base = Path("/mnt/data/FINAL_FF4_SEQUENTIAL_ARCHITECTURE_ABLATION_N10_RESULTS_v1")

zip_path = Path(shutil.make_archive(
    str(zip_base),
    "zip",
    root_dir=ROOT,
))

print(json.dumps(COMPLETE, indent=2, ensure_ascii=False))
print({"summary": str(summary_path), "result_zip": str(zip_path)})