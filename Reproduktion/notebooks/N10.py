# KI-Unterstuetzung: siehe Erklaerung der Arbeit.
# Ursprungsdatei: phreshphish_N10_CONFIRMATORY_20K_OOD(2).ipynb
# CELL 1

# 00 — Fester Replikationsplan / Konfiguration
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc, json, math, time, random, hashlib, itertools, shutil, warnings
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import t as student_t
from sklearn.metrics import average_precision_score, precision_recall_curve
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")

if not torch.cuda.is_available():
    raise RuntimeError("GPU erforderlich; Zielumgebung: Kaggle Tesla T4.")
DEVICE = torch.device("cuda")
AMP = True

VERSION = "N10_CONFIRMATORY_20K_OOD_v1"
MASTER_SEED = 20260815

# Zehn NEUE Deep-Replikationen: keine Überschneidung mit dem bisherigen Deep-N3 (42, 82, 122).
MODEL_SEEDS = [142, 162, 182, 202, 222, 242, 262, 282, 302, 322]

# Je Replikation eine eigene deterministische, balancierte 20k-Labelstichprobe.
# R0 und R1 erhalten innerhalb einer Replikation exakt dieselben Seiten.
LABEL_RANK_SEEDS = [9101, 9102, 9103, 9104, 9105, 9106, 9107, 9108, 9109, 9110]
assert len(MODEL_SEEDS) == len(LABEL_RANK_SEEDS) == 10

BUDGET = 20_000
PRIMARY_FPR = 0.005
SCENARIOS = [
    "OFFICIAL_TEST",
    "DOMAIN_OOD_EXACT",
    "TEMPLATE_OOD_EXACT",
    "DOMAIN_TEMPLATE_OOD_EXACT",
    "LATE_TEST_Q4",
]
OOD_FAMILY = [
    "DOMAIN_OOD_EXACT",
    "TEMPLATE_OOD_EXACT",
    "DOMAIN_TEMPLATE_OOD_EXACT",
    "LATE_TEST_Q4",
]

URL_MAX_LEN = 128
TEXT_MAX_LEN = 256
PROJ_DIM = 256
DEEP_LAST_N = 4

DEEP_EPOCHS = 1
DEEP_BATCH = 16
DEEP_ENCODER_LR = 1e-5
DEEP_HEAD_LR = 2e-4
DEEP_WEIGHT_DECAY = 0.01
DEEP_WARMUP_FRAC = 0.05
DEEP_AUX_TOTAL = 0.30

# Inferenz startet aggressiv; bei OOM automatische Wiederholung mit kleinerer Batchgröße.
SCORE_BATCH_CANDIDATES = [128, 96, 64, 48, 32, 16]

ROOT = Path("/kaggle/working/phreshphish_N10_CONFIRMATORY_20K_OOD")
RESULTS = ROOT / "results"
AUDIT = ROOT / "audit"
SCORES = ROOT / "scores"
CKPT = ROOT / "checkpoints"
for p in [ROOT, RESULTS, AUDIT, SCORES, CKPT]:
    p.mkdir(parents=True, exist_ok=True)

def seed_all(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))

def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
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
    "status": "FIXED_BEFORE_NEW_N10_EXECUTION",
    "scientific_role": "additional post-hoc replication/robustness extension after earlier FINAL results were known",
    "historical_final_known": True,
    "no_optional_stopping": True,
    "all_planned_replicates_must_be_reported": True,
    "budget": BUDGET,
    "model_seeds": MODEL_SEEDS,
    "label_rank_seeds": LABEL_RANK_SEEDS,
    "paired_R0_R1_same_rows_within_replicate": True,
    "replicates_vary_both_label_sample_and_training_seed": True,
    "architecture": "DUAL URL+HTML-text; same frozen v8.4/v7.2 hyperparameters",
    "R0_text_source": "roberta-base",
    "R1_text_source": "existing R1_DAPT_TEXT checkpoint",
    "primary_endpoint": {
        "dataset": "FINAL",
        "scenario": "OFFICIAL_TEST",
        "metric": "TPR",
        "threshold_rule": "threshold determined on benign CAL at target FPR 0.005",
        "target_fpr": PRIMARY_FPR,
        "test": "two-sided exact paired sign-flip permutation test on R1-R0 TPR differences",
        "multiplicity": "none; single predeclared primary comparison"
    },
    "ff3_secondary_family": {
        "scenarios": OOD_FAMILY,
        "metric": "TPR at the same CAL-derived 0.005 FPR threshold",
        "test": "two-sided exact paired sign-flip permutation test",
        "multiplicity": "Holm correction across exactly four scenario tests"
    }
}
atomic_json(AUDIT / "N10_FIXED_PROTOCOL.json", PROTOCOL)
seed_all(MASTER_SEED)

print(torch.cuda.get_device_name(0))
print(json.dumps(PROTOCOL, indent=2))


# CELL 2

# 01 — Inputs auflösen und Data Freeze prüfen
SEARCH_ROOT = Path("/kaggle/input")
FREEZE_ID = "PHRESHPHISH_FINAL_DATA_FREEZE_v3_HF_SHARDED_DOM"
ROLE_REL = {
    "SSL": Path("roles/ssl_pool_200k"),
    "CAL": Path("roles/fpr_calibration_benign"),
    "FINAL": Path("roles/final_test"),
}

# Korrigierter Metadata-Sidecar (typisch: freezeresume)
markers = []
for p in SEARCH_ROOT.rglob("FINAL_DATA_FREEZE_COMPLETE.json"):
    try:
        o = json.loads(p.read_text())
    except Exception:
        continue
    if o.get("status") == "COMPLETE" and o.get("freeze_id") == FREEZE_ID:
        markers.append(p)

if len(markers) != 1:
    raise RuntimeError(
        f"Erwartet genau einen v3.2 Freeze-Marker, gefunden: {len(markers)}. "
        "Bitte genau einen 'freezeresume'-Input anhängen."
    )

META_ROOT = markers[0].parent
MANIFEST_ROOT = META_ROOT / "manifests"
PRIVATE_MANIFEST = MANIFEST_ROOT / "train_role_manifest_PRIVATE_WITH_LABELS.parquet"
OOD_FLAGS = MANIFEST_ROOT / "final_test_exact_ood_flags_SEALED.parquet"
FINAL_MANIFEST = MANIFEST_ROOT / "final_test_manifest_SEALED.parquet"
for p in [PRIVATE_MANIFEST, OOD_FLAGS, FINAL_MANIFEST]:
    if not p.exists():
        raise RuntimeError(f"Fehlender Metadata-Input: {p}")

# Physische Rollen (typisch: finaldatafreeze)
roots = []
for r in SEARCH_ROOT.rglob("roles"):
    if r.is_dir():
        root = r.parent
        if all((root / rel).exists() for rel in ROLE_REL.values()):
            roots.append(root)
roots = list({str(x.resolve()): x for x in roots}.values())

if len(roots) != 1:
    raise RuntimeError(
        f"Erwartet genau einen physischen Data-Freeze-Root, gefunden: {[str(x) for x in roots]}"
    )

DATA_ROOT = roots[0]
ROLE_DIR = {k: DATA_ROOT / rel for k, rel in ROLE_REL.items()}

def nrows(d):
    fs = sorted(Path(d).glob("*.parquet"))
    if not fs:
        raise RuntimeError(f"Keine Parquet-Dateien: {d}")
    return sum(pq.ParquetFile(p).metadata.num_rows for p in fs)

COUNTS = {k: nrows(v) for k, v in ROLE_DIR.items()}
EXPECTED = {"SSL": 200000, "CAL": 50000, "FINAL": 168060}
if COUNTS != EXPECTED:
    raise RuntimeError(f"Role counts stimmen nicht: {COUNTS}")

# DAPT-Checkpoint (typisch: bigresults)
dapt_candidates = []
for p in SEARCH_ROOT.rglob("R1_DAPT_TEXT"):
    if p.is_dir() and (p / "config.json").exists():
        dapt_candidates.append(p)
# Bevorzugt exakt den bekannten v4.3-Pfad.
preferred_dapt = [p for p in dapt_candidates if "phreshphish_FINAL_DEEP_TRIMODAL_SSL_v4_3_LABELSET_ALIGNMENT_FIX" in str(p)]
if preferred_dapt:
    dapt_candidates = preferred_dapt
dapt_candidates = list({str(p.resolve()): p for p in dapt_candidates}.values())
if len(dapt_candidates) != 1:
    raise RuntimeError(f"R1_DAPT_TEXT nicht eindeutig gefunden: {[str(p) for p in dapt_candidates]}")
TEXT_DAPT_SRC = str(dapt_candidates[0])

# Lokales roberta-base (typisch: carmengeiss/roberta-base)
roberta_candidates = []
for cfg in SEARCH_ROOT.rglob("config.json"):
    p = cfg.parent
    s = str(p).lower()
    if p.name.lower() == "roberta-base" and "r1_dapt_text" not in s:
        roberta_candidates.append(p)
preferred_roberta = [p for p in roberta_candidates if "carmengeiss" in str(p).lower()]
if preferred_roberta:
    roberta_candidates = preferred_roberta
roberta_candidates = list({str(p.resolve()): p for p in roberta_candidates}.values())
if len(roberta_candidates) != 1:
    raise RuntimeError(
        f"roberta-base nicht eindeutig gefunden: {[str(p) for p in roberta_candidates]}. "
        "Bitte den bisherigen carmengeiss/roberta-base Input anhängen."
    )
TEXT_BASE_SRC = str(roberta_candidates[0])

# BERT URL-Encoder: lokal suchen, sonst öffentliches HF-Modell.
bert_candidates = []
for cfg in SEARCH_ROOT.rglob("config.json"):
    p = cfg.parent
    if p.name.lower() == "bert-base-uncased":
        bert_candidates.append(p)
bert_candidates = list({str(p.resolve()): p for p in bert_candidates}.values())
URL_SRC = str(bert_candidates[0]) if len(bert_candidates) == 1 else "bert-base-uncased"

INPUTS = {
    "data_root": str(DATA_ROOT),
    "meta_root": str(META_ROOT),
    "text_base_src": TEXT_BASE_SRC,
    "text_dapt_src": TEXT_DAPT_SRC,
    "url_src": URL_SRC,
    "counts": COUNTS,
}
atomic_json(AUDIT / "INPUT_RESOLUTION.json", INPUTS)
print(json.dumps(INPUTS, indent=2))


# CELL 3

# 02 — Rollen/Labels/OOD-Masken in exakt eingefrorener Reihenfolge
def y01(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s).astype(int).to_numpy()
    z = s.astype(str).str.lower().str.strip()
    pos = z.isin(["1", "true", "phish", "phishing", "malicious"])
    neg = z.isin(["0", "false", "benign", "legitimate", "legit"])
    if not (pos | neg).all():
        raise RuntimeError(f"Unbekannte Labels: {sorted(z[~(pos|neg)].unique())[:10]}")
    return pos.astype(int).to_numpy()

def read_role(key, cols):
    parts = [pd.read_parquet(p, columns=cols) for p in sorted(ROLE_DIR[key].glob("*.parquet"))]
    out = pd.concat(parts, ignore_index=True)
    if len(out) != COUNTS[key]:
        raise RuntimeError(f"Role row mismatch {key}: {len(out)}")
    return out

# Private Labels ausschließlich für den eingefrorenen 200k-SSL-Pool.
priv = pd.read_parquet(PRIVATE_MANIFEST, columns=["sha256", "label", "role"])
priv["sha256"] = priv.sha256.astype(str).str.lower()
sslmeta = priv[priv.role.eq("SSL_POOL")][["sha256", "label"]].reset_index(drop=True)
if len(sslmeta) != 200000 or sslmeta.sha256.duplicated().any():
    raise RuntimeError("SSL private manifest ungültig.")
sslmeta["y"] = y01(sslmeta.label)
SSL_LABEL_MAP = dict(zip(sslmeta.sha256, sslmeta.y.astype(int)))

SSL_DF = read_role("SSL", ["sha256", "url", "text"])
SSL_DF["sha256"] = SSL_DF.sha256.astype(str).str.lower()
SSL_DF["y"] = SSL_DF.sha256.map(SSL_LABEL_MAP)
if SSL_DF.y.isna().any():
    raise RuntimeError("SSL physical/private label mismatch.")
SSL_DF["y"] = SSL_DF.y.astype(int)

CAL_DF = read_role("CAL", ["sha256", "label", "url", "text"])
CAL_DF["sha256"] = CAL_DF.sha256.astype(str).str.lower()
CAL_DF["y"] = y01(CAL_DF.label)
if int(CAL_DF.y.sum()) != 0:
    raise RuntimeError("CAL muss benign-only sein.")

FINAL_DF = read_role("FINAL", ["sha256", "label", "date", "url", "text"])
FINAL_DF["sha256"] = FINAL_DF.sha256.astype(str).str.lower()
FINAL_DF["y"] = y01(FINAL_DF.label)
if (int((FINAL_DF.y == 0).sum()), int((FINAL_DF.y == 1).sum())) != (91260, 76800):
    raise RuntimeError("FINAL-Klassenverteilung unerwartet.")

# Disjunktheit.
sets = {
    "SSL": set(SSL_DF.sha256),
    "CAL": set(CAL_DF.sha256),
    "FINAL": set(FINAL_DF.sha256),
}
for a, b in itertools.combinations(sets, 2):
    if sets[a] & sets[b]:
        raise RuntimeError(f"SHA-Overlap {a}/{b}: {len(sets[a] & sets[b])}")

# OOD-Masken anhand versiegelter Flags.
flags = pd.read_parquet(OOD_FLAGS)
flags["sha256"] = flags.sha256.astype(str).str.lower()
flags = flags.set_index("sha256").loc[FINAL_DF.sha256].reset_index()

dt = pd.to_datetime(FINAL_DF.date, errors="coerce")
late_cutoff = dt.dropna().quantile(0.75)
late = (dt >= late_cutoff).fillna(False).to_numpy()

FINAL_MASKS = {
    "OFFICIAL_TEST": np.ones(len(FINAL_DF), dtype=bool),
    "DOMAIN_OOD_EXACT": flags.domain_ood_exact.to_numpy(bool),
    "TEMPLATE_OOD_EXACT": flags.template_ood_exact.to_numpy(bool),
    "DOMAIN_TEMPLATE_OOD_EXACT": flags.domain_template_ood_exact.to_numpy(bool),
    "LATE_TEST_Q4": late,
}

atomic_json(AUDIT / "DATA_PROTOCOL.json", {
    "status": "PASS",
    "counts": COUNTS,
    "sha_disjointness": "PASS",
    "final_class_counts": FINAL_DF.y.value_counts().sort_index().to_dict(),
    "late_q4_cutoff": str(late_cutoff),
    "scenario_rows": {k: int(v.sum()) for k, v in FINAL_MASKS.items()},
})
print("DATA_PROTOCOL PASS")
print({k: int(v.sum()) for k, v in FINAL_MASKS.items()})


# CELL 4

# 03 — Zehn feste, balancierte 20k-Labelstichproben
def keyed_rank(sha, seed):
    return hashlib.sha256(f"{seed}|{sha}".encode()).hexdigest()

def balanced_20k_indices(rank_seed):
    out = []
    for cls in [0, 1]:
        q = SSL_DF[SSL_DF.y.eq(cls)][["sha256"]].copy()
        q["idx"] = q.index.to_numpy()
        q["rank"] = [keyed_rank(x, rank_seed) for x in q.sha256]
        out.append(q.sort_values("rank").idx.to_numpy()[: BUDGET // 2])
    idx = np.sort(np.concatenate(out))
    if len(idx) != BUDGET or int(SSL_DF.y.iloc[idx].sum()) != BUDGET // 2:
        raise RuntimeError(f"20k-Budgetkonstruktion fehlgeschlagen für rank_seed={rank_seed}")
    return idx

REPLICATES = []
for rep_i, (model_seed, rank_seed) in enumerate(zip(MODEL_SEEDS, LABEL_RANK_SEEDS)):
    idx = balanced_20k_indices(rank_seed)
    REPLICATES.append({
        "replicate": rep_i,
        "model_seed": model_seed,
        "label_rank_seed": rank_seed,
        "indices": idx,
    })

# Overlap nur dokumentieren, nicht als Fehler werten.
ov = []
for i, j in itertools.combinations(range(10), 2):
    a = set(REPLICATES[i]["indices"])
    b = set(REPLICATES[j]["indices"])
    ov.append({
        "rep_i": i,
        "rep_j": j,
        "intersection": len(a & b),
        "jaccard": len(a & b) / len(a | b),
    })
pd.DataFrame(ov).to_csv(AUDIT / "LABEL_SAMPLE_OVERLAP.csv", index=False)

pd.DataFrame([
    {
        "replicate": r["replicate"],
        "model_seed": r["model_seed"],
        "label_rank_seed": r["label_rank_seed"],
        "rows": len(r["indices"]),
        "phish": int(SSL_DF.y.iloc[r["indices"]].sum()),
        "benign": int(len(r["indices"]) - SSL_DF.y.iloc[r["indices"]].sum()),
    }
    for r in REPLICATES
]).to_csv(AUDIT / "REPLICATE_PLAN.csv", index=False)

print("REPLICATE_PLAN PASS")


# CELL 5

# 04 — Modell, Tokenisierung, Low-FPR-Metriken
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        HF_TOKEN = None

url_tok = AutoTokenizer.from_pretrained(URL_SRC, token=HF_TOKEN)
text_tok = AutoTokenizer.from_pretrained(TEXT_BASE_SRC, token=HF_TOKEN)

def pooled(out, mask):
    h = out.last_hidden_state
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp_min(1.0)

def freeze_last(model, n):
    for p in model.parameters():
        p.requires_grad = False
    blocks = None
    for attr in ["encoder.layer", "transformer.layer"]:
        cur = model
        ok = True
        for a in attr.split("."):
            if not hasattr(cur, a):
                ok = False
                break
            cur = getattr(cur, a)
        if ok:
            blocks = cur
            break
    if blocks is not None:
        for b in blocks[-n:]:
            for p in b.parameters():
                p.requires_grad = True
    if hasattr(model, "pooler") and model.pooler is not None:
        for p in model.pooler.parameters():
            p.requires_grad = True

class DeepDual(nn.Module):
    def __init__(self, text_src):
        super().__init__()
        self.u = AutoModel.from_pretrained(URL_SRC, token=HF_TOKEN)
        self.t = AutoModel.from_pretrained(text_src, token=HF_TOKEN)
        freeze_last(self.u, DEEP_LAST_N)
        freeze_last(self.t, DEEP_LAST_N)

        self.ua = nn.Linear(int(self.u.config.hidden_size), PROJ_DIM)
        self.ta = nn.Linear(int(self.t.config.hidden_size), PROJ_DIM)
        self.gate = nn.Sequential(nn.Linear(PROJ_DIM, 64), nn.GELU(), nn.Linear(64, 1))
        self.head = nn.Sequential(
            nn.Linear(PROJ_DIM * 3, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        self.uaux = nn.Linear(PROJ_DIM, 1)
        self.taux = nn.Linear(PROJ_DIM, 1)

def dual_forward_all(m, u, t):
    zu = F.normalize(m.ua(pooled(m.u(**u), u["attention_mask"])), dim=-1)
    zt = F.normalize(m.ta(pooled(m.t(**t), t["attention_mask"])), dim=-1)
    st = torch.stack([zu, zt], 1)
    gw = torch.softmax(m.gate(st).squeeze(-1), 1)
    weighted = (st * gw[:, :, None]).sum(1)
    main = m.head(torch.cat([zu, zt, weighted], 1)).squeeze(-1)
    aux = [m.uaux(zu).squeeze(-1), m.taux(zt).squeeze(-1)]
    return main, aux

class WebTrainDataset(Dataset):
    def __init__(self, frame, indices):
        self.frame = frame.iloc[np.asarray(indices, dtype=int)].reset_index(drop=True)
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, i):
        r = self.frame.iloc[i]
        return (
            str(r.url) if pd.notna(r.url) else "",
            str(r.text) if pd.notna(r.text) else "",
            int(r.y),
        )

def collate_web(batch):
    urls = [x[0] for x in batch]
    texts = [x[1] for x in batch]
    y = torch.tensor([x[2] for x in batch], dtype=torch.float32, device=DEVICE)
    u = url_tok(urls, padding=True, truncation=True, max_length=URL_MAX_LEN, return_tensors="pt")
    t = text_tok(texts, padding=True, truncation=True, max_length=TEXT_MAX_LEN, return_tensors="pt")
    u = {k: v.to(DEVICE, non_blocking=True) for k, v in u.items()}
    t = {k: v.to(DEVICE, non_blocking=True) for k, v in t.items()}
    return u, t, y

def tok_url(rows):
    b = url_tok(
        rows.url.fillna("").astype(str).tolist(),
        padding=True, truncation=True, max_length=URL_MAX_LEN, return_tensors="pt"
    )
    return {k: v.to(DEVICE, non_blocking=True) for k, v in b.items()}

def tok_text(rows):
    b = text_tok(
        rows.text.fillna("").astype(str).tolist(),
        padding=True, truncation=True, max_length=TEXT_MAX_LEN, return_tensors="pt"
    )
    return {k: v.to(DEVICE, non_blocking=True) for k, v in b.items()}

def threshold_for_fpr(neg_scores, fpr):
    s = np.asarray(neg_scores, dtype=float)
    k = int(math.floor(float(fpr) * len(s) + 1e-12))
    if k <= 0:
        return float(np.nextafter(np.max(s), np.inf))
    ss = np.sort(s)
    return float(np.nextafter(ss[-k], np.inf))

def operating_metrics(y, score, th):
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    pred = score >= th
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    tn = int(((~pred) & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "tpr": tpr, "fpr": fpr, "precision": precision,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "fp_per_1000": 1000 * fpr,
    }

def ranking_metrics(y, score):
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    ap = float(average_precision_score(y, score)) if len(np.unique(y)) > 1 else np.nan
    if (y == 1).sum() == 0:
        return {"AP": ap, "P_at_R90": np.nan}
    p, r, _ = precision_recall_curve(y, score)
    ok = np.where(r >= 0.90)[0]
    return {"AP": ap, "P_at_R90": float(np.max(p[ok])) if len(ok) else np.nan}

print("MODEL_AND_METRICS PASS")


# CELL 6

# 05 — Training + adaptive Inferenz + condition-level Checkpoints
def model_paths(rep_i, rep_name):
    d = CKPT / f"rep{rep_i:02d}_{rep_name}"
    d.mkdir(parents=True, exist_ok=True)
    return d, d / "model_state.pt", d / "TRAIN_COMPLETE.json"

def score_paths(rep_i, rep_name):
    d = SCORES / f"rep{rep_i:02d}_{rep_name}"
    d.mkdir(parents=True, exist_ok=True)
    return d, d / "CAL_logit.npy", d / "FINAL_logit.npy"

def condition_marker(rep_i, rep_name):
    return RESULTS / f"rep{rep_i:02d}_{rep_name}_COMPLETE.json"

def condition_csv(rep_i, rep_name):
    return RESULTS / f"rep{rep_i:02d}_{rep_name}.csv"

def train_or_load_model(rep_i, model_seed, label_indices, rep_name):
    text_src = TEXT_BASE_SRC if rep_name == "R0" else TEXT_DAPT_SRC
    d, state_path, train_marker = model_paths(rep_i, rep_name)

    if state_path.exists() and train_marker.exists():
        m = DeepDual(text_src).to(DEVICE)
        m.load_state_dict(torch.load(state_path, map_location=DEVICE))
        return m, "RESUME_CHECKPOINT"

    seed_all(model_seed)
    m = DeepDual(text_src).to(DEVICE)
    ds = WebTrainDataset(SSL_DF, label_indices)
    dl = DataLoader(
        ds,
        batch_size=DEEP_BATCH,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_web,
        generator=torch.Generator().manual_seed(int(model_seed)),
        drop_last=False,
    )

    enc = []
    for module in [m.u, m.t]:
        enc.extend([p for p in module.parameters() if p.requires_grad])
    enc_ids = {id(p) for p in enc}
    heads = [p for p in m.parameters() if p.requires_grad and id(p) not in enc_ids]

    opt = torch.optim.AdamW(
        [
            {"params": enc, "lr": DEEP_ENCODER_LR},
            {"params": heads, "lr": DEEP_HEAD_LR},
        ],
        weight_decay=DEEP_WEIGHT_DECAY,
    )

    total_steps = max(1, len(dl) * DEEP_EPOCHS)
    warmup = int(round(DEEP_WARMUP_FRAC * total_steps))
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=warmup, num_training_steps=total_steps
    )

    losses = []
    m.train()
    step = 0
    t0 = time.time()
    for _epoch in range(DEEP_EPOCHS):
        for u, t, y in dl:
            opt.zero_grad(set_to_none=True)
            with amp_ctx():
                main, aux = dual_forward_all(m, u, t)
                main_loss = F.binary_cross_entropy_with_logits(main, y)
                aux_loss = torch.stack(
                    [F.binary_cross_entropy_with_logits(a, y) for a in aux]
                ).mean()
                loss = main_loss + DEEP_AUX_TOTAL * aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in m.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            sched.step()
            step += 1
            losses.append(float(loss.detach().cpu()))

            if step % 250 == 0 or step == total_steps:
                print({
                    "TRAIN": f"rep{rep_i:02d}_{rep_name}",
                    "step": step,
                    "of": total_steps,
                    "loss100": float(np.mean(losses[-100:])),
                    "elapsed_min": round((time.time() - t0) / 60, 2),
                })

    m.eval()
    atomic_torch(state_path, m.state_dict())
    atomic_json(train_marker, {
        "status": "COMPLETE",
        "replicate": rep_i,
        "rep": rep_name,
        "model_seed": model_seed,
        "rows": len(label_indices),
        "epochs": DEEP_EPOCHS,
        "batch": DEEP_BATCH,
        "encoder_lr": DEEP_ENCODER_LR,
        "head_lr": DEEP_HEAD_LR,
        "aux_total": DEEP_AUX_TOTAL,
        "elapsed_s": time.time() - t0,
    })

    del ds, dl, opt, sched
    gc.collect()
    torch.cuda.empty_cache()
    return m, "NEW_TRAIN"

@torch.no_grad()
def score_frame_once(m, df, batch):
    out = np.empty(len(df), dtype=np.float32)
    m.eval()
    for st in range(0, len(df), batch):
        en = min(st + batch, len(df))
        rows = df.iloc[st:en]
        u = tok_url(rows)
        t = tok_text(rows)
        with amp_ctx():
            main, _ = dual_forward_all(m, u, t)
        out[st:en] = main.float().detach().cpu().numpy()
    return out

def score_frame_adaptive(m, df, role):
    last_error = None
    for batch in SCORE_BATCH_CANDIDATES:
        try:
            torch.cuda.empty_cache()
            t0 = time.time()
            arr = score_frame_once(m, df, batch)
            print({
                "SCORE": role,
                "rows": len(df),
                "batch": batch,
                "elapsed_min": round((time.time() - t0) / 60, 2),
            })
            return arr, batch
        except torch.cuda.OutOfMemoryError as e:
            last_error = e
            print({"SCORE_OOM_RETRY": role, "failed_batch": batch})
            gc.collect()
            torch.cuda.empty_cache()
    raise RuntimeError(f"Scoring OOM selbst bei kleinster Batchgröße: {last_error}")

def run_condition(rep_info, rep_name):
    rep_i = rep_info["replicate"]
    model_seed = rep_info["model_seed"]
    rank_seed = rep_info["label_rank_seed"]
    idx = rep_info["indices"]

    marker = condition_marker(rep_i, rep_name)
    out_csv = condition_csv(rep_i, rep_name)
    if marker.exists() and out_csv.exists():
        print({"REUSE_COMPLETE": f"rep{rep_i:02d}_{rep_name}"})
        return

    score_dir, cal_path, final_path = score_paths(rep_i, rep_name)

    # Falls Scores schon vollständig vorliegen, Training muss nicht erneut geladen werden.
    if cal_path.exists() and final_path.exists():
        cal_scores = np.asarray(np.load(cal_path, mmap_mode="r"), dtype=np.float32)
        final_scores = np.asarray(np.load(final_path, mmap_mode="r"), dtype=np.float32)
        model_origin = "SCORES_REUSED"
        score_batches = {"CAL": None, "FINAL": None}
    else:
        m, model_origin = train_or_load_model(rep_i, model_seed, idx, rep_name)

        if cal_path.exists():
            cal_scores = np.asarray(np.load(cal_path, mmap_mode="r"), dtype=np.float32)
            cal_batch = None
        else:
            cal_scores, cal_batch = score_frame_adaptive(m, CAL_DF, "CAL")
            np.save(cal_path, cal_scores)

        if final_path.exists():
            final_scores = np.asarray(np.load(final_path, mmap_mode="r"), dtype=np.float32)
            final_batch = None
        else:
            final_scores, final_batch = score_frame_adaptive(m, FINAL_DF, "FINAL")
            np.save(final_path, final_scores)

        score_batches = {"CAL": cal_batch, "FINAL": final_batch}
        del m
        gc.collect()
        torch.cuda.empty_cache()

    threshold = threshold_for_fpr(cal_scores, PRIMARY_FPR)
    rows = []
    for scenario, mask in FINAL_MASKS.items():
        y = FINAL_DF.y.to_numpy(int)[mask]
        s = final_scores[mask]
        rows.append({
            "replicate": rep_i,
            "model_seed": model_seed,
            "label_rank_seed": rank_seed,
            "rep": rep_name,
            "budget": BUDGET,
            "dataset": "FINAL",
            "scenario": scenario,
            "target_fpr": PRIMARY_FPR,
            "threshold": threshold,
            **operating_metrics(y, s, threshold),
            **ranking_metrics(y, s),
        })

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    atomic_json(marker, {
        "status": "COMPLETE_SCORED",
        "replicate": rep_i,
        "rep": rep_name,
        "model_seed": model_seed,
        "label_rank_seed": rank_seed,
        "model_origin": model_origin,
        "score_batch": score_batches,
        "threshold": threshold,
    })

    # Nach sicher persistierten Scores wird der große Modellzustand gelöscht.
    _, state_path, _ = model_paths(rep_i, rep_name)
    if state_path.exists():
        state_path.unlink()

    gc.collect()
    torch.cuda.empty_cache()
    print({"CONDITION_COMPLETE": f"rep{rep_i:02d}_{rep_name}"})


# CELL 7

# 06 — Alle 10 × 2 Bedingungen ausführen
run_start = time.time()

for rep_info in REPLICATES:
    for rep_name in ["R0", "R1"]:
        run_condition(rep_info, rep_name)

print({
    "ALL_CONDITIONS_FINISHED": True,
    "elapsed_hours": round((time.time() - run_start) / 3600, 3),
})


# CELL 8

# 07 — Feste statistische Auswertung: Official primär, vier Shifts Holm-korrigiert
parts = sorted(RESULTS.glob("rep??_R[01].csv"))
if len(parts) != 20:
    raise RuntimeError(f"Erwartet 20 condition CSVs, gefunden: {len(parts)}")

ALL = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
ALL.to_csv(RESULTS / "DEEP_N10_REPLICATION_ALL.csv", index=False)

def exact_signflip_p(diff):
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan
    obs = abs(d.mean())
    vals = []
    for signs in itertools.product([-1, 1], repeat=len(d)):
        vals.append(abs(np.mean(d * np.asarray(signs))))
    return float(np.mean(np.asarray(vals) >= obs - 1e-15))

def mean_ci(diff):
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return (float(np.mean(d)) if len(d) else np.nan, np.nan, np.nan)
    m = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    q = float(student_t.ppf(0.975, len(d) - 1))
    return m, m - q * se, m + q * se

def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p))
    vals = p[finite]
    order = np.argsort(vals, kind="mergesort")
    m = len(vals)
    running = 0.0
    adjusted_sorted = np.empty(m, dtype=float)
    for rank, local_pos in enumerate(order):
        running = max(running, (m - rank) * vals[local_pos])
        adjusted_sorted[rank] = min(1.0, running)
    for rank, local_pos in enumerate(order):
        out[finite[local_pos]] = adjusted_sorted[rank]
    return out

delta_rows = []
for scenario in SCENARIOS:
    g = ALL[ALL.scenario.eq(scenario)]
    piv = g.pivot_table(index="replicate", columns="rep", values="tpr", aggfunc="first")
    if not {"R0", "R1"}.issubset(piv.columns) or len(piv.dropna()) != 10:
        raise RuntimeError(f"Unvollständige Paare für {scenario}")
    d = (piv["R1"] - piv["R0"]).dropna().to_numpy(float)
    m, lo, hi = mean_ci(d)
    delta_rows.append({
        "scenario": scenario,
        "n_pairs": len(d),
        "R0_tpr_mean": float(piv["R0"].mean()),
        "R1_tpr_mean": float(piv["R1"].mean()),
        "mean_delta_tpr_pp": 100 * m,
        "ci95_lo_pp": 100 * lo,
        "ci95_hi_pp": 100 * hi,
        "positive_pairs": int((d > 0).sum()),
        "negative_pairs": int((d < 0).sum()),
        "exact_signflip_p": exact_signflip_p(d),
    })

DELTA = pd.DataFrame(delta_rows)
DELTA["statistical_role"] = np.where(
    DELTA.scenario.eq("OFFICIAL_TEST"),
    "PRIMARY_SINGLE_TEST",
    "FF3_OOD_HOLM_FAMILY",
)
DELTA["holm_p"] = np.nan

ood_mask = DELTA.scenario.isin(OOD_FAMILY)
DELTA.loc[ood_mask, "holm_p"] = holm_adjust(
    DELTA.loc[ood_mask, "exact_signflip_p"].to_numpy(float)
)
DELTA["raw_p_lt_0p05"] = DELTA.exact_signflip_p < 0.05
DELTA["holm_p_lt_0p05"] = DELTA.holm_p < 0.05

DELTA.to_csv(RESULTS / "DEEP_N10_REPLICATION_DELTA.csv", index=False)
DELTA[DELTA.scenario.eq("OFFICIAL_TEST")].to_csv(
    RESULTS / "DEEP_N10_PRIMARY_OFFICIAL.csv", index=False
)
DELTA[ood_mask].to_csv(
    RESULTS / "DEEP_N10_OOD_HOLM.csv", index=False
)

# Deskriptive Aggregation der absoluten Systemwerte.
AGG = ALL.groupby(["rep", "scenario"], as_index=False).agg(
    n=("tpr", "size"),
    tpr_mean=("tpr", "mean"),
    tpr_std=("tpr", "std"),
    fpr_mean=("fpr", "mean"),
    fpr_std=("fpr", "std"),
    precision_mean=("precision", "mean"),
    AP_mean=("AP", "mean"),
    P_at_R90_mean=("P_at_R90", "mean"),
)
AGG.to_csv(RESULTS / "DEEP_N10_REPLICATION_AGG.csv", index=False)

print("\n=== N10 DELTA ===")
print(DELTA.to_string(index=False))
print("\n=== ABSOLUTE SYSTEMWERTE ===")
print(AGG.to_string(index=False))


# CELL 9

# 08 — Abschlussmarker und kleines Evidence-Package
required = [
    RESULTS / "DEEP_N10_REPLICATION_ALL.csv",
    RESULTS / "DEEP_N10_REPLICATION_DELTA.csv",
    RESULTS / "DEEP_N10_PRIMARY_OFFICIAL.csv",
    RESULTS / "DEEP_N10_OOD_HOLM.csv",
    RESULTS / "DEEP_N10_REPLICATION_AGG.csv",
    AUDIT / "N10_FIXED_PROTOCOL.json",
    AUDIT / "DATA_PROTOCOL.json",
    AUDIT / "INPUT_RESOLUTION.json",
    AUDIT / "REPLICATE_PLAN.csv",
    AUDIT / "LABEL_SAMPLE_OVERLAP.csv",
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise RuntimeError(f"Finalisierung unvollständig: {missing}")

complete = {
    "status": "COMPLETE",
    "version": VERSION,
    "replicates": 10,
    "conditions": 20,
    "budget": BUDGET,
    "primary_fpr": PRIMARY_FPR,
    "primary_test": "OFFICIAL_TEST",
    "ood_holm_family": OOD_FAMILY,
    "all_planned_replicates_reported": True,
}
atomic_json(ROOT / "N10_CONFIRMATORY_COMPLETE.json", complete)

package = Path("/kaggle/working/phreshphish_N10_CONFIRMATORY_RESULTS")
if package.exists():
    shutil.rmtree(package)
package.mkdir(parents=True)

shutil.copytree(RESULTS, package / "results")
shutil.copytree(AUDIT, package / "audit")
shutil.copy2(ROOT / "N10_CONFIRMATORY_COMPLETE.json", package / "N10_CONFIRMATORY_COMPLETE.json")

zip_base = "/kaggle/working/phreshphish_N10_CONFIRMATORY_RESULTS"
zip_path = shutil.make_archive(zip_base, "zip", root_dir=package)

print(json.dumps(complete, indent=2))
print({"EVIDENCE_PACKAGE": zip_path})
