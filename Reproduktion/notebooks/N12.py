# KI-Unterstuetzung: siehe Erklaerung der Arbeit.
# Ursprungsdatei: PUTRA_EXTERNAL_LABEL_EFFICIENCY_R0_VS_RUT_N10_v1_1.ipynb
# CELL 1
# 00 — Imports und festes Protokoll
import os, gc, re, json, math, time, random, hashlib, itertools, zipfile, shutil, warnings, html as html_std
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, roc_auc_score
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")

VERSION = "PUTRA_EXTERNAL_LABEL_EFFICIENCY_R0_VS_RUT_N10_v1_1"
MASTER_SEED = 20260911

# Identisch zum vorausgehenden Putra-Adaptationsexperiment.
EARLY_FRACTION = 0.75
LABEL_BUDGETS = [200, 500, 1000, 2000]
N_REPS = 10
MODEL_SEEDS = [142, 162, 182, 202, 222, 242, 262, 282, 302, 322]
LABEL_RANK_SEEDS = [9101, 9102, 9103, 9104, 9105, 9106, 9107, 9108, 9109, 9110]

URL_MAX_LEN = 128
TEXT_MAX_LEN = 256
TEXT_MAX_CHARS = 50_000
BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"

LINEAR_SPEC = {
    "C": 1.0,
    "solver": "liblinear",
    "class_weight": "balanced",
    "max_iter": 2000,
}

PUTRA_DOI = "10.5281/zenodo.8041387"
PUTRA_FILES = {
    "phishing.csv": {
        "url": "https://zenodo.org/records/8041387/files/phishing.csv?download=1",
        "md5": "513962464c413fc30b2030547a12868a",
        "label": 1,
    },
    "not-phishing.csv": {
        "url": "https://zenodo.org/records/8041387/files/not-phishing.csv?download=1",
        "md5": "f5d218eb67f5d7bd0571e8089a8fc392",
        "label": 0,
    },
}

SEARCH_ROOT = Path("/kaggle/input") if Path("/kaggle/input").exists() else Path("/mnt/data")
ROOT = Path("/kaggle/working/putra_r0_vs_rut_label_efficiency") if Path("/kaggle/working").exists() else Path("/mnt/data/putra_r0_vs_rut_label_efficiency")
DATA_DIR = ROOT / "data"
EMB_DIR = ROOT / "embeddings"
RESULTS_DIR = ROOT / "results"
AUDIT_DIR = ROOT / "audit"
FIG_DIR = ROOT / "figures"
CACHE_DIR = ROOT / "_cache"
for p in [ROOT, DATA_DIR, EMB_DIR, RESULTS_DIR, AUDIT_DIR, FIG_DIR, CACHE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("GPU erforderlich. In Kaggle unter Settings -> Accelerator eine GPU aktivieren.")
DEVICE = torch.device("cuda")

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
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)

def md5_file(path, chunk=8*1024*1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

PROTOCOL = {
    "version": VERSION,
    "scientific_status": "POST_HOC_EXTERNAL_LABEL_EFFICIENCY_ANALYSIS",
    "prior_putra_analyses_seen": True,
    "no_optional_stopping": True,
    "all_planned_replicates_reported": True,
    "temporal_design": f"global scan_date cutoff at empirical {EARLY_FRACTION:.2f} quantile",
    "variants": {
        "R0": "generic pretrained BERT URL + RoBERTa text encoders; no additional PhreshPhish TAPT",
        "RUT": "same encoder families after PhreshPhish URL- and text-TAPT",
    },
    "label_budgets": LABEL_BUDGETS,
    "model_seeds": MODEL_SEEDS,
    "label_rank_seeds": LABEL_RANK_SEEDS,
    "paired_same_labeled_rows_R0_RUT": True,
    "nested_budgets_within_seed": True,
    "primary_endpoint": "normalized area under AP learning curve over log10(label budget)",
    "primary_test": "two-sided exact paired sign-flip test over N10 AULC differences",
    "secondary_AP_family": "four paired AP tests, Holm-corrected across label budgets",
    "test_used_for_training": False,
    "putra_target_MLM_in_this_experiment": False,
}
atomic_json(AUDIT_DIR / "EXTERNAL_LABEL_EFFICIENCY_PROTOCOL_LOCK.json", PROTOCOL)
seed_all(MASTER_SEED)

print({
    "version": VERSION,
    "device": torch.cuda.get_device_name(0),
    "budgets": LABEL_BUDGETS,
    "n_reps": N_REPS,
    "search_root": str(SEARCH_ROOT),
})

# CELL 3
# 01 — Optional Putra-CSVs und früheres kleines Ergebnis-ZIP aus Kaggle Inputs lesen
CACHE_NAMES = {
    "phishing.csv",
    "not-phishing.csv",
    "PUTRA_TARGET_ADAPTATION_N10_RUN_METRICS.csv",
    "PUTRA_TEMPORAL_SPLIT_AUDIT.json",
}

cache_hits = []
for zpath in SEARCH_ROOT.rglob("*.zip"):
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
                    with zf.open(n) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
            cache_hits.append({"zip": str(zpath), "files": [Path(n).name for n in members]})
    except zipfile.BadZipFile:
        pass

atomic_json(AUDIT_DIR / "INPUT_CACHE_AUDIT.json", cache_hits)
print({"cache_archives_found": len(cache_hits)})

# CELL 4
# 02 — CSVs laden, Complete Cases bilden, temporal splitten

def resolve_putra_file(name, spec):
    hits = [p for p in SEARCH_ROOT.rglob(name) if p.is_file()]
    hits += [p for p in CACHE_DIR.rglob(name) if p.is_file()]
    if hits:
        p = sorted(hits, key=lambda x: (len(str(x)), str(x)))[0]
    else:
        p = DATA_DIR / name
        if not p.exists():
            print("Download", name)
            urlretrieve(spec["url"], p)

    digest = md5_file(p)
    if digest != spec["md5"]:
        raise RuntimeError(f"MD5 mismatch für {name}: {digest}")
    return p, digest

frames = []
input_audit = {}

for name, spec in PUTRA_FILES.items():
    p, digest = resolve_putra_file(name, spec)
    q = pd.read_csv(p, usecols=["url", "features.text", "scan_date"], low_memory=False)
    q["url"] = q["url"].fillna("").astype(str).str.strip()
    q["text"] = (
        q["features.text"].fillna("").astype(str)
        .map(html_std.unescape)
        .map(lambda s: re.sub(r"\s+", " ", s).strip()[:TEXT_MAX_CHARS])
    )
    q["scan_dt"] = pd.to_datetime(q["scan_date"], errors="coerce", utc=True)
    q["label"] = int(spec["label"])
    frames.append(q[["url", "text", "scan_dt", "label"]])
    input_audit[name] = {"path": str(p), "rows": len(q), "md5": digest}
    print({"putra_input": name, "rows": len(q), "md5": digest, "path": str(p)})

ALL = pd.concat(frames, ignore_index=True)
ALL = ALL[
    ALL["url"].str.len().gt(0)
    & ALL["text"].str.len().gt(0)
    & ALL["scan_dt"].notna()
].copy().reset_index(drop=True)

cutoff = ALL["scan_dt"].quantile(EARLY_FRACTION, interpolation="nearest")
EARLY = ALL[ALL["scan_dt"] <= cutoff].copy().reset_index(drop=True)
LATE_RAW = ALL[ALL["scan_dt"] > cutoff].copy().reset_index(drop=True)

early_urls = set(EARLY["url"].astype(str))
late_overlap = LATE_RAW["url"].astype(str).isin(early_urls)
LATE = LATE_RAW[~late_overlap].copy().reset_index(drop=True)

if min(EARLY["label"].value_counts()) < max(LABEL_BUDGETS)//2:
    raise RuntimeError("EARLY enthält für die geplanten balancierten Budgets zu wenige Fälle einer Klasse.")
if min(LATE["label"].value_counts()) < 200:
    raise RuntimeError("LATE-Test ist für eine Klasse unerwartet klein.")

SPLIT_AUDIT = {
    "putra_doi": PUTRA_DOI,
    "complete_rows": int(len(ALL)),
    "cutoff_utc": str(cutoff),
    "early_rows": int(len(EARLY)),
    "late_rows_before_overlap_filter": int(len(LATE_RAW)),
    "late_rows_after_overlap_filter": int(len(LATE)),
    "late_exact_url_overlaps_removed": int(late_overlap.sum()),
    "early_date_range": [str(EARLY.scan_dt.min()), str(EARLY.scan_dt.max())],
    "late_date_range": [str(LATE.scan_dt.min()), str(LATE.scan_dt.max())],
    "early_class_counts": {str(k): int(v) for k,v in EARLY.label.value_counts().sort_index().items()},
    "late_class_counts": {str(k): int(v) for k,v in LATE.label.value_counts().sort_index().items()},
    "features_text_used": True,
    "features_html_used_as_raw_html": False,
    "inputs": input_audit,
}
atomic_json(AUDIT_DIR / "PUTRA_TEMPORAL_SPLIT_AUDIT.json", SPLIT_AUDIT)

EARLY.to_parquet(DATA_DIR / "putra_early_pool.parquet", index=False)
LATE.to_parquet(DATA_DIR / "putra_late_test.parquet", index=False)

print(json.dumps(SPLIT_AUDIT, indent=2, ensure_ascii=False))

# CELL 6
# 03 — Encoderquellen robust auflösen

def resolve_checkpoint(preferred, needle):
    p = Path(preferred)
    if p.exists():
        return str(p)

    hits = []
    for cfg in SEARCH_ROOT.rglob("config.json"):
        d = cfg.parent
        if needle.lower() in str(d).lower() and ((d/"model.safetensors").exists() or (d/"pytorch_model.bin").exists()):
            hits.append(d)

    if not hits:
        raise RuntimeError(f"Checkpoint nicht gefunden: {needle}")
    hits.sort(key=lambda x: (len(str(x)), str(x)))
    return str(hits[0])

def resolve_base_model(name):
    hits = []
    for cfg in SEARCH_ROOT.rglob("config.json"):
        d = cfg.parent
        if d.name == name and ((d/"model.safetensors").exists() or (d/"pytorch_model.bin").exists()):
            hits.append(d)
    if hits:
        hits.sort(key=lambda x: (len(str(x)), str(x)))
        return str(hits[0])
    return name

URL_R0 = resolve_base_model("bert-base-uncased")
TEXT_R0 = resolve_base_model("roberta-base")

URL_RUT = resolve_checkpoint(
    "/kaggle/input/newdataset/final_url_ssl_integrated_v2/checkpoints/URL_DAPT_BERT_200K",
    "URL_DAPT_BERT_200K"
)
TEXT_RUT = resolve_checkpoint(
    "/kaggle/input/datasets/cristinakaufalt/bigresults/phreshphish_FINAL_DEEP_TRIMODAL_SSL_v4_3_LABELSET_ALIGNMENT_FIX/checkpoints/R1_DAPT_TEXT",
    "R1_DAPT_TEXT"
)

# Text-DAPT nutzt weiterhin den RoBERTa-Tokenizer.
TEXT_TOKENIZER = TEXT_R0

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN and Path("/kaggle/working").exists():
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        HF_TOKEN = None

SOURCE_AUDIT = {
    "R0": {"URL": URL_R0, "TEXT": TEXT_R0},
    "RUT": {"URL": URL_RUT, "TEXT": TEXT_RUT},
    "text_tokenizer": TEXT_TOKENIZER,
}
atomic_json(AUDIT_DIR / "REPRESENTATION_SOURCES.json", SOURCE_AUDIT)
print(json.dumps(SOURCE_AUDIT, indent=2))

# CELL 8
# 04 — Embeddings für EARLY und LATE

def masked_mean(h, m):
    mm = m.unsqueeze(-1).to(h.dtype)
    return (h*mm).sum(1) / mm.sum(1).clamp_min(1)

@torch.no_grad()
def embed_values(values, model_src, tokenizer_src, max_len, batch_size, out_path, revision=None):
    out_path = Path(out_path)

    if out_path.exists():
        a = np.load(out_path, mmap_mode="r")
        if a.shape == (len(values), 768):
            print({"REUSE_EMBED": out_path.name, "shape": a.shape})
            return out_path

    tok_kwargs = {"token": HF_TOKEN}
    mdl_kwargs = {"token": HF_TOKEN}
    if revision is not None and model_src == "bert-base-uncased":
        tok_kwargs["revision"] = revision
        mdl_kwargs["revision"] = revision

    tok = AutoTokenizer.from_pretrained(tokenizer_src, **tok_kwargs)
    model = AutoModel.from_pretrained(model_src, **mdl_kwargs).to(DEVICE).eval()

    if int(model.config.hidden_size) != 768:
        raise RuntimeError(f"Unerwartete Hidden-Dimension {model.config.hidden_size} für {model_src}")

    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(len(values), 768))
    vals = pd.Series(values).fillna("").astype(str).tolist()

    for st in range(0, len(vals), batch_size):
        en = min(st+batch_size, len(vals))
        b = tok(
            vals[st:en],
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        b = {k:v.to(DEVICE) for k,v in b.items()}

        with torch.autocast("cuda", dtype=torch.float16):
            o = model(**b)
            z = masked_mean(o.last_hidden_state, b["attention_mask"])

        arr[st:en] = z.float().cpu().numpy()

        if en % 2000 < batch_size or en == len(vals):
            arr.flush()
            print({"embed": out_path.name, "rows": en, "total": len(vals)})

    arr.flush()
    del arr, model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return out_path

VARIANTS = {
    "R0": {
        "url_model": URL_R0,
        "url_tok": URL_R0,
        "text_model": TEXT_R0,
        "text_tok": TEXT_TOKENIZER,
    },
    "RUT": {
        "url_model": URL_RUT,
        "url_tok": URL_RUT,
        "text_model": TEXT_RUT,
        "text_tok": TEXT_TOKENIZER,
    },
}

for split_name, frame in [("EARLY", EARLY), ("LATE", LATE)]:
    for variant, cfg in VARIANTS.items():
        embed_values(
            frame["url"], cfg["url_model"], cfg["url_tok"],
            URL_MAX_LEN, 128, EMB_DIR/f"{variant}_URL_{split_name}.npy",
            BERT_REVISION if (variant=="R0" and cfg["url_model"]=="bert-base-uncased") else None
        )
        embed_values(
            frame["text"], cfg["text_model"], cfg["text_tok"],
            TEXT_MAX_LEN, 64, EMB_DIR/f"{variant}_TEXT_{split_name}.npy"
        )

def X_of(variant, split):
    u = np.load(EMB_DIR/f"{variant}_URL_{split}.npy", mmap_mode="r")
    t = np.load(EMB_DIR/f"{variant}_TEXT_{split}.npy", mmap_mode="r")
    return np.concatenate([np.asarray(u,np.float32), np.asarray(t,np.float32)], axis=1)

XEARLY = {v:X_of(v,"EARLY") for v in VARIANTS}
XLATE = {v:X_of(v,"LATE") for v in VARIANTS}
y_early = EARLY["label"].to_numpy(int)
y_late = LATE["label"].to_numpy(int)

print({
    "EARLY": {k:v.shape for k,v in XEARLY.items()},
    "LATE": {k:v.shape for k,v in XLATE.items()},
})

# CELL 10
# 05 — Sampling, Linear-Probes und Late-Test-Metriken

def stable_u64(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)

EARLY_IDS = (
    EARLY["url"].astype(str)
    + "|"
    + EARLY["scan_dt"].astype(str)
    + "|"
    + EARLY["text"].astype(str).str.slice(0,256)
).tolist()

def ranked_indices(seed):
    by_class = {}
    for lab in [0,1]:
        idx = np.where(y_early == lab)[0]
        ranked = sorted(
            idx.tolist(),
            key=lambda i: stable_u64(f"PUTRA_TARGET|{seed}|{lab}|{EARLY_IDS[i]}")
        )
        by_class[lab] = ranked
    return by_class

def indices_for_budget(ranked, budget):
    n = budget//2
    idx = np.asarray(ranked[0][:n] + ranked[1][:n], dtype=np.int32)
    return np.sort(idx)

def sample_hash(idx):
    h = hashlib.sha256()
    for i in sorted(np.asarray(idx,dtype=int).tolist()):
        h.update(EARLY_IDS[i].encode())
        h.update(b"\n")
    return h.hexdigest()

def precision_at_recall(y, s, target=.90):
    p,r,_ = precision_recall_curve(y,s)
    ok = np.where(r>=target)[0]
    return float(np.max(p[ok])) if len(ok) else np.nan

def ranking_metrics(y, s):
    y=np.asarray(y,int)
    s=np.asarray(s,float)
    fpr,tpr,_=roc_curve(y,s)
    ii=np.where(tpr>=.90)[0]
    return {
        "AP": float(average_precision_score(y,s)),
        "AUC": float(roc_auc_score(y,s)),
        "P_at_R90": precision_at_recall(y,s,.90),
        "FPR_at_TPR90": float(fpr[ii[0]]) if len(ii) else np.nan,
    }

rows=[]
selection_audit=[]

for run,(rank_seed,model_seed) in enumerate(zip(LABEL_RANK_SEEDS,MODEL_SEEDS)):
    ranked = ranked_indices(rank_seed)
    previous = set()

    for budget in LABEL_BUDGETS:
        idx = indices_for_budget(ranked,budget)
        idxset = set(idx.tolist())

        if previous and not previous.issubset(idxset):
            raise RuntimeError(f"Nicht verschachtelte Budgets in Run {run}.")
        previous = idxset

        ytr = y_early[idx]
        if int((ytr==0).sum()) != budget//2 or int((ytr==1).sum()) != budget//2:
            raise RuntimeError(f"Klassenbalance verletzt: run={run}, budget={budget}")

        sh = sample_hash(idx)

        for variant in ["R0","RUT"]:
            clf = LogisticRegression(
                C=LINEAR_SPEC["C"],
                solver=LINEAR_SPEC["solver"],
                class_weight=LINEAR_SPEC["class_weight"],
                max_iter=LINEAR_SPEC["max_iter"],
                random_state=model_seed,
            )
            clf.fit(XEARLY[variant][idx], ytr)
            score = clf.decision_function(XLATE[variant]).astype(np.float32)

            rows.append({
                "run": run,
                "label_rank_seed": rank_seed,
                "model_seed": model_seed,
                "budget": budget,
                "variant": variant,
                "sample_hash": sh,
                "n_train": len(idx),
                "n_test": len(y_late),
                **ranking_metrics(y_late, score),
            })

        selection_audit.append({
            "run": run,
            "budget": budget,
            "sample_hash": sh,
            "rows": len(idx),
        })

    print({"run_complete": run, "max_budget_hash": selection_audit[-1]["sample_hash"][:12]})

RES = pd.DataFrame(rows).sort_values(["run","budget","variant"]).reset_index(drop=True)
RES.to_csv(RESULTS_DIR / "PUTRA_R0_VS_RUT_N10_RUN_METRICS.csv", index=False)
atomic_json(AUDIT_DIR / "LABEL_SELECTION_AUDIT.json", selection_audit)

pair = RES.pivot_table(
    index=["run","budget"],
    columns="variant",
    values="sample_hash",
    aggfunc="first"
)
if not (pair["R0"] == pair["RUT"]).all():
    raise RuntimeError("R0/RUT-Paarung verletzt.")

display(RES[["run","budget","variant","AP","AUC","P_at_R90","FPR_at_TPR90"]])

# CELL 11
# 06 — Optionaler Reproduktionscheck gegen den vorherigen Putra-E0_SOURCE_RUT-Zweig

prior_files = list(CACHE_DIR.rglob("PUTRA_TARGET_ADAPTATION_N10_RUN_METRICS.csv"))
REPRO_AUDIT = {"prior_result_found": bool(prior_files)}

if prior_files:
    prior = pd.read_csv(prior_files[0])
    prior = prior[prior["variant"].astype(str).eq("E0_SOURCE_RUT")].copy()

    cur = RES[RES["variant"].eq("RUT")].copy()

    merged = cur.merge(
        prior,
        on=["run","budget"],
        suffixes=("_current","_prior"),
        how="inner"
    )

    metric_checks = {}
    for metric in ["AP","AUC","P_at_R90","FPR_at_TPR90"]:
        if f"{metric}_current" in merged and f"{metric}_prior" in merged:
            diff = np.abs(
                merged[f"{metric}_current"].to_numpy(float)
                - merged[f"{metric}_prior"].to_numpy(float)
            )
            metric_checks[metric] = {
                "n": len(diff),
                "max_abs_difference": float(np.max(diff)) if len(diff) else None,
            }

    REPRO_AUDIT.update({
        "prior_file": str(prior_files[0]),
        "matched_run_budget_rows": int(len(merged)),
        "metric_checks": metric_checks,
        "reproduces_within_1e-8": all(
            v["n"] == N_REPS*len(LABEL_BUDGETS)
            and v["max_abs_difference"] <= 1e-8
            for v in metric_checks.values()
        ) if metric_checks else False,
    })

atomic_json(AUDIT_DIR / "RUT_REPRODUCTION_AUDIT.json", REPRO_AUDIT)
print(json.dumps(REPRO_AUDIT, indent=2))

# CELL 13
# 07 — Exakter Sign-Flip, 95%-KI und Holm

def exact_signflip_p(diff):
    d=np.asarray(diff,float)
    d=d[np.isfinite(d)]
    obs=abs(d.mean())
    vals=[
        abs(np.mean(d*np.asarray(signs)))
        for signs in itertools.product([-1,1], repeat=len(d))
    ]
    return float(np.mean(np.asarray(vals) >= obs-1e-15))

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

# Primärer Endpunkt: AP-AULC über log10(Budget)
x=np.log10(np.asarray(LABEL_BUDGETS,float))
x=(x-x.min())/(x.max()-x.min())

aulc_rows=[]
for run in range(N_REPS):
    for variant in ["R0","RUT"]:
        q=RES[(RES.run==run)&(RES.variant==variant)].set_index("budget").loc[LABEL_BUDGETS]
        aulc=float(np.trapezoid(q["AP"].to_numpy(float), x))
        aulc_rows.append({"run":run,"variant":variant,"AP_AULC":aulc})

AULC=pd.DataFrame(aulc_rows)
piv=AULC.pivot(index="run",columns="variant",values="AP_AULC").sort_index()
d=(piv["RUT"]-piv["R0"]).to_numpy(float)
m,lo,hi=mean_ci95(d)

PRIMARY=pd.DataFrame([{
    "endpoint":"AP_AULC_log_budget",
    "n_pairs":N_REPS,
    "R0_mean":float(piv["R0"].mean()),
    "RUT_mean":float(piv["RUT"].mean()),
    "mean_delta":m,
    "mean_delta_pp":100*m,
    "ci95_lo":lo,
    "ci95_hi":hi,
    "ci95_lo_pp":100*lo,
    "ci95_hi_pp":100*hi,
    "positive_pairs":int((d>0).sum()),
    "negative_pairs":int((d<0).sum()),
    "zero_pairs":int((d==0).sum()),
    "exact_signflip_p":exact_signflip_p(d),
}])
PRIMARY.to_csv(RESULTS_DIR / "PUTRA_R0_VS_RUT_PRIMARY_AP_AULC.csv", index=False)

# Sekundäre AP-Familie je Budget
sec=[]
for budget in LABEL_BUDGETS:
    q=RES[RES.budget.eq(budget)].pivot(index="run",columns="variant",values="AP").sort_index()
    dd=(q["RUT"]-q["R0"]).to_numpy(float)
    mm,ll,hh=mean_ci95(dd)
    sec.append({
        "budget":budget,
        "R0_AP_mean":float(q["R0"].mean()),
        "RUT_AP_mean":float(q["RUT"].mean()),
        "delta_AP":mm,
        "delta_AP_pp":100*mm,
        "ci95_lo":ll,
        "ci95_hi":hh,
        "ci95_lo_pp":100*ll,
        "ci95_hi_pp":100*hh,
        "positive_pairs":int((dd>0).sum()),
        "negative_pairs":int((dd<0).sum()),
        "p_raw":exact_signflip_p(dd),
    })

SECONDARY=pd.DataFrame(sec)
SECONDARY["p_holm"]=holm_adjust(SECONDARY["p_raw"].to_numpy(float))
SECONDARY.to_csv(RESULTS_DIR / "PUTRA_R0_VS_RUT_AP_BY_BUDGET_STATS.csv", index=False)

# Deskriptive Lernkurven
AGG=RES.groupby(["budget","variant"],as_index=False).agg(
    AP_mean=("AP","mean"),
    AP_sd=("AP","std"),
    AUC_mean=("AUC","mean"),
    P_at_R90_mean=("P_at_R90","mean"),
    FPR_at_TPR90_mean=("FPR_at_TPR90","mean"),
)
AGG.to_csv(RESULTS_DIR / "PUTRA_R0_VS_RUT_LEARNING_CURVES.csv", index=False)

display(PRIMARY)
display(SECONDARY)
display(AGG)

# CELL 14
# 08 — Deskriptive Label-Äquivalenz und Abbildungen

# Kleinster RUT-Budgetpunkt, dessen mittlere AP mindestens R0@2000 erreicht.
ref = float(
    AGG[
        (AGG.budget==max(LABEL_BUDGETS))
        & (AGG.variant=="R0")
    ]["AP_mean"].iloc[0]
)

eligible = AGG[
    (AGG.variant=="RUT")
    & (AGG.AP_mean >= ref)
].sort_values("budget")

smallest_equiv = int(eligible.iloc[0].budget) if len(eligible) else None

LABEL_EQUIV = {
    "reference": "R0 at 2000 labeled Putra target instances",
    "reference_mean_AP": ref,
    "smallest_RUT_budget_with_mean_AP_at_least_reference": smallest_equiv,
    "inferential_status": "DESCRIPTIVE_ONLY_NO_NONINFERIORITY_CLAIM",
}
atomic_json(RESULTS_DIR / "PUTRA_R0_VS_RUT_LABEL_EQUIVALENCE_DESCRIPTIVE.json", LABEL_EQUIV)
print(LABEL_EQUIV)

import matplotlib.pyplot as plt

fig,ax=plt.subplots(figsize=(8,5))
for variant in ["R0","RUT"]:
    q=AGG[AGG.variant.eq(variant)].sort_values("budget")
    ax.plot(q["budget"],q["AP_mean"],marker="o",label=variant)
ax.set_xscale("log")
ax.set_xlabel("Labeled Putra target instances")
ax.set_ylabel("Average Precision on later Putra test")
ax.set_title("External label efficiency: R0 vs. RUT")
ax.grid(alpha=.2)
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "PUTRA_R0_VS_RUT_AP_LEARNING_CURVE.png",dpi=180)
plt.show()

fig,ax=plt.subplots(figsize=(8,5))
for variant in ["R0","RUT"]:
    q=AGG[AGG.variant.eq(variant)].sort_values("budget")
    ax.plot(q["budget"],100*q["FPR_at_TPR90_mean"],marker="o",label=variant)
ax.set_xscale("log")
ax.set_xlabel("Labeled Putra target instances")
ax.set_ylabel("FPR required for TPR=90% (%)")
ax.set_title("External low-FPR ranking: R0 vs. RUT")
ax.grid(alpha=.2)
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "PUTRA_R0_VS_RUT_FPR_AT_TPR90.png",dpi=180)
plt.show()

# CELL 16
# 09 — Finales Manifest und kompaktes Ergebnispaket

primary = PRIMARY.iloc[0].to_dict()

SUMMARY = {
    "status":"COMPLETE",
    "version":VERSION,
    "scientific_status":"POST_HOC_EXTERNAL_LABEL_EFFICIENCY_ANALYSIS",
    "putra_split":SPLIT_AUDIT,
    "representation_sources":SOURCE_AUDIT,
    "label_budgets":LABEL_BUDGETS,
    "n_pairs":N_REPS,
    "primary":primary,
    "secondary_AP_by_budget":SECONDARY.to_dict(orient="records"),
    "learning_curves":AGG.to_dict(orient="records"),
    "label_equivalence_descriptive":LABEL_EQUIV,
    "rut_reproduction_audit":REPRO_AUDIT,
    "guardrails":[
        "R0 is generically self-supervised pretrained; contrast is additional PhreshPhish task-adaptive representation learning.",
        "Putra late test is never used for probe fitting.",
        "R0 and RUT receive identical labeled target rows within each run and budget.",
        "N10 quantifies target label-sample/probe variation at fixed encoder checkpoints.",
        "No Putra target MLM is performed in this experiment.",
        "Putra features.text is provider-preprocessed text, not raw HTML processed by the PhreshPhish pipeline.",
        "Label equivalence is descriptive only.",
        "Prior Putra analyses were known before this experiment; interpret as post-hoc external validation.",
    ],
}
atomic_json(RESULTS_DIR / "PUTRA_R0_VS_RUT_EXTERNAL_LABEL_EFFICIENCY_SUMMARY.json", SUMMARY)
print(json.dumps(SUMMARY, indent=2, ensure_ascii=False))

zip_path = ROOT / "PUTRA_EXTERNAL_LABEL_EFFICIENCY_R0_VS_RUT_N10_RESULTS_v1_1.zip"
with zipfile.ZipFile(zip_path,"w",compression=zipfile.ZIP_DEFLATED) as zf:
    for folder in [RESULTS_DIR,AUDIT_DIR,FIG_DIR]:
        for p in folder.rglob("*"):
            if p.is_file():
                zf.write(p,p.relative_to(ROOT))

print({"RESULTS_ZIP":str(zip_path)})