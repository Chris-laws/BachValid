# KI-Unterstuetzung: siehe Erklaerung der Arbeit.
# Ursprungsdatei: FINAL_SSL_LABEL_SUFFICIENCY_N10_FF1_FF3_RESOURCE_LINK_v5_DIRECT_CACHE_RESOLVER.ipynb
# CELL 1
# 00 — Imports, immutable analysis plan, directories

import os, gc, re, json, math, time, random, hashlib, itertools, shutil, zipfile, warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score, roc_curve,
    precision_recall_curve
)
from scipy.stats import beta as beta_dist
from scipy.stats import t as student_t

warnings.filterwarnings("ignore")

IS_KAGGLE = Path("/kaggle/working").exists()
SEARCH_ROOT = Path("/kaggle/input") if Path("/kaggle/input").exists() else Path("/mnt/data")
WORK_BASE = Path("/kaggle/working") if IS_KAGGLE else Path("/mnt/data")

VERSION = "FINAL_SSL_LABEL_SUFFICIENCY_N10_FF1_FF3_RESOURCE_LINK_v5_DIRECT_CACHE_RESOLVER"
SCIENTIFIC_STATUS = "N5_TO_N10_EXTENSION_WITH_PRE_SPECIFIED_SUFFICIENCY_MARGIN_DIRECT_CACHE_RESOLVER_V5"

ROOT = WORK_BASE / "final_ssl_label_sufficiency_n10_v1"
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
AUDIT = ROOT / "audit"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"
PARTS = ROOT / "new_n5_parts"

for p in [ROOT, RESULTS, TABLES, AUDIT, MODELS, FIGURES, PARTS]:
    p.mkdir(parents=True, exist_ok=True)

PRIMARY_FPR = 0.005
NI_MARGIN_TPR = 0.01               # 1.0 percentage point
NI_MARGIN_TPR_PP = 1.0
ALPHA = 0.05

REFERENCE_REP = "R0"
REFERENCE_BUDGET = 200_000
CANDIDATE_REP = "RUT"
CANDIDATE_BUDGETS = [10_000, 20_000, 50_000, 100_000]
EQUAL_BUDGET = 20_000

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

# Existing canonical N5 curve seeds.
HIST_MODEL_SEEDS = [42, 62, 82, 102, 122]
HIST_LABEL_RANK_SEEDS = [20260813, 20260823, 20260833, 20260843, 20260853]

# Five genuinely new paired runs for this extension.
NEW_MODEL_SEEDS = [342, 362, 382, 402, 422]
NEW_LABEL_RANK_SEEDS = [9301, 9302, 9303, 9304, 9305]

if set(HIST_MODEL_SEEDS) & set(NEW_MODEL_SEEDS):
    raise RuntimeError("Historical/new model seeds overlap.")
if set(HIST_LABEL_RANK_SEEDS) & set(NEW_LABEL_RANK_SEEDS):
    raise RuntimeError("Historical/new label-rank seeds overlap.")

NEW_CONDITIONS = [
    ("R0", 20_000),
    ("R0", 200_000),
    ("RUT", 10_000),
    ("RUT", 20_000),
    ("RUT", 50_000),
    ("RUT", 100_000),
]

EXPECTED_COUNTS = {
    "SSL": 200_000,
    "CAL": 50_000,
    "FINAL": 168_060,
}
EXPECTED_FINAL_CLASSES = (91_260, 76_800)

def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)

ANALYSIS_PLAN = {
    "version": VERSION,
    "scientific_status": SCIENTIFIC_STATUS,
    "written_before_new_fits": True,
    "primary_target_fpr": PRIMARY_FPR,
    "equal_budget_confirmation": {
        "contrast": "RUT_20k - R0_20k",
        "primary_scenario": "OFFICIAL_TEST",
        "test": "paired exact two-sided sign-flip",
        "secondary_scenarios": OOD_FAMILY,
        "secondary_adjustment": "Holm across four shift/Late scenarios",
    },
    "label_sufficiency": {
        "reference": "R0_200k",
        "candidates": [f"RUT_{b}" for b in CANDIDATE_BUDGETS],
        "margin_tpr_pp": NI_MARGIN_TPR_PP,
        "component_test": "paired exact one-sided sign-flip on d + 0.01",
        "alpha": ALPHA,
        "global_decision": "intersection-union: all five scenario component tests must reject",
        "fpr_role": "reported secondary operational guardrail; not part of primary NI decision",
    },
    "n10_construction": {
        "historical_n": 5,
        "new_n": 5,
        "historical_model_seeds": HIST_MODEL_SEEDS,
        "historical_label_rank_seeds": HIST_LABEL_RANK_SEEDS,
        "new_model_seeds": NEW_MODEL_SEEDS,
        "new_label_rank_seeds": NEW_LABEL_RANK_SEEDS,
        "independent_external_replication": False,
        "first_n5_observed_before_current_sufficiency_plan": True,
    },
    "method_lock": {
        "representation": "canonical integrated R0/RUT URL+text pooled embeddings",
        "probe": "sklearn LogisticRegression C=1 max_iter=2000 solver=liblinear class_weight=balanced",
        "threshold": "CAL benign-only threshold at target FPR 0.5%",
        "transformers_retrained": False,
        "missing_embedding_policy": "FAIL_FAST",
    },
}
atomic_json(AUDIT / "ANALYSIS_PLAN_LOCK.json", ANALYSIS_PLAN)

print(json.dumps(ANALYSIS_PLAN, indent=2))
print({
    "SEARCH_ROOT": str(SEARCH_ROOT),
    "NEW_FITS_PLANNED": len(NEW_MODEL_SEEDS) * len(NEW_CONDITIONS),
    "analysis_plan_locked": True,
})

# CELL 2
# 01 — Canonical roles + private labels from finaldatafreeze; sealed OOD flags resolved separately
#
# Verified from prior successful thesis runs:
# DATA_ROOT:
# /kaggle/input/datasets/cristinakaufalt/finaldatafreeze/phreshphish_FINAL_DATA_FREEZE_v3
#
# OOD flags:
# /kaggle/input/datasets/cristinakaufalt/freezeresume/manifests/final_test_exact_ood_flags_SEALED.parquet
#
# v2 incorrectly required the flag file to live inside finaldatafreeze and therefore rejected
# a perfectly valid attached canonical freeze. v3 separates these provenance roots.

ROLE_REL = {
    "SSL": Path("roles/ssl_pool_200k"),
    "CAL": Path("roles/fpr_calibration_benign"),
    "FINAL": Path("roles/final_test"),
}

def role_nrows(d):
    fs = sorted(Path(d).glob("*.parquet"))
    if not fs:
        raise RuntimeError(f"No parquet shards under {d}")
    return sum(pq.ParquetFile(f).metadata.num_rows for f in fs)

def validate_freeze_without_flags(root):
    root = Path(root)
    dirs = {k: root / v for k, v in ROLE_REL.items()}
    if not all(p.exists() and p.is_dir() for p in dirs.values()):
        return None, "ROLE_DIR_MISSING"

    cnt = {k: role_nrows(v) for k, v in dirs.items()}
    if cnt != EXPECTED_COUNTS:
        return None, f"ROLE_COUNTS_MISMATCH:{cnt}"

    priv = root / "manifests/train_role_manifest_PRIVATE_WITH_LABELS.parquet"
    if not priv.exists():
        return None, "PRIVATE_MANIFEST_MISSING"

    return cnt, "PASS"

def add_freeze_candidate(cands, root, score, reason):
    root = Path(root)
    if not root.exists():
        return
    try:
        cnt, why = validate_freeze_without_flags(root)
        if cnt is not None:
            cands.append((score, root, cnt, reason))
        else:
            print({"freeze_candidate_rejected": str(root), "reason": why})
    except Exception as e:
        print({"freeze_candidate_rejected": str(root), "reason": repr(e)})

try:
    top_level_inputs = sorted(str(p) for p in SEARCH_ROOT.iterdir())
except Exception:
    top_level_inputs = []
print({"KAGGLE_INPUT_TOP_LEVEL": top_level_inputs[:100]})

# -------------------------------------------------------------------------
# A. Canonical role freeze
# -------------------------------------------------------------------------
freeze_candidates = []

KNOWN_FREEZE_PATHS = [
    Path("/kaggle/input/datasets/cristinakaufalt/finaldatafreeze/phreshphish_FINAL_DATA_FREEZE_v3"),
    Path("/kaggle/input/finaldatafreeze/phreshphish_FINAL_DATA_FREEZE_v3"),
    SEARCH_ROOT / "datasets/cristinakaufalt/finaldatafreeze/phreshphish_FINAL_DATA_FREEZE_v3",
    SEARCH_ROOT / "finaldatafreeze/phreshphish_FINAL_DATA_FREEZE_v3",
]
for p in KNOWN_FREEZE_PATHS:
    add_freeze_candidate(freeze_candidates, p, 100_000, "KNOWN_CANONICAL_PATH")

manifest_hits = []
try:
    manifest_hits = list(SEARCH_ROOT.rglob("train_role_manifest_PRIVATE_WITH_LABELS.parquet"))
except Exception as e:
    print({"manifest_rglob_error": repr(e)})

for priv in manifest_hits:
    add_freeze_candidate(
        freeze_candidates,
        priv.parent.parent,
        90_000,
        "INFERRED_FROM_PRIVATE_MANIFEST",
    )

for role_dir_name in ["ssl_pool_200k", "fpr_calibration_benign", "final_test"]:
    try:
        for p in SEARCH_ROOT.rglob(role_dir_name):
            if p.is_dir() and p.parent.name == "roles":
                add_freeze_candidate(
                    freeze_candidates,
                    p.parent.parent,
                    80_000,
                    f"INFERRED_FROM_ROLE_{role_dir_name}",
                )
    except Exception as e:
        print({"role_rglob_error": role_dir_name, "error": repr(e)})

if not freeze_candidates:
    diagnostic = {
        "status": "MISSING_CANONICAL_ROLE_FREEZE",
        "expected": str(KNOWN_FREEZE_PATHS[0]),
        "top_level_inputs": top_level_inputs[:100],
        "private_manifest_hits": [str(x) for x in manifest_hits[:20]],
        "action": "Attach finaldatafreeze. No new fit has started.",
    }
    atomic_json(AUDIT / "MISSING_ROLE_FREEZE_DIAGNOSTIC.json", diagnostic)
    print(json.dumps(diagnostic, indent=2))
    raise RuntimeError("Canonical finaldatafreeze roles/private-label manifest could not be validated.")

uniq = {}
for score, root, cnt, reason in freeze_candidates:
    key = str(root.resolve())
    if key not in uniq or score > uniq[key][0]:
        uniq[key] = (score, root, cnt, reason)
freeze_candidates = sorted(
    uniq.values(),
    key=lambda z: (-z[0], len(str(z[1])), str(z[1])),
)

_, DATA_ROOT, FREEZE_COUNTS, FREEZE_RESOLUTION_REASON = freeze_candidates[0]
ROLE_DIR = {k: DATA_ROOT / v for k, v in ROLE_REL.items()}
PRIVATE_MANIFEST = DATA_ROOT / "manifests/train_role_manifest_PRIVATE_WITH_LABELS.parquet"

# -------------------------------------------------------------------------
# B. Sealed OOD flags: separate provenance asset
# -------------------------------------------------------------------------
def validate_ood_flags(path):
    path = Path(path)
    if not path.exists():
        return False, "MISSING"
    try:
        pf = pq.ParquetFile(path)
        cols = set(pf.schema_arrow.names)
        required = {
            "sha256",
            "domain_ood_exact",
            "template_ood_exact",
            "domain_template_ood_exact",
        }
        if not required.issubset(cols):
            return False, f"MISSING_COLUMNS:{sorted(required-cols)}"
        if pf.metadata.num_rows != EXPECTED_COUNTS["FINAL"]:
            return False, f"ROW_COUNT:{pf.metadata.num_rows}"
        return True, "PASS"
    except Exception as e:
        return False, repr(e)

flag_candidates = []
KNOWN_FLAG_PATHS = [
    Path("/kaggle/input/datasets/cristinakaufalt/freezeresume/manifests/final_test_exact_ood_flags_SEALED.parquet"),
    Path("/kaggle/input/freezeresume/manifests/final_test_exact_ood_flags_SEALED.parquet"),
    DATA_ROOT / "manifests/final_test_exact_ood_flags_SEALED.parquet",
]
for p in KNOWN_FLAG_PATHS:
    ok, why = validate_ood_flags(p)
    if ok:
        score = 100_000 if "freezeresume" in str(p).lower() else 80_000
        flag_candidates.append((score, p, "KNOWN_PATH"))
    elif p.exists():
        print({"ood_flag_candidate_rejected": str(p), "reason": why})

try:
    global_flag_hits = list(SEARCH_ROOT.rglob("final_test_exact_ood_flags_SEALED.parquet"))
except Exception as e:
    global_flag_hits = []
    print({"ood_flag_rglob_error": repr(e)})

for p in global_flag_hits:
    ok, why = validate_ood_flags(p)
    if ok:
        score = 90_000 if "freezeresume" in str(p).lower() else 70_000
        flag_candidates.append((score, p, "GLOBAL_DISCOVERY"))
    else:
        print({"ood_flag_candidate_rejected": str(p), "reason": why})

if not flag_candidates:
    diagnostic = {
        "status": "MISSING_SEALED_OOD_FLAGS",
        "expected_primary": str(KNOWN_FLAG_PATHS[0]),
        "hits": [str(x) for x in global_flag_hits],
        "action": "Attach freezeresume containing the sealed FINAL OOD flags. No new fit has started.",
    }
    atomic_json(AUDIT / "MISSING_OOD_FLAGS_DIAGNOSTIC.json", diagnostic)
    print(json.dumps(diagnostic, indent=2))
    raise RuntimeError(
        "Sealed FINAL OOD flags not found. Attach freezeresume; "
        "do not substitute or regenerate masks."
    )

flag_candidates.sort(key=lambda z: (-z[0], len(str(z[1])), str(z[1])))
_, OOD_FLAGS, OOD_FLAG_RESOLUTION_REASON = flag_candidates[0]

# -------------------------------------------------------------------------
# C. Integrated frozen R0/RUT embedding cache — DIRECT FILE VALIDATION
# -------------------------------------------------------------------------

STREAMS = ["URL_BASE", "URL_DAPT", "TEXT_BASE", "TEXT_DAPT"]
ROLES = ["SSL", "CAL", "FINAL"]
EXPECTED_ROWS = {"SSL": 200_000, "CAL": 50_000, "FINAL": 168_060}

def cache_file_set(root):
    root = Path(root)
    emb_root = root / "embeddings"
    paths = {(s, r): emb_root / f"{s}_{r}.npy" for s in STREAMS for r in ROLES}
    return emb_root, paths

def validate_integrated_arrays(root):
    root = Path(root)
    emb_root, paths = cache_file_set(root)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        return False, {"missing": missing}

    audit = {}
    dims = set()
    try:
        for (stream, role), p in paths.items():
            a = np.load(p, mmap_mode="r")
            if a.ndim != 2:
                return False, {"bad_ndim": str(p), "shape": list(a.shape)}
            if a.shape[0] != EXPECTED_ROWS[role]:
                return False, {
                    "bad_rows": str(p),
                    "shape": list(a.shape),
                    "expected_rows": EXPECTED_ROWS[role],
                }
            dims.add(int(a.shape[1]))
            audit[f"{stream}_{role}"] = {
                "path": str(p),
                "shape": list(a.shape),
                "dtype": str(a.dtype),
            }
        if len(dims) != 1:
            return False, {"hidden_dims": sorted(dims)}
    except Exception as ex:
        return False, {"exception": repr(ex)}

    marker = root / "FINAL_INTEGRATED_COMPLETE.json"
    marker_meta = None
    marker_status = "ABSENT_BUT_NOT_REQUIRED"
    if marker.exists():
        try:
            marker_meta = json.loads(marker.read_text())
            marker_status = "PRESENT"
        except Exception as ex:
            marker_status = f"PRESENT_UNREADABLE:{repr(ex)}"

    return True, {
        "embedding_audit": audit,
        "hidden_dim": int(next(iter(dims))),
        "marker": str(marker) if marker.exists() else None,
        "marker_status": marker_status,
        "marker_meta": marker_meta,
    }

integrated_candidates = []

# Exact roots documented by prior successful thesis runs.
KNOWN_INTEGRATED_PATHS = [
    Path("/kaggle/input/datasets/cristinakaufalt/newdataset/final_url_ssl_integrated_v2"),
    Path("/kaggle/input/newdataset/final_url_ssl_integrated_v2"),
    SEARCH_ROOT / "datasets/cristinakaufalt/newdataset/final_url_ssl_integrated_v2",
    SEARCH_ROOT / "newdataset/final_url_ssl_integrated_v2",
]
for p in KNOWN_INTEGRATED_PATHS:
    if not p.exists():
        continue
    ok, meta = validate_integrated_arrays(p)
    if ok:
        integrated_candidates.append((100_000, p, "KNOWN_CANONICAL_ARRAY_ROOT", meta))
    else:
        print({"known_integrated_root_rejected": str(p), "details": meta})

# Directly discover the actual canonical SSL array if Kaggle changes the outer mount path.
try:
    ssl_hits = list(SEARCH_ROOT.rglob("URL_DAPT_SSL.npy"))
except Exception as ex:
    ssl_hits = []
    print({"URL_DAPT_SSL_rglob_error": repr(ex)})

for hit in ssl_hits:
    # .../final_url_ssl_integrated_v2/embeddings/URL_DAPT_SSL.npy
    candidate = hit.parent.parent
    ok, meta = validate_integrated_arrays(candidate)
    if ok:
        score = 95_000
        if candidate.name == "final_url_ssl_integrated_v2":
            score += 2_000
        integrated_candidates.append((score, candidate, "INFERRED_FROM_URL_DAPT_SSL_ARRAY", meta))
    else:
        print({"array_inferred_root_rejected": str(candidate), "details": meta})

if not integrated_candidates:
    diagnostic = {
        "status": "MISSING_COMPLETE_CANONICAL_EMBEDDING_SET",
        "known_roots_checked": [str(x) for x in KNOWN_INTEGRATED_PATHS],
        "URL_DAPT_SSL_hits": [str(x) for x in ssl_hits[:50]],
        "required_arrays": [
            f"{s}_{r}.npy" for s in STREAMS for r in ROLES
        ],
        "expected_shapes": {
            "SSL": [200000, "same_hidden_dim"],
            "CAL": [50000, "same_hidden_dim"],
            "FINAL": [168060, "same_hidden_dim"],
        },
        "note": (
            "FINAL_INTEGRATED_COMPLETE.json is intentionally NOT required in v5. "
            "The immutable arrays themselves are validated."
        ),
        "action": "The canonical newdataset mount must contain the 12 frozen R0/RUT arrays.",
    }
    atomic_json(AUDIT / "MISSING_INTEGRATED_ARRAYS_DIAGNOSTIC.json", diagnostic)
    print(json.dumps(diagnostic, indent=2))
    raise RuntimeError("Complete canonical R0/RUT embedding array set not found.")

# Deduplicate by resolved root.
iunq = {}
for score, p, reason, meta in integrated_candidates:
    key = str(p.resolve())
    if key not in iunq or score > iunq[key][0]:
        iunq[key] = (score, p, reason, meta)

integrated_candidates = sorted(
    iunq.values(),
    key=lambda z: (-z[0], len(str(z[1])), str(z[1])),
)
_, INTEGRATED_ROOT, INTEGRATED_RESOLUTION_REASON, INTEGRATED_META = integrated_candidates[0]

EMB_ROOT = INTEGRATED_ROOT / "embeddings"
EMB_PATHS = {(s, r): EMB_ROOT / f"{s}_{r}.npy" for s in STREAMS for r in ROLES}
EMB_AUDIT = INTEGRATED_META["embedding_audit"]
HIDDEN_DIM = int(INTEGRATED_META["hidden_dim"])
FEATURE_DIM = 2 * HIDDEN_DIM
INTEGRATED_MARKER_STATUS = INTEGRATED_META["marker_status"]

print({
    "CANONICAL_EMBEDDINGS": "PASS",
    "root": str(INTEGRATED_ROOT),
    "resolution": INTEGRATED_RESOLUTION_REASON,
    "marker_status": INTEGRATED_MARKER_STATUS,
    "hidden_dim": HIDDEN_DIM,
    "arrays": len(EMB_PATHS),
})

# -------------------------------------------------------------------------
# D. Historical canonical N5 curve — filename-independent schema resolver
# -------------------------------------------------------------------------

HIST_REQUIRED_COLS = {
    "run","model_seed","label_rank_seed","budget","rep","probe",
    "scenario","target_fpr","tpr","fpr","AP","P_at_R90"
}

def historical_table_is_usable(p):
    try:
        q = pd.read_csv(p)
    except Exception:
        return False, None
    if not HIST_REQUIRED_COLS.issubset(set(q.columns)):
        return False, None

    # Must contain the exact historical N5 seed triplets and planned R0/RUT conditions.
    small = q[
        q.probe.astype(str).eq("LINEAR")
        & q.rep.astype(str).isin(["R0","RUT"])
        & q.budget.astype(int).isin([10_000,20_000,50_000,100_000,200_000])
    ].copy()
    got_pairs = set(
        tuple(x) for x in
        small[["run","model_seed","label_rank_seed"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    expected_pairs = set(
        zip(range(5), HIST_MODEL_SEEDS, HIST_LABEL_RANK_SEEDS)
    )
    if not expected_pairs.issubset(got_pairs):
        return False, None
    return True, q

hist_candidates = []

# Preferred known/canonical filenames first.
preferred_names = [
    "FF2_LABEL_CURVE_FINAL_ALL.csv",
    "FF3_STRICT_SHIFT_ALL_BUDGETS_FINAL.csv",
    "FF3_STRICT_SHIFT_ALL_BUDGETS.csv",
    "LABEL_CURVE_FINAL_ALL.csv",
]
for name in preferred_names:
    try:
        hits = list(SEARCH_ROOT.rglob(name))
    except Exception:
        hits = []
    for p in hits:
        ok, _ = historical_table_is_usable(p)
        if ok:
            score = 100_000 if name == "FF2_LABEL_CURVE_FINAL_ALL.csv" else 95_000
            hist_candidates.append((score, p, f"KNOWN_FILENAME:{name}"))

# If exact names differ, inspect only likely curve/shift CSVs, not every CSV in Kaggle.
if not hist_candidates:
    likely = []
    for pattern in [
        "*LABEL*CURVE*.csv", "*label*curve*.csv",
        "*STRICT*SHIFT*.csv", "*strict*shift*.csv",
        "*ALL*BUDGET*.csv", "*all*budget*.csv",
    ]:
        try:
            likely.extend(SEARCH_ROOT.rglob(pattern))
        except Exception:
            pass

    seen = set()
    for p in likely:
        sp = str(p)
        if sp in seen:
            continue
        seen.add(sp)
        ok, _ = historical_table_is_usable(p)
        if ok:
            hist_candidates.append((80_000, p, "SCHEMA_AND_SEED_VALIDATED_DISCOVERY"))

if not hist_candidates:
    diagnostic = {
        "status": "MISSING_HISTORICAL_N5_TABLE",
        "required_columns": sorted(HIST_REQUIRED_COLS),
        "required_historical_seed_pairs": list(
            zip(range(5), HIST_MODEL_SEEDS, HIST_LABEL_RANK_SEEDS)
        ),
        "action": (
            "Attach the canonical integrated/strict-shift result containing the historical "
            "N5 all-budget R0/RUT linear table. No historical row will be fabricated."
        ),
    }
    atomic_json(AUDIT / "MISSING_HISTORICAL_N5_DIAGNOSTIC.json", diagnostic)
    print(json.dumps(diagnostic, indent=2))
    raise RuntimeError("No schema+seed validated historical N5 curve table found.")

hist_candidates.sort(key=lambda z: (-z[0], len(str(z[1])), str(z[1])))
_, HIST_CURVE_PATH, HIST_RESOLUTION_REASON = hist_candidates[0]

print({
    "HISTORICAL_N5_TABLE": "PASS",
    "path": str(HIST_CURVE_PATH),
    "resolution": HIST_RESOLUTION_REASON,
})

INPUT_RESOLUTION = {
    "data_root": str(DATA_ROOT),
    "freeze_resolution_reason": FREEZE_RESOLUTION_REASON,
    "freeze_counts": FREEZE_COUNTS,
    "private_manifest": str(PRIVATE_MANIFEST),
    "ood_flags": str(OOD_FLAGS),
    "ood_flag_resolution_reason": OOD_FLAG_RESOLUTION_REASON,
    "integrated_root": str(INTEGRATED_ROOT),
    "integrated_resolution_reason": INTEGRATED_RESOLUTION_REASON,
    "integrated_marker_status": INTEGRATED_MARKER_STATUS,
    "embedding_root": str(EMB_ROOT),
    "historical_curve": str(HIST_CURVE_PATH),
    "historical_curve_resolution_reason": HIST_RESOLUTION_REASON,
    "hidden_dim": HIDDEN_DIM,
    "feature_dim": FEATURE_DIM,
    "embedding_cache_complete": True,
    "transformer_recompute_allowed": False,
}
atomic_json(AUDIT / "INPUT_RESOLUTION.json", INPUT_RESOLUTION)
atomic_json(AUDIT / "EMBEDDING_CACHE_AUDIT.json", EMB_AUDIT)

print("INPUT RESOLUTION: PASS")
print(json.dumps(INPUT_RESOLUTION, indent=2))

# CELL 3
# 02 — Labels, exact physical SSL order, frozen FINAL scenario masks

def y01(s):
    x = pd.Series(s)
    if pd.api.types.is_numeric_dtype(x):
        return x.astype(int).to_numpy()
    m = {
        "benign":0, "legitimate":0, "legit":0, "0":0, "false":0,
        "phish":1, "phishing":1, "malicious":1, "1":1, "true":1,
    }
    return x.astype(str).str.strip().str.lower().map(m).astype(int).to_numpy()

def read_role(role, columns):
    parts = []
    for p in sorted(ROLE_DIR[role].glob("*.parquet")):
        names = set(pq.ParquetFile(p).schema_arrow.names)
        miss = [c for c in columns if c not in names]
        if miss:
            raise RuntimeError(f"{p}: missing {miss}")
        parts.append(pd.read_parquet(p, columns=columns))
    q = pd.concat(parts, ignore_index=True)
    if len(q) != EXPECTED_ROWS[role]:
        raise RuntimeError(f"{role} row mismatch")
    return q

# Private labels are rejoined only for downstream supervision.
priv = pd.read_parquet(PRIVATE_MANIFEST, columns=["sha256","label","role"])
priv["sha256"] = priv.sha256.astype(str).str.lower()
role_upper = priv.role.astype(str).str.upper()
ssl_priv = priv[role_upper.isin(["SSL_POOL","SSL","SSL_UNLABELED_200K"])][["sha256","label"]].copy()
if len(ssl_priv) != 200_000 or ssl_priv.sha256.duplicated().any():
    raise RuntimeError("Private SSL label manifest invalid.")
label_map = dict(zip(ssl_priv.sha256, y01(ssl_priv.label)))

ssl_sha = []
for p in sorted(ROLE_DIR["SSL"].glob("*.parquet")):
    names = set(pq.ParquetFile(p).schema_arrow.names)
    if "label" in names:
        raise RuntimeError("Physical SSL shard unexpectedly contains labels.")
    q = pd.read_parquet(p, columns=["sha256"])
    ssl_sha.extend(q.sha256.astype(str).str.lower().tolist())

if len(ssl_sha) != 200_000 or len(set(ssl_sha)) != 200_000:
    raise RuntimeError("Physical SSL SHA audit failed.")
if set(ssl_sha) != set(label_map):
    raise RuntimeError("SSL physical/private SHA sets differ.")

TRAIN_Y = np.asarray([label_map[s] for s in ssl_sha], dtype=np.int8)

CAL = read_role("CAL", ["sha256","label"])
CAL["sha256"] = CAL.sha256.astype(str).str.lower()
CAL_Y = y01(CAL.label)
if int(CAL_Y.sum()) != 0:
    raise RuntimeError("CAL is expected to contain only benign pages.")

FINAL = read_role("FINAL", ["sha256","label","date"])
FINAL["sha256"] = FINAL.sha256.astype(str).str.lower()
FINAL_Y = y01(FINAL.label)
class_counts = (int((FINAL_Y==0).sum()), int((FINAL_Y==1).sum()))
if class_counts != EXPECTED_FINAL_CLASSES:
    raise RuntimeError({"final_class_counts": class_counts, "expected": EXPECTED_FINAL_CLASSES})

if set(ssl_sha) & set(CAL.sha256) or set(ssl_sha) & set(FINAL.sha256) or set(CAL.sha256) & set(FINAL.sha256):
    raise RuntimeError("Role SHA overlap detected.")

flags = pd.read_parquet(OOD_FLAGS)
flags["sha256"] = flags.sha256.astype(str).str.lower()
flags = flags.set_index("sha256").loc[FINAL.sha256].reset_index()

dt = pd.to_datetime(FINAL.date, errors="coerce")
late_cut = dt.dropna().quantile(.75)
late = (dt >= late_cut).fillna(False).to_numpy()

FINAL_MASKS = {
    "OFFICIAL_TEST": np.ones(len(FINAL), dtype=bool),
    "DOMAIN_OOD_EXACT": flags.domain_ood_exact.to_numpy(bool),
    "TEMPLATE_OOD_EXACT": flags.template_ood_exact.to_numpy(bool),
    "DOMAIN_TEMPLATE_OOD_EXACT": flags.domain_template_ood_exact.to_numpy(bool),
    "LATE_TEST_Q4": late,
}
EXPECTED_SCENARIO_COUNTS = {
    "OFFICIAL_TEST": 168060,
    "DOMAIN_OOD_EXACT": 115917,
    "TEMPLATE_OOD_EXACT": 161223,
    "DOMAIN_TEMPLATE_OOD_EXACT": 110095,
    "LATE_TEST_Q4": 46784,
}
got = {k:int(v.sum()) for k,v in FINAL_MASKS.items()}
if got != EXPECTED_SCENARIO_COUNTS:
    raise RuntimeError({"scenario_counts": got, "expected": EXPECTED_SCENARIO_COUNTS})

DATA_AUDIT = {
    "ssl_rows": len(TRAIN_Y),
    "ssl_positive_rate": float(TRAIN_Y.mean()),
    "cal_rows": len(CAL_Y),
    "cal_phish": int(CAL_Y.sum()),
    "final_rows": len(FINAL_Y),
    "final_class_counts": class_counts,
    "scenario_counts": got,
    "late_q4_cutoff": str(late_cut),
    "role_sha_disjoint": True,
}
atomic_json(AUDIT / "DATA_AND_SCENARIO_AUDIT.json", DATA_AUDIT)
print(json.dumps(DATA_AUDIT, indent=2))

# CELL 4
# 03 — Metrics, paired exact tests, nested balanced label selections

def thr_fpr(neg_scores, f):
    s = np.asarray(neg_scores, dtype=np.float64)
    if not np.isfinite(s).all():
        raise RuntimeError("Non-finite CAL scores.")
    k = int(math.floor(float(f) * len(s) + 1e-12))
    if k <= 0:
        return float(np.nextafter(s.max(), np.inf))
    ss = np.sort(s)
    return float(np.nextafter(ss[-k], np.inf))

def ci_fpr(fp, n):
    if n == 0:
        return np.nan, np.nan
    return (
        0.0 if fp == 0 else float(beta_dist.ppf(.025, fp, n-fp+1)),
        1.0 if fp == n else float(beta_dist.ppf(.975, fp+1, n-fp)),
    )

def op(y, score, th):
    y = np.asarray(y, int)
    s = np.asarray(score, float)
    p = s >= th
    tp = int((p & (y==1)).sum())
    fp = int((p & (y==0)).sum())
    tn = int((~p & (y==0)).sum())
    fn = int((~p & (y==1)).sum())
    tpr = tp / max(tp+fn, 1)
    fpr = fp / max(fp+tn, 1)
    precision = tp / max(tp+fp, 1)
    lo, hi = ci_fpr(fp, fp+tn)
    return {
        "tpr": tpr, "fpr": fpr, "precision": precision,
        "f1": 2*precision*tpr/max(precision+tpr,1e-12),
        "tp":tp, "fp":fp, "tn":tn, "fn":fn,
        "fp_per_1000":1000*fpr, "fpr_ci_lo":lo, "fpr_ci_hi":hi,
    }

def precision_at_recall(y, s, target=.90):
    p, r, _ = precision_recall_curve(y, s)
    ok = np.where(r >= target)[0]
    return float(np.max(p[ok])) if len(ok) else np.nan

def curves(y, s):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    fpr, tpr, _ = roc_curve(y, s)
    def fat(target):
        ii = np.where(tpr >= target)[0]
        return float(fpr[ii[0]]) if len(ii) else np.nan
    return {
        "AP": float(average_precision_score(y, s)),
        "AUC": float(roc_auc_score(y, s)),
        "P_at_R90": precision_at_recall(y, s, .90),
        "FPR_at_TPR90": fat(.90),
        "FPR_at_TPR95": fat(.95),
    }

def mean_ci95(d):
    d = np.asarray(d, float)
    m = float(d.mean())
    if len(d) < 2:
        return m, np.nan, np.nan
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    q = float(student_t.ppf(.975, len(d)-1))
    return m, m-q*se, m+q*se

def lower_ci95_one_sided(d):
    d = np.asarray(d, float)
    m = float(d.mean())
    if len(d) < 2:
        return np.nan
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    q = float(student_t.ppf(.95, len(d)-1))
    return m - q*se

def exact_signflip_p_two_sided(d):
    d = np.asarray(d, float)
    obs = abs(float(d.mean()))
    vals = []
    for signs in itertools.product([-1,1], repeat=len(d)):
        vals.append(abs(float(np.mean(d*np.asarray(signs)))))
    return float(np.mean(np.asarray(vals) >= obs - 1e-15))

def exact_signflip_p_greater(d):
    d = np.asarray(d, float)
    obs = float(d.mean())
    vals = []
    for signs in itertools.product([-1,1], repeat=len(d)):
        vals.append(float(np.mean(d*np.asarray(signs))))
    return float(np.mean(np.asarray(vals) >= obs - 1e-15))

def holm_adjust(p_values):
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    out = np.empty(len(p), float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p)-rank)*p[idx])
        out[idx] = min(1.0, running)
    return out

def keyed_rank(sha, seed):
    return hashlib.sha256(f"{seed}|{sha}".encode()).hexdigest()

# Precompute class indices once.
SSL_META = pd.DataFrame({"sha256": ssl_sha, "y": TRAIN_Y})
CLASS_RANK_CACHE = {}

def balanced_indices(rank_seed, budget):
    if budget == 200_000:
        return np.arange(200_000, dtype=np.int64)
    key = int(rank_seed)
    if key not in CLASS_RANK_CACHE:
        by = {}
        for cls in [0,1]:
            q = SSL_META[SSL_META.y.eq(cls)][["sha256"]].copy()
            q["idx"] = q.index.to_numpy()
            q["rank"] = [keyed_rank(x, key) for x in q.sha256]
            by[cls] = q.sort_values("rank").idx.to_numpy()
        CLASS_RANK_CACHE[key] = by
    by = CLASS_RANK_CACHE[key]
    n = budget // 2
    ii = np.sort(np.concatenate([by[0][:n], by[1][:n]])).astype(np.int64)
    if len(ii) != budget:
        raise RuntimeError(f"Budget selection failed: seed={rank_seed}, budget={budget}")
    return ii

print("METRICS / STATISTICS / LABEL SELECTIONS: READY")

# CELL 5
# 04 — Load and strictly validate the historical canonical N5 subset

HIST_RAW = pd.read_csv(HIST_CURVE_PATH)

HIST = HIST_RAW[
    HIST_RAW.probe.eq("LINEAR")
    & np.isclose(HIST_RAW.target_fpr.astype(float), PRIMARY_FPR)
    & HIST_RAW.rep.isin(["R0","RUT"])
    & HIST_RAW.budget.isin([10_000,20_000,50_000,100_000,200_000])
    & HIST_RAW.scenario.isin(SCENARIOS)
].copy()

# Canonical N5 must contain exactly the expected seed pairs.
seed_pairs = (
    HIST[["run","model_seed","label_rank_seed"]]
    .drop_duplicates()
    .sort_values("run")
    .reset_index(drop=True)
)
expected_pairs = pd.DataFrame({
    "run": np.arange(5),
    "model_seed": HIST_MODEL_SEEDS,
    "label_rank_seed": HIST_LABEL_RANK_SEEDS,
})
if not seed_pairs.equals(expected_pairs):
    raise RuntimeError({
        "historical_seed_pairs_observed": seed_pairs.to_dict("records"),
        "expected": expected_pairs.to_dict("records"),
    })

# Required historical conditions for the planned analysis.
required_hist = []
for run in range(5):
    for scenario in SCENARIOS:
        required_hist.append((run,"R0",20_000,scenario))
        required_hist.append((run,"R0",200_000,scenario))
        for b in CANDIDATE_BUDGETS:
            required_hist.append((run,"RUT",b,scenario))

observed_hist = set(
    tuple(x) for x in HIST[["run","rep","budget","scenario"]].itertuples(index=False, name=None)
)
missing = [x for x in required_hist if x not in observed_hist]
if missing:
    raise RuntimeError(f"Historical N5 missing planned rows, first={missing[:10]}")

HIST["source"] = "HISTORICAL_CANONICAL_N5"
HIST["prospective_under_current_plan"] = False

HIST.to_csv(RESULTS / "HISTORICAL_N5_REUSED_UNCHANGED.csv", index=False)

atomic_json(AUDIT / "HISTORICAL_N5_AUDIT.json", {
    "source": str(HIST_CURVE_PATH),
    "rows_loaded_for_analysis": len(HIST),
    "runs": sorted(HIST.run.unique().astype(int).tolist()),
    "seed_pairs_match_canonical": True,
    "historical_results_recomputed": False,
})

print(seed_pairs.to_string(index=False))
print({"HISTORICAL_N5": "PASS", "rows": len(HIST)})

# CELL 6
# 05 — Frozen embedding access and memory-efficient linear-probe training/scoring

def rep_streams(rep):
    if rep == "R0":
        return "URL_BASE", "TEXT_BASE"
    if rep == "RUT":
        return "URL_DAPT", "TEXT_DAPT"
    raise KeyError(rep)

def emb(stream, role):
    return np.load(EMB_PATHS[(stream, role)], mmap_mode="r")

def training_matrix(rep, idx):
    us, ts = rep_streams(rep)
    u = emb(us, "SSL")
    t = emb(ts, "SSL")
    idx = np.asarray(idx, dtype=np.int64)
    X = np.empty((len(idx), FEATURE_DIM), dtype=np.float32)
    X[:, :HIDDEN_DIM] = np.asarray(u[idx], dtype=np.float32)
    X[:, HIDDEN_DIM:] = np.asarray(t[idx], dtype=np.float32)
    return X

def manual_linear_score(coef, intercept, rep, role, chunk=32768):
    us, ts = rep_streams(rep)
    u = emb(us, role)
    t = emb(ts, role)
    wu = np.asarray(coef[:HIDDEN_DIM], dtype=np.float32)
    wt = np.asarray(coef[HIDDEN_DIM:], dtype=np.float32)
    n = len(u)
    out = np.empty(n, dtype=np.float32)
    for st in range(0, n, chunk):
        en = min(st+chunk, n)
        uu = np.asarray(u[st:en], dtype=np.float32)
        tt = np.asarray(t[st:en], dtype=np.float32)
        out[st:en] = (
            uu @ wu + tt @ wt + float(intercept)
        ).astype(np.float32)
    return out

def condition_paths(run, rep, budget):
    cdir = MODELS / f"run{run:02d}" / f"{rep}_B{budget}"
    cdir.mkdir(parents=True, exist_ok=True)
    return {
        "dir": cdir,
        "model": cdir / "linear.npz",
        "part": PARTS / f"run{run:02d}_{rep}_B{budget}.csv",
        "done": cdir / "COMPLETE.json",
    }

def fit_and_evaluate_new_condition(run, model_seed, rank_seed, rep, budget):
    P = condition_paths(run, rep, budget)
    spec = {
        "run": int(run),
        "model_seed": int(model_seed),
        "label_rank_seed": int(rank_seed),
        "rep": rep,
        "budget": int(budget),
        "probe": "LINEAR",
        "solver": "liblinear",
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 2000,
        "primary_fpr": PRIMARY_FPR,
        "feature_dim": FEATURE_DIM,
    }

    if P["part"].exists() and P["done"].exists():
        old = json.loads(P["done"].read_text())
        if old.get("spec") == spec:
            return

    idx = balanced_indices(rank_seed, budget)
    ytr = TRAIN_Y[idx].astype(int)

    t0 = time.time()
    Xtr = training_matrix(rep, idx)
    clf = LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver="liblinear",
        class_weight="balanced",
        random_state=int(model_seed),
    ).fit(Xtr, ytr)

    coef = clf.coef_.reshape(-1).astype(np.float32)
    intercept = float(clf.intercept_.reshape(-1)[0])
    if len(coef) != FEATURE_DIM:
        raise RuntimeError("Linear coefficient dimension mismatch.")

    np.savez(P["model"], coef=coef, intercept=np.asarray([intercept], dtype=np.float32))

    del Xtr, clf
    gc.collect()

    cal_score = manual_linear_score(coef, intercept, rep, "CAL")
    final_score = manual_linear_score(coef, intercept, rep, "FINAL")
    th = thr_fpr(cal_score, PRIMARY_FPR)

    rows = []
    for sn, mask in FINAL_MASKS.items():
        rows.append({
            "run": int(run),
            "model_seed": int(model_seed),
            "label_rank_seed": int(rank_seed),
            "budget": int(budget),
            "rep": rep,
            "probe": "LINEAR",
            "dataset": "FINAL",
            "scenario": sn,
            "target_fpr": PRIMARY_FPR,
            "threshold": th,
            **op(FINAL_Y[mask], final_score[mask], th),
            **curves(FINAL_Y[mask], final_score[mask]),
            "source": "NEW_N5_CURRENT_PLAN",
            "prospective_under_current_plan": True,
        })

    pd.DataFrame(rows).to_csv(P["part"], index=False)
    atomic_json(P["done"], {
        "status": "COMPLETE",
        "spec": spec,
        "elapsed_seconds": time.time()-t0,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    })

    del cal_score, final_score, coef
    gc.collect()

    print({
        "NEW_CONDITION_COMPLETE": True,
        "run": run,
        "rep": rep,
        "budget": budget,
        "elapsed_min": round((time.time()-t0)/60, 2),
    })

print("LINEAR PROBE ENGINE: READY")

# CELL 7
# 06 — Add five new paired runs; resumable at condition level

NEW_START = 5
for j, (mseed, rseed) in enumerate(zip(NEW_MODEL_SEEDS, NEW_LABEL_RANK_SEEDS)):
    run = NEW_START + j
    rt0 = time.time()

    for rep, budget in NEW_CONDITIONS:
        fit_and_evaluate_new_condition(
            run=run,
            model_seed=mseed,
            rank_seed=rseed,
            rep=rep,
            budget=budget,
        )

    print({
        "NEW_REPLICATE_COMPLETE": run,
        "model_seed": mseed,
        "label_rank_seed": rseed,
        "replicate_elapsed_min": round((time.time()-rt0)/60, 2),
    })

NEW = pd.concat(
    [
        pd.read_csv(condition_paths(5+j, rep, budget)["part"])
        for j in range(5)
        for rep, budget in NEW_CONDITIONS
    ],
    ignore_index=True,
)

if NEW.run.nunique() != 5:
    raise RuntimeError("New N5 extension incomplete.")
if set(NEW.run.unique()) != set(range(5,10)):
    raise RuntimeError(f"Unexpected new run IDs: {sorted(NEW.run.unique())}")

NEW.to_csv(RESULTS / "NEW_N5_CURRENT_PLAN_ALL.csv", index=False)

print({
    "NEW_N5": "COMPLETE",
    "new_runs": sorted(NEW.run.unique().tolist()),
    "conditions_per_run": len(NEW_CONDITIONS),
    "scenario_rows": len(NEW),
})

# CELL 8
# 07 — Assemble the matched N10 analysis dataset

# Keep only rows needed for the two linked analyses.
HIST_KEEP = HIST[
    (
        (HIST.rep.eq("R0") & HIST.budget.isin([20_000, 200_000]))
        |
        (HIST.rep.eq("RUT") & HIST.budget.isin(CANDIDATE_BUDGETS))
    )
].copy()

N10 = pd.concat([HIST_KEEP, NEW], ignore_index=True, sort=False)

# Exact completeness: each run has 6 conditions x 5 scenarios.
expected_condition_set = {
    ("R0",20_000),
    ("R0",200_000),
    ("RUT",10_000),
    ("RUT",20_000),
    ("RUT",50_000),
    ("RUT",100_000),
}
for run in range(10):
    q = N10[N10.run.eq(run)]
    cset = set(tuple(x) for x in q[["rep","budget"]].drop_duplicates().itertuples(index=False, name=None))
    if cset != expected_condition_set:
        raise RuntimeError({"run":run, "conditions":sorted(cset), "expected":sorted(expected_condition_set)})
    for cond in expected_condition_set:
        z = q[q.rep.eq(cond[0]) & q.budget.eq(cond[1])]
        if set(z.scenario) != set(SCENARIOS) or len(z) != 5:
            raise RuntimeError({"run":run, "condition":cond, "scenario_rows":len(z)})

N10["n10_source_group"] = np.where(N10.run < 5, "HISTORICAL_N5", "NEW_N5")
N10.to_csv(RESULTS / "N10_MATCHED_ANALYSIS_ALL.csv", index=False)

N10_AUDIT = {
    "n_runs": int(N10.run.nunique()),
    "historical_runs": sorted(N10[N10.run<5].run.unique().astype(int).tolist()),
    "new_runs": sorted(N10[N10.run>=5].run.unique().astype(int).tolist()),
    "conditions_per_run": 6,
    "scenarios_per_condition": 5,
    "expected_rows": 10*6*5,
    "observed_rows": len(N10),
    "complete": len(N10) == 10*6*5,
}
atomic_json(AUDIT / "N10_COMPLETENESS_AUDIT.json", N10_AUDIT)
if not N10_AUDIT["complete"]:
    raise RuntimeError(N10_AUDIT)

print(json.dumps(N10_AUDIT, indent=2))

# CELL 9
# 08 — A: Equal-budget RUT20k vs R0 20k N10 confirmation

eq_rows = []
for sn in SCENARIOS:
    q = N10[
        N10.scenario.eq(sn)
        & N10.budget.eq(EQUAL_BUDGET)
        & N10.rep.isin(["R0","RUT"])
    ]

    piv_tpr = q.pivot(index="run", columns="rep", values="tpr").sort_index()
    piv_fpr = q.pivot(index="run", columns="rep", values="fpr").sort_index()
    piv_ap  = q.pivot(index="run", columns="rep", values="AP").sort_index()

    if len(piv_tpr) != 10 or not {"R0","RUT"}.issubset(piv_tpr.columns):
        raise RuntimeError(f"Equal-budget N10 pairing failed: {sn}")

    d = (piv_tpr["RUT"] - piv_tpr["R0"]).to_numpy(float)
    dfpr = (piv_fpr["RUT"] - piv_fpr["R0"]).to_numpy(float)
    dap = (piv_ap["RUT"] - piv_ap["R0"]).to_numpy(float)

    m, lo, hi = mean_ci95(d)
    eq_rows.append({
        "scenario": sn,
        "contrast": "RUT_20k - R0_20k",
        "n_pairs": 10,
        "RUT_tpr_mean": float(piv_tpr["RUT"].mean()),
        "R0_tpr_mean": float(piv_tpr["R0"].mean()),
        "mean_delta_tpr_pp": 100*m,
        "ci95_lo_pp": 100*lo,
        "ci95_hi_pp": 100*hi,
        "positive_pairs": int((d>0).sum()),
        "negative_pairs": int((d<0).sum()),
        "exact_signflip_p_two_sided": exact_signflip_p_two_sided(d),
        "mean_delta_fpr_pp": 100*float(dfpr.mean()),
        "mean_delta_AP_pp": 100*float(dap.mean()),
    })

EQ = pd.DataFrame(eq_rows)
EQ["holm_p_shift_family"] = np.nan
mask = EQ.scenario.isin(OOD_FAMILY)
EQ.loc[mask, "holm_p_shift_family"] = holm_adjust(
    EQ.loc[mask, "exact_signflip_p_two_sided"].to_numpy(float)
)
EQ["primary_official"] = EQ.scenario.eq("OFFICIAL_TEST")
EQ.to_csv(TABLES / "TABLE_A_EQUAL_BUDGET_RUT20K_VS_R020K_N10.csv", index=False)

PRIMARY_EQ = EQ[EQ.primary_official].copy()
PRIMARY_EQ.to_csv(TABLES / "TABLE_A_PRIMARY_EQUAL_BUDGET_OFFICIAL_N10.csv", index=False)

print("=== EQUAL-BUDGET N10 ===")
print(EQ.to_string(index=False))

# CELL 10
# 09 — B: Statistical label sufficiency vs R0@200k under 1 pp margin

ni_rows = []

for B in CANDIDATE_BUDGETS:
    for sn in SCENARIOS:
        cand = N10[
            N10.rep.eq("RUT")
            & N10.budget.eq(B)
            & N10.scenario.eq(sn)
        ].sort_values("run")

        ref = N10[
            N10.rep.eq("R0")
            & N10.budget.eq(REFERENCE_BUDGET)
            & N10.scenario.eq(sn)
        ].sort_values("run")

        if cand.run.tolist() != ref.run.tolist() or len(cand) != 10:
            raise RuntimeError(f"NI pairing failed B={B}, scenario={sn}")

        d_tpr = cand.tpr.to_numpy(float) - ref.tpr.to_numpy(float)
        d_fpr = cand.fpr.to_numpy(float) - ref.fpr.to_numpy(float)
        d_ap = cand.AP.to_numpy(float) - ref.AP.to_numpy(float)
        d_pr = cand.P_at_R90.to_numpy(float) - ref.P_at_R90.to_numpy(float)

        shifted = d_tpr + NI_MARGIN_TPR
        m, lo, hi = mean_ci95(d_tpr)
        lower1 = lower_ci95_one_sided(d_tpr)
        p_ni = exact_signflip_p_greater(shifted)

        # New-N5-only sensitivity: prospective part of current plan.
        d_new = d_tpr[5:]
        p_new = exact_signflip_p_greater(d_new + NI_MARGIN_TPR)

        ni_rows.append({
            "budget": B,
            "reference": "R0_200k",
            "candidate": f"RUT_{B}",
            "scenario": sn,
            "n_pairs": 10,
            "margin_tpr_pp": NI_MARGIN_TPR_PP,
            "RUT_tpr_mean": float(cand.tpr.mean()),
            "R0_200k_tpr_mean": float(ref.tpr.mean()),
            "mean_delta_tpr_pp": 100*m,
            "ci95_two_sided_lo_pp": 100*lo,
            "ci95_two_sided_hi_pp": 100*hi,
            "ci95_one_sided_lower_pp": 100*lower1,
            "exact_signflip_p_NI_one_sided": p_ni,
            "NI_reject_alpha_0_05": bool(p_ni < ALPHA),
            "pairs_within_margin": int((d_tpr > -NI_MARGIN_TPR).sum()),
            "candidate_better_pairs": int((d_tpr > 0).sum()),
            "mean_delta_fpr_pp": 100*float(d_fpr.mean()),
            "mean_delta_AP_pp": 100*float(d_ap.mean()),
            "mean_delta_P_at_R90_pp": 100*float(d_pr.mean()),
            "NEW_N5_exact_signflip_p_NI_one_sided": p_new,
            "NEW_N5_pairs_within_margin": int((d_new > -NI_MARGIN_TPR).sum()),
        })

NI = pd.DataFrame(ni_rows)
NI.to_csv(TABLES / "TABLE_B_LABEL_SUFFICIENCY_COMPONENT_N10.csv", index=False)

# Intersection-union budget-level decision: ALL five scenarios must pass.
budget_rows = []
for B, g in NI.groupby("budget", sort=True):
    all5 = bool(g.NI_reject_alpha_0_05.all())
    descriptive_all5 = bool((g.mean_delta_tpr_pp > -NI_MARGIN_TPR_PP).all())
    budget_rows.append({
        "budget": int(B),
        "reference_budget": REFERENCE_BUDGET,
        "label_reduction_pct": 100*(1-B/REFERENCE_BUDGET),
        "reference_to_candidate_factor": REFERENCE_BUDGET/B,
        "all_5_scenarios_exact_NI_alpha_0_05": all5,
        "all_5_scenario_means_within_1pp": descriptive_all5,
        "worst_scenario_mean_delta_tpr_pp": float(g.mean_delta_tpr_pp.min()),
        "worst_scenario_one_sided_lower_ci_pp": float(g.ci95_one_sided_lower_pp.min()),
        "largest_mean_fpr_increase_pp": float(g.mean_delta_fpr_pp.max()),
        "smallest_component_NI_p": float(g.exact_signflip_p_NI_one_sided.min()),
        "largest_component_NI_p": float(g.exact_signflip_p_NI_one_sided.max()),
        "component_tests_rejected": int(g.NI_reject_alpha_0_05.sum()),
        "component_tests_total": 5,
    })

BUDGET_DECISION = pd.DataFrame(budget_rows).sort_values("budget")
BUDGET_DECISION.to_csv(TABLES / "TABLE_B_LABEL_SUFFICIENCY_BUDGET_DECISION_N10.csv", index=False)

q = BUDGET_DECISION[BUDGET_DECISION.all_5_scenarios_exact_NI_alpha_0_05]
MIN_STAT_SUFFICIENT = int(q.budget.min()) if len(q) else None

qd = BUDGET_DECISION[BUDGET_DECISION.all_5_scenario_means_within_1pp]
MIN_DESCRIPTIVE_SUFFICIENT = int(qd.budget.min()) if len(qd) else None

SUFFICIENCY_HEADLINE = {
    "statistically_sufficient_budget_all_5_scenarios": MIN_STAT_SUFFICIENT,
    "descriptively_within_margin_budget_all_5_scenarios": MIN_DESCRIPTIVE_SUFFICIENT,
    "reference_budget": REFERENCE_BUDGET,
    "margin_tpr_pp": NI_MARGIN_TPR_PP,
    "decision_rule": "all five exact one-sided margin-shifted sign-flip component tests p<0.05",
    "intersection_union": True,
    "independent_external_replication": False,
}

if MIN_STAT_SUFFICIENT is not None:
    SUFFICIENCY_HEADLINE["label_reduction_pct"] = 100*(1-MIN_STAT_SUFFICIENT/REFERENCE_BUDGET)
    SUFFICIENCY_HEADLINE["label_reduction_factor"] = REFERENCE_BUDGET/MIN_STAT_SUFFICIENT
else:
    SUFFICIENCY_HEADLINE["label_reduction_pct"] = None
    SUFFICIENCY_HEADLINE["label_reduction_factor"] = None

atomic_json(RESULTS / "LABEL_SUFFICIENCY_N10_HEADLINE.json", SUFFICIENCY_HEADLINE)

print("=== LABEL SUFFICIENCY COMPONENT TESTS ===")
print(NI.to_string(index=False))
print("\n=== BUDGET DECISION ===")
print(BUDGET_DECISION.to_string(index=False))
print("\n=== HEADLINE ===")
print(json.dumps(SUFFICIENCY_HEADLINE, indent=2))

# CELL 11
# 10 — Stability views: per-seed primary contrasts and historical-vs-new sensitivity

# Per-seed equal-budget official.
q = N10[
    N10.scenario.eq("OFFICIAL_TEST")
    & N10.budget.eq(20_000)
    & N10.rep.isin(["R0","RUT"])
]
p = q.pivot(index="run", columns="rep", values=["tpr","fpr","AP"]).sort_index()
PER_SEED_EQ = pd.DataFrame({
    "run": p.index.to_numpy(),
    "source_group": ["HISTORICAL_N5" if i<5 else "NEW_N5" for i in p.index],
    "R0_tpr": p["tpr"]["R0"].to_numpy(),
    "RUT_tpr": p["tpr"]["RUT"].to_numpy(),
    "delta_tpr_pp": 100*(p["tpr"]["RUT"]-p["tpr"]["R0"]).to_numpy(),
    "delta_fpr_pp": 100*(p["fpr"]["RUT"]-p["fpr"]["R0"]).to_numpy(),
    "delta_AP_pp": 100*(p["AP"]["RUT"]-p["AP"]["R0"]).to_numpy(),
})
PER_SEED_EQ.to_csv(TABLES / "TABLE_PRIMARY_EQUAL_BUDGET_PER_SEED_N10.csv", index=False)

# Budget-level historical-N5 vs new-N5 descriptive split.
split_rows = []
for group_name, runs in [("HISTORICAL_N5", range(0,5)), ("NEW_N5", range(5,10))]:
    z = N10[N10.run.isin(list(runs))]
    for B in CANDIDATE_BUDGETS:
        deltas = []
        for sn in SCENARIOS:
            a = z[z.rep.eq("RUT") & z.budget.eq(B) & z.scenario.eq(sn)].sort_values("run")
            b = z[z.rep.eq("R0") & z.budget.eq(200_000) & z.scenario.eq(sn)].sort_values("run")
            d = a.tpr.to_numpy(float)-b.tpr.to_numpy(float)
            deltas.extend(d.tolist())
        split_rows.append({
            "source_group": group_name,
            "budget": B,
            "mean_delta_tpr_pp_across_seed_scenario_cells": 100*float(np.mean(deltas)),
            "min_delta_tpr_pp_across_seed_scenario_cells": 100*float(np.min(deltas)),
            "cells_within_1pp_margin": int((np.asarray(deltas)>-NI_MARGIN_TPR).sum()),
            "cells_total": len(deltas),
        })

SPLIT = pd.DataFrame(split_rows)
SPLIT.to_csv(TABLES / "TABLE_HISTORICAL_VS_NEW_N5_SENSITIVITY.csv", index=False)

print(PER_SEED_EQ.to_string(index=False))
print(SPLIT.to_string(index=False))

# CELL 12
# 11 — Optional FF4 resource-table linkage; no recomputation

def find_csv(name):
    hits = list(SEARCH_ROOT.rglob(name))
    if not hits:
        return None
    hits.sort(key=lambda p: (
        0 if "adaptive_resource" in str(p).lower() else 1,
        0 if "final" in str(p).lower() else 1,
        len(str(p)),
        str(p),
    ))
    return hits[0]

H1_PATH = find_csv("TABLE_H1_SSL_COMPUTE_PAIRED.csv")
FRONTIER_PATH = find_csv("TABLE_SSL_CASCADE_FRONTIER_FINAL.csv")
OP_PATH = find_csv("TABLE_OPERATIONAL_BENCHMARK.csv")

RESOURCE_SYNTHESIS = {
    "label_sufficiency_available": True,
    "minimal_statistically_sufficient_budget": MIN_STAT_SUFFICIENT,
    "reference_labels": REFERENCE_BUDGET,
    "label_reduction_factor": (
        None if MIN_STAT_SUFFICIENT is None else REFERENCE_BUDGET/MIN_STAT_SUFFICIENT
    ),
    "label_reduction_pct": (
        None if MIN_STAT_SUFFICIENT is None else 100*(1-MIN_STAT_SUFFICIENT/REFERENCE_BUDGET)
    ),
    "quality_tolerance_tpr_pp": NI_MARGIN_TPR_PP,
    "ff4_h1_table": str(H1_PATH) if H1_PATH else None,
    "ff4_frontier_table": str(FRONTIER_PATH) if FRONTIER_PATH else None,
    "ff4_operational_table": str(OP_PATH) if OP_PATH else None,
}

if H1_PATH is not None:
    h1 = pd.read_csv(H1_PATH)
    if {"scenario","mean_delta_escalation_pp"}.issubset(h1.columns):
        RESOURCE_SYNTHESIS["DAPT_minus_BASE_escalation_pp_mean_5scenarios"] = float(
            h1[h1.scenario.isin(SCENARIOS)].mean_delta_escalation_pp.mean()
        )
        off = h1[h1.scenario.eq("OFFICIAL_TEST")]
        if len(off):
            RESOURCE_SYNTHESIS["DAPT_minus_BASE_escalation_pp_OFFICIAL"] = float(
                off.iloc[0].mean_delta_escalation_pp
            )

if FRONTIER_PATH is not None:
    f = pd.read_csv(FRONTIER_PATH)
    need = {"stream","allowed_tpr_loss_pp","scenario","escalation_mean"}
    if need.issubset(f.columns):
        q = f[
            np.isclose(f.allowed_tpr_loss_pp.astype(float), NI_MARGIN_TPR_PP)
            & f.scenario.eq("OFFICIAL_TEST")
        ]
        for stream in ["URL_BASE","URL_DAPT"]:
            z = q[q.stream.eq(stream)]
            if len(z):
                RESOURCE_SYNTHESIS[f"{stream}_escalation_OFFICIAL_pct"] = 100*float(z.iloc[0].escalation_mean)

if OP_PATH is not None:
    opdf = pd.read_csv(OP_PATH)
    if {"system","latency_b1_median_ms","throughput_pages_s_mean"}.issubset(opdf.columns):
        full = opdf[opdf.system.eq("TRI_RUT_200K_FULL")]
        cas = opdf[opdf.system.eq("TRI_RUT_200K_CASCADE")]
        if len(full) and len(cas):
            frow = full.iloc[0]
            crow = cas.iloc[0]
            RESOURCE_SYNTHESIS["full_median_latency_ms"] = float(frow.latency_b1_median_ms)
            RESOURCE_SYNTHESIS["cascade_median_latency_ms"] = float(crow.latency_b1_median_ms)
            RESOURCE_SYNTHESIS["latency_reduction_pct"] = 100*(
                1-float(crow.latency_b1_median_ms)/float(frow.latency_b1_median_ms)
            )
            RESOURCE_SYNTHESIS["full_throughput_pages_s"] = float(frow.throughput_pages_s_mean)
            RESOURCE_SYNTHESIS["cascade_throughput_pages_s"] = float(crow.throughput_pages_s_mean)
            RESOURCE_SYNTHESIS["throughput_speedup_x"] = float(
                crow.throughput_pages_s_mean/frow.throughput_pages_s_mean
            )
            if "cascade_escalation_rate_sample" in opdf.columns:
                RESOURCE_SYNTHESIS["operational_cascade_escalation_pct"] = 100*float(
                    crow.cascade_escalation_rate_sample
                )

atomic_json(RESULTS / "CROSS_RESOURCE_SYNTHESIS.json", RESOURCE_SYNTHESIS)
pd.DataFrame([RESOURCE_SYNTHESIS]).to_csv(
    TABLES / "TABLE_CROSS_RESOURCE_SYNTHESIS.csv", index=False
)

print(json.dumps(RESOURCE_SYNTHESIS, indent=2))

# CELL 13
# 12 — Thesis-ready figures

import matplotlib.pyplot as plt

# Figure 1: scenario-wise N10 label sufficiency frontier.
fig, ax = plt.subplots(figsize=(8.0, 4.8))
for sn in SCENARIOS:
    q = NI[NI.scenario.eq(sn)].sort_values("budget")
    ax.plot(
        q.budget.to_numpy(),
        q.mean_delta_tpr_pp.to_numpy(),
        marker="o",
        label=sn,
    )
ax.axhline(-NI_MARGIN_TPR_PP, linestyle="--", linewidth=1.2, label="-1 pp margin")
ax.axhline(0.0, linestyle=":", linewidth=1.0)
ax.set_xscale("log")
ax.set_xticks(CANDIDATE_BUDGETS)
ax.set_xticklabels([f"{b//1000}k" for b in CANDIDATE_BUDGETS])
ax.set_xlabel("RUT label budget")
ax.set_ylabel("ΔTPR vs. R0 @ 200k (pp)")
ax.set_title("Label sufficiency under five evaluation conditions (N10)")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIGURES / "FIG_LABEL_SUFFICIENCY_N10.png", dpi=220)
plt.close(fig)

# Figure 2: equal-budget representation advantage.
q = EQ.copy()
fig, ax = plt.subplots(figsize=(8.0, 4.4))
x = np.arange(len(q))
y = q.mean_delta_tpr_pp.to_numpy()
lo = y - q.ci95_lo_pp.to_numpy()
hi = q.ci95_hi_pp.to_numpy() - y
ax.errorbar(x, y, yerr=np.vstack([lo,hi]), fmt="o", capsize=4)
ax.axhline(0.0, linestyle="--", linewidth=1.0)
ax.set_xticks(x)
ax.set_xticklabels(q.scenario.tolist(), rotation=25, ha="right")
ax.set_ylabel("RUT 20k − R0 20k ΔTPR (pp)")
ax.set_title("Equal-label representation advantage (N10)")
fig.tight_layout()
fig.savefig(FIGURES / "FIG_EQUAL_BUDGET_RUT_R0_N10.png", dpi=220)
plt.close(fig)

print({
    "figures": [
        str(FIGURES / "FIG_LABEL_SUFFICIENCY_N10.png"),
        str(FIGURES / "FIG_EQUAL_BUDGET_RUT_R0_N10.png"),
    ]
})

# CELL 14
# 13 — Final interpretation object, coverage audit, compact results ZIP

primary_eq = PRIMARY_EQ.iloc[0]

HEADLINE = {
    "status": "COMPLETE",
    "version": VERSION,
    "scientific_status": SCIENTIFIC_STATUS,
    "equal_budget_20k": {
        "RUT_minus_R0_OFFICIAL_delta_tpr_pp": float(primary_eq.mean_delta_tpr_pp),
        "ci95_pp": [float(primary_eq.ci95_lo_pp), float(primary_eq.ci95_hi_pp)],
        "exact_signflip_p_two_sided": float(primary_eq.exact_signflip_p_two_sided),
        "positive_pairs": int(primary_eq.positive_pairs),
        "n_pairs": 10,
    },
    "label_sufficiency": SUFFICIENCY_HEADLINE,
    "resource_synthesis": RESOURCE_SYNTHESIS,
    "claim_boundaries": [
        "The N10 combines five previously observed canonical runs with five new runs generated after the current analysis-plan lock.",
        "This is not an independent external replication.",
        "The 1 pp TPR margin is an analytical sufficiency tolerance shared with the thesis' primary cascade resource policy; it is not a universal business threshold.",
        "A statistically sufficient label budget is declared only if all five scenario-specific one-sided exact non-inferiority tests reject at alpha 0.05.",
        "Realized FINAL FPR differences are reported separately and must be discussed alongside TPR sufficiency.",
        "No transformer representation or architecture is retrained in this notebook.",
    ],
}
atomic_json(RESULTS / "FINAL_N10_FF1_FF3_RESOURCE_HEADLINE.json", HEADLINE)

coverage = pd.DataFrame([
    {"check":"analysis plan written before new fits", "pass":(AUDIT/"ANALYSIS_PLAN_LOCK.json").exists()},
    {"check":"canonical frozen embeddings reused", "pass":all(p.exists() for p in EMB_PATHS.values())},
    {"check":"historical N5 preserved unchanged", "pass":HIST.run.nunique()==5},
    {"check":"five new runs completed", "pass":NEW.run.nunique()==5},
    {"check":"N10 paired runs complete", "pass":N10.run.nunique()==10},
    {"check":"six planned conditions per run", "pass":len(N10)==10*6*5},
    {"check":"five frozen evaluation scenarios", "pass":set(N10.scenario)==set(SCENARIOS)},
    {"check":"equal-budget N10 table", "pass":len(EQ)==5},
    {"check":"four candidate budgets x five NI scenarios", "pass":len(NI)==20},
    {"check":"intersection-union budget table", "pass":len(BUDGET_DECISION)==4},
    {"check":"no transformer retraining", "pass":True},
    {"check":"no external benchmark dependency", "pass":True},
])
coverage.to_csv(AUDIT / "COMPLETE_COVERAGE_AUDIT.csv", index=False)
if not coverage["pass"].all():
    raise RuntimeError(coverage[~coverage["pass"]].to_dict("records"))

atomic_json(ROOT / "FINAL_SSL_LABEL_SUFFICIENCY_N10_COMPLETE.json", {
    "status": "COMPLETE",
    "version": VERSION,
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "coverage_pass": True,
    "minimal_statistically_sufficient_budget": MIN_STAT_SUFFICIENT,
})

package = ROOT / "_package"
if package.exists():
    shutil.rmtree(package)

for name in ["results","tables","audit","figures"]:
    shutil.copytree(ROOT/name, package/name)

shutil.copy2(
    ROOT / "FINAL_SSL_LABEL_SUFFICIENCY_N10_COMPLETE.json",
    package / "FINAL_SSL_LABEL_SUFFICIENCY_N10_COMPLETE.json",
)

zip_base = WORK_BASE / "FINAL_SSL_LABEL_SUFFICIENCY_N10_FF1_FF3_RESOURCE_LINK_RESULTS_v5"
zip_path = Path(shutil.make_archive(str(zip_base), "zip", package))
shutil.rmtree(package)

sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
atomic_json(ROOT / "RESULTS_ZIP_SHA256.json", {
    "file": str(zip_path),
    "sha256": sha,
    "size_bytes": zip_path.stat().st_size,
})

print(json.dumps(HEADLINE, indent=2, default=str))
print({
    "RESULT_ZIP": str(zip_path),
    "size_mb": round(zip_path.stat().st_size/1024**2, 2),
    "sha256": sha,
})
print("COMPLETE")