# KI-Unterstuetzung: siehe Erklaerung der Arbeit.
# Ursprungsdatei: FINAL_SSL_CROSS_PARADIGM_LABEL_BENCHMARK_N10_v1_FULL_GRID_XGB200K.ipynb
# CELL 1
# 00 — Imports, immutable benchmark plan, output directories

import os, gc, re, json, math, time, random, hashlib, itertools, shutil, zipfile, warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score, roc_auc_score, roc_curve,
    precision_recall_curve
)
from scipy.stats import beta as beta_dist
from scipy.stats import t as student_t
from scipy.sparse import hstack, csr_matrix, load_npz, save_npz
import joblib
import xgboost as xgb

warnings.filterwarnings("ignore")

IS_KAGGLE = Path("/kaggle/working").exists()
SEARCH_ROOT = Path("/kaggle/input") if Path("/kaggle/input").exists() else Path("/mnt/data")
WORK_BASE = Path("/kaggle/working") if IS_KAGGLE else Path("/mnt/data")

VERSION = "FINAL_SSL_CROSS_PARADIGM_LABEL_BENCHMARK_N10_v1_FULL_GRID_XGB200K"
SCIENTIFIC_STATUS = "POST_HOC_CROSS_PARADIGM_EXTENSION_WITH_FULL_PREDEFINED_BUDGET_GRID"

ROOT = WORK_BASE / "final_ssl_cross_paradigm_label_benchmark_n10_v1"
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
AUDIT = ROOT / "audit"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"
LINEAR_PARTS = ROOT / "linear_parts"
XGB_PARTS = ROOT / "xgb_parts"
XGB_MODELS = ROOT / "xgb_models"
XGB_CACHE = ROOT / "xgb_feature_cache"

for p in [ROOT, RESULTS, TABLES, AUDIT, MODELS, FIGURES, LINEAR_PARTS, XGB_PARTS, XGB_MODELS, XGB_CACHE]:
    p.mkdir(parents=True, exist_ok=True)

PRIMARY_FPR = 0.005
NI_MARGIN_TPR = 0.01
NI_MARGIN_TPR_PP = 1.0
ALPHA = 0.05

# Full canonical label grid. No budget is selected in advance.
BUDGETS = [2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000]
REFERENCE_BUDGET = 200_000

SCENARIOS = [
    "OFFICIAL_TEST",
    "DOMAIN_OOD_EXACT",
    "TEMPLATE_OOD_EXACT",
    "DOMAIN_TEMPLATE_OOD_EXACT",
    "LATE_TEST_Q4",
]

# The same canonical N10 run design as the preceding label-sufficiency analysis.
HIST_MODEL_SEEDS = [42, 62, 82, 102, 122]
HIST_LABEL_RANK_SEEDS = [20260813, 20260823, 20260833, 20260843, 20260853]
NEW_MODEL_SEEDS = [342, 362, 382, 402, 422]
NEW_LABEL_RANK_SEEDS = [9301, 9302, 9303, 9304, 9305]
MODEL_SEEDS = HIST_MODEL_SEEDS + NEW_MODEL_SEEDS
LABEL_RANK_SEEDS = HIST_LABEL_RANK_SEEDS + NEW_LABEL_RANK_SEEDS
RUN_SPECS = list(zip(range(10), MODEL_SEEDS, LABEL_RANK_SEEDS))

# Independent XGB seeds: intentionally not paired to transformer/label-rank replicates.
XGB_N10_SEEDS = [1042, 1062, 1082, 1102, 1122, 1342, 1362, 1382, 1402, 1422]

EXPECTED_COUNTS = {"SSL": 200_000, "CAL": 50_000, "FINAL": 168_060}
EXPECTED_FINAL_CLASSES = (91_260, 76_800)

# Exact canonical v8.4 TF-IDF/XGB system-reference configuration.
XGB_URL_MAX_FEATURES = 25_000
XGB_TEXT_MAX_FEATURES = 50_000
XGB_N_ESTIMATORS = 400
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.08
XGB_MIN_CHILD_WEIGHT = 2
XGB_SUBSAMPLE = 0.90
XGB_COLSAMPLE = 0.65
XGB_REG_LAMBDA = 2.0
XGB_MAX_BIN = 128

def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)

ANALYSIS_PLAN = {
    "version": VERSION,
    "scientific_status": SCIENTIFIC_STATUS,
    "prior_rut_r0_results_already_observed": True,
    "prior_50k_sufficiency_result_already_observed": True,
    "claim": "post-hoc cross-paradigm benchmark extension; not independent prospective replication",
    "budget_grid": BUDGETS,
    "budget_preselection": None,
    "primary_target_fpr": PRIMARY_FPR,
    "scenarios": SCENARIOS,
    "references": {
        "internal_mechanistic": "R0_200k_LINEAR",
        "cross_paradigm_system": "TFIDF_XGB_200k",
    },
    "primary_reporting": "complete RUT budget-performance curve against both 200k references",
    "internal_reference_inference": {
        "design": "paired by canonical run",
        "margin_tpr_pp": NI_MARGIN_TPR_PP,
        "component_test": "exact one-sided sign-flip on (RUT_B - R0_200k + margin)",
        "global_within_budget": "intersection-union over all five scenarios",
        "budget_family_correction": "Holm over seven global budget p-values",
    },
    "cross_paradigm_inference": {
        "design": "independent N10 run distributions; no artificial seed pairing",
        "role": "secondary/supportive system benchmark only",
        "margin_tpr_pp": NI_MARGIN_TPR_PP,
        "component_test": "one-sided Welch non-inferiority screen",
        "global_within_budget": "intersection-union over all five scenarios",
        "budget_family_correction": "Holm over seven global budget p-values",
        "causal_ssl_attribution_allowed": False,
    },
    "cross_paradigm_primary_descriptive_rule": {
        "rule": "all-scenario mean dominance is reported if mean TPR_RUT >= mean TPR_XGB and mean FPR_RUT <= mean FPR_XGB in every scenario",
        "formal_test": False,
    },
    "xgb_protocol": {
        "tfidf_fit": "SSL 200k inputs only; labels unused",
        "url": {"analyzer":"char","ngram_range":[3,5],"min_df":2,"max_features":XGB_URL_MAX_FEATURES,"sublinear_tf":True,"norm":"l2"},
        "text": {"analyzer":"word","ngram_range":[1,2],"min_df":3,"max_df":0.995,"max_features":XGB_TEXT_MAX_FEATURES,"sublinear_tf":True,"strip_accents":"unicode","norm":"l2"},
        "xgb": {
            "n_estimators":XGB_N_ESTIMATORS,"max_depth":XGB_MAX_DEPTH,
            "learning_rate":XGB_LEARNING_RATE,"min_child_weight":XGB_MIN_CHILD_WEIGHT,
            "subsample":XGB_SUBSAMPLE,"colsample_bytree":XGB_COLSAMPLE,
            "reg_lambda":XGB_REG_LAMBDA,"max_bin":XGB_MAX_BIN,
            "tree_method":"hist",
        },
        "n10_seeds": XGB_N10_SEEDS,
        "retrain_all_xgb_models_in_current_runtime": True,
        "reason": "avoid mixing historical and newly fitted XGB models across library/runtime versions",
    },
    "resource_interpretation_boundary": "label demand only; TAPT pretraining compute is not included in the label-budget comparison",
}
atomic_json(AUDIT / "ANALYSIS_PLAN_LOCK.json", ANALYSIS_PLAN)

RUNTIME = {
    "python": __import__("sys").version,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "sklearn": __import__("sklearn").__version__,
    "xgboost": xgb.__version__,
}
atomic_json(AUDIT / "RUNTIME_VERSIONS.json", RUNTIME)

print(json.dumps(ANALYSIS_PLAN, indent=2))
print(json.dumps(RUNTIME, indent=2))


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
# D. Historical canonical N5 curve — optional reuse, never a hard dependency
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
    small = q[
        q.probe.astype(str).eq("LINEAR")
        & q.rep.astype(str).isin(["R0","RUT"])
    ].copy()
    got_pairs = set(
        tuple(x) for x in
        small[["run","model_seed","label_rank_seed"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    expected_pairs = set(zip(range(5), HIST_MODEL_SEEDS, HIST_LABEL_RANK_SEEDS))
    if not expected_pairs.issubset(got_pairs):
        return False, None
    return True, q

hist_candidates = []
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
        if str(p) in seen:
            continue
        seen.add(str(p))
        ok, _ = historical_table_is_usable(p)
        if ok:
            hist_candidates.append((80_000, p, "SCHEMA_AND_SEED_VALIDATED_DISCOVERY"))

HIST_CURVE_PATH = None
HIST_RESOLUTION_REASON = "NOT_FOUND_RECOMPUTE_MISSING_PROBES"
if hist_candidates:
    hist_candidates.sort(key=lambda z: (-z[0], len(str(z[1])), str(z[1])))
    _, HIST_CURVE_PATH, HIST_RESOLUTION_REASON = hist_candidates[0]
    print({
        "HISTORICAL_N5_TABLE": "REUSE_AVAILABLE",
        "path": str(HIST_CURVE_PATH),
        "resolution": HIST_RESOLUTION_REASON,
    })
else:
    print({
        "HISTORICAL_N5_TABLE": "NOT_FOUND",
        "policy": "Missing linear-probe rows will be recomputed from frozen embeddings.",
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
    "historical_curve": str(HIST_CURVE_PATH) if HIST_CURVE_PATH else None,
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
# 04 — Reuse any already-computed canonical probe rows; recompute only missing conditions

# Historical N5 curve, if present.
existing_frames = []
if HIST_CURVE_PATH is not None:
    h = pd.read_csv(HIST_CURVE_PATH)
    h = h[
        h.probe.astype(str).eq("LINEAR")
        & np.isclose(h.target_fpr.astype(float), PRIMARY_FPR)
        & h.rep.astype(str).isin(["R0","RUT"])
        & h.scenario.astype(str).isin(SCENARIOS)
    ].copy()
    h["reuse_source"] = "HISTORICAL_CANONICAL_CURVE"
    h["known_before_current_benchmark"] = True
    existing_frames.append(h)

# Reuse the preceding v5 N10 master if it is mounted or still in /kaggle/working.
def prior_n10_is_usable(p):
    try:
        q = pd.read_csv(p)
    except Exception:
        return False, None
    req = {
        "run","model_seed","label_rank_seed","budget","rep","probe","scenario",
        "target_fpr","tpr","fpr","AP","P_at_R90"
    }
    if not req.issubset(q.columns):
        return False, None
    q = q[
        q.probe.astype(str).eq("LINEAR")
        & np.isclose(q.target_fpr.astype(float), PRIMARY_FPR)
        & q.rep.astype(str).isin(["R0","RUT"])
        & q.scenario.astype(str).isin(SCENARIOS)
    ].copy()
    if set(range(10)).issubset(set(q.run.astype(int).unique())):
        return True, q
    return False, None

prior_candidates = []
local_prior = WORK_BASE / "final_ssl_label_sufficiency_n10_v1/results/N10_MATCHED_ANALYSIS_ALL.csv"
if local_prior.exists():
    ok, q = prior_n10_is_usable(local_prior)
    if ok:
        prior_candidates.append((100_000, local_prior, q, "LOCAL_PRECEDING_RUN"))

try:
    for p in SEARCH_ROOT.rglob("N10_MATCHED_ANALYSIS_ALL.csv"):
        ok, q = prior_n10_is_usable(p)
        if ok:
            prior_candidates.append((90_000, p, q, "MOUNTED_PRECEDING_RUN"))
except Exception:
    pass

if prior_candidates:
    prior_candidates.sort(key=lambda z: (-z[0], len(str(z[1])), str(z[1])))
    _, PRIOR_N10_PATH, q, PRIOR_REASON = prior_candidates[0]
    q["reuse_source"] = "PRIOR_LABEL_SUFFICIENCY_N10"
    q["known_before_current_benchmark"] = True
    existing_frames.append(q)
    print({"PRIOR_N10_REUSE": str(PRIOR_N10_PATH), "reason": PRIOR_REASON})
else:
    PRIOR_N10_PATH = None
    print({"PRIOR_N10_REUSE": None, "policy": "recompute missing probe rows only"})

EXISTING = pd.concat(existing_frames, ignore_index=True, sort=False) if existing_frames else pd.DataFrame()

# Deduplicate by experimental key, preferring the N10 master over older curve rows when both exist.
if len(EXISTING):
    priority = {"PRIOR_LABEL_SUFFICIENCY_N10": 2, "HISTORICAL_CANONICAL_CURVE": 1}
    EXISTING["_priority"] = EXISTING.reuse_source.map(priority).fillna(0)
    EXISTING = (
        EXISTING.sort_values("_priority", ascending=False)
        .drop_duplicates(["run","rep","budget","scenario","target_fpr"], keep="first")
        .drop(columns="_priority")
    )

atomic_json(AUDIT / "PROBE_REUSE_AUDIT.json", {
    "historical_curve": str(HIST_CURVE_PATH) if HIST_CURVE_PATH else None,
    "prior_n10": str(PRIOR_N10_PATH) if PRIOR_N10_PATH else None,
    "reused_rows": int(len(EXISTING)),
})
print({"EXISTING_PROBE_ROWS": len(EXISTING)})


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
        "part": LINEAR_PARTS / f"run{run:02d}_{rep}_B{budget}.csv",
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
            "source": "RECOMPUTED_MISSING_PROBE_CONDITION",
            "known_before_current_benchmark": False,
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
# 06 — Build the complete N10 RUT budget grid + R0@200k reference

def existing_condition_rows(run, rep, budget):
    if EXISTING.empty:
        return None
    q = EXISTING[
        EXISTING.run.astype(int).eq(int(run))
        & EXISTING.rep.astype(str).eq(str(rep))
        & EXISTING.budget.astype(int).eq(int(budget))
        & EXISTING.scenario.astype(str).isin(SCENARIOS)
        & np.isclose(EXISTING.target_fpr.astype(float), PRIMARY_FPR)
    ].copy()
    if len(q) != len(SCENARIOS) or set(q.scenario.astype(str)) != set(SCENARIOS):
        return None
    return q

required_conditions = []
for run, mseed, rseed in RUN_SPECS:
    required_conditions.append((run,mseed,rseed,"R0",REFERENCE_BUDGET))
    for b in BUDGETS:
        required_conditions.append((run,mseed,rseed,"RUT",b))

rows = []
fit_count = 0
reuse_count = 0

for run, mseed, rseed, rep, budget in required_conditions:
    q = existing_condition_rows(run, rep, budget)
    if q is not None:
        reuse_count += 1
        rows.append(q)
        continue

    fit_and_evaluate_new_condition(
        run=run,
        model_seed=mseed,
        rank_seed=rseed,
        rep=rep,
        budget=budget,
    )
    q = pd.read_csv(condition_paths(run, rep, budget)["part"])
    q["reuse_source"] = "RECOMPUTED_MISSING_PROBE_CONDITION"
    q["known_before_current_benchmark"] = False
    rows.append(q)
    fit_count += 1

PROBES = pd.concat(rows, ignore_index=True, sort=False)

expected_conditions_per_run = 1 + len(BUDGETS)
for run in range(10):
    q = PROBES[PROBES.run.astype(int).eq(run)]
    cset = set(tuple(x) for x in q[["rep","budget"]].drop_duplicates().itertuples(index=False, name=None))
    expected = {("R0",REFERENCE_BUDGET)} | {("RUT",b) for b in BUDGETS}
    if cset != expected:
        raise RuntimeError({"run":run,"conditions":sorted(cset),"expected":sorted(expected)})
    for rep,b in expected:
        z = q[q.rep.astype(str).eq(rep) & q.budget.astype(int).eq(b)]
        if len(z) != len(SCENARIOS) or set(z.scenario.astype(str)) != set(SCENARIOS):
            raise RuntimeError({"run":run,"rep":rep,"budget":b,"rows":len(z)})

PROBES.to_csv(RESULTS / "N10_RUT_FULL_GRID_AND_R0_200K.csv", index=False)

atomic_json(AUDIT / "PROBE_COVERAGE_AUDIT.json", {
    "runs": 10,
    "rut_budgets": BUDGETS,
    "r0_reference_budget": REFERENCE_BUDGET,
    "scenarios": SCENARIOS,
    "rows_expected": 10 * (len(BUDGETS)+1) * len(SCENARIOS),
    "rows_observed": len(PROBES),
    "conditions_reused": reuse_count,
    "conditions_recomputed": fit_count,
    "complete": len(PROBES) == 10 * (len(BUDGETS)+1) * len(SCENARIOS),
})
print({
    "PROBES_COMPLETE": True,
    "conditions_reused": reuse_count,
    "conditions_recomputed": fit_count,
    "rows": len(PROBES),
})


# CELL 8
# 07 — Canonical v8.4 TF-IDF feature pipeline for the cross-paradigm reference

def validate_xgb_cache_dir(d):
    d = Path(d)
    req = [
        d/"url_tfidf.joblib", d/"text_tfidf.joblib",
        d/"X_SSL.npz", d/"X_CAL.npz", d/"X_FINAL.npz",
    ]
    if not all(p.exists() for p in req):
        return False, None
    try:
        uv = joblib.load(d/"url_tfidf.joblib")
        tv = joblib.load(d/"text_tfidf.joblib")
        xs = load_npz(d/"X_SSL.npz")
        xc = load_npz(d/"X_CAL.npz")
        xf = load_npz(d/"X_FINAL.npz")
        canonical_path = "phreshphish_v8_4_FINAL/xgb_baseline/feature_cache" in str(d).replace("\\","/")
        exact_vocab = (
            len(uv.vocabulary_) == XGB_URL_MAX_FEATURES
            and len(tv.vocabulary_) == XGB_TEXT_MAX_FEATURES
        )
        ok = (
            canonical_path
            and xs.shape[0] == EXPECTED_COUNTS["SSL"]
            and xc.shape[0] == EXPECTED_COUNTS["CAL"]
            and xf.shape[0] == EXPECTED_COUNTS["FINAL"]
            and xs.shape[1] == xc.shape[1] == xf.shape[1] == (XGB_URL_MAX_FEATURES + XGB_TEXT_MAX_FEATURES)
            and exact_vocab
        )
        if not ok:
            return False, None
        return True, (uv,tv,xs,xc,xf)
    except Exception as e:
        print({"XGB_CACHE_REJECTED":str(d),"error":repr(e)})
        return False, None

cache_candidates = []
try:
    for p in SEARCH_ROOT.rglob("url_tfidf.joblib"):
        d = p.parent
        ok, payload = validate_xgb_cache_dir(d)
        if ok:
            # Only an exact canonical v8.4 workroot cache is accepted.
            # Anything else is rebuilt from the currently validated role data.
            score = 100_000
            cache_candidates.append((score,d,payload))
except Exception as e:
    print({"XGB_CACHE_SEARCH_ERROR":repr(e)})

if cache_candidates:
    cache_candidates.sort(key=lambda z: (-z[0], len(str(z[1])), str(z[1])))
    _, XGB_CACHE_SOURCE, payload = cache_candidates[0]
    url_vec, text_vec, X_SSL, X_CAL, X_FINAL = payload
    cache_origin = "CANONICAL_CACHE_REUSE"
    print({"XGB_FEATURE_CACHE":"REUSED","path":str(XGB_CACHE_SOURCE)})
else:
    print("No validated canonical XGB feature cache found; rebuilding the exact v8.4 TF-IDF representation.")

    SSL_XGB = read_role("SSL", ["sha256","url","text"])
    CAL_XGB = read_role("CAL", ["sha256","url","text"])
    FINAL_XGB = read_role("FINAL", ["sha256","url","text"])

    for q in [SSL_XGB,CAL_XGB,FINAL_XGB]:
        q["sha256"] = q.sha256.astype(str).str.lower()
        q["url"] = q.url.fillna("").astype(str)
        q["text"] = q.text.fillna("").astype(str)

    if not np.array_equal(SSL_XGB.sha256.to_numpy(), np.asarray(ssl_sha, dtype=str)):
        raise RuntimeError("XGB SSL SHA alignment mismatch.")
    if not np.array_equal(CAL_XGB.sha256.to_numpy(), CAL.sha256.to_numpy()):
        raise RuntimeError("XGB CAL SHA alignment mismatch.")
    if not np.array_equal(FINAL_XGB.sha256.to_numpy(), FINAL.sha256.to_numpy()):
        raise RuntimeError("XGB FINAL SHA alignment mismatch.")

    url_vec = TfidfVectorizer(
        analyzer="char", ngram_range=(3,5), min_df=2,
        max_features=XGB_URL_MAX_FEATURES,
        sublinear_tf=True, norm="l2", dtype=np.float32,
    )
    text_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1,2), min_df=3, max_df=.995,
        max_features=XGB_TEXT_MAX_FEATURES,
        sublinear_tf=True, strip_accents="unicode",
        norm="l2", dtype=np.float32,
    )

    t0 = time.perf_counter()
    Xu = url_vec.fit_transform(SSL_XGB.url)
    Xt = text_vec.fit_transform(SSL_XGB.text)
    fit_seconds = time.perf_counter() - t0

    X_SSL = csr_matrix(hstack([Xu,Xt],format="csr"), dtype=np.float32)
    del Xu, Xt
    gc.collect()

    def tfidf_pair(frame):
        xu = url_vec.transform(frame.url)
        xt = text_vec.transform(frame.text)
        return csr_matrix(hstack([xu,xt],format="csr"), dtype=np.float32)

    X_CAL = tfidf_pair(CAL_XGB)
    X_FINAL = tfidf_pair(FINAL_XGB)

    joblib.dump(url_vec, XGB_CACHE/"url_tfidf.joblib", compress=3)
    joblib.dump(text_vec, XGB_CACHE/"text_tfidf.joblib", compress=3)
    save_npz(XGB_CACHE/"X_SSL.npz", X_SSL, compressed=True)
    save_npz(XGB_CACHE/"X_CAL.npz", X_CAL, compressed=True)
    save_npz(XGB_CACHE/"X_FINAL.npz", X_FINAL, compressed=True)
    XGB_CACHE_SOURCE = XGB_CACHE
    cache_origin = "REBUILT_EXACT_V8_4_PROTOCOL"

    del SSL_XGB, CAL_XGB, FINAL_XGB
    gc.collect()

atomic_json(AUDIT / "XGB_FEATURE_PROTOCOL_AUDIT.json", {
    "status":"PASS",
    "origin":cache_origin,
    "source":str(XGB_CACHE_SOURCE),
    "tfidf_labels_used":False,
    "url_vocab_size":int(len(url_vec.vocabulary_)),
    "text_vocab_size":int(len(text_vec.vocabulary_)),
    "shapes":{
        "SSL":list(X_SSL.shape),
        "CAL":list(X_CAL.shape),
        "FINAL":list(X_FINAL.shape),
    },
})
print({
    "XGB_FEATURES_READY":True,
    "origin":cache_origin,
    "shape":X_SSL.shape,
    "url_vocab":len(url_vec.vocabulary_),
    "text_vocab":len(text_vec.vocabulary_),
})


# CELL 9
# 08 — Train a homogeneous XGB@200k N10 reference in the current runtime

def xgb_params(seed):
    p = dict(
        objective="binary:logistic",
        eval_metric="logloss",
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
        tree_method="hist",
    )
    try:
        major = int(str(xgb.__version__).split(".")[0])
    except Exception:
        major = 2
    if major >= 2:
        p["device"] = "cpu"
    return p

def xgb_part_paths(seed):
    return {
        "part": XGB_PARTS / f"seed{seed}_B200000.csv",
        "done": XGB_PARTS / f"seed{seed}_B200000_COMPLETE.json",
        "model": XGB_MODELS / f"seed{seed}_B200000.ubj",
    }

def xgb_margin(model, X):
    return np.asarray(model.predict(X, output_margin=True), dtype=np.float64)

for seed in XGB_N10_SEEDS:
    P = xgb_part_paths(seed)
    spec = {
        "seed":int(seed),
        "budget":REFERENCE_BUDGET,
        "xgboost_version":xgb.__version__,
        "params":xgb_params(seed),
        "feature_protocol":"canonical_v8_4_tfidf_url25k_text50k",
        "target_fpr":PRIMARY_FPR,
    }

    if P["part"].exists() and P["done"].exists():
        old = json.loads(P["done"].read_text())
        if old.get("spec") == spec:
            print({"XGB_REUSE_CURRENT_RUNTIME_PART":seed})
            continue

    t0 = time.perf_counter()
    model = xgb.XGBClassifier(**xgb_params(seed))
    model.fit(X_SSL, TRAIN_Y.astype(int), verbose=False)
    train_s = time.perf_counter() - t0
    model.save_model(P["model"])

    t1 = time.perf_counter()
    cal_score = xgb_margin(model, X_CAL)
    final_score = xgb_margin(model, X_FINAL)
    score_s = time.perf_counter() - t1

    th = thr_fpr(cal_score, PRIMARY_FPR)
    rows = []
    for scenario, mask in FINAL_MASKS.items():
        rows.append({
            "xgb_seed":int(seed),
            "budget":REFERENCE_BUDGET,
            "system":"TFIDF_XGB",
            "dataset":"FINAL",
            "scenario":scenario,
            "target_fpr":PRIMARY_FPR,
            "threshold":th,
            **op(FINAL_Y[mask], final_score[mask], th),
            **curves(FINAL_Y[mask], final_score[mask]),
        })
    pd.DataFrame(rows).to_csv(P["part"], index=False)
    atomic_json(P["done"], {
        "status":"COMPLETE",
        "spec":spec,
        "train_seconds":float(train_s),
        "score_seconds":float(score_s),
        "completed_utc":datetime.now(timezone.utc).isoformat(),
    })

    print({
        "XGB_N10_SEED_COMPLETE":seed,
        "train_min":round(train_s/60,2),
        "score_min":round(score_s/60,2),
    })
    del model, cal_score, final_score
    gc.collect()

XGB_N10 = pd.concat([pd.read_csv(xgb_part_paths(s)["part"]) for s in XGB_N10_SEEDS], ignore_index=True)
if len(XGB_N10) != len(XGB_N10_SEEDS)*len(SCENARIOS):
    raise RuntimeError("XGB N10 reference incomplete.")
if set(XGB_N10.scenario) != set(SCENARIOS):
    raise RuntimeError("XGB scenario coverage mismatch.")
XGB_N10.to_csv(RESULTS / "XGB_200K_N10_ALL_SCENARIOS.csv", index=False)

timing_rows = []
for s in XGB_N10_SEEDS:
    d = json.loads(xgb_part_paths(s)["done"].read_text())
    timing_rows.append({
        "seed":s,
        "train_seconds":d.get("train_seconds",0.0),
        "score_seconds":d.get("score_seconds",0.0),
        "xgboost_version":d["spec"]["xgboost_version"],
    })
pd.DataFrame(timing_rows).to_csv(RESULTS / "XGB_200K_N10_TIMING.csv", index=False)

print({
    "XGB_N10_COMPLETE":True,
    "n_seeds":len(XGB_N10_SEEDS),
    "rows":len(XGB_N10),
    "xgboost_version":xgb.__version__,
})


# CELL 10
# 09 — Statistical helpers for the two deliberately different reference roles

def exact_signflip_p_greater_shifted(diff, margin=0.0):
    # H0: location(diff) <= -margin; test sign-flips of diff + margin against > 0.
    z = np.asarray(diff, float) + float(margin)
    obs = float(z.mean())
    vals = []
    for signs in itertools.product([-1,1], repeat=len(z)):
        vals.append(float(np.mean(z*np.asarray(signs))))
    return float(np.mean(np.asarray(vals) >= obs - 1e-15))

def holm_adjust_with_nan(p_values):
    p = np.asarray(p_values, float)
    out = np.full(len(p), np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if len(finite) == 0:
        return out
    vals = p[finite]
    order = np.argsort(vals, kind="mergesort")
    m = len(vals)
    running = 0.0
    for rank, local in enumerate(order):
        running = max(running, (m-rank)*vals[local])
        out[finite[local]] = min(1.0, running)
    return out

def welch_noninferiority(a, b, margin=0.01):
    # Independent-sample supportive screen only.
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ma, mb = float(a.mean()), float(b.mean())
    va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
    na, nb = len(a), len(b)
    se2 = va/na + vb/nb
    if se2 <= 0:
        return {
            "diff":ma-mb,"se":0.0,"df":np.inf,
            "lower_one_sided":ma-mb,
            "p_ni":0.0 if ma-mb > -margin else 1.0,
        }
    se = math.sqrt(se2)
    num = se2**2
    den = (va/na)**2/max(na-1,1) + (vb/nb)**2/max(nb-1,1)
    df = num/den if den > 0 else np.inf
    tstat = ((ma-mb) + margin)/se
    p = float(student_t.sf(tstat, df))
    lower = float((ma-mb) - student_t.ppf(.95, df)*se)
    return {"diff":ma-mb,"se":se,"df":df,"lower_one_sided":lower,"p_ni":p}

RUT = PROBES[PROBES.rep.astype(str).eq("RUT")].copy()
R0_REF = PROBES[
    PROBES.rep.astype(str).eq("R0")
    & PROBES.budget.astype(int).eq(REFERENCE_BUDGET)
].copy()

print({
    "RUT_ROWS":len(RUT),
    "R0_REF_ROWS":len(R0_REF),
    "XGB_REF_ROWS":len(XGB_N10),
})


# CELL 11
# 10 — Internal mechanistic sufficiency: full RUT grid vs R0@200k (paired N10)

internal_components = []
for b in BUDGETS:
    for scenario in SCENARIOS:
        a = RUT[(RUT.budget.astype(int)==b) & RUT.scenario.astype(str).eq(scenario)].sort_values("run")
        r = R0_REF[R0_REF.scenario.astype(str).eq(scenario)].sort_values("run")
        if len(a) != 10 or len(r) != 10 or not np.array_equal(a.run.astype(int), r.run.astype(int)):
            raise RuntimeError({"internal_pairing_failure":(b,scenario)})
        d = a.tpr.to_numpy(float) - r.tpr.to_numpy(float)
        m,lo,hi = mean_ci95(d)
        p = exact_signflip_p_greater_shifted(d, NI_MARGIN_TPR)
        internal_components.append({
            "budget":b,
            "scenario":scenario,
            "n":10,
            "rut_tpr_mean":float(a.tpr.mean()),
            "ref_tpr_mean":float(r.tpr.mean()),
            "delta_tpr_mean":m,
            "delta_tpr_ci95_lo":lo,
            "delta_tpr_ci95_hi":hi,
            "delta_fpr_mean":float(a.fpr.mean()-r.fpr.mean()),
            "ni_margin_tpr_pp":NI_MARGIN_TPR_PP,
            "ni_p_one_sided_exact":p,
            "component_reject":bool(p < ALPHA),
        })

INTERNAL_COMPONENTS = pd.DataFrame(internal_components)
INTERNAL_COMPONENTS.to_csv(TABLES / "INTERNAL_RUT_VS_R0_200K_COMPONENTS.csv", index=False)

internal_global = (
    INTERNAL_COMPONENTS.groupby("budget")
    .agg(
        global_iut_p=("ni_p_one_sided_exact","max"),
        all_5_component_reject=("component_reject","all"),
        worst_mean_delta_tpr=("delta_tpr_mean","min"),
        worst_mean_delta_fpr=("delta_fpr_mean","max"),
    )
    .reset_index()
    .sort_values("budget")
)
internal_global["holm_p_over_7_budgets"] = holm_adjust_with_nan(internal_global.global_iut_p.to_numpy(float))
internal_global["holm_global_reject"] = internal_global.holm_p_over_7_budgets < ALPHA
internal_global.to_csv(TABLES / "INTERNAL_RUT_VS_R0_200K_GLOBAL_BUDGETS.csv", index=False)

supported_internal = internal_global[internal_global.holm_global_reject].budget.astype(int).tolist()
min_internal = min(supported_internal) if supported_internal else None

print(internal_global.to_string(index=False))
print({"MIN_INTERNAL_SUPPORTED_BUDGET":min_internal})


# CELL 12
# 11 — Cross-paradigm benchmark: full RUT grid vs independently trained TF-IDF/XGB@200k N10

cross_rows = []
for b in BUDGETS:
    for scenario in SCENARIOS:
        a = RUT[(RUT.budget.astype(int)==b) & RUT.scenario.astype(str).eq(scenario)]
        x = XGB_N10[XGB_N10.scenario.astype(str).eq(scenario)]
        if len(a) != 10 or len(x) != 10:
            raise RuntimeError({"cross_n_failure":(b,scenario), "rut_n":len(a), "xgb_n":len(x)})

        w = welch_noninferiority(a.tpr.to_numpy(float), x.tpr.to_numpy(float), NI_MARGIN_TPR)
        cross_rows.append({
            "budget":b,
            "scenario":scenario,
            "rut_n":len(a),
            "xgb_n":len(x),
            "rut_tpr_mean":float(a.tpr.mean()),
            "xgb_tpr_mean":float(x.tpr.mean()),
            "delta_tpr_mean":float(a.tpr.mean()-x.tpr.mean()),
            "rut_fpr_mean":float(a.fpr.mean()),
            "xgb_fpr_mean":float(x.fpr.mean()),
            "delta_fpr_mean":float(a.fpr.mean()-x.fpr.mean()),
            "strict_mean_tpr_at_least_xgb":bool(a.tpr.mean() >= x.tpr.mean()),
            "strict_mean_fpr_no_worse":bool(a.fpr.mean() <= x.fpr.mean()),
            "strict_bivariate_mean_dominance":bool((a.tpr.mean() >= x.tpr.mean()) and (a.fpr.mean() <= x.fpr.mean())),
            "welch_ni_margin_tpr_pp":NI_MARGIN_TPR_PP,
            "welch_ni_lower_one_sided":w["lower_one_sided"],
            "welch_ni_p_one_sided":w["p_ni"],
            "welch_df":w["df"],
            "supportive_component_ni":bool(w["p_ni"] < ALPHA),
        })

CROSS_COMPONENTS = pd.DataFrame(cross_rows)
CROSS_COMPONENTS.to_csv(TABLES / "CROSS_PARADIGM_RUT_VS_XGB200K_COMPONENTS.csv", index=False)

cross_global = (
    CROSS_COMPONENTS.groupby("budget")
    .agg(
        all_5_strict_mean_dominance=("strict_bivariate_mean_dominance","all"),
        all_5_tpr_mean_at_least_xgb=("strict_mean_tpr_at_least_xgb","all"),
        all_5_fpr_mean_no_worse=("strict_mean_fpr_no_worse","all"),
        supportive_global_iut_p=("welch_ni_p_one_sided","max"),
        all_5_supportive_component_ni=("supportive_component_ni","all"),
        worst_mean_delta_tpr=("delta_tpr_mean","min"),
        worst_mean_delta_fpr=("delta_fpr_mean","max"),
    )
    .reset_index()
    .sort_values("budget")
)
cross_global["supportive_holm_p_over_7_budgets"] = holm_adjust_with_nan(
    cross_global.supportive_global_iut_p.to_numpy(float)
)
cross_global["supportive_holm_global_reject"] = cross_global.supportive_holm_p_over_7_budgets < ALPHA
cross_global.to_csv(TABLES / "CROSS_PARADIGM_RUT_VS_XGB200K_GLOBAL_BUDGETS.csv", index=False)

strict = cross_global[cross_global.all_5_strict_mean_dominance].budget.astype(int).tolist()
supportive = cross_global[cross_global.supportive_holm_global_reject].budget.astype(int).tolist()
min_strict = min(strict) if strict else None
min_supportive = min(supportive) if supportive else None

print(cross_global.to_string(index=False))
print({
    "MIN_CROSS_PARADIGM_STRICT_MEAN_DOMINANCE_BUDGET":min_strict,
    "MIN_CROSS_PARADIGM_SUPPORTIVE_NI_BUDGET":min_supportive,
    "IMPORTANT":"Cross-paradigm results are system-level; do not attribute them causally to TAPT alone.",
})


# CELL 13
# 12 — Full-grid tables and thesis-ready figures

# Aggregated RUT curve.
rut_agg = (
    RUT.groupby(["budget","scenario"])
    .agg(
        n=("tpr","size"),
        tpr_mean=("tpr","mean"),
        tpr_std=("tpr","std"),
        fpr_mean=("fpr","mean"),
        fpr_std=("fpr","std"),
        AP_mean=("AP","mean"),
    )
    .reset_index()
)
xgb_agg = (
    XGB_N10.groupby("scenario")
    .agg(
        n=("tpr","size"),
        tpr_mean=("tpr","mean"),
        tpr_std=("tpr","std"),
        fpr_mean=("fpr","mean"),
        fpr_std=("fpr","std"),
        AP_mean=("AP","mean"),
    )
    .reset_index()
)
r0_agg = (
    R0_REF.groupby("scenario")
    .agg(
        n=("tpr","size"),
        tpr_mean=("tpr","mean"),
        tpr_std=("tpr","std"),
        fpr_mean=("fpr","mean"),
        fpr_std=("fpr","std"),
        AP_mean=("AP","mean"),
    )
    .reset_index()
)

rut_agg.to_csv(TABLES/"RUT_N10_FULL_LABEL_CURVE.csv", index=False)
xgb_agg.to_csv(TABLES/"XGB200K_N10_REFERENCE.csv", index=False)
r0_agg.to_csv(TABLES/"R0_200K_N10_REFERENCE.csv", index=False)

import matplotlib.pyplot as plt

# Figure A: Official full label curve; reference levels are horizontal because both use 200k labels.
official = rut_agg[rut_agg.scenario.eq("OFFICIAL_TEST")].sort_values("budget")
xoff = xgb_agg[xgb_agg.scenario.eq("OFFICIAL_TEST")].iloc[0]
r0off = r0_agg[r0_agg.scenario.eq("OFFICIAL_TEST")].iloc[0]

fig, ax = plt.subplots(figsize=(8.2,4.7))
ax.errorbar(
    official.budget, 100*official.tpr_mean,
    yerr=100*official.tpr_std/np.sqrt(official.n),
    marker="o", capsize=3, label="RUT + Linear (N10)"
)
ax.axhline(100*xoff.tpr_mean, linestyle="--", label="TF-IDF/XGB @ 200k (N10)")
ax.axhline(100*r0off.tpr_mean, linestyle=":", label="R0 + Linear @ 200k (N10)")
ax.set_xscale("log")
ax.set_xticks(BUDGETS, labels=["2k","5k","10k","20k","50k","100k","200k"])
ax.set_xlabel("Gelabelte Trainingsbeispiele")
ax.set_ylabel("TPR am CAL-basierten 0,5%-FPR-Betriebspunkt [%]")
ax.set_title("Vollständige Labelkurve ohne vorselektiertes Kandidatenbudget")
ax.grid(True, alpha=.25)
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES/"01_OFFICIAL_FULL_GRID_VS_200K_REFERENCES.png", dpi=200)
plt.close(fig)

# Figure B: worst-case mean TPR gap over all five scenarios.
ig = internal_global.set_index("budget").loc[BUDGETS]
cg = cross_global.set_index("budget").loc[BUDGETS]
fig, ax = plt.subplots(figsize=(8.2,4.7))
ax.plot(BUDGETS, 100*ig.worst_mean_delta_tpr, marker="o", label="RUT minus R0@200k")
ax.plot(BUDGETS, 100*cg.worst_mean_delta_tpr, marker="o", label="RUT minus XGB@200k")
ax.axhline(0.0, linestyle="--")
ax.axhline(-NI_MARGIN_TPR_PP, linestyle=":")
ax.set_xscale("log")
ax.set_xticks(BUDGETS, labels=["2k","5k","10k","20k","50k","100k","200k"])
ax.set_xlabel("Gelabelte Trainingsbeispiele für RUT")
ax.set_ylabel("Schlechteste mittlere ΔTPR über fünf Szenarien [pp]")
ax.set_title("Worst-Case-Abstand zu zwei 200k-Referenzen")
ax.grid(True, alpha=.25)
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES/"02_WORST_CASE_GAP_TWO_REFERENCES.png", dpi=200)
plt.close(fig)

print({
    "TABLES_WRITTEN": True,
    "FIGURES_WRITTEN": [
        "01_OFFICIAL_FULL_GRID_VS_200K_REFERENCES.png",
        "02_WORST_CASE_GAP_TWO_REFERENCES.png",
    ],
})


# CELL 14
# 13 — Sensitivity and claim-boundary audit

# Margin sensitivity is descriptive/supportive only; the 1 pp margin remains primary.
sensitivity = []
for margin_pp in [0.0, 0.5, 1.0, 2.0]:
    margin = margin_pp/100.0
    for b in BUDGETS:
        ps = []
        lows = []
        for scenario in SCENARIOS:
            a = RUT[(RUT.budget.astype(int)==b) & RUT.scenario.astype(str).eq(scenario)].tpr.to_numpy(float)
            x = XGB_N10[XGB_N10.scenario.astype(str).eq(scenario)].tpr.to_numpy(float)
            w = welch_noninferiority(a,x,margin)
            ps.append(w["p_ni"])
            lows.append(w["lower_one_sided"])
        sensitivity.append({
            "margin_pp":margin_pp,
            "budget":b,
            "global_iut_p_unadjusted":max(ps),
            "worst_one_sided_lower_delta_tpr":min(lows),
        })
SENS = pd.DataFrame(sensitivity)
SENS.to_csv(TABLES/"CROSS_PARADIGM_MARGIN_SENSITIVITY.csv", index=False)

DECISION = {
    "version":VERSION,
    "scientific_status":SCIENTIFIC_STATUS,
    "no_budget_preselected":True,
    "full_budget_grid":BUDGETS,
    "internal_mechanistic_reference":{
        "reference":"R0@200k + Linear",
        "minimum_holm_supported_budget":int(min_internal) if min_internal is not None else None,
        "causal_role":"supports representation-adaptation attribution because architecture/probe are matched",
    },
    "cross_paradigm_reference":{
        "reference":"TF-IDF/XGB@200k",
        "minimum_strict_all_scenario_mean_dominance_budget":int(min_strict) if min_strict is not None else None,
        "minimum_supportive_welch_ni_holm_budget":int(min_supportive) if min_supportive is not None else None,
        "causal_role":"none; system-level benchmark only",
    },
    "interpretation_limits":[
        "Do not state that the cross-paradigm gap is caused solely by TAPT/SSL.",
        "Do not describe this notebook as a prospective or independent replication.",
        "Do not convert label-count reduction into monetary savings without a cost model.",
        "Do not include TAPT pretraining compute in a label-efficiency claim unless it is measured separately.",
        "Report realized FPR beside TPR; CAL target 0.5% does not imply identical realized FINAL FPR.",
    ],
}
atomic_json(RESULTS/"FINAL_BENCHMARK_DECISION.json", DECISION)
print(json.dumps(DECISION, indent=2))


# CELL 15
# 14 — Reproducibility audit and compact output package

required_outputs = [
    AUDIT/"ANALYSIS_PLAN_LOCK.json",
    AUDIT/"RUNTIME_VERSIONS.json",
    AUDIT/"INPUT_RESOLUTION.json",
    AUDIT/"DATA_AND_SCENARIO_AUDIT.json",
    AUDIT/"PROBE_COVERAGE_AUDIT.json",
    AUDIT/"XGB_FEATURE_PROTOCOL_AUDIT.json",
    RESULTS/"N10_RUT_FULL_GRID_AND_R0_200K.csv",
    RESULTS/"XGB_200K_N10_ALL_SCENARIOS.csv",
    TABLES/"INTERNAL_RUT_VS_R0_200K_COMPONENTS.csv",
    TABLES/"INTERNAL_RUT_VS_R0_200K_GLOBAL_BUDGETS.csv",
    TABLES/"CROSS_PARADIGM_RUT_VS_XGB200K_COMPONENTS.csv",
    TABLES/"CROSS_PARADIGM_RUT_VS_XGB200K_GLOBAL_BUDGETS.csv",
    RESULTS/"FINAL_BENCHMARK_DECISION.json",
]
missing = [str(p) for p in required_outputs if not p.exists()]
if missing:
    raise RuntimeError({"missing_required_outputs":missing})

coverage = {
    "status":"COMPLETE",
    "version":VERSION,
    "rut_runs":int(RUT.run.nunique()),
    "rut_budgets":sorted(RUT.budget.astype(int).unique().tolist()),
    "xgb_n10_seeds":sorted(XGB_N10.xgb_seed.astype(int).unique().tolist()),
    "scenario_counts":{s:int(FINAL_MASKS[s].sum()) for s in SCENARIOS},
    "all_required_outputs_present":True,
    "completed_utc":datetime.now(timezone.utc).isoformat(),
}
atomic_json(AUDIT/"FINAL_COVERAGE_AUDIT.json", coverage)

zip_path = WORK_BASE / "FINAL_SSL_CROSS_PARADIGM_LABEL_BENCHMARK_N10_RESULTS_v1.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in ROOT.rglob("*"):
        if p.is_file():
            z.write(p, arcname=str(p.relative_to(ROOT)))

print(json.dumps(coverage, indent=2))
print({"RESULT_ZIP":str(zip_path)})
