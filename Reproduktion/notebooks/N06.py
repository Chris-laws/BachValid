# KI-Unterstuetzung: siehe Erklaerung der Arbeit.
# Ursprungsdatei: phreshphish_FINAL_DATA_FREEZE_BUILDER_v3_2_SCHEMA_FIX_RESUME(2).ipynb
# CELL 3

# ============================================================
# 00 — HF/Kaggle networking before Hugging Face imports
# ============================================================
import os
from pathlib import Path

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"]="600"
os.environ["HF_HUB_ETAG_TIMEOUT"]="60"
os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"

HF_HOME=Path("/kaggle/working/hf_transient") if Path("/kaggle/working").exists() else Path("/mnt/data/hf_transient")
os.environ["HF_HOME"]=str(HF_HOME)

HF_TOKEN=None
if Path("/kaggle/working").exists():
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN=UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        HF_TOKEN=os.environ.get("HF_TOKEN")
else:
    HF_TOKEN=os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError(
        "STOP: Kaggle Secret 'HF_TOKEN' fehlt. Einen Hugging-Face Read Token als "
        "Kaggle Secret HF_TOKEN anlegen und dem Notebook freigeben."
    )

os.environ["HF_TOKEN"]=HF_TOKEN
print({
    "HF_TOKEN":"AVAILABLE_NOT_PRINTED",
    "HF_HUB_DOWNLOAD_TIMEOUT":os.environ["HF_HUB_DOWNLOAD_TIMEOUT"],
    "HF_HOME":str(HF_HOME)
})


# CELL 4

# ============================================================
# 01 — Imports / immutable freeze
# ============================================================
import io, gc, re, json, html as html_std, math, time, pickle, hashlib, zipfile, warnings, shutil
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from huggingface_hub import HfApi, hf_hub_download

try:
    from lxml import html as lxml_html
    from lxml import etree
    LXML_VERSION=".".join(map(str,etree.LXML_VERSION))
except Exception as e:
    raise RuntimeError("lxml wird benötigt") from e

INPUT_ROOT=Path("/kaggle/input") if Path("/kaggle/input").exists() else Path("/mnt/data")
ROOT=(Path("/kaggle/working/phreshphish_FINAL_DATA_FREEZE_v3")
      if Path("/kaggle/working").exists()
      else Path("/mnt/data/phreshphish_FINAL_DATA_FREEZE_v3_output"))

TMP_RAW=ROOT/"_tmp_raw"
TMP_TRAIN=ROOT/"_tmp_processed_train"
META_SHARDS=ROOT/"_metadata_shards"
ROLES=ROOT/"roles"
MANIFEST=ROOT/"manifests"
LABELSETS=ROOT/"labelsets"
AUDIT=ROOT/"audit"
MARKERS=ROOT/"_markers"

for p in [ROOT,TMP_RAW,TMP_TRAIN,META_SHARDS,ROLES,MANIFEST,LABELSETS,AUDIT,MARKERS]:
    p.mkdir(parents=True,exist_ok=True)

DATASET_ID="phreshphish/phreshphish"
PINNED_REVISION="eabec4b7a66324b79cc8a0ad856d1731dc26fe1a"

EXPECTED={
    "train_rows":498255,
    "test_rows":168060,
    "train_benign":276729,
    "train_phish":221526,
    "test_benign":91260,
    "test_phish":76800
}

# Internal consistency guard: class totals must equal split totals.
if EXPECTED["train_benign"] + EXPECTED["train_phish"] != EXPECTED["train_rows"]:
    raise RuntimeError("STOP inconsistent EXPECTED train class totals")
if EXPECTED["test_benign"] + EXPECTED["test_phish"] != EXPECTED["test_rows"]:
    raise RuntimeError("STOP inconsistent EXPECTED test class totals")

SUPERVISED_N=4000
DEVELOPMENT_N=20000
CALIBRATION_BENIGN_N=50000
SSL_MAX_N=200000
SSL_NESTED=[40000,100000,200000]
FINAL_SEEDS=[42,52,62,72,82,242,252,262,272,282]

HTML_PREFIX_BYTES=4_000_000
TEXT_MAX_CHARS=50_000
STRUCT_MAX_TOKENS=50_000
DOM_MAX_NODES=4096
READ_BATCH_ROWS=256
WRITE_CHUNK_ROWS=2500

FREEZE={
    "freeze_id":"PHRESHPHISH_FINAL_DATA_FREEZE_v3_HF_SHARDED_DOM",
    "dataset_id":DATASET_ID,
    "revision":PINNED_REVISION,
    "expected":EXPECTED,
    "roles":{
        "supervised":SUPERVISED_N,
        "development":DEVELOPMENT_N,
        "calibration_benign":CALIBRATION_BENIGN_N,
        "ssl":SSL_MAX_N,
        "final_test":"official complete test"
    },
    "ssl_nested":SSL_NESTED,
    "network":"one download per official raw shard; process locally then delete raw shard",
    "static_dom":{
        "parser":"lxml.html.HTMLParser(recover=True, remove_comments=True)",
        "max_nodes":DOM_MAX_NODES,
        "order":"preorder",
        "tags":"normalized strings; later vocab fitted without FINAL_TEST"
    }
}
(AUDIT/"DATA_FREEZE_SPEC.json").write_text(json.dumps(FREEZE,indent=2),encoding="utf-8")
print(json.dumps(FREEZE,indent=2))


# CELL 5

# ============================================================
# 01b — Optional recovery from an attached failed v3 Kaggle output
# ============================================================
# Preferred path: same Kaggle working directory still exists -> nothing to do.
# Fallback: attach the failed v3 output as a Kaggle Dataset. We detect a snapshot
# containing _markers/TRAIN_ROLES_COMPLETE.json and copy only the persisted freeze
# state required to continue. This is local disk I/O, not a 36.6 GB HF redownload.

if Path("/kaggle/input").exists():
    current_ready=(MARKERS/"TRAIN_ROLES_COMPLETE.json").exists()
    if not current_ready:
        candidates=[]
        for p in Path("/kaggle/input").rglob("TRAIN_ROLES_COMPLETE.json"):
            if p.parent.name=="_markers":
                snap=p.parent.parent
                # Require evidence that this is the same v3 freeze.
                spec=snap/"audit"/"DATA_FREEZE_SPEC.json"
                if spec.exists():
                    try:
                        obj=json.loads(spec.read_text())
                        if obj.get("freeze_id")=="PHRESHPHISH_FINAL_DATA_FREEZE_v3_HF_SHARDED_DOM":
                            candidates.append(snap)
                    except Exception:
                        pass

        # De-duplicate nested references.
        uniq=[]
        for p in candidates:
            if str(p) not in {str(x) for x in uniq}:
                uniq.append(p)

        if len(uniq)>1:
            raise RuntimeError(f"STOP multiple v3 resume snapshots attached: {[str(x) for x in uniq]}")
        if len(uniq)==1:
            snap=uniq[0]
            print({"RESUME_SNAPSHOT_FOUND":str(snap)})
            for name in ["_markers","_metadata_shards","roles","manifests","labelsets","audit"]:
                src=snap/name
                dst=ROOT/name
                if src.exists():
                    print({"copy_resume_component":name})
                    shutil.copytree(src,dst,dirs_exist_ok=True)
            print({"RESUME_SNAPSHOT_IMPORT":"COMPLETE"})
        else:
            print("No attached v3 resume snapshot found; normal execution.")
    else:
        print("Existing /kaggle/working v3 state found; direct resume.")


# CELL 6

# ============================================================
# 02 — Official shard list from pinned HF revision
# ============================================================
api=HfApi(token=HF_TOKEN)
repo_info=api.dataset_info(DATASET_ID,revision=PINNED_REVISION,token=HF_TOKEN)
files=api.list_repo_files(DATASET_ID,repo_type="dataset",revision=PINNED_REVISION,token=HF_TOKEN)

TRAIN_FILES=sorted([x for x in files if re.fullmatch(r"data/train-\d+\.parquet",x)])
TEST_FILES=sorted([x for x in files if re.fullmatch(r"data/test-\d+\.parquet",x)])

if not TRAIN_FILES or not TEST_FILES:
    raise RuntimeError("STOP: offizielle train/test Parquet-Shards nicht gefunden")

source={
    "dataset_id":DATASET_ID,
    "requested_revision":PINNED_REVISION,
    "resolved_sha":repo_info.sha,
    "train_shards":TRAIN_FILES,
    "test_shards":TEST_FILES
}
(AUDIT/"SOURCE_PROVENANCE.json").write_text(json.dumps(source,indent=2),encoding="utf-8")
print({"resolved_sha":repo_info.sha,"train_shards":len(TRAIN_FILES),"test_shards":len(TEST_FILES)})


# CELL 7

# ============================================================
# 03 — HTML / URL / Static DOM processor
# ============================================================
from urllib.parse import urlsplit

COMMENT_RE=re.compile(r"<!--.*?-->",re.I|re.S)
SCRIPT_STYLE_RE=re.compile(r"<(script|style|noscript|template)\b[^>]*>.*?</\1\s*>",re.I|re.S)
TAG_RE=re.compile(r"<\s*(/?)\s*([a-zA-Z][a-zA-Z0-9:_-]*)\b[^>]*>",re.S)
TAG_STRIP_RE=re.compile(r"<[^>]+>",re.S)
WS_RE=re.compile(r"\s+")

def normalize_label(x):
    s=str(x).strip().lower()
    if s in {"phish","phishing","1","true"}:return "phish"
    if s in {"benign","legitimate","0","false"}:return "benign"
    return s

def domain_key(url):
    try:return (urlsplit(str(url)).hostname or "").strip(".").lower()
    except Exception:return ""

def normalize_tag(tag):
    if not isinstance(tag,str):return "#nonstring"
    z=tag.strip().lower()
    if z.startswith("{") and "}" in z:z=z.split("}",1)[1]
    return z or "#empty"

def clipped_html(raw):
    s="" if raw is None else str(raw)
    b=s.encode("utf-8","ignore")
    trunc=len(b)>HTML_PREFIX_BYTES
    if trunc:s=b[:HTML_PREFIX_BYTES].decode("utf-8","ignore")
    return s,trunc

def parse_dom(s):
    try:
        parser=lxml_html.HTMLParser(recover=True,remove_comments=True,encoding="utf-8")
        root=lxml_html.fromstring(s if s.strip() else "<html></html>",parser=parser)
    except Exception:
        root=lxml_html.fromstring("<html></html>")

    tags=[];parents=[];depths=[];attrs=[];child_counts=[];trunc=False
    stack=[(root,-1,0)]
    while stack:
        node,parent,depth=stack.pop()
        if len(tags)>=DOM_MAX_NODES:
            trunc=True
            break
        if not isinstance(getattr(node,"tag",None),str):
            continue
        idx=len(tags)
        tags.append(normalize_tag(node.tag))
        parents.append(int(parent))
        depths.append(int(min(depth,65535)))
        attrs.append(int(min(len(getattr(node,"attrib",{})),65535)))
        children=[ch for ch in node if isinstance(getattr(ch,"tag",None),str)]
        child_counts.append(int(min(len(children),65535)))
        for ch in reversed(children):
            stack.append((ch,idx,depth+1))
    return tags,parents,depths,attrs,child_counts,trunc

def process_row(row,include_label=True):
    s,prefix_trunc=clipped_html(row.get("html"))
    no_comments=COMMENT_RE.sub(" ",s)

    toks=[]
    for slash,tag in TAG_RE.findall(no_comments):
        toks.append(("/" if slash else "")+normalize_tag(tag))
        if len(toks)>=STRUCT_MAX_TOKENS:
            break
    struct=" ".join(toks)

    txt=SCRIPT_STYLE_RE.sub(" ",no_comments)
    txt=TAG_STRIP_RE.sub(" ",txt)
    txt=html_std.unescape(txt)
    txt=WS_RE.sub(" ",txt).strip()[:TEXT_MAX_CHARS]

    tags,parents,depths,attrs,children,trunc=parse_dom(s)

    out={
        "sha256":str(row.get("sha256")).strip().lower(),
        "url":"" if row.get("url") is None else str(row.get("url")),
        "date":row.get("date"),
        "target":row.get("target"),
        "lang":row.get("lang"),
        "lang_score":row.get("lang_score"),
        "domain":domain_key(row.get("url")),
        "text":txt,
        "html_struct":struct,
        "template_hash":hashlib.sha256(struct.encode("utf-8","ignore")).hexdigest(),
        "html_prefix_truncated":bool(prefix_trunc),
        "struct_token_count":len(toks),
        "dom_tag":tags,
        "dom_parent_idx":parents,
        "dom_depth":depths,
        "dom_attr_count":attrs,
        "dom_child_count":children,
        "dom_node_count":len(tags),
        "dom_max_depth":max(depths) if depths else 0,
        "dom_truncated":bool(trunc)
    }
    if include_label:
        out["label"]=normalize_label(row.get("label"))
    return out

test=process_row({
    "sha256":"a"*64,
    "url":"https://a.example.com/x",
    "html":"<html><body><form><input><button>Go</button></form></body></html>",
    "label":"phish"
})
assert test["dom_parent_idx"][0]==-1
assert len(test["dom_tag"])==test["dom_node_count"]
assert test["domain"]=="a.example.com"
print({"STATIC_DOM_SELFTEST":"PASS","nodes":test["dom_node_count"],"lxml":LXML_VERSION})


# CELL 8

# ============================================================
# 04 — Robust shard downloader + local processors
# ============================================================
RAW_COLS=["sha256","url","label","date","target","lang","lang_score","html"]

def stem(repo_file):
    return Path(repo_file).stem

def download_shard(repo_file,max_attempts=8):
    last=None
    for attempt in range(1,max_attempts+1):
        try:
            path=hf_hub_download(
                repo_id=DATASET_ID,
                filename=repo_file,
                repo_type="dataset",
                revision=PINNED_REVISION,
                token=HF_TOKEN,
                local_dir=TMP_RAW
            )
            return Path(path)
        except Exception as e:
            last=e
            wait=min(60,2**attempt)
            print({"download_retry":repo_file,"attempt":attempt,"sleep_s":wait,"error":repr(e)[:250]})
            time.sleep(wait)
    raise RuntimeError(f"STOP download failed: {repo_file}") from last

def cleanup_raw(path):
    try:Path(path).unlink(missing_ok=True)
    except Exception:pass
    gc.collect()

def process_train_shard(repo_file):
    s=stem(repo_file)
    marker=MARKERS/f"train_{s}.json"
    if marker.exists():
        print({"reuse_train_shard":s})
        return

    for p in TMP_TRAIN.glob(f"{s}_part*.parquet"):
        p.unlink()
    (META_SHARDS/f"{s}.parquet").unlink(missing_ok=True)

    raw=download_shard(repo_file)
    pf=pq.ParquetFile(raw)
    cols=[c for c in RAW_COLS if c in pf.schema.names]
    required={"sha256","url","label","date","html"}
    if not required.issubset(cols):
        raise RuntimeError(f"STOP schema {repo_file}: missing {sorted(required-set(cols))}")

    meta=[];buf=[];part=0;rows=0;t0=time.time()
    for batch in pf.iter_batches(columns=cols,batch_size=READ_BATCH_ROWS):
        d=batch.to_pydict()
        n=len(next(iter(d.values()))) if d else 0
        for i in range(n):
            row={c:(d[c][i] if c in d else None) for c in RAW_COLS}
            out=process_row(row,include_label=True)
            buf.append(out)
            meta.append({k:out.get(k) for k in ["sha256","url","label","date","target","lang","lang_score"]})
            rows+=1
            if len(buf)>=WRITE_CHUNK_ROWS:
                pd.DataFrame(buf).to_parquet(
                    TMP_TRAIN/f"{s}_part{part:03d}.parquet",
                    index=False,compression="zstd"
                )
                buf.clear();part+=1
        if rows and rows%5000<READ_BATCH_ROWS:
            print({"train_shard":s,"rows":rows,"minutes":round((time.time()-t0)/60,2)})

    if buf:
        pd.DataFrame(buf).to_parquet(
            TMP_TRAIN/f"{s}_part{part:03d}.parquet",
            index=False,compression="zstd"
        )
        part+=1

    pd.DataFrame(meta).to_parquet(META_SHARDS/f"{s}.parquet",index=False,compression="zstd")
    marker.write_text(json.dumps({"repo_file":repo_file,"rows":rows,"parts":part},indent=2))
    cleanup_raw(raw)
    print({"completed_train_shard":s,"rows":rows})

def process_test_shard(repo_file,final_dir):
    s=stem(repo_file)
    marker=MARKERS/f"test_{s}.json"
    if marker.exists():
        print({"reuse_test_shard":s})
        return

    for p in final_dir.glob(f"{s}_part*.parquet"):
        p.unlink()
    (META_SHARDS/f"{s}.parquet").unlink(missing_ok=True)

    raw=download_shard(repo_file)
    pf=pq.ParquetFile(raw)
    cols=[c for c in RAW_COLS if c in pf.schema.names]

    meta=[];buf=[];part=0;rows=0;t0=time.time()
    for batch in pf.iter_batches(columns=cols,batch_size=READ_BATCH_ROWS):
        d=batch.to_pydict()
        n=len(next(iter(d.values()))) if d else 0
        for i in range(n):
            row={c:(d[c][i] if c in d else None) for c in RAW_COLS}
            out=process_row(row,include_label=True)
            buf.append(out)
            meta.append({k:out.get(k) for k in [
                "sha256","url","label","date","target","lang","lang_score","domain","template_hash"
            ]})
            rows+=1
            if len(buf)>=WRITE_CHUNK_ROWS:
                pd.DataFrame(buf).to_parquet(
                    final_dir/f"{s}_part{part:03d}.parquet",
                    index=False,compression="zstd"
                )
                buf.clear();part+=1

    if buf:
        pd.DataFrame(buf).to_parquet(
            final_dir/f"{s}_part{part:03d}.parquet",
            index=False,compression="zstd"
        )
        part+=1

    pd.DataFrame(meta).to_parquet(META_SHARDS/f"{s}.parquet",index=False,compression="zstd")
    marker.write_text(json.dumps({"repo_file":repo_file,"rows":rows,"parts":part},indent=2))
    cleanup_raw(raw)
    print({"completed_test_shard":s,"rows":rows,"minutes":round((time.time()-t0)/60,2)})


# CELL 9

# ============================================================
# 05 — ONE raw-network pass over official TRAIN
# ============================================================
for i,repo_file in enumerate(TRAIN_FILES,1):
    print({"TRAIN_SHARD":f"{i}/{len(TRAIN_FILES)}","file":repo_file})
    process_train_shard(repo_file)

train_meta=pd.concat(
    [pd.read_parquet(META_SHARDS/f"{stem(x)}.parquet") for x in TRAIN_FILES],
    ignore_index=True
)
train_meta["sha256"]=train_meta.sha256.astype(str).str.lower()

if train_meta.sha256.duplicated().any():
    raise RuntimeError("STOP duplicate train SHA")

counts=train_meta.label.value_counts().to_dict()
if (
    len(train_meta)!=EXPECTED["train_rows"]
    or counts.get("benign",0)!=EXPECTED["train_benign"]
    or counts.get("phish",0)!=EXPECTED["train_phish"]
):
    raise RuntimeError({
        "STOP":"unexpected official train counts",
        "rows":len(train_meta),
        "labels":counts
    })

train_meta.to_parquet(MANIFEST/"official_train_metadata.parquet",index=False,compression="zstd")
print({"TRAIN_PASS_COMPLETE":len(train_meta),"labels":counts})


# CELL 10

# ============================================================
# 06 — Legacy anchor + deterministic train-role allocation
# ============================================================
def stable_rank(series,salt):
    return np.fromiter(
        (
            int(hashlib.sha256((salt+"|"+x).encode()).hexdigest()[:16],16)
            for x in series.astype(str)
        ),
        dtype=np.uint64,count=len(series)
    )

def find_legacy():
    hits=list(INPUT_ROOT.rglob("split_roles_and_holdout_cache_v2_ram_safe.pkl"))
    if not hits:
        return None,{"available":False,"used":False}

    p=sorted(hits,key=lambda x:len(str(x)))[0]
    with open(p,"rb") as f:
        obj=pickle.load(f)

    old=obj.get("train_df") if isinstance(obj,dict) else None
    audit={"available":True,"path":str(p),"used":False}
    if not isinstance(old,pd.DataFrame) or len(old)!=4000 or not {"sha256","label"}.issubset(old.columns):
        return None,audit

    x=old[["sha256","label"]].copy()
    x["sha256"]=x.sha256.astype(str).str.lower()
    x["legacy_label"]=x.label.map(
        lambda z:"phish" if str(z).lower() in {"1","phish","phishing","true"} else "benign"
    )
    off=train_meta.set_index("sha256")
    audit["found"]=int(x.sha256.isin(off.index).sum())

    if audit["found"]==4000:
        mapped=x.sha256.map(off.label)
        audit["label_mismatch"]=int((mapped.to_numpy()!=x.legacy_label.to_numpy()).sum())
        if audit["label_mismatch"]==0 and x.sha256.nunique()==4000:
            audit["used"]=True
            return x.sha256.tolist(),audit

    return None,audit

legacy,legacy_audit=find_legacy()
(AUDIT/"LEGACY_ANCHOR_AUDIT.json").write_text(json.dumps(legacy_audit,indent=2),encoding="utf-8")

roles=train_meta.copy()
roles["role"]="UNASSIGNED"
roles["ssl_rank"]=-1

if legacy is not None:
    sup=set(legacy)
else:
    sup=set()
    for lab in ["benign","phish"]:
        q=roles[roles.label.eq(lab)].copy()
        q["_r"]=stable_rank(q.sha256,f"SUP_{lab}_v3")
        sup.update(q.nsmallest(SUPERVISED_N//2,"_r").sha256.tolist())

roles.loc[roles.sha256.isin(sup),"role"]="SUPERVISED_POOL"

dev=set()
for lab in ["benign","phish"]:
    q=roles[(roles.role=="UNASSIGNED")&(roles.label==lab)].copy()
    q["_r"]=stable_rank(q.sha256,f"DEV_{lab}_v3")
    dev.update(q.nsmallest(DEVELOPMENT_N//2,"_r").sha256.tolist())
roles.loc[roles.sha256.isin(dev),"role"]="DEVELOPMENT"

q=roles[(roles.role=="UNASSIGNED")&(roles.label=="benign")].copy()
q["_r"]=stable_rank(q.sha256,"CAL_BENIGN_v3")
cal=set(q.nsmallest(CALIBRATION_BENIGN_N,"_r").sha256.tolist())
roles.loc[roles.sha256.isin(cal),"role"]="FPR_CALIBRATION_BENIGN"

q=roles[roles.role=="UNASSIGNED"].copy()
q["_r"]=stable_rank(q.sha256,"SSL_UNLABELED_v3")
ssl=q.nsmallest(SSL_MAX_N,"_r").sort_values("_r")
ssl_map={s:i for i,s in enumerate(ssl.sha256.tolist())}
roles.loc[roles.sha256.isin(ssl_map),"role"]="SSL_POOL"
roles.loc[roles.sha256.isin(ssl_map),"ssl_rank"]=roles.loc[
    roles.sha256.isin(ssl_map),"sha256"
].map(ssl_map).astype(int)

roles.loc[roles.role=="UNASSIGNED","role"]="RESERVE"

for n in SSL_NESTED:
    roles[f"ssl_{n//1000}k"]=(roles.role.eq("SSL_POOL")&(roles.ssl_rank<n))

expected_roles={
    "SUPERVISED_POOL":4000,
    "DEVELOPMENT":20000,
    "FPR_CALIBRATION_BENIGN":50000,
    "SSL_POOL":200000
}
for r,n in expected_roles.items():
    got=int((roles.role==r).sum())
    if got!=n:
        raise RuntimeError(f"STOP {r}: {got}!={n}")

roles.to_parquet(
    MANIFEST/"train_role_manifest_PRIVATE_WITH_LABELS.parquet",
    index=False,compression="zstd"
)
roles.groupby(["role","label"]).size().rename("n").reset_index().to_csv(
    AUDIT/"ROLE_CLASS_COUNTS.csv",index=False
)
print(roles.groupby(["role","label"]).size())


# CELL 11

# ============================================================
# 07 — Build train role Parquets from compact temp train shards
# ============================================================
ROLE_DIRS={
    "SUPERVISED_POOL":ROLES/"supervised_pool",
    "DEVELOPMENT":ROLES/"development",
    "FPR_CALIBRATION_BENIGN":ROLES/"fpr_calibration_benign",
    "SSL_POOL":ROLES/"ssl_pool_200k"
}
ROLE_BUILD_MARKER=MARKERS/"TRAIN_ROLES_COMPLETE.json"

if not ROLE_BUILD_MARKER.exists():
    for d in ROLE_DIRS.values():
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True,exist_ok=True)

    role_map=roles.set_index("sha256")[["role","ssl_rank"]]
    counters=Counter()

    for p in sorted(TMP_TRAIN.glob("train-*_part*.parquet")):
        df=pd.read_parquet(p)
        df["_role"]=df.sha256.astype(str).map(role_map["role"]).to_numpy()
        df["_ssl_rank"]=df.sha256.astype(str).map(role_map["ssl_rank"]).to_numpy()

        for role,outdir in ROLE_DIRS.items():
            x=df[df._role.eq(role)].copy()
            if not len(x):
                continue
            if role=="SSL_POOL":
                x["ssl_rank"]=x["_ssl_rank"].astype(int)
                x=x.drop(columns=["label"],errors="ignore")
            x=x.drop(columns=["_role","_ssl_rank"],errors="ignore")
            x.to_parquet(
                outdir/f"part-{counters[role]:05d}.parquet",
                index=False,compression="zstd"
            )
            counters[role]+=1

    row_counts={}
    for role,d in ROLE_DIRS.items():
        row_counts[role]=sum(
            pq.ParquetFile(p).metadata.num_rows for p in d.glob("*.parquet")
        )
        if row_counts[role]!=expected_roles[role]:
            raise RuntimeError(
                f"STOP materialized {role}: {row_counts[role]}!={expected_roles[role]}"
            )

    ROLE_BUILD_MARKER.write_text(
        json.dumps({"status":"COMPLETE","rows":row_counts},indent=2),
        encoding="utf-8"
    )
    print({"TRAIN_ROLE_BUILD":"COMPLETE","rows":row_counts})

if ROLE_BUILD_MARKER.exists():
    shutil.rmtree(TMP_TRAIN,ignore_errors=True)
    TMP_TRAIN.mkdir(parents=True,exist_ok=True)
    print("Temporary all-train processed shards deleted.")


# CELL 12

# ============================================================
# 08 — Label-budget freezes
# ============================================================
sup=roles[roles.role.eq("SUPERVISED_POOL")].sort_values("sha256").reset_index(drop=True)
sup["supervised_index"]=np.arange(len(sup),dtype=np.int32)
sup.to_parquet(MANIFEST/"supervised_pool_manifest.parquet",index=False,compression="zstd")

def exact_budget(seed,n):
    if n==4000:
        return np.arange(4000,dtype=np.int32)
    out=[]
    for lab,nlab in [("benign",n//2),("phish",n-n//2)]:
        q=sup[sup.label.eq(lab)].copy()
        q["_r"]=stable_rank(q.sha256,f"LABEL_{seed}_{n}_{lab}")
        out.extend(q.nsmallest(nlab,"_r").supervised_index.tolist())
    return np.sort(np.asarray(out,dtype=np.int32))

audit=[]
for seed in FINAL_SEEDS:
    for name,n in [("B10",400),("B25R",1000),("B100",4000)]:
        idx=exact_budget(seed,n)
        np.save(LABELSETS/f"{name}_seed{seed}.npy",idx)
        yy=sup.iloc[idx].label
        audit.append({
            "seed":seed,"budget":name,"n":n,
            "benign":int((yy=="benign").sum()),
            "phish":int((yy=="phish").sum()),
            "idx_sha256":hashlib.sha256(idx.tobytes()).hexdigest()
        })

pd.DataFrame(audit).to_csv(AUDIT/"LABELSET_AUDIT.csv",index=False)
(AUDIT/"ACTIVE_LEARNING_NOTE.txt").write_text(
    "B25AL wird nicht vom Data Freeze Builder erzeugt. Active Learning ist ein separates Verfahren. "
    "Legacy AL25 darf nur bei exakt wiederverwendetem Legacy-Supervised-Pool übernommen werden.",
    encoding="utf-8"
)
print(pd.DataFrame(audit).head())


# CELL 13

# ============================================================
# 09 — ONE raw-network pass over official TEST → FINAL_TEST
# ============================================================
FINAL_DIR=ROLES/"final_test"
FINAL_DIR.mkdir(parents=True,exist_ok=True)

for i,repo_file in enumerate(TEST_FILES,1):
    print({"TEST_SHARD":f"{i}/{len(TEST_FILES)}","file":repo_file})
    process_test_shard(repo_file,FINAL_DIR)

test_meta=pd.concat(
    [pd.read_parquet(META_SHARDS/f"{stem(x)}.parquet") for x in TEST_FILES],
    ignore_index=True
)
test_meta["sha256"]=test_meta.sha256.astype(str).str.lower()

if test_meta.sha256.duplicated().any():
    raise RuntimeError("STOP duplicate test SHA")
if set(test_meta.sha256)&set(train_meta.sha256):
    raise RuntimeError("STOP exact train/test overlap")

counts=test_meta.label.value_counts().to_dict()
if (
    len(test_meta)!=EXPECTED["test_rows"]
    or counts.get("benign",0)!=EXPECTED["test_benign"]
    or counts.get("phish",0)!=EXPECTED["test_phish"]
):
    raise RuntimeError({
        "STOP":"unexpected official test counts",
        "rows":len(test_meta),
        "labels":counts
    })

test_meta["role"]="FINAL_TEST"
test_meta.to_parquet(
    MANIFEST/"final_test_manifest_SEALED.parquet",
    index=False,compression="zstd"
)
print({"TEST_PASS_COMPLETE":len(test_meta),"labels":counts})


# CELL 14

# ============================================================
# 10 — Low-FPR resolution + exact OOD metadata
# ============================================================
def resolution(name,nneg,npos):
    out={
        "set":name,
        "n_neg":int(nneg),
        "n_pos":int(npos),
        "one_fp_fpr":1/max(nneg,1)
    }
    for f in [0.0001,0.001,0.0025,0.005,0.01,0.02]:
        out[f"expected_fp_at_{f}"]=f*nneg
        out[f"supports_10fp_{f}"]=bool(f*nneg>=10)
        out[f"supports_50fp_{f}"]=bool(f*nneg>=50)
    return out

dev=roles[roles.role.eq("DEVELOPMENT")]
FPR=pd.DataFrame([
    resolution("FPR_CALIBRATION_BENIGN",CALIBRATION_BENIGN_N,0),
    resolution(
        "DEVELOPMENT",
        int((dev.label=="benign").sum()),
        int((dev.label=="phish").sum())
    ),
    resolution(
        "FINAL_TEST",
        int((test_meta.label=="benign").sum()),
        int((test_meta.label=="phish").sum())
    )
])
FPR.to_csv(AUDIT/"LOW_FPR_RESOLUTION.csv",index=False)

exposure_roles={"SUPERVISED_POOL","DEVELOPMENT","FPR_CALIBRATION_BENIGN","SSL_POOL"}
ex=roles[roles.role.isin(exposure_roles)]
seen_domains=set(ex.url.map(domain_key).fillna("").astype(str))
seen_domains.discard("")

seen_templates=set()
for d in ROLE_DIRS.values():
    for p in d.glob("*.parquet"):
        q=pd.read_parquet(p,columns=["template_hash"])
        seen_templates.update(q.template_hash.fillna("").astype(str))
seen_templates.discard("")

test_struct=test_meta.copy()
test_struct["domain_ood_exact"]=~test_struct.domain.fillna("").astype(str).isin(seen_domains)
test_struct["template_ood_exact"]=~test_struct.template_hash.fillna("").astype(str).isin(seen_templates)
test_struct["domain_template_ood_exact"]=(
    test_struct.domain_ood_exact & test_struct.template_ood_exact
)

test_struct[
    ["sha256","domain_ood_exact","template_ood_exact","domain_template_ood_exact"]
].to_parquet(
    MANIFEST/"final_test_exact_ood_flags_SEALED.parquet",
    index=False,compression="zstd"
)

ood={
    "domain_ood_exact":int(test_struct.domain_ood_exact.sum()),
    "template_ood_exact":int(test_struct.template_ood_exact.sum()),
    "domain_template_ood_exact":int(test_struct.domain_template_ood_exact.sum()),
    "near_duplicate_filter":"NOT YET APPLIED; algorithm+threshold must be frozen before score inspection"
}
(AUDIT/"EXACT_OOD_METADATA_COUNTS.json").write_text(json.dumps(ood,indent=2),encoding="utf-8")
display(FPR)
print(ood)


# CELL 15

# ============================================================
# 10b — v3.2 nested DOM schema preflight
# ============================================================
_probe_files=sorted(ROLE_DIRS["SUPERVISED_POOL"].glob("*.parquet"))
if not _probe_files:
    raise RuntimeError("STOP no SUPERVISED_POOL parquet for schema preflight")

_pf=pq.ParquetFile(_probe_files[0])
_physical_names=set(_pf.schema.names)
_arrow_names=set(_pf.schema_arrow.names)

_nested={"dom_tag","dom_parent_idx","dom_depth","dom_attr_count","dom_child_count"}
if not _nested.issubset(_arrow_names):
    raise RuntimeError(
        f"STOP v3.2 preflight: nested DOM fields truly absent from Arrow schema: "
        f"{sorted(_nested-_arrow_names)}"
    )

# Actual read proves these logical columns can be projected from the parquet file.
_probe=pd.read_parquet(_probe_files[0],columns=sorted(_nested)).head(1)

print({
    "DOM_ARROW_SCHEMA_PREFLIGHT":"PASS",
    "file":_probe_files[0].name,
    "nested_fields_in_arrow_schema":sorted(_nested),
    "physical_schema_missing_nested_names":sorted(_nested-_physical_names),
    "logical_column_read":"PASS"
})
del _pf,_probe


# CELL 16

# ============================================================
# 11 — Role / SSL-label / DOM integrity
# v3.2: use logical Arrow top-level schema for nested list columns
# ============================================================
def _safe_list(x):
    if x is None:
        return []
    if isinstance(x,np.ndarray):
        return x.tolist()
    if isinstance(x,(list,tuple)):
        return list(x)
    # Defensive fallback for Arrow/Pandas scalar wrappers.
    try:
        return list(x)
    except Exception:
        return []

def _validate_dom_sample(p,sample_n=32):
    cols=[
        "dom_tag","dom_parent_idx","dom_depth","dom_attr_count","dom_child_count",
        "dom_node_count","dom_max_depth","dom_truncated"
    ]
    q=pd.read_parquet(p,columns=cols)
    if len(q)>sample_n:
        # Deterministic evenly-spaced sample; no randomness and no label dependency.
        idx=np.linspace(0,len(q)-1,sample_n,dtype=int)
        q=q.iloc[idx]

    checked=0
    for _,r in q.iterrows():
        n=int(r["dom_node_count"])
        tags=_safe_list(r["dom_tag"])
        parents=[int(x) for x in _safe_list(r["dom_parent_idx"])]
        depths=_safe_list(r["dom_depth"])
        attrs=_safe_list(r["dom_attr_count"])
        children=_safe_list(r["dom_child_count"])

        lengths={
            "dom_tag":len(tags),
            "dom_parent_idx":len(parents),
            "dom_depth":len(depths),
            "dom_attr_count":len(attrs),
            "dom_child_count":len(children)
        }
        bad={k:v for k,v in lengths.items() if v!=n}
        if bad:
            raise RuntimeError(
                f"STOP DOM sample length mismatch in {p.name}: "
                f"dom_node_count={n}, lengths={lengths}"
            )

        if parents:
            if parents[0] != -1:
                raise RuntimeError(
                    f"STOP DOM root parent must be -1 in {p.name}; got {parents[0]}"
                )
            if any(x < -1 or x >= n for x in parents):
                raise RuntimeError(
                    f"STOP invalid DOM parent index in {p.name}: n={n}"
                )

        if n and int(r["dom_max_depth"]) < 0:
            raise RuntimeError(f"STOP invalid negative DOM depth in {p.name}")

        checked+=1
    return checked

def scan_role(directory,expect_label):
    files=sorted(directory.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"STOP no files: {directory}")

    rows=0
    common_cols=None
    trunc=0
    node_chunks=[]
    sample_checked=0

    required={
        "dom_tag","dom_parent_idx","dom_depth","dom_attr_count","dom_child_count",
        "dom_node_count","dom_max_depth","dom_truncated"
    }

    for p in files:
        pf=pq.ParquetFile(p)
        rows+=pf.metadata.num_rows

        # IMPORTANT:
        # pf.schema is the physical/unconverted Parquet schema and nested LIST fields
        # can appear as leaf-level names. schema_arrow reconstructs the logical
        # top-level Arrow columns written by pandas/pyarrow.
        names=set(pf.schema_arrow.names)
        common_cols=names if common_cols is None else common_cols & names

        missing_here=required-names
        if missing_here:
            raise RuntimeError(
                f"STOP DOM fields genuinely missing from Arrow schema in {p.name}: "
                f"{sorted(missing_here)}"
            )

        q=pd.read_parquet(p,columns=["dom_node_count","dom_truncated"])
        trunc+=int(q.dom_truncated.sum())
        node_chunks.append(q.dom_node_count.to_numpy())

        # Real content validation on a small deterministic sample per file.
        sample_checked+=_validate_dom_sample(p,sample_n=32)

    if expect_label and "label" not in common_cols:
        raise RuntimeError(f"STOP label missing: {directory}")
    if not expect_label and "label" in common_cols:
        raise RuntimeError(f"STOP SSL leaked label: {directory}")

    if not required.issubset(common_cols):
        raise RuntimeError(
            f"STOP DOM fields missing from role-common Arrow schema: "
            f"{sorted(required-common_cols)}"
        )

    nodes=np.concatenate(node_chunks)
    return {
        "rows":rows,
        "columns":sorted(common_cols),
        "dom_sample_rows_checked":int(sample_checked),
        "dom_truncated_n":trunc,
        "dom_truncated_rate":trunc/max(rows,1),
        "dom_nodes_median":float(np.median(nodes)),
        "dom_nodes_p99":float(np.quantile(nodes,.99))
    }

scan={
    "SUPERVISED_POOL":scan_role(ROLE_DIRS["SUPERVISED_POOL"],True),
    "DEVELOPMENT":scan_role(ROLE_DIRS["DEVELOPMENT"],True),
    "FPR_CALIBRATION_BENIGN":scan_role(ROLE_DIRS["FPR_CALIBRATION_BENIGN"],True),
    "SSL_POOL":scan_role(ROLE_DIRS["SSL_POOL"],False),
    "FINAL_TEST":scan_role(FINAL_DIR,True)
}

(AUDIT/"MATERIALIZED_ROLE_INTEGRITY.json").write_text(
    json.dumps(scan,indent=2),encoding="utf-8"
)
(AUDIT/"DOM_PARSER_PROVENANCE.json").write_text(json.dumps({
    "parser":"lxml.html.HTMLParser",
    "recover":True,
    "remove_comments":True,
    "lxml_version":LXML_VERSION,
    "order":"preorder",
    "max_nodes":DOM_MAX_NODES,
    "dom_tag_storage":"normalized string",
    "browser_rendered":False,
    "integrity_schema":"ParquetFile.schema_arrow.names",
    "integrity_content_check":"deterministic sample up to 32 rows per parquet file"
},indent=2),encoding="utf-8")

print(json.dumps(scan,indent=2))


# CELL 17

# ============================================================
# 12 — Completion + compact handoff ZIP
# ============================================================
def sha256_file(path,chunk=8*1024*1024):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        while True:
            b=f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

small=[]
for base in [MANIFEST,LABELSETS,AUDIT]:
    for p in sorted(base.rglob("*")):
        if p.is_file():
            small.append({
                "path":str(p.relative_to(ROOT)),
                "bytes":p.stat().st_size,
                "sha256":sha256_file(p)
            })
pd.DataFrame(small).to_csv(AUDIT/"SMALL_ARTIFACT_SHA256.csv",index=False)

role_files=[]
for p in sorted(ROLES.rglob("*.parquet")):
    role_files.append({
        "path":str(p.relative_to(ROOT)),
        "bytes":p.stat().st_size,
        "rows":pq.ParquetFile(p).metadata.num_rows
    })
pd.DataFrame(role_files).to_csv(AUDIT/"ROLE_FILE_MANIFEST.csv",index=False)

completion={
    "status":"COMPLETE",
    "freeze_id":"PHRESHPHISH_FINAL_DATA_FREEZE_v3_HF_SHARDED_DOM",
    "revision":PINNED_REVISION,
    "hf_authenticated":True,
    "network_design":"one raw download per official shard; local processing; raw shard deleted",
    "legacy_anchor":legacy_audit,
    "ssl_export_contains_label":False,
    "final_test_used_for_model_selection":False,
    "static_dom":True,
    "browser_rendered_dom":False,
    "near_duplicate_ood":"not yet frozen",
    "roles":{
        "supervised":"roles/supervised_pool",
        "development":"roles/development",
        "calibration":"roles/fpr_calibration_benign",
        "ssl":"roles/ssl_pool_200k",
        "final_test":"roles/final_test"
    }
}
(ROOT/"FINAL_DATA_FREEZE_COMPLETE.json").write_text(
    json.dumps(completion,indent=2),encoding="utf-8"
)

handoff=(
    Path("/kaggle/working/PHRESHPHISH_FINAL_DATA_FREEZE_v3_HANDOFF.zip")
    if Path("/kaggle/working").exists()
    else ROOT.parent/"PHRESHPHISH_FINAL_DATA_FREEZE_v3_HANDOFF.zip"
)
with zipfile.ZipFile(handoff,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for base in [MANIFEST,LABELSETS,AUDIT]:
        for p in base.rglob("*"):
            if p.is_file():
                z.write(p,arcname=str(p.relative_to(ROOT)))
    z.write(
        ROOT/"FINAL_DATA_FREEZE_COMPLETE.json",
        arcname="FINAL_DATA_FREEZE_COMPLETE.json"
    )

with zipfile.ZipFile(handoff) as z:
    bad=z.testzip()
    if bad:
        raise RuntimeError(bad)

shutil.rmtree(TMP_RAW,ignore_errors=True)
shutil.rmtree(HF_HOME,ignore_errors=True)

print(json.dumps(completion,indent=2))
print({
    "HANDOFF_ZIP":str(handoff),
    "SAVE_FULL_KAGGLE_OUTPUT_AS_DATASET":True
})
