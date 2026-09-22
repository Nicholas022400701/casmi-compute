#!/bin/bash
# shard.sh — GitHub Actions runner for ONE shard of a sharded ICE scoring job (same module/inputs/outputs as the sandbox pool's pool.sh, no heartbeats).
# Inputs (env): KAGGLE_API_TOKEN (secret)  JOB (name/tag, e.g. ranker)  SHARD  N  JOBS_DS (private Kaggle dataset)  INPUTS (comma list of files in JOBS_DS)
#   [JOBS_MD5 comma list file=md5 — verified after download; mismatch, or any INPUTS file left unpinned while JOBS_MD5 is set -> exit 1 (D78)]  [ARGS module args]  [MODULE casmi.fwdsim.runShard]
#   [SRC_DS nicholasooo/casmi-c1-pool-jobs] [SRC_DIR src_4a3d6d5] [SRC_MD5 tree md5 pin of the snapshot's .py files; default set for src_4a3d6d5]  [CKPT_DS/GEN/INTEN]  DSPREFIX (output dataset nicholasooo/<DSPREFIX>-sNN)  [THREADS 4] [CHUNK 512] [BATCH 16] [MAXNODES 100] [SPARSEK 100]
# Output dataset files are prefixed <JOB>_sNN_ exactly like the pool (chunk*.parquet, scores/pairs/summary/selftest, pool_summary.json with cpu model + secPerJob).
# Logging rule: counts and timings only — never SMILES, keys, tokens.
set -uo pipefail
: "${KAGGLE_API_TOKEN:?}" "${JOB:?}" "${SHARD:?}" "${N:?}" "${JOBS_DS:?}" "${INPUTS:?}" "${DSPREFIX:?}"; export KAGGLE_API_TOKEN
MODULE=${MODULE:-casmi.fwdsim.runShard}; ARGS=${ARGS:---rows jobs/rows.parquet --queries jobs/queries.parquet --head all --tag $JOB}
SRC_DS=${SRC_DS:-nicholasooo/casmi-c1-pool-jobs}; SRC_DIR=${SRC_DIR:-src_4a3d6d5}; JOBS_MD5=${JOBS_MD5:-}
CKPT_DS=${CKPT_DS:-nicholasooo/casmi-m2-fwdsim-assets}; GEN=${GEN:-iceberg_msg_all/gen/best.ckpt}; INTEN=${INTEN:-iceberg_msg_all/inten_contr/best.ckpt}
THREADS=${THREADS:-4}; CHUNK=${CHUNK:-512}; BATCH=${BATCH:-16}; MAXNODES=${MAXNODES:-100}; SPARSEK=${SPARSEK:-100}
ROOT=${ROOT:-$HOME/ice}; S=$(printf 's%02d' "$SHARD"); DS=nicholasooo/$DSPREFIX-$S; OUT=$ROOT/out/$S; T_BOOT=$(date +%s)
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
# every Kaggle call goes through kg(): up to 8 attempts with growing jittered backoff (matrix runs start 20-40 jobs at once -> the API answers 429 without this)
kg() { local i R; for i in 1 2 3 4 5 6 7 8; do R=$(kaggle "$@" 2>&1) && { echo "$R"; return 0; }; case "$R" in *429*|*"Too Many"*|*timed*out*|*"Connection"*|*"503"*|*"502"*) log "kaggle ${1} ${2}: transient error (attempt $i)"; sleep $(( i * 15 + RANDOM % 30 ));; *) echo "$R"; return 1;; esac; done; echo "$R"; return 1; }
dl() { local want=$1 i; shift; for i in 1 2 3; do [ -f "$want" ] && return 0; kg datasets download "$@" >/dev/null; [ -f "$want" ] || sleep $(( 20 + RANDOM % 40 )); done; [ -f "$want" ]; }   # dl <expected file> <kaggle download args>
mkdir -p "$ROOT/log" "$OUT" "$ROOT/jobs" "$ROOT/ds" && cd "$ROOT"
STAGGER=${STAGGER:-$(( RANDOM % 90 ))}; log "start stagger ${STAGGER}s (shard $SHARD)"; sleep "$STAGGER"
# ---- setup (identical package set to pool.sh) ----
export PATH=$HOME/.local/bin:$PATH
which uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -d venv ] || uv venv -q --python 3.12 venv; source venv/bin/activate
uv pip install -q kaggle
if [ ! -f wh/.done ]; then
  kg datasets download nicholasooo/casmi-m2-fwdsim-wheels -p wh --unzip -q >/dev/null || { log 'wheels download failed'; exit 1; }
  (cd wh && for f in *2.6.0cpu*.whl; do mv "$f" "${f/2.6.0cpu/2.6.0+cpu}"; done; for f in *pt26cpu*.whl; do mv "$f" "${f/2.1.2pt26cpu/2.1.2+pt26cpu}"; done) && touch wh/.done
fi
uv pip install -q sympy==1.13.1 filelock jinja2 fsspec networkx typing-extensions setuptools packaging requests pydantic numpy pandas h5py scipy scikit-learn tqdm pyyaml einops psutil joblib matplotlib seaborn pathos easydict appdirs aiohttp pillow omegaconf polars pyarrow
uv pip install -q --no-index --no-deps --find-links wh torch dgl torch_scatter pygmtools pytorch_lightning torchmetrics lightning_utilities platformdirs multiprocess dill rdkit ms_pred
# pinned source snapshot (zip uploaded to SRC_DS, unpacked by Kaggle as <SRC_DIR>/src) and model checkpoints
[ -f src/casmi/fwdsim/runShard.py ] || { rm -rf srcdl && dl "srcdl/$SRC_DIR/src/casmi/fwdsim/runShard.py" "$SRC_DS" -p srcdl -q --unzip && mv "srcdl/$SRC_DIR/src" src && rm -rf srcdl; }
[ -f src/casmi/fwdsim/runShard.py ] || { log 'src snapshot missing'; exit 1; }
CK=ck/${CKPT_DS#*/}; for f in "$GEN" "$INTEN"; do dl "$CK/$f" "$CKPT_DS" -f "$f" -p "$CK/$(dirname "$f")" -q --unzip; [ -f "$CK/$f" ] || { log "ckpt missing: $f"; exit 1; }; done
# inputs + md5 pins
for f in ${INPUTS//,/ }; do dl "jobs/$f" "$JOBS_DS" -f "$f" -p jobs -q --unzip; [ -f "jobs/$f" ] || { log "input missing: $f"; exit 1; }; done
PINNED=,; for kv in ${JOBS_MD5//,/ }; do f=${kv%%=*}; want=$(echo "${kv#*=}" | tr 'A-F' 'a-f'); [ -f "jobs/$f" ] || { log "md5 pin for unknown input $f"; exit 1; }; have=$(md5sum "jobs/$f" | cut -d' ' -f1); [ "$have" = "$want" ] || { log "input md5 mismatch $f: have $have want $want -> refusing"; exit 1; }; log "md5 ok $f"; PINNED="$PINNED$f,"; done
if [ -n "$JOBS_MD5" ]; then for f in ${INPUTS//,/ }; do case "$PINNED" in *",$f,"*) ;; *) log "input $f has no md5 pin while JOBS_MD5 is set -> refusing"; exit 1;; esac; done; fi
SRCC=$(head -1 src/COMMIT.txt 2>/dev/null | cut -d' ' -f1); [ -n "$SRCC" ] || { log "src/COMMIT.txt missing -> refusing"; exit 1; }; case "$SRCC" in "${SRC_DIR#src_}"*) ;; *) log "src snapshot commit $SRCC does not match $SRC_DIR -> refusing"; exit 1;; esac
[ -n "${SRC_MD5:-}" ] || [ "$SRC_DIR" != src_4a3d6d5 ] || SRC_MD5=ae7e29626a48f443ad1813fd188a2151   # tree md5 of the 4a3d6d5 snapshot's non-empty .py files (sorted paths, md5sum | md5sum)
TREE=$(cd src && find . -name '*.py' -size +0 | LC_ALL=C sort | xargs md5sum | md5sum | cut -d' ' -f1); if [ -n "${SRC_MD5:-}" ]; then [ "$TREE" = "$SRC_MD5" ] || { log "src tree md5 $TREE != pinned $SRC_MD5 -> refusing"; exit 1; }; fi; log "src $SRCC tree md5 $TREE ($(find src -name '*.py' | wc -l) py files)"
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //'); log "setup done in $(( $(date +%s) - T_BOOT ))s: $(nproc) cpus ($CPU); torch $(python -c 'import torch;print(torch.__version__)'); src $(head -1 src/COMMIT.txt 2>/dev/null | cut -d' ' -f1); shard $SHARD/$N job $JOB"
# ---- run ----
T0=$(date +%s)
PYTHONPATH=src env -u KAGGLE_API_TOKEN python -m "$MODULE" $ARGS --out "$OUT" --gen "$CK/$GEN" --inten "$CK/$INTEN" --device cpu --threads "$THREADS" --batch "$BATCH" --chunk "$CHUNK" --maxNodes "$MAXNODES" --sparseK "$SPARSEK" --shard "$SHARD/$N" > "log/ice_$S.log" 2>&1 &
PID=$!; while kill -0 $PID 2>/dev/null; do sleep 60; log "chunks done: $(ls "$OUT"/chunk*.parquet 2>/dev/null | wc -l); $(grep -o '[0-9.]* s/job' "log/ice_$S.log" | tail -1)"; done
wait $PID; RC=$?; WALL=$(( $(date +%s) - T0 )); log "module rc=$RC wall ${WALL}s"; grep -E '^shard |^selftest:' "log/ice_$S.log" | tail -3 | cut -c1-200; log "module log: $(wc -l < "log/ice_$S.log") lines, $(grep -c '^fail ' "log/ice_$S.log") fail lines (kept private)"
python - "$OUT" "$SHARD" "$N" "$RC" "$WALL" "${MODULE##*.}" "$JOB" <<'PY'
import glob, json, os, sys, polars as pl
out, shard, n, rc, wall, module, job = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7]
files = sorted(glob.glob(out + '/chunk*.parquet')); df = pl.concat([pl.read_parquet(f) for f in files]) if files else None
rows = df.height if df is not None else 0; empty = int((df['mzs'].list.len() == 0).sum()) if df is not None and 'mzs' in df.columns else None
ms = json.load(open(out + '/summary.json')) if os.path.exists(out + '/summary.json') else None
cpu = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), None)
s = {'job': job, 'shard': f'{shard}/{n}', 'module': module, 'cpu': cpu, 'worker': 'gha-' + os.environ.get('GITHUB_RUN_ID', 'local'), 'threads': int(os.environ.get('THREADS', 0) or 0), 'rc': rc, 'chunksDone': len(files),
     'complete': rc == 0 and (ms is None or bool(ms.get('pass', True))), 'rows': rows, 'emptyPredictions': empty, 'secPerJob': round(float(df['seconds'].mean()), 4) if df is not None and 'seconds' in df.columns else None,
     'wallSec': wall, 'moduleSummary': ms, 'files': sorted(os.path.basename(f) for f in glob.glob(out + '/*') if not f.endswith('pool_summary.json'))}
json.dump(s, open(out + '/pool_summary.json', 'w'), indent=1); print('summary:', json.dumps({k: s[k] for k in ('job', 'shard', 'rc', 'chunksDone', 'complete', 'rows', 'emptyPredictions', 'secPerJob', 'wallSec')}))
PY
# ---- publish: nicholasooo/<DSPREFIX>-sNN with pool-style file names ----
for f in "$OUT"/*; do [ -f "$f" ] && ln -f "$f" "ds/${JOB}_${S}_$(basename "$f")"; done
printf '{"title": "%s", "id": "%s", "licenses": [{"name": "other"}]}\n' "$DSPREFIX-$S" "$DS" > ds/dataset-metadata.json
# publish with retries: create -> (exists) version; any other answer (403/429/5xx, seen 'Forbidden' on CreateDatasetVersion at 18:01 when 3 datasets were created within 70 s) -> wait and retry; then verify by file listing
PUB=0; for i in 1 2 3 4 5 6 7 8; do
  R=$(kg datasets create -p ds -q); case "$R" in *"being created"*|*uccess*) PUB=1;; *exists*|*already*) R=$(kg datasets version -p ds -q -m "$JOB $S $(date -u +%H:%M)"); case "$R" in *"being created"*|*uccess*) PUB=1;; esac;; esac
  log "publish attempt $i: ${R: -90}"; [ "$PUB" = 1 ] && break; sleep $(( 30 * i + RANDOM % 30 )); done
for w in 60 60 90 120 150 180; do sleep $w; kg datasets files "$DS" --page-size 500 | grep -q "_${S}_pool_summary.json" && { log "dataset ready: $DS"; echo "DONE rc=$RC"; exit $RC; }; done   # 6 API calls per shard (shared account budget)
# last resort: keep the output as an encrypted artifact (workflow uploads $ROOT/art) so a failed publish never loses the shard
if [ -n "${CASMI_ENC_KEY:-}" ]; then mkdir -p art && tar cf - -C ds . | gzip -1 | openssl enc -aes-256-cbc -pbkdf2 -salt -pass env:CASMI_ENC_KEY -out "art/${JOB}_${S}.tar.gz.enc" && log "encrypted fallback artifact written ($(du -k "art/${JOB}_${S}.tar.gz.enc" | cut -f1) KB)"; fi
log "dataset NOT verified after 11 min: $DS"; echo "DONE (upload unverified) rc=$RC"; exit 1
