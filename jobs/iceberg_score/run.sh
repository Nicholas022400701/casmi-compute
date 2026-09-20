#!/bin/bash
# C1 pool: resumable ICEBERG CPU scoring of one shard of a job table (casmi.fwdsim.runIceberg --shard i/n; D40/D41).
# Input: JOBS_DS/JOBS_FILE (private Kaggle dataset, parquet from casmi.fwdsim.jobs.jobTable: ik14, smiles, adductIce, ceEv, instrumentIce);
# model ckpts from nicholasooo/casmi-m2-fwdsim-assets, wheels from casmi-m2-fwdsim-wheels, package from nicholasooo/casmi-src.
# Output: private dataset nicholasooo/<DSPREFIX>-s<ii> with chunk*.parquet (runIceberg output columns) + summary.json; versioned every CKPT_MIN
# minutes while running, so a reset loses <= CKPT_MIN minutes; on restart existing chunks are downloaded, validated and skipped.
# usage: KAGGLE_API_TOKEN=... JOBS_DS=nicholasooo/... JOBS_FILE=jobs.parquet SHARD=i N=n DSPREFIX=casmi-c1-ice-<job> [THREADS=2] [CHUNK=512] [CKPT_MIN=10] [LIMIT=0] bash run.sh
set -uo pipefail
: "${SHARD:?}" "${N:?}" "${JOBS_DS:?}" "${JOBS_FILE:?}" "${DSPREFIX:?}" "${KAGGLE_API_TOKEN:?}"
THREADS=${THREADS:-2}; CHUNK=${CHUNK:-512}; CKPT_MIN=${CKPT_MIN:-10}; LIMIT=${LIMIT:-0}; MAXNODES=${MAXNODES:-100}; SPARSEK=${SPARSEK:-100}; BATCH=${BATCH:-16}
ROOT=${ROOT:-/data/c1w/ice}; S=$(printf '%02d' "$SHARD"); DS=nicholasooo/$DSPREFIX-s$S; OUT=$ROOT/out_$DSPREFIX/s$S
mkdir -p "$ROOT/log" "$OUT" && cd "$ROOT"
export PATH=$HOME/.local/bin:$PATH
which uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -d venv ] || uv venv -q --python 3.12 venv
source venv/bin/activate
uv pip install -q kaggle
if [ ! -f wh/.done ]; then
  kaggle datasets download nicholasooo/casmi-m2-fwdsim-wheels -p wh --unzip -q
  (cd wh && for f in *2.6.0cpu*.whl; do mv "$f" "${f/2.6.0cpu/2.6.0+cpu}"; done; for f in *pt26cpu*.whl; do mv "$f" "${f/2.1.2pt26cpu/2.1.2+pt26cpu}"; done) && touch wh/.done
fi
uv pip install -q sympy==1.13.1 filelock jinja2 fsspec networkx typing-extensions setuptools packaging requests pydantic numpy pandas h5py scipy scikit-learn tqdm pyyaml einops psutil joblib matplotlib seaborn pathos easydict appdirs aiohttp pillow omegaconf polars pyarrow
uv pip install -q --no-index --no-deps --find-links wh torch dgl torch_scatter pygmtools pytorch_lightning torchmetrics lightning_utilities platformdirs multiprocess dill rdkit ms_pred
[ -f src/casmi/fwdsim/runIceberg.py ] || kaggle datasets download nicholasooo/casmi-src -p . --unzip -q
for f in iceberg_msg_all/gen/best.ckpt iceberg_msg_all/inten_contr/best.ckpt; do [ -f "$f" ] || kaggle datasets download nicholasooo/casmi-m2-fwdsim-assets -f "$f" -p "$(dirname "$f")" -q --unzip; done
[ -f "jobs/$JOBS_FILE" ] || kaggle datasets download "$JOBS_DS" -f "$JOBS_FILE" -p jobs -q --unzip
# resume: fetch chunks already uploaded by an earlier incarnation of this worker, drop unreadable ones
if kaggle datasets files "$DS" >/dev/null 2>&1 && [ -z "$(ls "$OUT"/chunk*.parquet 2>/dev/null)" ]; then kaggle datasets download "$DS" -p "$OUT" --unzip -q 2>/dev/null || true; fi
python - "$OUT" <<'PY'
import glob, os, sys, polars as pl
n = 0
for f in glob.glob(sys.argv[1] + '/chunk*.parquet'):
    try: pl.read_parquet(f); n += 1
    except Exception: os.remove(f)
print(f'resume: {n} finished chunks on disk')
PY
echo "[$(date -u +%H:%M:%S)] setup done: $(nproc) cpus; shard $SHARD/$N of $JOBS_FILE"
printf '{"title": "%s", "id": "%s", "licenses": [{"name": "other"}]}\n' "$DSPREFIX-s$S" "$DS" > "$OUT/dataset-metadata.json"
upload() {  # version (or create) the output dataset from the finished chunk files
  R=$(kaggle datasets version -p "$OUT" -q -m "$1 $(date -u +%H:%M)" 2>&1); case "$R" in *rror*|*"not found"*|*404*) R=$(kaggle datasets create -p "$OUT" -q 2>&1);; esac; echo "[$(date -u +%H:%M:%S)] upload ($1): ${R: -90}"
}
T0=$(date +%s)
PYTHONPATH=src nohup python -m casmi.fwdsim.runIceberg --jobs "jobs/$JOBS_FILE" --out "$OUT" --gen iceberg_msg_all/gen/best.ckpt --inten iceberg_msg_all/inten_contr/best.ckpt --device cpu --threads "$THREADS" --batch "$BATCH" --chunk "$CHUNK" --maxNodes "$MAXNODES" --sparseK "$SPARSEK" --limit "$LIMIT" --shard "$SHARD/$N" > "$ROOT/log/ice_$DSPREFIX-s$S.log" 2>&1 &
PID=$!; LAST=$(date +%s)
while kill -0 $PID 2>/dev/null; do
  sleep 30
  if [ $(( $(date +%s) - LAST )) -ge $(( CKPT_MIN * 60 )) ]; then upload checkpoint; LAST=$(date +%s); echo "  chunks done: $(ls "$OUT"/chunk*.parquet 2>/dev/null | wc -l), $(grep -c '^chunk' "$ROOT/log/ice_$DSPREFIX-s$S.log") this run"; fi
done
wait $PID; RC=$?
python - "$OUT" "$SHARD" "$N" "$RC" "$(( $(date +%s) - T0 ))" "jobs/$JOBS_FILE" "$CHUNK" "$LIMIT" <<'PY'
import glob, json, sys, polars as pl
out, shard, n, rc, wall, jobs, chunk, limit = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], int(sys.argv[7]), int(sys.argv[8])
total = pl.scan_parquet(jobs).select(pl.len()).collect().item(); total = min(total, limit) if limit else total; nChunks = (total + chunk - 1) // chunk; mine = len(range(shard, nChunks, n))
files = sorted(glob.glob(out + '/chunk*.parquet')); df = pl.concat([pl.read_parquet(f) for f in files]) if files else None
rows = df.height if df is not None else 0; empty = int((df['mzs'].list.len() == 0).sum()) if df is not None else 0
s = {'shard': f'{shard}/{n}', 'rc': rc, 'chunksDone': len(files), 'chunksExpected': mine, 'rows': rows, 'emptyPredictions': empty, 'secPerJob': round(float(df['seconds'].mean()), 4) if df is not None else None, 'wallSec': wall}
json.dump(s, open(out + '/summary.json', 'w'), indent=1); print('summary:', json.dumps(s))
PY
upload final
for i in $(seq 1 20); do sleep 30; kaggle datasets files "$DS" 2>/dev/null | grep -q summary.json && { echo "[$(date -u +%H:%M:%S)] dataset ready: $DS"; echo "DONE rc=$RC"; exit $RC; }; done
echo "DONE (upload unverified) rc=$RC"
