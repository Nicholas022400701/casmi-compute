#!/bin/bash
# kpool.sh — Kaggle CPU-kernel worker (static shard list, no secrets, no heartbeats).  Inputs come from datasets attached to the kernel
# (/kaggle/input/datasets/nicholasooo/<slug>): casmi-m2-fwdsim-wheels, casmi-m2-fwdsim-assets, casmi-c1-pool-jobs (src snapshot), <jobs dataset>.  Outputs land in
# /kaggle/working/out/<job>_sNN/ (chunk*.parquet, module files, pool_summary.json); C1 pulls them with `kaggle kernels output` and publishes casmi-c1-pool-<KID>.
# env: KID=k0 JOB=ranker N=100 SHARDS="90 91 92" JOBS_DS=casmi-m2-ranker-jobs ARGS="--rows jobs/rows.parquet --queries jobs/queries.parquet --head all --tag ranker"
#      [MODULE=casmi.fwdsim.runShard] [SRC_DIR=src_4a3d6d5] [PAR=2] [THREADS=2] [BATCH=16] [CHUNK=512] [MAXNODES=100] [SPARSEK=100] [IN=/kaggle/input] [OUT=/kaggle/working/out] [ROOT=/kaggle/working/w]
set -uo pipefail
: "${KID:?}" "${JOB:?}" "${N:?}" "${SHARDS:?}" "${JOBS_DS:?}" "${ARGS:?}"
MODULE=${MODULE:-casmi.fwdsim.runShard}; SRC_DIR=${SRC_DIR:-src_4a3d6d5}; PAR=${PAR:-2}; THREADS=${THREADS:-2}; BATCH=${BATCH:-16}; CHUNK=${CHUNK:-512}; MAXNODES=${MAXNODES:-100}; SPARSEK=${SPARSEK:-100}
IN=${IN:-/kaggle/input/datasets/nicholasooo}; OUT=${OUT:-/kaggle/working/out}; ROOT=${ROOT:-/kaggle/working/w}; T_BOOT=$(date +%s)
GEN=iceberg_msg_all/gen/best.ckpt; INTEN=iceberg_msg_all/inten_contr/best.ckpt; CK=$IN/casmi-m2-fwdsim-assets
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
mkdir -p "$ROOT" "$OUT" && cd "$ROOT"
log "inputs under $IN:"; find "$IN" -maxdepth 3 2>/dev/null | head -60; nproc; free -g | head -2; python3 --version 2>/dev/null
# ---- setup: with internet -> python 3.12 venv + full ICE wheel set (as pool.sh); without -> Kaggle stock python/torch + wheels only (compat shims for torch_scatter/pygmtools) ----
NET=0; curl -sI --max-time 8 https://pypi.org >/dev/null 2>&1 && NET=1; log "internet: $NET"
mkdir -p wh; [ -f wh/.done ] || { cp "$IN"/casmi-m2-fwdsim-wheels/*.whl wh/ && (cd wh && for f in *2.6.0cpu*.whl; do [ -f "$f" ] && mv "$f" "${f/2.6.0cpu/2.6.0+cpu}"; done; for f in *pt26cpu*.whl; do [ -f "$f" ] && mv "$f" "${f/2.1.2pt26cpu/2.1.2+pt26cpu}"; done; true) && touch wh/.done; }
if [ "$NET" = 1 ]; then
  export PATH=$HOME/.local/bin:$PATH; which uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  [ -d venv ] || uv venv -q --python 3.12 venv; source venv/bin/activate; PY=python
  uv pip install -q sympy==1.13.1 filelock jinja2 fsspec networkx typing-extensions setuptools packaging requests pydantic numpy pandas h5py scipy scikit-learn tqdm pyyaml einops psutil joblib matplotlib polars pyarrow
  uv pip install -q --no-index --no-deps --find-links wh torch dgl torch_scatter pygmtools pytorch_lightning torchmetrics lightning_utilities platformdirs multiprocess dill rdkit ms_pred
else
  PY=python3; $PY -m pip install -q --no-index --no-deps --find-links wh ms_pred dgl rdkit dill multiprocess platformdirs lightning_utilities torchmetrics pytorch_lightning 2>&1 | grep -v -i 'wrapt\|sitecustomize' | tail -3
fi
$PY - <<'PYCHK' || { log 'missing modules'; exit 1; }
import importlib, sys
miss = [m for m in ('torch', 'dgl', 'rdkit', 'ms_pred', 'polars', 'pyarrow', 'numpy', 'pandas', 'pytorch_lightning') if importlib.util.find_spec(m) is None]
print('missing:', miss) if miss else print('modules ok'); sys.exit(1 if miss else 0)
PYCHK
[ -d src ] || cp -r "$IN/casmi-c1-pool-jobs/$SRC_DIR/src" src; [ -f src/casmi/fwdsim/runShard.py ] || { log 'src missing'; exit 1; }
[ -e jobs ] || ln -s "$IN/$JOBS_DS" jobs
for f in "$GEN" "$INTEN"; do [ -f "$CK/$f" ] || { log "ckpt missing $f"; exit 1; }; done
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //'); log "setup done in $(( $(date +%s) - T_BOOT ))s: $(nproc) cpus ($CPU); torch $($PY -c 'import torch;print(torch.__version__)'); src $(head -1 src/COMMIT.txt | cut -d' ' -f1)"
cat > summ.py <<'PY'
import glob, json, os, sys, polars as pl
out, shard, n, rc, wall, module, job, kid = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7], sys.argv[8]
files = sorted(glob.glob(out + '/chunk*.parquet')); df = pl.concat([pl.read_parquet(f) for f in files]) if files else None
rows = df.height if df is not None else 0; empty = int((df['mzs'].list.len() == 0).sum()) if df is not None and 'mzs' in df.columns else None
ms = json.load(open(out + '/summary.json')) if os.path.exists(out + '/summary.json') else None
cpu = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), None)
s = {'job': job, 'shard': f'{shard}/{n}', 'module': module, 'cpu': cpu, 'worker': kid, 'rc': rc, 'chunksDone': len(files), 'complete': rc == 0 and (ms is None or bool(ms.get('pass', True))), 'rows': rows, 'emptyPredictions': empty,
     'secPerJob': round(float(df['seconds'].mean()), 4) if df is not None and 'seconds' in df.columns else None, 'wallSec': wall, 'moduleSummary': ms, 'files': sorted(os.path.basename(f) for f in glob.glob(out + '/*') if not f.endswith('pool_summary.json'))}
json.dump(s, open(out + '/pool_summary.json', 'w'), indent=1); print('summary:', json.dumps(s)[:400])
PY
runOne() {  # $1 shard
  local s=$1 S; S=$(printf 's%02d' "$s"); local O="$OUT/${JOB}_$S"; mkdir -p "$O"; local T0; T0=$(date +%s)
  log "shard $s start"; PYTHONPATH=src $PY -m "$MODULE" $ARGS --out "$O" --gen "$CK/$GEN" --inten "$CK/$INTEN" --device cpu --threads "$THREADS" --batch "$BATCH" --chunk "$CHUNK" --maxNodes "$MAXNODES" --sparseK "$SPARSEK" --shard "$s/$N" > "$O/run.log" 2>&1; local RC=$?
  $PY summ.py "$O" "$s" "$N" "$RC" $(( $(date +%s) - T0 )) "${MODULE##*.}" "$JOB" "$KID" | tail -1; log "shard $s done rc=$RC $(( $(date +%s) - T0 ))s"
  tar -cf "$OUT/${JOB}_$S.tar" -C "$OUT" "${JOB}_$S" && rm -rf "$O"   # one file per shard (kernel output file-count limit); C1 untars before publishing
  echo "$JOB:$s rc=$RC" >> "$OUT/done.txt"
}
# ---- run SHARDS, PAR at a time ----
for s in $SHARDS; do
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 20; done
  runOne "$s" &
  sleep 5
done
wait; log "all shards finished: $(wc -l < "$OUT/done.txt") in $(( $(date +%s) - T_BOOT ))s"; cat "$OUT/done.txt"
