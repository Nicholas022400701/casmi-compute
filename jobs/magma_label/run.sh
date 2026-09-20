#!/bin/bash
# C1 pool job 2: one shard of M2b's MAGMa/subformula labelling (labelShard.py from the private dataset casmi-m2-finetune-data).
# usage: KAGGLE_API_TOKEN=... SHARD=<i> N=<n> [WORKERS=2] [ROOT=/data/c1w/magma] [DSPREFIX=casmi-c1-magma] [TEST_N=0] bash run.sh
# Idempotent: re-running skips finished stages; output = flat private dataset nicholasooo/<DSPREFIX>-s<ii>:
# labels_shard.tsv, magma_tree.hdf5, no_subform.hdf5, timing.json, check.json. Console prints counts/timings only; full log in $ROOT/log.
set -uo pipefail
: "${SHARD:?}" "${N:?}"; WORKERS=${WORKERS:-2}; ROOT=${ROOT:-/data/c1w/magma}; DSPREFIX=${DSPREFIX:-casmi-c1-magma}
mkdir -p "$ROOT/log" && cd "$ROOT"
export PATH=$HOME/.local/bin:$PATH
which uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -d venv ] || uv venv -q --python 3.12 venv
source venv/bin/activate
uv pip install -q kaggle
if [ ! -f wh/.done ]; then
  kaggle datasets download nicholasooo/casmi-m2-fwdsim-wheels -p wh --unzip -q
  (cd wh && for f in *2.6.0cpu*.whl; do mv "$f" "${f/2.6.0cpu/2.6.0+cpu}"; done; for f in *pt26cpu*.whl; do mv "$f" "${f/2.1.2pt26cpu/2.1.2+pt26cpu}"; done) && touch wh/.done
fi
uv pip install -q sympy==1.13.1 filelock jinja2 fsspec networkx typing-extensions setuptools packaging requests pydantic numpy pandas h5py scipy scikit-learn tqdm pyyaml einops psutil joblib matplotlib seaborn pathos easydict appdirs aiohttp pillow omegaconf
uv pip install -q --no-index --no-deps --find-links wh torch dgl torch_scatter pygmtools pytorch_lightning torchmetrics lightning_utilities platformdirs multiprocess dill rdkit ms_pred
[ -f ftdata/labels.tsv ] || kaggle datasets download nicholasooo/casmi-m2-finetune-data -p ftdata --unzip -q
python -c "import sys; sys.path.insert(0, 'ftdata'); import compat; compat.install(); import torch, dgl, ms_pred.magma.run_magma" 2>&1 | tail -3 | sed -E 's/[A-Za-z0-9_-]{12,}/<id>/g'
echo "[$(date -u +%H:%M:%S)] setup done: $(nproc) cpus, $(free -g | awk '/Mem/{print $2}') GB"
S=$(printf '%02d' "$SHARD"); OUT=$ROOT/out/shard_$S; mkdir -p "$OUT"
if [ ! -f "$OUT/timing.json" ]; then
  T=$(date +%s)
  python ftdata/labelShard.py --shard "$SHARD/$N" --data ftdata --out "$OUT" --workers "$WORKERS" > "$ROOT/log/shard_$S.log" 2>&1; RC=$?
  echo "[$(date -u +%H:%M:%S)] labelShard rc=$RC, $(( $(date +%s) - T )) s wall, skipped=$(grep -c -i 'skipping' "$ROOT/log/shard_$S.log")"
fi
echo "timing: $(tr -d '\n ' < "$OUT/timing.json" | cut -c1-400)"
# checks (labelShard.py rule relaxed per orchestrator/M2b: subform must cover every record, MAGMa may miss <= 2 % of records; misses + skipped counts go to check.json only)
SKIPPED=$(grep -c -i 'skipping spec' "$ROOT/log/shard_$S.log" 2>/dev/null || echo 0)
python - "$OUT" "$SKIPPED" <<'PY' || { echo "shard checks FAILED; last log lines (ids redacted):"; tail -12 "$ROOT/log/shard_$S.log" | sed -E 's/[A-Za-z0-9_-]{12,}/<id>/g'; exit 1; }
import json, sys, h5py, pandas as pd
out = sys.argv[1]; t = json.load(open(out + '/timing.json')); skipped = int(sys.argv[2])
specs = set(pd.read_csv(out + '/labels_shard.tsv', sep='\t')['spec']); c = {'records': len(specs)}
for name, path in [('magma', out + '/magma_outputs/magma_tree.hdf5'), ('subform', out + '/subformulae/no_subform.hdf5')]:
    try:
        with h5py.File(path, 'r') as h: keys = set(h.keys())
        got = {k.split('_collision')[0].split('.')[0] for k in keys} & specs
        c[name] = {'keys': len(keys), 'matched': len(got), 'missing': sorted(specs - got), 'ok': len(got) == len(specs)}
    except Exception as e:
        c[name] = {'ok': False, 'error': str(e)[:300]}
c['skippedSpecs'] = skipped; c['magmaMissing'] = len(c['magma'].get('missing', []))
magmaOk = c['magmaMissing'] <= 0.02 * len(specs)
ok = t.get('magma_rc') == 0 and t.get('subform_rc') == 0 and c['subform'].get('ok') is True and magmaOk
c['accepted'] = ok; c['rule'] = 'rc==0, subform all records, magmaMissing <= 2 % of records; skippedSpecs = tool "Skipping spec" lines (spectra without a label row)'
json.dump(c, open(out + '/check.json', 'w'), indent=1)
print('check: records', len(specs), 'magma matched', c['magma'].get('matched'), 'missing', len(c['magma'].get('missing', [])), 'subform matched', c['subform'].get('matched'), '->', 'CHECKS OK' if ok else 'CHECKS FAILED')
sys.exit(0 if ok else 1)
PY
UP=$ROOT/up/shard_$S; rm -rf "$UP"; mkdir -p "$UP"
cp "$OUT/labels_shard.tsv" "$OUT/timing.json" "$OUT/check.json" "$UP/" && cp "$OUT"/magma_outputs/magma_tree.hdf5 "$UP/magma_tree.hdf5" && cp "$OUT"/subformulae/no_subform.hdf5 "$UP/no_subform.hdf5" || { echo "missing output files:"; find "$OUT" -maxdepth 2 | sed -E 's/[A-Za-z0-9_-]{12,}/<id>/g' | head -20; exit 1; }
DS=nicholasooo/$DSPREFIX-s$S
printf '{"title": "%s", "id": "%s", "licenses": [{"name": "other"}]}\n' "$DSPREFIX-s$S" "$DS" > "$UP/dataset-metadata.json"
du -sh "$UP" | cut -f1
R=$(kaggle datasets create -p "$UP" -q 2>&1); echo "create: ${R: -160}"
case "$R" in *"already in use"*|*rror*) R=$(kaggle datasets version -p "$UP" -q -m "rerun $(date -u +%H:%M)" 2>&1); echo "version: ${R: -160}";; esac
for i in $(seq 1 30); do sleep 30; kaggle datasets files "$DS" 2>/dev/null | grep -q magma_tree && { echo "[$(date -u +%H:%M:%S)] dataset ready: $DS"; echo DONE; exit 0; }; done
echo "WARNING: dataset files not listed yet: $DS"; echo "DONE (upload unverified)"
