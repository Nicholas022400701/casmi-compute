#!/bin/bash
# C1 pool job 2/4: encrypted labelling shard for GitHub Actions runners (also runs on a sandbox).
# Input: one private Kaggle dataset file <IN_DS>/<IN_FILE> = tar (labels.tsv, spec_files.hdf5, split.tsv, labelShard.py, compat.py, assign_subformulae.py)
# encrypted with `openssl enc -aes-256-cbc -pbkdf2 -salt -pass env:CASMI_ENC_KEY`. Output: <DSPREFIX>-s<ii> private dataset with shard_<ii>.tar.zst.enc
# (labels_shard.tsv, magma_tree.hdf5, no_subform.hdf5, timing.json, check.json) + summary.json (counts/timings only). Console: counts/timings only.
# usage: KAGGLE_API_TOKEN=... CASMI_ENC_KEY=... IN_DS=nicholasooo/... IN_FILE=x.tar.enc|ft_full_s{S}.tar.enc SHARD=i N=n [INNER=0/1] [WORKERS=4] [ROOT=$RUNNER_TEMP/w] [DSPREFIX=casmi-c1-ftlab] bash run_enc.sh
set -uo pipefail
: "${SHARD:?}" "${N:?}" "${IN_DS:?}" "${IN_FILE:?}" "${CASMI_ENC_KEY:?}" "${KAGGLE_API_TOKEN:?}"
WORKERS=${WORKERS:-4}; ROOT=${ROOT:-${RUNNER_TEMP:-/data/c1w}/ftlab}; DSPREFIX=${DSPREFIX:-casmi-c1-ftlab}; S=$(printf '%02d' "$SHARD")
IN_FILE=${IN_FILE//\{S\}/$S}; INNER=${INNER:-$SHARD/$N}   # per-shard input tars: IN_FILE=ft_full_s{S}.tar.enc with INNER=0/1 (the tar holds only that shard)
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
uv pip install -q sympy==1.13.1 filelock jinja2 fsspec networkx typing-extensions setuptools packaging requests pydantic numpy pandas h5py scipy scikit-learn tqdm pyyaml einops psutil joblib matplotlib seaborn pathos easydict appdirs aiohttp pillow omegaconf zstandard
uv pip install -q --no-index --no-deps --find-links wh torch dgl torch_scatter pygmtools pytorch_lightning torchmetrics lightning_utilities platformdirs multiprocess dill rdkit ms_pred
if [ ! -f data/labels.tsv ]; then
  kaggle datasets download "$IN_DS" -f "$IN_FILE" -p in -q; [ -f "in/$IN_FILE.zip" ] && (cd in && unzip -q -o "$IN_FILE.zip" && rm "$IN_FILE.zip")
  mkdir -p data && openssl enc -d -aes-256-cbc -pbkdf2 -pass env:CASMI_ENC_KEY -in "in/$IN_FILE" | tar xf - -C data && rm -rf in
fi
python -c "import sys; sys.path.insert(0, 'data'); import compat; compat.install(); import torch, dgl, ms_pred.magma.run_magma" 2>&1 | tail -2 | sed -E 's/[A-Za-z0-9_-]{12,}/<id>/g'
echo "[$(date -u +%H:%M:%S)] setup done: $(nproc) cpus, $(free -g | awk '/Mem/{print $2}') GB, input rows $(($(wc -l < data/labels.tsv) - 1))"
OUT=$ROOT/out/shard_$S; mkdir -p "$OUT"
if [ ! -f "$OUT/timing.json" ]; then
  T=$(date +%s)
  python data/labelShard.py --shard "$INNER" --data data --out "$OUT" --workers "$WORKERS" > "$ROOT/log/shard_$S.log" 2>&1; RC=$?
  echo "[$(date -u +%H:%M:%S)] labelShard rc=$RC, $(( $(date +%s) - T )) s wall"
fi
echo "timing: $(tr -d '\n ' < "$OUT/timing.json" | cut -c1-400)"
python - "$OUT" <<'PY' || { echo "shard checks FAILED; last log lines (ids redacted):"; tail -12 "$ROOT/log/shard_$S.log" | sed -E 's/[A-Za-z0-9_-]{12,}/<id>/g'; exit 1; }
import json, sys, h5py, pandas as pd
out = sys.argv[1]; t = json.load(open(out + '/timing.json'))
specs = set(pd.read_csv(out + '/labels_shard.tsv', sep='\t')['spec']); c = {'records': len(specs)}
for name, path in [('magma', out + '/magma_outputs/magma_tree.hdf5'), ('subform', out + '/subformulae/no_subform.hdf5')]:
    try:
        with h5py.File(path, 'r') as h: keys = set(h.keys())
        got = {k.split('_collision')[0].split('.')[0] for k in keys} & specs
        c[name] = {'keys': len(keys), 'matched': len(got), 'missing': sorted(specs - got), 'ok': len(got) == len(specs)}
    except Exception as e:
        c[name] = {'ok': False, 'error': str(e)[:300]}
c['magmaMissing'] = len(c['magma'].get('missing', [])); c['skippedSpecs'] = None
ok = t.get('magma_rc') == 0 and t.get('subform_rc') == 0 and c['subform'].get('ok') is True and c['magmaMissing'] <= 0.02 * len(specs)
c['accepted'] = ok; c['rule'] = 'rc==0, subform all records, magmaMissing <= 2 % of records'
json.dump(c, open(out + '/check.json', 'w'), indent=1)
json.dump({'shard': f'{t["shard"]}', 'records': len(specs), 'spectra': t.get('spectra'), 'magmaMatched': c['magma'].get('matched'), 'magmaMissing': c['magmaMissing'], 'subformMatched': c['subform'].get('matched'),
           'magma_s': t.get('magma_s'), 'subform_s': t.get('subform_s'), 'magma_s_per_record': t.get('magma_s_per_record'), 'accepted': ok}, open(out + '/summary.json', 'w'), indent=1)
print('check: records', len(specs), 'magma matched', c['magma'].get('matched'), 'missing', c['magmaMissing'], 'subform matched', c['subform'].get('matched'), '->', 'CHECKS OK' if ok else 'CHECKS FAILED')
sys.exit(0 if ok else 1)
PY
UP=$ROOT/up/shard_$S; rm -rf "$UP"; mkdir -p "$UP/pack"
cp "$OUT/labels_shard.tsv" "$OUT/timing.json" "$OUT/check.json" "$UP/pack/" && cp "$OUT"/magma_outputs/magma_tree.hdf5 "$OUT"/subformulae/no_subform.hdf5 "$UP/pack/" || { echo "missing output files"; exit 1; }
T=$(date +%s); tar cf - -C "$UP/pack" . | zstd -q -T0 -3 | openssl enc -aes-256-cbc -pbkdf2 -salt -pass env:CASMI_ENC_KEY -out "$UP/shard_$S.tar.zst.enc"; rm -rf "$UP/pack"
cp "$OUT/summary.json" "$UP/"; echo "[$(date -u +%H:%M:%S)] packed+encrypted in $(( $(date +%s) - T )) s: $(du -sh "$UP/shard_$S.tar.zst.enc" | cut -f1) (plain $(du -sh "$OUT" | cut -f1))"
DS=nicholasooo/$DSPREFIX-s$S
printf '{"title": "%s", "id": "%s", "licenses": [{"name": "other"}]}\n' "$DSPREFIX-s$S" "$DS" > "$UP/dataset-metadata.json"
R=$(kaggle datasets create -p "$UP" -q 2>&1); echo "create: ${R: -120}"
case "$R" in *"already in use"*|*rror*) R=$(kaggle datasets version -p "$UP" -q -m "rerun $(date -u +%H:%M)" 2>&1); echo "version: ${R: -120}";; esac
for i in $(seq 1 30); do sleep 30; kaggle datasets files "$DS" 2>/dev/null | grep -q "tar.zst.enc" && { echo "[$(date -u +%H:%M:%S)] dataset ready: $DS"; echo DONE; exit 0; }; done
echo "WARNING: dataset files not listed yet: $DS"; echo "DONE (upload unverified)"
