#!/bin/bash
# zenodo_diff/run.sh — LABEL-FREE COUNT job for casmi-compute's run-script workflow (runs inside the private bundle dir; counts and timings only in the log).
# Downloads a public CC BY 4.0 spectral-library release (Zenodo 21346580: full + filtered parquet) and the competition files, then runs zenodoDiff.py
# (set diff vs train, adduct-filter test, kernel lib-gate matches, standardisation collapse, profiles, visible-test match counts). Results -> out/ (private output dataset).
set -uo pipefail
T0=$(date +%s); log() { echo "[$(date -u +%H:%M:%S) +$(( $(date +%s) - T0 ))s] $*"; }; fail() { log "ERROR: $*"; exit 1; }
: "${CELL:?}"; B=$PWD; OUT=$B/out; mkdir -p "$OUT"; ROOT=$(dirname "$B"); JD=${JOB_DIR:-$GITHUB_WORKSPACE/jobs/zenodo_diff}; KG=$ROOT/kvenv/bin/kaggle
[ -x "$KG" ] || { python3 -m venv "$ROOT/kvenv" && "$ROOT/kvenv/bin/pip" install -q kaggle >/dev/null 2>&1; }
( sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc /usr/local/.ghcup 2>/dev/null ) &   # ~15 GB of toolchains we do not need; the release is 2.6 GB gz
log "start: cpus $(nproc) mem $(free -g | awk '/Mem/{print $2}')G disk $(df -BG . | awk 'NR==2{print $4}')"
python3 -m venv "$B/venv" && . "$B/venv/bin/activate" && pip install -q --upgrade pip >/dev/null 2>&1
for i in 1 2 3; do pip install -q numpy==2.0.2 pandas==2.3.3 pyarrow numba scipy rdkit >/dev/null 2>&1 && break; sleep $((20*i)); done
python -c "import numpy, pandas, pyarrow, numba, rdkit" >/dev/null 2>&1 || fail "python env failed"; log "python env ready"
D=$ROOT/zen; mkdir -p "$D"; COMP=enveda-CASMI26-molecule-id-mass-spectra; CD=$ROOT/comp; mkdir -p "$CD"
REC=https://zenodo.org/api/records/21346580/files
dlz() { local f=$1 want=$2 i; for i in 1 2 3 4; do curl -sL --retry 3 -o "$D/$f" "$REC/$f/content" && [ "$(md5sum "$D/$f" | cut -c1-32)" = "$want" ] && return 0; log "release file download attempt $i failed ($(du -m "$D/$f" 2>/dev/null | cut -f1) MB)"; sleep $((30*i)); done; return 1; }
( dlz enveda-180.parquet.gz fa8732cd62b148c7d690f949ee6a3008 && log "full release downloaded $(du -m "$D/enveda-180.parquet.gz" | cut -f1) MB" || log "full release download FAILED" ) &
P1=$!
( dlz enveda-180-filtered.parquet.gz 8a802a360735c9c330c5a1fbf75f00f2 && log "filtered release downloaded" || log "filtered release download FAILED" ) &
P2=$!
for f in train.parquet test.parquet; do for i in $(seq 1 10); do sl=$(( 30 * (i < 6 ? i : 6) + RANDOM % 30 )); "$KG" competitions download -c $COMP -f $f -p "$CD" -q > "$OUT/kg_err.txt" 2>&1 && break; log "retry competition file $i sleep $sl sec"; sleep $sl; done; done
for z in "$CD"/*.zip; do [ -e "$z" ] && { unzip -o -q "$z" -d "$CD" && rm -f "$z"; }; done; rm -f "$OUT/kg_err.txt"
[ -s "$CD/train.parquet" ] && [ -s "$CD/test.parquet" ] || fail "competition files missing"; log "competition files ok ($(du -sm "$CD" | cut -f1) MB)"
wait $P1 $P2; [ -s "$D/enveda-180.parquet.gz" ] || fail "release missing"
gunzip -f "$D/enveda-180.parquet.gz" || fail "gunzip failed"; [ -s "$D/enveda-180-filtered.parquet.gz" ] && gunzip -f "$D/enveda-180-filtered.parquet.gz"
log "release unpacked: $(du -sm "$D" | cut -f1) MB; disk left $(df -BG . | awk 'NR==2{print $4}')"
ZF=""; [ -s "$D/enveda-180-filtered.parquet" ] && ZF="--zenFiltered $D/enveda-180-filtered.parquet"
python "$JD/zenodoDiff.py" --zen "$D/enveda-180.parquet" $ZF --train "$CD/train.parquet" --test "$CD/test.parquet" --bundle "$B" --cell "$CELL" --out "$OUT" --nControl 3 --seed 0 > "$OUT/zenodoDiff.log" 2>&1; RC=$?
grep -viE 'smiles|inchi' "$OUT/zenodoDiff.log" | tail -30 | cut -c1-200
log "done rc=$RC; output files $(ls "$OUT" | wc -l), $(du -sm "$OUT" | cut -f1) MB"; exit $RC
