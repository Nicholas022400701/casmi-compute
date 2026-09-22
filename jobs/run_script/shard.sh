#!/bin/bash
# run_script/shard.sh — generic Actions runner: fetch a private Kaggle bundle (run.sh + inputs), run one shard of it, encrypt its output dir into one artifact file.
# env: KAGGLE_API_TOKEN CASMI_ENC_KEY BUNDLE_DS SHARD N [ENVX "K=V K=V" extra env for the script] [PROCS nproc] [CMD "bash run.sh"] [OUT_DIR out] [ROOT $RUNNER_TEMP/rs]
# Logging rule (public repo): counts and timings only — the script's own stdout is filtered to lines without SMILES/InChI/keys and truncated.
set -uo pipefail
: "${KAGGLE_API_TOKEN:?}" "${CASMI_ENC_KEY:?}" "${BUNDLE_DS:?}" "${SHARD:?}" "${N:?}"; export KAGGLE_API_TOKEN CASMI_ENC_KEY
ENVX=${ENVX:-}; PROCS=${PROCS:-$(nproc)}; CMD=${CMD:-bash run.sh}; OUT_DIR=${OUT_DIR:-out}; ROOT=${ROOT:-${RUNNER_TEMP:-/tmp}/rs}; S=$(printf 's%02d' "$SHARD"); T0=$(date +%s)
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
mkdir -p "$ROOT/bundle" "$ROOT/art" && cd "$ROOT"
export PATH=$HOME/.local/bin:$PATH; which uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
which zstd >/dev/null 2>&1 || sudo apt-get install -y -qq zstd >/dev/null 2>&1
uv venv -q --python 3.12 kvenv && uv pip install -q --python kvenv/bin/python kaggle; K=$ROOT/kvenv/bin/kaggle   # kaggle CLI in its own venv; system python/pip stay untouched for the bundle's run.sh
$K datasets download "$BUNDLE_DS" -p bundle --unzip -q || { log "bundle download failed"; exit 1; }
log "bundle: $(find bundle -type f | wc -l) files, $(du -sm bundle | cut -f1) MB; $(nproc) cpus; setup $(( $(date +%s) - T0 ))s"
cd bundle; [ -f run.sh ] || { log "run.sh missing in bundle"; exit 1; }
T1=$(date +%s); env $ENVX SHARD="$SHARD" N="$N" PROCS="$PROCS" bash -c "$CMD" > "$ROOT/script.log" 2>&1; RC=$?; WALL=$(( $(date +%s) - T1 ))
log "script rc=$RC wall ${WALL}s; log lines $(wc -l < "$ROOT/script.log"); output files $(find "$OUT_DIR" -type f 2>/dev/null | wc -l), $(du -sm "$OUT_DIR" 2>/dev/null | cut -f1) MB"
grep -viE 'smiles|inchi|token|key' "$ROOT/script.log" | grep -iE 'done|rows|count|min|sec|s/|shard|part|error|traceback' | tail -25 | cut -c1-160
[ -d "$OUT_DIR" ] && [ "$(find "$OUT_DIR" -type f | wc -l)" -gt 0 ] || { log "no output dir/files"; exit 1; }
printf 'shard=%s n=%s rc=%s wall=%s run=%s\n' "$SHARD" "$N" "$RC" "$WALL" "${GITHUB_RUN_ID:-local}" > "$OUT_DIR/_gha_$S.txt"
tar cf - -C "$OUT_DIR" . | zstd -q -T0 -3 | openssl enc -aes-256-cbc -pbkdf2 -salt -pass env:CASMI_ENC_KEY -out "$ROOT/art/$S.tar.zst.enc"
log "artifact $S: $(du -k "$ROOT/art/$S.tar.zst.enc" | cut -f1) KB; DONE rc=$RC"; exit $RC
