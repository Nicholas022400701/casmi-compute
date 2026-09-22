#!/bin/bash
# run_script/collect.sh — decrypt all shard artifacts (art/*/sNN.tar.zst.enc) into one folder and publish ONE private Kaggle dataset nicholasooo/<OUT_DS> (create, or new version if it exists).
set -uo pipefail
: "${KAGGLE_API_TOKEN:?}" "${CASMI_ENC_KEY:?}" "${OUT_DS:?}"; export KAGGLE_API_TOKEN CASMI_ENC_KEY; ROOT=${ROOT:-${RUNNER_TEMP:-/tmp}/rs}; ART=${ART:-art}
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
mkdir -p "$ROOT/ds" && export PATH=$HOME/.local/bin:$PATH; which uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
which zstd >/dev/null 2>&1 || sudo apt-get install -y -qq zstd >/dev/null 2>&1
uv venv -q --python 3.12 "$ROOT/kvenv" && uv pip install -q --python "$ROOT/kvenv/bin/python" kaggle; K=$ROOT/kvenv/bin/kaggle
n=0; for f in $(find "$ART" -name '*.tar.zst.enc' | sort); do openssl enc -d -aes-256-cbc -pbkdf2 -pass env:CASMI_ENC_KEY -in "$f" | zstd -dq | tar xf - -C "$ROOT/ds" && n=$((n+1)); done
log "shards collected: $n; files $(find "$ROOT/ds" -type f | wc -l); $(du -sm "$ROOT/ds" | cut -f1) MB"; [ "$n" -gt 0 ] || { log "nothing to publish"; exit 1; }
printf '{"title": "%s", "id": "nicholasooo/%s", "licenses": [{"name": "other"}]}\n' "$OUT_DS" "$OUT_DS" > "$ROOT/ds/dataset-metadata.json"
R=$($K datasets create -p "$ROOT/ds" -q --dir-mode zip 2>&1); case "$R" in *rror*|*exists*|*already*) R=$($K datasets version -p "$ROOT/ds" -q --dir-mode zip -m "collect $(date -u +%H:%M) run ${GITHUB_RUN_ID:-local}" 2>&1);; esac; log "publish: ${R: -120}"
for i in $(seq 1 20); do sleep 30; $K datasets files "nicholasooo/$OUT_DS" 2>/dev/null | grep -q '_gha_' && { log "dataset ready: nicholasooo/$OUT_DS"; exit 0; }; done
log "dataset not verified yet (may still be processing): nicholasooo/$OUT_DS"; exit 0
