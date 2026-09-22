#!/bin/bash
# C1 pool worker (swarm100 protocol, casmi2026-gold docs/plans/swarm100_pool.md): autonomous ICEBERG CPU scoring of shards claimed from pool/manifest.json.
# env: ID=p00 IDX=0 KAGGLE_API_TOKEN GH_TOKEN  [KILLTEST=1: self-kill after first chunk of first shard, restart expected (resume test)]  [DIE=1: stop after first chunk, no restart (stale test)]
# reads casmi-compute main: pool/manifest.json, pool/KILL (raw, <=5 min cache); heartbeat = branch hb/<ID> file hb.json (fetched at each claim for done/stale detection).
# writes private dataset nicholasooo/<dsPrefix>-<ID>: every file of the shard's out dir prefixed <job>_s<NN>_ (chunk*.parquet, module summary/scores/pairs, pool_summary.json), one version per finished shard (no checkpoints).
# manifest: job N W chunk jobsDataset inputs[] module args srcCommit dsPrefix staleMin graceMin hbMin steal ckptDs gen inten ice{threads,batch,maxNodes,sparseK} selftest{file,tol}|{jobsFile,limit,expectedFile,tol}
set -uo pipefail
: "${ID:?}" "${IDX:?}" "${KAGGLE_API_TOKEN:?}" "${GH_TOKEN:?}"; export GH_TOKEN KAGGLE_API_TOKEN
REPO=Nicholas022400701/casmi-compute; RAW=https://raw.githubusercontent.com/$REPO/main/pool; MANIFEST_URL=${MANIFEST_URL:-$RAW/manifest.json}; KILL_URL=${KILL_URL:-$RAW/KILL}
ROOT=${ROOT:-/data/c1w/ice}; mkdir -p "$ROOT/log" "$ROOT/hb" "$ROOT/ds" && cd "$ROOT"
KILLTEST=${KILLTEST:-0}; DIE=${DIE:-0}; T_BOOT=$(date +%s); ERR429=0; JOBS_DONE=0; SETUP_SEC=0; SELFTEST=none; JOB=none; RATE=0
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
zz() { local n=$1 c; while [ "$n" -gt 0 ]; do c=$(( n > 600 ? 600 : n )); sleep "$c" || true; n=$(( n - c )); date -u; echo tick; done; }
gitc() { git -c credential.helper= -c 'credential.helper=!f() { echo username=x; echo "password=$GH_TOKEN"; }; f' "$@"; }
c429() { case "$1" in *429*|*"Too Many"*) ERR429=$((ERR429+1)); log "429 seen (total $ERR429)";; esac; }
cat > hbw.py <<'PY'
import json, os, sys, time
k = ['id','idx','job','status','shard','chunk','jobsDone','rate','err429','setupSec','selftest','killtest']
d = dict(zip(k, sys.argv[1:])); d['ts'] = time.time(); d['tsUtc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
for f in ('idx','chunk','jobsDone','err429','setupSec'): d[f] = int(d[f])
d['shard'] = None if d['shard'] == '-' else int(d['shard']); d['rate'] = float(d['rate'])
d['done'] = sorted(set(open('hb/done.txt').read().split())) if os.path.exists('hb/done.txt') else []
print(json.dumps(d))
PY
cat > claim.py <<'PY'
import json, subprocess, sys, time
id_, idx, boot = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]); mf = json.load(open('manifest.json'))
job, N, W, stale, grace, steal = mf['job'], int(mf['N']), int(mf['W']), float(mf.get('staleMin', 30)), float(mf.get('graceMin', 20)), mf.get('steal', 'stale')
def git(*a): return subprocess.run(['git', *a], cwd='hb', capture_output=True, text=True).stdout
git('fetch', '-q', '-p', 'origin', '+refs/heads/hb/*:refs/remotes/hb/*')
hbs = []
for r in git('for-each-ref', '--format=%(refname)', 'refs/remotes/hb/').split():
    try: hbs.append(json.loads(git('show', f'{r}:hb.json')))
    except Exception: pass
now = time.time(); active = set(); byIdx = {}
try: done = {int(d.split(':')[1]) for d in open('hb/done.txt').read().split() if d.startswith(job + ':')}
except Exception: done = set()
for h in hbs:
    done |= {int(d.split(':')[1]) for d in h.get('done', []) if d.startswith(job + ':')}
    if h.get('id') == id_: continue
    byIdx.setdefault(h.get('idx'), []).append(h)
    if h.get('job') == job and h.get('shard') is not None and h.get('status') in ('running', 'publishing', 'paused') and now - h.get('ts', 0) < stale * 60: active.add(int(h['shard']))
own = [s for s in range(N) if s % W == idx and s not in done and s not in active]
def stealable(s):
    if s in done or s in active or s % W == idx: return False
    o = byIdx.get(s % W)
    if not o: return now - boot > grace * 60   # owner never heartbeated
    h = max(o, key=lambda x: x.get('ts', 0))
    if h.get('status') in ('idle', 'killed', 'err', 'selftestFail') or now - h.get('ts', 0) > stale * 60 or h.get('job') != job: return True
    return steal == 'any' and h.get('status') == 'running' and h.get('shard') != s   # owner busy elsewhere: take its pending shard (highest first)
rest = [s for s in range(N - 1, -1, -1) if stealable(s)]
print(own[0] if own else (rest[0] if rest else -1), len(done), len(active), len(hbs))
PY
cat > conflict.py <<'PY'
import json, subprocess, sys, time
id_, idx, shard, job = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]; mine = json.load(open('hb/hb.json'))['ts']
def git(*a): return subprocess.run(['git', *a], cwd='hb', capture_output=True, text=True).stdout
git('fetch', '-q', '-p', 'origin', '+refs/heads/hb/*:refs/remotes/hb/*'); now = time.time(); lose = 0
for r in git('for-each-ref', '--format=%(refname)', 'refs/remotes/hb/').split():
    try: h = json.loads(git('show', f'{r}:hb.json'))
    except Exception: continue
    if h.get('id') == id_ or h.get('job') != job or h.get('shard') != shard or h.get('status') not in ('running', 'publishing') or now - h.get('ts', 0) > 900: continue
    if h['ts'] < mine - 1 or (abs(h['ts'] - mine) <= 1 and h.get('idx', 99) < idx): lose = 1; print(f"conflict: {h['id']} announced shard {shard} first", file=sys.stderr)
print(lose)
PY
cat > summ.py <<'PY'
import glob, json, sys, polars as pl
import os
out, shard, n, rc, wall, module, chunk, job = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], int(sys.argv[7]), sys.argv[8]
files = sorted(glob.glob(out + '/chunk*.parquet')); df = pl.concat([pl.read_parquet(f) for f in files]) if files else None
rows = df.height if df is not None else 0; empty = int((df['mzs'].list.len() == 0).sum()) if df is not None and 'mzs' in df.columns else None
ms = json.load(open(out + '/summary.json')) if os.path.exists(out + '/summary.json') else None   # module's own summary (runShard) is kept as is
s = {'job': job, 'shard': f'{shard}/{n}', 'module': module, 'rc': rc, 'chunksDone': len(files), 'complete': rc == 0 and (ms is None or bool(ms.get('pass', True))), 'rows': rows, 'emptyPredictions': empty,
     'secPerJob': round(float(df['seconds'].mean()), 4) if df is not None and 'seconds' in df.columns else None, 'wallSec': wall, 'moduleSummary': ms, 'files': sorted(os.path.basename(f) for f in glob.glob(out + '/*') if not f.endswith('pool_summary.json'))}
json.dump(s, open(out + '/pool_summary.json', 'w'), indent=1); print('summary:', json.dumps(s)[:600])
PY
cat > adapt.py <<'PY'
# adapt a job table to ICE columns: settings struct/json -> adductIce/ceEv/instrumentIce; jobId (if present) -> ik14 so outputs are keyed by jobId
import os, sys, polars as pl
src, dst = sys.argv[1], sys.argv[2]
if os.path.exists(dst): sys.exit(0)
j = pl.read_parquet(src)
if 'settings' in j.columns and 'adductIce' not in j.columns:
    if j['settings'].dtype == pl.Utf8: j = j.with_columns(pl.col('settings').str.json_decode())
    j = j.unnest('settings')
if 'jobId' in j.columns: j = j.with_columns(pl.col('jobId').cast(pl.Utf8).alias('ik14'))
need = ['ik14', 'smiles', 'adductIce', 'ceEv', 'instrumentIce']; miss = [c for c in need if c not in j.columns]
if miss: print('jobs table missing columns', miss); sys.exit(1)
j.with_columns(pl.col('adductIce').cast(pl.Utf8), pl.col('ceEv').cast(pl.Float64), pl.col('instrumentIce').cast(pl.Utf8)).select(need).write_parquet(dst)
PY
cat > stprep.py <<'PY'
# self-test job table: selftest.json {jobs:[...]} (argv1) or head(limit) of a jobs file (argv2, argv3)
import json, sys, polars as pl
stf, jf, lim = sys.argv[1], sys.argv[2], int(sys.argv[3]); st = json.load(open(stf)) if stf else {}
(pl.DataFrame(st['jobs']) if st.get('jobs') else pl.read_parquet(jf).head(lim)).write_parquet('st_raw.parquet')
PY
cat > stcmp.py <<'PY'
# compare self-test predictions with expected values -> "pass|fail <maxAbsDiff> <nCompared>"
import glob, hashlib, json, sys, polars as pl
mf = json.load(open('manifest.json'))['selftest']; stf, ef = sys.argv[1], sys.argv[2]; st = json.load(open(stf)) if stf else {}
exp = st.get('expected') if st else (json.load(open(ef)) if ef else None); tol = float(st.get('tol', mf.get('tol', 1e-6)))
f = sorted(glob.glob('st/chunk*.parquet')); d = pl.concat([pl.read_parquet(x) for x in f]) if f else None
if d is None or d.height == 0: print('fail noOutput 0'); sys.exit(0)
if exp is None:
    h = hashlib.sha1(json.dumps([[round(x, 3) for x in r] for r in d['mzs'].to_list()] + [[round(x, 3) for x in r] for r in d['intens'].to_list()]).encode()).hexdigest()
    print('pass' if h == mf.get('sha1') else 'fail', 'sha1', d.height); sys.exit(0)
if isinstance(exp, list): exp = {str(i): e for i, e in enumerate(exp)}
keyed = bool(st.get('jobs')) and 'jobId' in st['jobs'][0]
md, n = 0.0, 0
for i, r in enumerate(d.iter_rows(named=True)):
    e = exp.get(str(r['ik14'])) if keyed else exp.get(str(i))
    if e is None: continue
    n += 1
    for a, b in ((r['mzs'], e['mzs']), (r['intens'], e['intens'])):
        if len(a) != len(b): md = float('inf'); continue
        for p, q in zip(a, b): md = max(md, abs(float(p) - float(q)))
print('pass' if (n > 0 and md <= tol) else 'fail', md, n)
PY
hb() {  # status [shard] [chunksDone]
  python hbw.py "$ID" "$IDX" "$JOB" "$1" "${2:--}" "${3:-0}" "$JOBS_DONE" "$RATE" "$ERR429" "$SETUP_SEC" "$SELFTEST" "$KILLTEST$DIE" > hb/hb.json 2>/dev/null || return 0
  ( cd hb && git add hb.json && { git commit -q --amend -m "hb $ID" >/dev/null 2>&1 || git commit -q -m "hb $ID" >/dev/null 2>&1; } && gitc push -qf origin "HEAD:refs/heads/hb/$ID" >/dev/null 2>&1 ) || log "hb push failed"
}
killState() {  # run | pause | kill  (pool/KILL on main: empty=run, PAUSE|KILL [id] per line)
  local k; k=$(curl -sf --max-time 20 "$KILL_URL" 2>/dev/null || true); local st=run
  while read -r a b; do [ -z "$a" ] && continue; [ -n "$b" ] && [ "$b" != "$ID" ] && continue; case "$a" in KILL) st=kill;; PAUSE) [ "$st" = kill ] || st=pause;; esac; done <<< "$k"; echo $st
}
mf() { curl -sf --max-time 20 "$MANIFEST_URL" -o manifest.new && mv manifest.new manifest.json || log "manifest fetch failed"; python -c "import json;print(json.load(open('manifest.json'))['job'])" 2>/dev/null; }
mget() { python -c "import json,sys;m=json.load(open('manifest.json'));v=m;[v:=v[k] for k in sys.argv[1].split('.')];print(v)" "$1"; }
# ---- setup (same assets as run.sh) ----
( cd hb && [ -d .git ] || { git init -q && git checkout -q --orphan "hb/$ID" && git remote add origin "https://github.com/$REPO.git" && git config user.email "$ID@pool" && git config user.name "$ID"; } )
export PATH=$HOME/.local/bin:$PATH
which uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -d venv ] && ! venv/bin/python -c "import sys" >/dev/null 2>&1 && rm -rf venv
[ -d venv ] || uv venv -q --python 3.12 venv
source venv/bin/activate
uv pip install -q kaggle
[ -n "$(mf)" ] || { log 'no manifest'; exit 1; }
JOB=$(mget job); hb setup
if [ ! -f wh/.done ]; then
  kaggle datasets download nicholasooo/casmi-m2-fwdsim-wheels -p wh --unzip -q
  (cd wh && for f in *2.6.0cpu*.whl; do mv "$f" "${f/2.6.0cpu/2.6.0+cpu}"; done; for f in *pt26cpu*.whl; do mv "$f" "${f/2.1.2pt26cpu/2.1.2+pt26cpu}"; done) && touch wh/.done
fi
uv pip install -q sympy==1.13.1 filelock jinja2 fsspec networkx typing-extensions setuptools packaging requests pydantic numpy pandas h5py scipy scikit-learn tqdm pyyaml einops psutil joblib matplotlib seaborn pathos easydict appdirs aiohttp pillow omegaconf polars pyarrow
uv pip install -q --no-index --no-deps --find-links wh torch dgl torch_scatter pygmtools pytorch_lightning torchmetrics lightning_utilities platformdirs multiprocess dill rdkit ms_pred
WANT=$(mget srcCommit 2>/dev/null); HAVE=$(cat src/COMMIT.txt casmi_src/COMMIT.txt 2>/dev/null | head -1)
[ -n "$WANT" ] && [ "$WANT" != "$HAVE" ] && { log "casmi-src refresh: have '$HAVE' want '$WANT'"; rm -rf src casmi_src; }
[ -f src/casmi/fwdsim/runIceberg.py ] || kaggle datasets download nicholasooo/casmi-src -p . --unzip -q
[ -f src/casmi/fwdsim/runIceberg.py ] || { [ -d casmi_src/src ] && ln -sfn casmi_src/src src; }
[ -f src/casmi/fwdsim/runIceberg.py ] || { log 'casmi package missing'; hb err; exit 1; }
CKPT_DS=$(mget ckptDs); GEN=$(mget gen); INTEN=$(mget inten); CK=ck/${CKPT_DS#*/}
for f in "$GEN" "$INTEN"; do [ -f "$CK/$f" ] || kaggle datasets download "$CKPT_DS" -f "$f" -p "$CK/$(dirname "$f")" -q --unzip; [ -f "$CK/$f" ] || { log "ckpt missing: $f"; hb err; exit 1; }; done
SETUP_SEC=$(( $(date +%s) - T_BOOT )); log "setup done in ${SETUP_SEC}s: $(nproc) cpus; torch $(python -c 'import torch;print(torch.__version__)')"
printf '{"title": "%s", "id": "%s", "licenses": [{"name": "other"}]}\n' "$(mget dsPrefix)-$ID" "nicholasooo/$(mget dsPrefix)-$ID" > ds/dataset-metadata.json
getJobs() {  # download every input of the manifest from jobsDataset (once per job), adapt the ICE job table if the module is runIceberg
  JOBS_DS=$(mget jobsDataset); JOBS_FILE=$(mget jobsFile 2>/dev/null); local f
  for f in $(python -c "import json;m=json.load(open('manifest.json'));print(' '.join(m.get('inputs') or [m['jobsFile']]))"); do
    [ -f "jobs/$f" ] || kaggle datasets download "$JOBS_DS" -f "$f" -p jobs -q --unzip; [ -f "jobs/$f" ] || { log "input missing $JOBS_DS/$f"; return 1; }
  done
  MODULE=$(mget module 2>/dev/null); MODULE=${MODULE:-casmi.fwdsim.runIceberg}; MARGS=$(mget args 2>/dev/null)
  if [ "$MODULE" = casmi.fwdsim.runIceberg ]; then python adapt.py "jobs/$JOBS_FILE" "jobs/ice_$JOBS_FILE" || return 1; MARGS="${MARGS:---jobs jobs/ice_$JOBS_FILE}"; fi
}
iceArgs() { echo "--gen $CK/$GEN --inten $CK/$INTEN --device cpu --threads $(mget ice.threads) --batch $(mget ice.batch) --chunk $(mget chunk) --maxNodes $(mget ice.maxNodes) --sparseK $(mget ice.sparseK)"; }
selftest() {  # 8 known-answer ICE jobs; SELFTEST=pass|fail|none; a worker never claims while fail
  SELFTEST=none; local F JF EF L f; F=$(mget selftest.file 2>/dev/null); JF=$(mget selftest.jobsFile 2>/dev/null); EF=$(mget selftest.expectedFile 2>/dev/null); L=$(mget selftest.limit 2>/dev/null)
  [ -z "$F$JF$L" ] && return 0
  [ -n "$F$JF" ] || JF=$JOBS_FILE   # legacy: first L jobs of the job table, sha1 of rounded values
  for f in $F $JF $EF; do [ -f "jobs/$f" ] || kaggle datasets download "$JOBS_DS" -f "$f" -p jobs -q --unzip; [ -f "jobs/$f" ] || { log "selftest input missing $f"; SELFTEST=fail; return; }; done
  rm -f st_raw.parquet st_jobs.parquet; python stprep.py "${F:+jobs/$F}" "${JF:+jobs/$JF}" "${L:-8}" && python adapt.py st_raw.parquet st_jobs.parquet || { SELFTEST=fail; log 'selftest prep failed'; return; }
  rm -rf st && PYTHONPATH=src python -m casmi.fwdsim.runIceberg --jobs st_jobs.parquet --out st $(iceArgs) --shard 0/1 > log/selftest.log 2>&1
  read -r SELFTEST STD STN <<< "$(python stcmp.py "${F:+jobs/$F}" "${EF:+jobs/$EF}" 2>>log/selftest.log | tail -1)"; SELFTEST=${SELFTEST:-fail}; log "selftest $SELFTEST maxAbsDiff ${STD:-?} jobs ${STN:-0} tol $(mget selftest.tol 2>/dev/null || echo 1e-6)"
}
getJobs && selftest; ST_JOB=$JOB
hb ready
publish() {  # $1 out dir $2 shard tag  -> version (or create) my dataset with all finished shards
  for f in "$1"/*; do [ -f "$f" ] && ln -f "$f" "ds/${JOB}_$2_$(basename "$f")"; done
  sleep $(( IDX * 7 % 60 ))
  local R; R=$(kaggle datasets version -p ds -q -m "$JOB $2 $(date -u +%H:%M)" 2>&1); c429 "$R"
  case "$R" in *rror*|*"not found"*|*404*) R=$(kaggle datasets create -p ds -q 2>&1); c429 "$R";; esac; log "publish $2: ${R: -80}"
  case "$R" in *rror*|*"failed"*) return 1;; esac; return 0
}
runShard() {  # $1 shard -> 0 done, 2 paused, 3 killed, 9 killtest/die exit
  local s=$1 S; S=$(printf 's%02d' "$s"); local OUT=out_$JOB/$S; mkdir -p "$OUT"; local N; N=$(mget N)
  python - "$OUT" <<'PY'
import glob, os, sys, polars as pl
n = 0
for f in glob.glob(sys.argv[1] + '/chunk*.parquet'):
    try: pl.read_parquet(f); n += 1
    except Exception: os.remove(f)
print(f'resume: {n} finished chunks on disk')
PY
  local T0; T0=$(date +%s); local C0; C0=$(ls "$OUT"/chunk*.parquet 2>/dev/null | wc -l)
  PYTHONPATH=src nohup python -m "$MODULE" $MARGS --out "$OUT" $(iceArgs) --shard "$s/$N" > "log/ice_${JOB}_$S.log" 2>&1 &
  local PID=$! LASTHB; LASTHB=$(date +%s); hb running "$s" "$C0"; local ST
  sleep $(( 15 + RANDOM % 30 )); if [ "$(python conflict.py "$ID" "$IDX" "$s" "$JOB" 2>>log/conflict.log)" = 1 ]; then kill $PID 2>/dev/null; sleep 2; kill -9 $PID 2>/dev/null; log "shard $s: claimed earlier by another worker, backing off"; hb ready; return 4; fi
  while kill -0 $PID 2>/dev/null; do
    sleep 30 || true; local C; C=$(ls "$OUT"/chunk*.parquet 2>/dev/null | wc -l)
    if [ "$KILLTEST$DIE" != 00 ] && [ ! -f killtest.done ] && [ "$C" -gt "$C0" ]; then
      touch killtest.done; kill $PID 2>/dev/null; sleep 2; kill -9 $PID 2>/dev/null
      if [ "$DIE" = 1 ]; then log "DIE: runner stopped mid-shard $s after $C chunk(s); do NOT restart"; hb died "$s" "$C"; return 9; fi
      log "KILLTEST: runner killed mid-shard $s after $C chunk(s) — restart the same command now"; hb killtest "$s" "$C"; return 9
    fi
    if [ $(( $(date +%s) - LASTHB )) -ge $(( $(mget hbMin) * 60 )) ]; then RATE=$(grep -o '[0-9.]* s/job' "log/ice_${JOB}_$S.log" | tail -1 | cut -d' ' -f1); RATE=${RATE:-0}; hb running "$s" "$C"; LASTHB=$(date +%s)
      ST=$(killState); [ "$ST" = run ] || { kill $PID 2>/dev/null; sleep 2; kill -9 $PID 2>/dev/null; [ "$ST" = pause ] && { hb paused "$s" "$C"; return 2; }; hb killed "$s" "$C"; return 3; }
    fi
  done
  wait $PID; local RC=$?; RATE=$(grep -o '[0-9.]* s/job' "log/ice_${JOB}_$S.log" | tail -1 | cut -d' ' -f1); RATE=${RATE:-0}
  python summ.py "$OUT" "$s" "$N" "$RC" "$(( $(date +%s) - T0 ))" "${MODULE##*.}" "$(mget chunk)" "$JOB"
  local ROWS; ROWS=$(python -c "import json;print(json.load(open('$OUT/pool_summary.json'))['rows'])"); JOBS_DONE=$(( JOBS_DONE + ROWS ))
  hb publishing "$s" "$(ls "$OUT"/chunk*.parquet | wc -l)"
  if publish "$OUT" "$S"; then echo "$JOB:$s" >> hb/done.txt; hb published "$s"; log "shard $s done: $ROWS rows, $(( $(date +%s) - T0 ))s"; else log "publish failed shard $s (kept on disk)"; hb err "$s"; fi
  return 0
}
# ---- main loop: claim -> run -> publish; idle when nothing to claim; KILL/PAUSE honoured every chunk ----
IDLE=0
while :; do
  ST=$(killState); if [ "$ST" = kill ]; then hb killed; log 'KILL: exiting'; exit 0; elif [ "$ST" = pause ]; then hb paused; log 'PAUSE'; zz 300; continue; fi
  NEWJOB=$(mf); [ -n "$NEWJOB" ] && [ "$NEWJOB" != "$JOB" ] && { JOB=$NEWJOB; log "manifest job now $JOB"; rm -f killtest.done; }
  getJobs || { hb err; zz 300; continue; }
  [ "$ST_JOB" = "$JOB" ] || { selftest; ST_JOB=$JOB; }
  [ "$SELFTEST" = fail ] && { log 'selftest failed: refusing to claim'; hb selftestFail; zz 600; ST_JOB=; continue; }
  read -r PICK ND NA NH <<< "$(python claim.py "$ID" "$IDX" "$T_BOOT" 2>&1 | tail -1)"; log "claim: shard $PICK (done $ND active $NA heartbeats $NH)"
  [[ "$PICK" =~ ^-?[0-9]+$ ]] || { log 'claim failed'; hb err; zz 120; continue; }
  if [ "$PICK" = -1 ]; then hb idle; IDLE=1; zz 300; continue; fi
  IDLE=0; runShard "$PICK"; RC=$?
  case $RC in 9) exit 9;; 2) zz 300;; 3) log 'KILL: exiting'; exit 0;; 4) zz 30;; esac
done
