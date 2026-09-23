#!/usr/bin/env bash
# Bootstrap / refresh csRunner (+ keep-alive heartbeat) on a CloudStudio box.
# usage: curl -sL <raw>/tools/csBoot.sh | bash -s [<runner-token>] [port]
#   runner token: written to /workspace/casmi/.token when given; otherwise the existing file is used.
#   keep-alive:   started only if /workspace/casmi/.cs_healthz_token (CloudStudio access token) exists.
set -e
R=/workspace/casmi; PORT="${2:-8787}"; RAW=https://raw.githubusercontent.com/Nicholas022400701/casmi-compute/main/tools
mkdir -p "$R/jobs" && cd "$R"
umask 077
if [ -n "$1" ]; then echo "$1" > .token; fi
[ -s .token ] || { echo "runner token required (arg 1 or $R/.token)"; exit 1; }
curl -sL "$RAW/csRunner.py" -o csRunner.py.new && python3 -m py_compile csRunner.py.new && mv csRunner.py.new csRunner.py
curl -sL "$RAW/csKeepalive.sh" -o csKeepalive.sh && chmod +x csKeepalive.sh
# restart the runner loop (jobs started via /start live in their own sessions and survive this)
pkill -f 'csRunner.py' 2>/dev/null || true
sleep 1
nohup bash -c "while true; do CS_TOKEN=\$(cat $R/.token) python3 $R/csRunner.py $PORT; sleep 3; done" > "$R/runner.log" 2>&1 < /dev/null &
if [ -s .cs_healthz_token ]; then
  pkill -f 'csKeepalive.sh' 2>/dev/null || true
  nohup "$R/csKeepalive.sh" >> "$R/keepalive.log" 2>&1 < /dev/null &
  echo "keepalive started"
else
  echo "no .cs_healthz_token -> keepalive NOT started (space may idle-sleep)"
fi
sleep 2; tail -n 1 "$R/runner.log"; (ss -ltnp 2>/dev/null || netstat -ltnp 2>/dev/null) | grep -q ":$PORT " && echo "runner listening on $PORT" || echo "port $PORT not listening yet"
