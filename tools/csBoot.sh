#!/usr/bin/env bash
# Bootstrap csRunner on a CloudStudio box.  usage: curl -sL <raw>/tools/csBoot.sh | bash -s <token> [port]
set -e
TOK="$1"; PORT="${2:-8787}"; R=/workspace/casmi
[ -n "$TOK" ] || { echo "token required"; exit 1; }
mkdir -p "$R" && cd "$R"
curl -sL https://raw.githubusercontent.com/Nicholas022400701/casmi-compute/main/tools/csRunner.py -o csRunner.py
umask 077; echo "$TOK" > .token
pkill -f 'csRunner.py' 2>/dev/null || true
sleep 1
nohup bash -c "while true; do CS_TOKEN=\$(cat $R/.token) python3 $R/csRunner.py $PORT; sleep 3; done" > "$R/runner.log" 2>&1 < /dev/null &
sleep 2; tail -n 2 "$R/runner.log"; (ss -ltnp 2>/dev/null || netstat -ltnp 2>/dev/null) | grep ":$PORT " || echo "port $PORT not listening yet"
