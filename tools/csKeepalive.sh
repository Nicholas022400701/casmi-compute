#!/usr/bin/env bash
# Keep a CloudStudio space from idle-sleeping: heartbeat its /healthz every 5 s with the user's access token
# (same mechanism as the workspace's own `nosleep` helper). Token file: /workspace/casmi/.cs_healthz_token
Y=/var/run/cloudstudio/space.yaml
sk=$(grep '^spacekey:' $Y | sed 's/spacekey: *//' | tr -d '\r')
rg=$(grep '^region:' $Y | sed 's/region: *//' | tr -d '\r')
h=$(grep '^host:' $Y | sed 's/host: *//' | tr -d '\r')
u="https://${sk}--api.${rg}.${h}/healthz"
echo "$(date -u +%FT%TZ) keepalive start $u"
fails=0; n=0
while true; do
  code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $(cat /workspace/casmi/.cs_healthz_token)" "$u" || echo 000)
  if [ "$code" != "200" ]; then fails=$((fails+1)); echo "$(date -u +%FT%TZ) heartbeat $code (fails=$fails)"; else fails=0; fi
  n=$((n+1)); [ $((n % 720)) -eq 0 ] && echo "$(date -u +%FT%TZ) ok ($n beats)"
  sleep 5
done
