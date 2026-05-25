#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/isoHC/results/0525_fe_fair_deep48_p33013_20m
LOG="$ROOT/autopoweroff.log"
METHODS=(baseline identity-hc unconstrained mhc isohc)
echo "[$(date +%F %T)] autopoweroff watcher start" >> "$LOG"
while true; do
  done_count=0
  for m in "${METHODS[@]}"; do
    if [[ -f "$ROOT/fe-deep-48l-512_${m}_seed0/run_summary.json" ]]; then
      done_count=$((done_count + 1))
    fi
  done
  if [[ -f "$ROOT/overnight_summary.md" && "$done_count" -eq 5 ]]; then
    echo "[$(date +%F %T)] all results complete; poweroff" >> "$LOG"
    sync
    /sbin/poweroff || poweroff || shutdown -h now
    exit 0
  fi
  if ! pgrep -af "fe-deep-48l-512" >/dev/null; then
    echo "[$(date +%F %T)] training process not found; done_count=$done_count; no poweroff unless summary exists" >> "$LOG"
  else
    echo "[$(date +%F %T)] still running; done_count=$done_count" >> "$LOG"
  fi
  sleep 300
done
