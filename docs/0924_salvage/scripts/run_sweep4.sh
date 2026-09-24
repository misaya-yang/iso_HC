#!/bin/bash
cd "$(dirname "$0")"
CONFIGS=( "baseline 0.01" "identity 0.01" "leaky 0.01" "unconstrained 0.01" "exchange 0.01" "isohc 0.01" )
run() { set -- $1; m=$1; lam=$2; tag=${m}_lam${lam}_scaledinit_s0
  [ -f ../cpu_probe_runs/$tag.json ] && return
  python3 access_probe.py --method $m --lam $lam --scaled_init --steps 1200 --out ../cpu_probe_runs/$tag.json --threads 1 > ../cpu_probe_runs/$tag.log 2>&1; }
export -f run
printf '%s\n' "${CONFIGS[@]}" | xargs -P 4 -I{} bash -c 'run "{}"'
echo ALLDONE4
