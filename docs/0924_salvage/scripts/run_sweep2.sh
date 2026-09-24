#!/bin/bash
cd "$(dirname "$0")"
CONFIGS=(
 "orthogonal 0.5 4" "exchange 0.5 4" "leaky 0.5 4"
 "orthogonal 0.01 4" "exchange 0.01 4" "leaky 0.01 4"
)
run() { set -- $1; m=$1; lam=$2; db=$3; tag=${m}_lam${lam}_db${db}_s0
  [ -f ../cpu_probe_runs/$tag.json ] && return
  python3 access_probe.py --method $m --lam $lam --diag_bias $db --steps 1200 --out ../cpu_probe_runs/$tag.json --threads 1 > ../cpu_probe_runs/$tag.log 2>&1; }
export -f run
printf '%s\n' "${CONFIGS[@]}" | xargs -P 4 -I{} bash -c 'run "{}"'
echo ALLDONE2
