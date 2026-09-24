#!/bin/bash
cd "$(dirname "$0")"
CONFIGS=(
 "baseline 0.01 4 1" "identity 0.01 4 1" "isohc 0.01 4 1" "orthogonal 0.01 4 1"
 "exchange 0.01 4 1" "mhc 0.01 4 1" "leaky 0.01 4 1" "unconstrained 0.01 4 1"
)
run() { set -- $1; m=$1; lam=$2; db=$3; s=$4; tag=${m}_lam${lam}_db${db}_s${s}
  [ -f ../cpu_probe_runs/$tag.json ] && return
  python3 access_probe.py --method $m --lam $lam --diag_bias $db --seed $s --steps 1200 --out ../cpu_probe_runs/$tag.json --threads 1 > ../cpu_probe_runs/$tag.log 2>&1; }
export -f run
printf '%s\n' "${CONFIGS[@]}" | xargs -P 4 -I{} bash -c 'run "{}"'
echo ALLDONE3
