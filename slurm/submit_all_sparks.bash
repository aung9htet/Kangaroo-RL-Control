#!/usr/bin/env bash
# Whole programme on TWO Sparks (spark-2c1b, spark-91d0; spark-c3d8 is excluded in the sbatch) as fast as they allow (run from the repo root on spark-c3d8).
#
# Every job takes one whole node (20 cores). P1 and P2 are cut into 12 chunks each (one action x two states, slowest
# action first); Slurm starts a chunk on whichever Spark is free, so the nodes stay busy and finish together.
# The global locks (rewards after P0, HKS scalers after P1 or P2) run as single steps in between, over all cells.
#   P0 (+ reward lock) -> 12 x P1 -> scaler lock -> 12 x P2 -> scaler lock if still needed -> P3 | P4
#   P0_JOB=<jobid> SQSH_JOB=<jobid> ./slurm/submit_all_sparks.bash   # reuse a queued P0 job and wait for the image import
#   WALL=7-00:00:00 ./slurm/submit_all_sparks.bash    # per-job limit
#   ./slurm/submit_all_sparks.bash --dry-run
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
wall=${WALL:-7-00:00:00}
dry=0; [[ ${1:-} == --dry-run ]] && dry=1
mkdir -p logs
NRL='python -u -m natural_rl'
RA="$NRL run-all --no-lock-scalers --device cuda --threads 2"
ids() { echo "\$($NRL catalog programme --phase $1 --commands | grep -o -- '--run-id [^ ]*' | cut -d' ' -f2 | paste -sd,)"; }
LOCK_P1="$NRL calibrate scalers --runs \"$(ids P1)\""
LOCK_P2="$NRL calibrate scalers --runs \"$(ids P2)\""
ANALYSE_P2="$NRL catalog programme --phase P2 --commands | grep -o -- '--run-id [^ ]*' | cut -d' ' -f2 | while read -r id; do $NRL analyse --run-id \"\$id\"; done"

sub() {  # sub NAME DEPENDENCY COMMAND -> job id
    local name=$1 dep=$2 cmd=$3 args=(--parsable --job-name="nrl-$1" --time="$wall")
    [[ -n $dep ]] && args+=(--dependency="afterok:$dep")
    if (( dry )); then echo "sbatch ${args[*]} '${cmd:0:90}'" >&2; echo "<$name>"; else sbatch "${args[@]}" dgx_spark_slurm.sbatch "$cmd"; fi
}
chunks() {  # chunks PHASE DEP -> colon-joined job ids; torque (slowest) first
    local phase=$1 dep=$2 out=() a pair
    for a in torque velocity position; do
        for pair in v1,v2 v3,v4 v5,v6 v7,v8; do
            out+=("$(sub "${phase,,}-$a-${pair//,/}" "$dep" "$RA --phases $phase --actions $a --states $pair")")
        done
    done
    (IFS=:; echo "${out[*]}")
}
j0=${P0_JOB:-$(sub p0 "" "$RA --phases P0")}
p1=$(chunks P1 "$j0${SQSH_JOB:+:$SQSH_JOB}")   # SQSH_JOB: wait for the shared image import, so no job re-imports from Docker Hub
jl1=$(sub lock-scalers "$p1" "{ $LOCK_P1; } || echo 'P1 could not fix the scalers: fitted from P2 instead'")
p2=$(chunks P2 "$jl1")
jl2=$(sub lock-scalers-p2 "$p2" "[ -f configs/metrics/hks_scalers_v1.yaml ] && grep -q 'status: locked' configs/metrics/hks_scalers_v1.yaml || { $LOCK_P2 && $ANALYSE_P2; }")
sub p3 "$jl2" "$NRL run-all --phases P3 --device cuda --threads 2" >/dev/null
sub p4 "$jl2" "$NRL run-all --phases P4 --device cuda --threads 2" >/dev/null
echo "submitted. Progress:  python3 slurm/progress.py --watch"
