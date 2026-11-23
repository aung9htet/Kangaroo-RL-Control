#!/usr/bin/env bash
# Run the whole programme on TWO Sparks with chained Slurm jobs (run from the repo root on spark-c3d8).
#
# run-all is a single-node program, so the programme is split by state: v1-v4 on one Spark, v5-v8 on the other.
# The two global locks need ALL cells, so they run as separate single steps in between:
#   P0 (+ reward lock)  ->  P1 half A | P1 half B  ->  scaler lock  ->  P2 half A | P2 half B
#   ->  scaler lock if P1 could not fix them  ->  P3 | P4  (P3 and P4 are independent, one Spark each)
# Every job is safe to resubmit: finished runs are skipped, interrupted runs start again from step 0.
#   WALL=48:00:00 ./slurm/submit_two_sparks.bash          # per-job wall-clock limit (check the cluster maximum first)
#   ./slurm/submit_two_sparks.bash --dry-run              # print the sbatch lines only
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
wall=${WALL:-48:00:00}
dry=0; [[ ${1:-} == --dry-run ]] && dry=1
mkdir -p logs
A=v1,v2,v3,v4
B=v5,v6,v7,v8
RA='python -u -m natural_rl run-all --no-lock-scalers --device cuda --threads 2'
NRL='python -u -m natural_rl'
LOCK_P1="$NRL calibrate scalers --runs \"\$($NRL catalog programme --phase P1 --commands | grep -o -- '--run-id [^ ]*' | cut -d' ' -f2 | paste -sd,)\""
LOCK_P2="$NRL calibrate scalers --runs \"\$($NRL catalog programme --phase P2 --commands | grep -o -- '--run-id [^ ]*' | cut -d' ' -f2 | paste -sd,)\""
ANALYSE_P2="$NRL catalog programme --phase P2 --commands | grep -o -- '--run-id [^ ]*' | cut -d' ' -f2 | while read -r id; do $NRL analyse --run-id \"\$id\"; done"

sub() {  # sub NAME DEPENDENCY COMMAND  -> prints the job id
    local name=$1 dep=$2 cmd=$3 args=(--parsable --job-name="nrl-$1" --time="$wall")
    [[ -n $dep ]] && args+=(--dependency="afterok:$dep")
    if (( dry )); then echo "sbatch ${args[*]} dgx_spark_slurm.sbatch '$cmd'" >&2; echo "<$name>"; else sbatch "${args[@]}" dgx_spark_slurm.sbatch "$cmd"; fi
}
j0=$(sub p0 "" "$RA --phases P0")
j1a=$(sub p1-a "$j0" "$RA --phases P1 --states $A")
j1b=$(sub p1-b "$j0" "$RA --phases P1 --states $B")
# If P1 has too few successes the fit refuses and writes nothing; the job still succeeds, and the P2-based fit below takes over.
jl1=$(sub lock-scalers "$j1a:$j1b" "{ $LOCK_P1; } || echo 'P1 could not fix the scalers: they are fitted from P2 instead'")
j2a=$(sub p2-a "$jl1" "$RA --phases P2 --states $A")
j2b=$(sub p2-b "$jl1" "$RA --phases P2 --states $B")
jl2=$(sub lock-scalers-p2 "$j2a:$j2b" "[ -f configs/metrics/hks_scalers_v1.yaml ] && grep -q 'status: locked' configs/metrics/hks_scalers_v1.yaml || { $LOCK_P2 && $ANALYSE_P2; }")
sub p3 "$jl2" "$NRL run-all --phases P3 --device cuda --threads 2" >/dev/null
sub p4 "$jl2" "$NRL run-all --phases P4 --device cuda --threads 2" >/dev/null
echo "submitted: watch with  squeue -u \$USER   and   tail -f logs/nrl-*-<id>.out"
