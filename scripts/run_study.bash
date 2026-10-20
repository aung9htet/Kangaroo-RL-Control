#!/usr/bin/env bash
# The whole study, inside the container:  ./run_rl_docker.bash --headless -- bash scripts/run_study.bash
# Stages can be selected:                 bash scripts/run_study.bash train evaluate figures
set -euo pipefail
cd "$(dirname "$0")/.."
stages=("$@")
((${#stages[@]})) || stages=(probe train evaluate baselines dp demo figures)
threads=${STUDY_THREADS:-5}
log() { echo "[$(date +%H:%M:%S)] $*"; }
run_parallel() {   # run the given commands in parallel, at most $STUDY_JOBS at a time
    local jobs=${STUDY_JOBS:-4}
    for command in "$@"; do
        while (( $(jobs -rp | wc -l) >= jobs )); do sleep 5; done
        log "start: $command"
        bash -c "$command" > "runs/logs/$(echo "$command" | md5sum | cut -c1-8).log" 2>&1 &
    done
    wait
}
mkdir -p runs/logs
for stage in "${stages[@]}"; do
    case $stage in
        probe) log 'probe'; python scripts/probe_model.py > runs/logs/probe.log ;;
        train)
            run_parallel \
                "python -m kangaroo_rl.train --config configs/reach.yaml --seed 0 --threads $threads" \
                "python -m kangaroo_rl.train --config configs/reach.yaml --seed 1 --threads $threads" \
                "python -m kangaroo_rl.train --config configs/reach.yaml --seed 2 --threads $threads" \
                "python -m kangaroo_rl.train --config configs/slice2.yaml --seed 0 --threads $threads" \
                "python -m kangaroo_rl.train --config configs/reach_raw.yaml --seed 0 --threads $threads" \
                "python -m kangaroo_rl.train --config configs/reach_raw.yaml --seed 1 --threads $threads" \
                "python -m kangaroo_rl.train --config configs/reach_raw.yaml --seed 2 --threads $threads" \
                "python -m kangaroo_rl.train --config configs/reach_noher.yaml --seed 0 --threads $threads" ;;
        evaluate)
            run_parallel \
                "python -m kangaroo_rl.evaluate --run runs/reach/seed0 --backend kinematic && python -m kangaroo_rl.evaluate --run runs/reach/seed0 --backend libero --checkpoints all final.pt" \
                "python -m kangaroo_rl.evaluate --run runs/reach/seed1 --backend kinematic && python -m kangaroo_rl.evaluate --run runs/reach/seed1 --backend libero --checkpoints all final.pt" \
                "python -m kangaroo_rl.evaluate --run runs/reach/seed2 --backend kinematic && python -m kangaroo_rl.evaluate --run runs/reach/seed2 --backend libero --checkpoints all final.pt" ;;
        baselines)
            for b in random dls dls_continuous; do
                python -m kangaroo_rl.evaluate --config configs/reach.yaml --baseline $b --backend kinematic
                python -m kangaroo_rl.evaluate --config configs/reach.yaml --baseline $b --backend libero
            done ;;
        dp) log 'dp check'; python -m kangaroo_rl.dp_check --run runs/slice2/seed0 ;;
        demo) log 'retarget demo'; python -m kangaroo_rl.retarget_demo --run runs/reach/seed0 ;;
        figures) log 'figures'; python scripts/make_figures.py ;;
        *) echo "unknown stage $stage" >&2; exit 2 ;;
    esac
done
log done
