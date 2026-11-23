#!/bin/bash
#SBATCH --job-name=natural-rl
#SBATCH --account=hpc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=spark-c3d8   # keep the manager free: it is shared and where you log in
#SBATCH --cpus-per-task=20
#SBATCH --mem=110G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# One job = one Spark. Submit from the repo root on spark-c3d8 (under your shared $HOME).
#   sbatch dgx_spark_slurm.sbatch                        # the whole programme on one Spark
#   sbatch dgx_spark_slurm.sbatch 'python -u -m natural_rl run-all --phases P1 --states v1,v2,v3,v4 --no-lock-scalers'
# To use two Sparks, run ./slurm/submit_two_sparks.bash instead (it chains the jobs).
# Re-submitting the same script continues an interrupted programme (run-all resumes from results/).
#
# ARM64 image (DGX Spark is aarch64). Build and push it from a machine with Docker, not the Spark:
#   docker login -u aung9htet && ./docker/build_arm64.bash
# Pyxis syntax: registry#path:tag. Override with IMAGE=... sbatch ... (a .sqsh path also works).
# Prefer the shared squashfs in $HOME (imported once; every node can read it) over a fresh registry import per job.
if [[ -z "${IMAGE:-}" && -f "$HOME/containers/kangaroo-natural-rl-arm64.sqsh" ]]; then
    IMAGE="$HOME/containers/kangaroo-natural-rl-arm64.sqsh"
fi
IMAGE="${IMAGE:-docker.io#aung9htet/kangaroo-natural-rl:0.3.0-arm64}"

set -euo pipefail

CMD="${1:-python -u -m natural_rl run-all --device cuda --threads 2}"

echo "Job $SLURM_JOB_ID on $(hostname) started $(date -Is)"
echo "Command: $CMD"

# PATH, KANGAROO_REPO, PYTHONPATH etc. are baked into the image (docker/Dockerfile.natural-arm64); Pyxis does not run
# an ENTRYPOINT, and this image does not need one. Only the run-time choices are set here.
srun --mpi=none \
     --container-image="$IMAGE" \
     --no-container-remap-root \
     --container-mounts="$SLURM_SUBMIT_DIR:/workspace/Kangaroo-RL-Control,$SLURM_SUBMIT_DIR/pal-assets/kangaroo_mujoco:/opt/pal/alum/share/kangaroo_mujoco:ro,$SLURM_SUBMIT_DIR/pal-assets/kangaroo-assets:/opt/kangaroo-assets:ro" \
     --container-workdir=/workspace/Kangaroo-RL-Control \
     --export=ALL,MUJOCO_GL=egl,PYTHONHASHSEED=0,OMP_NUM_THREADS=2,NATURAL_RL_IMAGE_REF=kangaroo-natural-rl:0.3.0-arm64 \
     bash -c '
       set -euo pipefail
       test -f "$KANGAROO_REPO/src/kangaroo_libero/__init__.py" || { echo "Simulator submodule missing: run git submodule update --init" >&2; exit 2; }
       python -c "import torch; assert torch.cuda.is_available(), \"no CUDA\"; print(\"GPU:\", torch.cuda.get_device_name(0))"
       eval "$1"
     ' bash "$CMD"

echo "Job $SLURM_JOB_ID finished $(date -Is)"
