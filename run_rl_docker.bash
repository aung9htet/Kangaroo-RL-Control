#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
image=${KANGAROO_RL_IMAGE:-kangaroo-rl:0.1.0}
base_image=${KANGAROO_IMAGE:-aung9htet/kangaroo-libero:0.1.0-ubuntu-22.04}
gpu=auto
headless=0
build=0
container_name="kangaroo-rl-$(id -u)-$$"
usage() {
    cat <<'HELP'
Usage: ./run_rl_docker.bash [options] [-- command ...]
  --build                Build the RL image (simulator image + PyTorch) before starting
  --image NAME           Use another local RL image or tag
  --gpu auto|nvidia|software
  --software             Use Mesa software rendering and CPU-only compute
  --headless             Do not mount a desktop display (MUJOCO_GL=egl or osmesa)
  --name NAME            Choose a container name for docker exec
  -h, --help             Show this help
Default command: bash, in the mounted Kangaroo-RL-Control checkout.
The simulator source comes from the sim/Kangaroo_Libero submodule of this checkout.
HELP
}
die() { echo "Error: $*" >&2; exit 1; }
need_value() { [[ $# -ge 2 && -n $2 ]] || die "$1 needs a value"; }
while (($#)); do
    case "$1" in
        --build) build=1; shift ;;
        --image) need_value "$@"; image=$2; shift 2 ;;
        --gpu) need_value "$@"; gpu=$2; shift 2 ;;
        --software) gpu=software; shift ;;
        --headless) headless=1; shift ;;
        --name) need_value "$@"; container_name=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
        *) die "Unknown option: $1 (use -- before a container command)" ;;
    esac
done
[[ $gpu == auto || $gpu == nvidia || $gpu == software ]] || die 'Choose --gpu auto, nvidia, or software'
command -v docker >/dev/null || die 'Install Docker Engine on the host first.'
docker info >/dev/null 2>&1 || die 'Docker Engine is unavailable or your user cannot access it. Check the daemon and docker group membership.'
[[ $(uname -m) == x86_64 ]] || die 'This project supports Linux amd64 hosts.'
[[ -f $repo_dir/sim/Kangaroo_Libero/src/kangaroo_libero/__init__.py ]] \
    || die 'The simulator submodule is missing. Run: git submodule update --init'
[[ $repo_dir != *,* ]] || die 'Docker mount paths cannot contain commas; move the checkout to another path.'
if ((build)) || ! docker image inspect "$image" >/dev/null 2>&1; then
    docker image inspect "$base_image" >/dev/null 2>&1 \
        || die "Base image $base_image is missing. Pull or build it from sim/Kangaroo_Libero first (see docs/integration.md)."
    echo "Building $image from $base_image"
    docker build --network host --platform linux/amd64 --build-arg "BASE_IMAGE=$base_image" \
        -f "$repo_dir/docker/Dockerfile" -t "$image" "$repo_dir/docker"
fi
if [[ $gpu == auto ]]; then
    if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1 && docker info --format '{{json .Runtimes}}' | grep -q nvidia; then
        gpu=nvidia
    else
        gpu=software
        echo 'Using software rendering and CPU compute; select --gpu nvidia after configuring NVIDIA Container Toolkit.'
    fi
fi
args=(run --rm --init --name "$container_name" --shm-size 1g
      --mount "type=bind,src=$repo_dir,dst=/workspace/Kangaroo-RL-Control"
      -e "LOCAL_USER_ID=$(id -u)" -e "LOCAL_GROUP_ID=$(id -g)"
      -e "KANGAROO_RL_IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$image")")
[[ -t 0 && -t 1 ]] && args+=(-it)
if [[ $gpu == nvidia ]]; then
    command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1 || die 'The host NVIDIA driver is not working; check nvidia-smi or use --software.'
    docker info --format '{{json .Runtimes}}' | grep -q nvidia || die 'Configure NVIDIA Container Toolkit for Docker, or use --software.'
    args+=(--gpus all -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute,display
           -e __NV_PRIME_RENDER_OFFLOAD=1 -e __GLX_VENDOR_LIBRARY_NAME=nvidia)
else
    args+=(-e LIBGL_ALWAYS_SOFTWARE=1 -e CUDA_VISIBLE_DEVICES=)
fi
xauth_file=
cleanup() { [[ -z $xauth_file ]] || rm -f -- "$xauth_file"; }
trap cleanup EXIT
if ((headless)); then
    if [[ $gpu == nvidia ]]; then args+=(-e MUJOCO_GL=egl); else args+=(-e MUJOCO_GL=osmesa); fi
    args+=(-e MPLBACKEND=Agg)
else
    [[ -n ${DISPLAY:-} ]] || die 'DISPLAY is missing. Use a Linux desktop with X11/XWayland, or --headless.'
    command -v xauth >/dev/null || die 'Install xauth on the host for GUI access.'
    xauth_file=$(mktemp /tmp/kangaroo-rl-xauth.XXXXXX)
    cookie=$(xauth nlist "$DISPLAY" 2>/dev/null || true)
    [[ -n $cookie ]] || die 'No X11 authorization cookie found. Check DISPLAY and XAUTHORITY in your desktop terminal.'
    printf '%s\n' "$cookie" | sed 's/^..../ffff/' | xauth -f "$xauth_file" nmerge -
    args+=(-e "DISPLAY=$DISPLAY" -e XAUTHORITY=/tmp/kangaroo.xauth -e MUJOCO_GL=glfw
           -e QT_X11_NO_MITSHM=1
           --mount 'type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix,readonly'
           --mount "type=bind,src=$xauth_file,dst=/tmp/kangaroo.xauth,readonly")
fi
(($#)) || set -- bash
echo "Starting $container_name ($gpu); repository: $repo_dir"
docker "${args[@]}" "$image" "$@"
