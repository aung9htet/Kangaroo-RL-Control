#!/usr/bin/env bash
# Build the ARM64 natural_rl image for the DGX Spark and push it to Docker Hub.
# Run on any machine with Docker (the Spark itself has no Docker). On x86 the build runs under QEMU: slow, but works.
#   docker login -u aung9htet
#   ./docker/build_arm64.bash                 # build + push aung9htet/kangaroo-natural-rl:0.3.0-arm64
#   ./docker/build_arm64.bash --no-push       # build only (loads nothing; just checks the build succeeds)
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
image=${IMAGE:-aung9htet/kangaroo-natural-rl:0.3.0-arm64}
action=--push
[[ ${1:-} == --no-push ]] && action=

if [[ ! -f $repo_dir/sim/Kangaroo_Libero/pyproject.toml ]]; then
    git -C "$repo_dir" submodule update --init
fi
if [[ $(uname -m) != aarch64 ]]; then
    docker run --privileged --rm tonistiigi/binfmt --install arm64 >/dev/null
fi
docker buildx inspect natural-arm64 >/dev/null 2>&1 || docker buildx create --name natural-arm64 --driver docker-container >/dev/null
# shellcheck disable=SC2086
docker buildx build --builder natural-arm64 --platform linux/arm64 \
    -f "$repo_dir/docker/Dockerfile.natural-arm64" -t "$image" $action "$repo_dir"
[[ -n $action ]] && docker buildx imagetools inspect "$image"
