#!/usr/bin/env bash
# Runs after Kangaroo_Libero's entrypoint has set the user, ROS, the venv and PYTHONPATH:
# check the simulator submodule, then start in the RL checkout.
set -e
if [[ ! -f ${KANGAROO_REPO}/src/kangaroo_libero/__init__.py ]]; then
    echo "Simulator submodule missing at ${KANGAROO_REPO}. On the host run: git submodule update --init" >&2
    exit 2
fi
cd "${KANGAROO_RL_REPO:-/workspace/Kangaroo-RL-Control}"
exec "$@"
