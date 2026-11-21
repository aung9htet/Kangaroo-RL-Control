# Kangaroo-RL-Control runtime: the Kangaroo_Libero simulator image plus PyTorch.
# Build context is this directory only (see run_rl_docker.bash --build).
ARG BASE_IMAGE=aung9htet/kangaroo-libero:0.1.0-ubuntu-22.04
FROM ${BASE_IMAGE}
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG TORCH_VERSION=2.5.1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu121
COPY constraints.txt /opt/kangaroo-rl/constraints.txt
# The simulator's pins (numpy 1.26.4) must survive the torch install.
RUN /opt/kangaroo-venv/bin/pip install --no-cache-dir -c /opt/kangaroo-rl/constraints.txt \
      --index-url "${TORCH_INDEX}" --extra-index-url https://pypi.org/simple "torch==${TORCH_VERSION}" \
    && /opt/kangaroo-venv/bin/python -c "import numpy, torch; assert numpy.__version__ == '1.26.4', numpy.__version__" \
    && /opt/kangaroo-venv/bin/pip freeze --local > /opt/kangaroo-rl/runtime.freeze.txt
COPY enter.bash /opt/kangaroo-rl/enter.bash
RUN chmod 755 /opt/kangaroo-rl/enter.bash
# The simulator entrypoint reads KANGAROO_REPO (cd, PYTHONPATH, config); point it at the submodule
# inside the mounted RL checkout so the pinned simulator source is the one that runs.
ENV KANGAROO_RL_REPO=/workspace/Kangaroo-RL-Control \
    KANGAROO_REPO=/workspace/Kangaroo-RL-Control/sim/Kangaroo_Libero \
    PYTHONPATH=/workspace/Kangaroo-RL-Control
WORKDIR /workspace/Kangaroo-RL-Control
ENTRYPOINT ["/opt/kangaroo-libero/docker/entrypoint.bash", "/opt/kangaroo-rl/enter.bash"]
CMD ["bash"]
LABEL org.opencontainers.image.title="Kangaroo RL Control" \
      org.opencontainers.image.source="https://github.com/aung9htet/Kangaroo-RL-Control" \
      org.opencontainers.image.version="0.1.0"
