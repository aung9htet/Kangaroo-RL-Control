# Kangaroo_Libero and Kangaroo-RL-Control: how the two repositories fit together

The question was whether to clone this repository into Kangaroo_Libero, so that Kangaroo_Libero is the
simulator setup and this repository holds only the RL model. The recommendation is the other way round:
**this repository is the project, and Kangaroo_Libero is a pinned git submodule inside it.** This page says
why, how it is set up, and how to work with it day to day.

## 1. The recommendation in one picture

```
Kangaroo-RL-Control/                 <- this repository: the experiment (RL code, configs, docs, results)
├── sim/Kangaroo_Libero/             <- git submodule, pinned to one commit (c7be874)
├── docker/Dockerfile                <- FROM aung9htet/kangaroo-libero:0.1.0-ubuntu-22.04 + PyTorch
├── run_rl_docker.bash               <- mounts this checkout; the submodule comes with it
├── kangaroo_rl/ configs/ scripts/ tests/ docs/
```

Inside the container, the simulator entrypoint is pointed at the submodule
(`KANGAROO_REPO=/workspace/Kangaroo-RL-Control/sim/Kangaroo_Libero`). The Kangaroo_Libero code that runs is
therefore exactly the submodule's commit, with its `src/` first on `PYTHONPATH`, as when Kangaroo_Libero runs
on its own. The image supplies only the dependencies (ROS 2, PAL packages, MuJoCo, robosuite, LIBERO, the
prepared Kangaroo model) plus PyTorch.

## 2. Why not clone the RL repository into Kangaroo_Libero

| | RL inside the simulator (the suggestion) | Simulator as a submodule of the RL repository (chosen) |
| --- | --- | --- |
| Direction of dependency | the simulator would contain its user | the user pins the simulator it was run with |
| Simulator image | its Dockerfile does `COPY . /opt/kangaroo-libero`, and `.dockerignore` does not exclude a nested repository, so the next simulator image would bake in the RL code, its runs and checkpoints | unchanged; the RL image is built `FROM` it |
| Simulator dependencies | Kangaroo_Libero deliberately ships no training dependencies (`docker/requirements.txt`: "Training and demonstration dependencies are intentionally absent"); torch would have to go into it | torch lives only in the RL image |
| Git | a nested repository the simulator must ignore or track | a submodule: one line in `.gitmodules`, one pinned commit |
| Reproducibility | an RL result is not tied to a simulator version | every run records the RL commit, the simulator commit and the image ID (`run.json`) |
| Other users of the simulator | see an RL project inside it | see a clean simulator |
| Cloning | two clones, nested | `git clone --recursive` once |

Your suggestion has one real advantage: the existing launcher would mount the RL code automatically. The
RL repository's own launcher, `run_rl_docker.bash` (modelled on `run_kangaroo_docker.bash`, same options),
gives the same convenience the other way round.

## 3. How it is set up

- `.gitmodules` uses a relative URL, `../Kangaroo_Libero.git`, resolved against this repository's own
  remote. It works with your SSH alias (`github.com-personal`) and with anyone else's.
- `docker/Dockerfile`:
  - Base: `aung9htet/kangaroo-libero:0.1.0-ubuntu-22.04`, the image Kangaroo_Libero builds and publishes.
  - Adds `torch==2.5.1` (CUDA 12.1 wheels; numpy kept at the simulator's 1.26.4 by `docker/constraints.txt`).
  - Copies `docker/enter.bash`, which runs after the simulator's entrypoint, checks that the submodule is
    present and starts in this checkout.
  - The build context is only `docker/`, so builds are fast and do not copy the repository.
- `run_rl_docker.bash`:
  - Options `--build`, `--image`, `--gpu auto|nvidia|software`, `--software`, `--headless`, `--name`.
  - Mounts this checkout at `/workspace/Kangaroo-RL-Control` and refuses to start if the submodule is
    missing.
  - Passes the image ID into the container for the run metadata.
- Simulator settings for this experiment (the robot base 0.25 m behind the table instead of 0.55 m) are an
  override file written at runtime from the submodule's `config/simulation.yaml` and read through
  `KANGAROO_CONFIG`. Kangaroo_Libero is not modified.

## 4. Working with it

```bash
# first time
git clone --recursive git@github.com-personal:aung9htet/Kangaroo-RL-Control.git
cd Kangaroo-RL-Control
./run_rl_docker.bash --build --headless -- python -m pytest -q

# a checkout made without --recursive
git submodule update --init

# use a newer simulator commit
git -C sim/Kangaroo_Libero fetch && git -C sim/Kangaroo_Libero checkout <commit>
git add sim/Kangaroo_Libero && git commit -m "Use Kangaroo_Libero <commit>"

# change the simulator and the experiment together
#   edit sim/Kangaroo_Libero (it is a normal clone), commit and push there,
#   then commit the new submodule pointer here
```

Rebuild the RL image (`--build`) only when the base image or `docker/` changes. Python edits in either
repository take effect on the next run.

**One caveat.** The image `0.1.0` was built from Kangaroo_Libero before its first commit (`c7be874`) was
made. The Python code and settings that run come from the submodule. The prepared robot model
(`/opt/kangaroo-assets`) and the patched LIBERO come from the image. If the simulator's model preparation
or patches change, rebuild the simulator image, tag it, and update `BASE_IMAGE` / `KANGAROO_IMAGE` here.
Both IDs are recorded in every run.
