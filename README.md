# Kangaroo-RL-Control

Reinforcement learning for the PAL Kangaroo right arm (7 joints) in LIBERO/MuJoCo: reach the place where an object appears and stay
there. The repository has two parts.

* **`natural_rl`**, a code-only benchmark of ten RL controllers and twelve reward profiles. It compares the *kinematics* of the executed
  reaching movements (smoothness, timing, directness, pauses), not only whether the goal is reached, with a declared proxy score
  (HKS). Everything is launched by hand from a Docker shell, and every saved training checkpoint also yields a lossless, unedited
  raw video.
* **`kangaroo_rl`**, the original single-experiment code (goal-conditioned DecQN with hindsight relabelling), reused by the benchmark as
  its baseline.

**Everything in one command.** All ten controllers (DecQN, Growing Q, PPO-MultiDiscrete, PPO-Continuous, SAC, TD3, TQC, CrossQ, TD-MPC2,
SAC-PID-Lagrangian) x three actions (joint velocity, joint position step, joint torque) x eight states (joint angles and goal; plus previous command
and hold state; plus joint velocity; plus hand position and goal error from forward kinematics; plus hand velocity; plus joint-limit margins and table
geometry; hand-centric without joint angles; everything) x the reward profiles, with every phase, calibration lock and selection in between, one overall progress display, and every CPU core and the GPU the machine has:

```bash
python -m natural_rl run-all
```

The display (format only):

```
[overall] ███████████░░░░░░░░░░░░░░░░░  38.7% | P2 | runs 1,641/4,053 | 1,172.3M/2,901M transitions | 10 running | 3,150 tr/s | 1d 14:20:11 elapsed, ETA 2d 1:02:47
  sac        r7   torque/v3      s02  41.2%  train
  decqn      r1   position/v1    s04  88.0%  eval val_v1
```

The question it answers is which state and action produce the most human-like reaching movement, and what that depends on. `--actions`,
`--states`, `--phases`, `--workers`, `--device` and `--dry-run` narrow or inspect it (section 1).

HKS measures resemblance to a few simple point-to-point reaching priors (one smooth speed peak, minimum-jerk timing, a straight path, no
pauses or reversals). It is a computational proxy, not a validated measure of perceived naturalness. "Speed effort" and "range margin"
are command/geometry proxies, not energy or joint stress.

## Contents

- [1. Quick start](#1-quick-start): build and enter the container, enable the GPU, run everything with one command, or phase by phase
- [2. The problem: state, action, transition, reward](#2-the-problem-state-action-transition-reward): the eleven state components and eight states, the three actions, the dynamics, the reward
- [3. The ten controllers](#3-the-ten-controllers): how to run each part and its settings, then what each controller does, its equations, settings and commands
  - [How to run each part](#how-to-run-each-part): the commands, in five batches, and every run setting
  - [`decqn`](#decqn-decqn): DecQN
  - [`growing_q`](#growing_q-growing-q-networks): Growing Q-Networks
  - [`ppo_md`](#ppo_md-ppo-with-multidiscrete-actions): PPO with MultiDiscrete actions
  - [`ppo_box`](#ppo_box-ppo-with-continuous-actions): PPO with continuous actions
  - [`sac`](#sac-soft-actor-critic): Soft Actor-Critic
  - [`td3`](#td3-twin-delayed-ddpg): Twin Delayed DDPG
  - [`tqc`](#tqc-truncated-quantile-critics): Truncated Quantile Critics
  - [`crossq`](#crossq-crossq): CrossQ
  - [`tdmpc2`](#tdmpc2-td-mpc2): TD-MPC2
  - [`sac_pidlag`](#sac_pidlag-sac-with-pid-lagrangian-cost-budgets): SAC with PID-Lagrangian cost budgets
- [4. Reward profiles and cost budgets](#4-reward-profiles-and-cost-budgets): the reward terms, the twelve profiles, coefficient calibration
- [5. Metrics, HKS and statistics](#5-metrics-hks-and-statistics): how movements are scored and compared
- [6. Tasks and evaluation](#6-tasks-and-evaluation): splits, backends, checkpoint selection
- [7. Training programme](#7-training-programme): the action x state grid, phases, seeds, run counts
- [8. Checkpoints, videos and results](#8-checkpoints-videos-and-results): raw videos, verification, results layout
- [9. Tests](#9-tests): what the automated tests check
- [10. Repository layout](#10-repository-layout): where things are
- [11. The original experiment](#11-the-original-experiment): the preserved single-experiment code
- [12. Running on the DGX Sparks (Slurm)](#12-running-on-the-dgx-sparks-slurm): the ARM64 image, the two-Spark pipeline and the progress bar

---

## 1. Quick start

**Set up (host, once).** `git submodule update --init` (the simulator, pinned at `c7be874`) and `docker login` with an account that can pull the
private image `aung9htet/kangaroo-libero:0.1.0-ubuntu-22.04`. Then:

```bash
docker info
docker compose build rl
docker compose run --rm rl bash        # lands in /workspace/Kangaroo-RL-Control as your user; nothing starts by itself
```

**Check the environment (container).**

```bash
python -m natural_rl verify-env --suite train_v1        # libraries, simulator, manifests, scene, rendering, FFV1, disk
python -m natural_rl catalog models
python -m natural_rl catalog rewards
python -m natural_rl catalog suites
```

`results/` and `data/` are bind-mounted from the host and survive the container. Copy `.env.example` to `.env` to change the host paths
(`RESULTS_DIR`), the thread count or the user ids.

**GPU.** With the NVIDIA Container Toolkit's CDI spec on the host (`/var/run/cdi/nvidia.yaml` or `/etc/cdi/nvidia.yaml`; no Docker runtime
or daemon restart is needed), uncomment `COMPOSE_FILE=compose.yaml:compose.gpu.yaml` in `.env`. Inside the container
`torch.cuda.is_available()` is then true, `train ... --device cuda` runs any controller on CUDA, and `MUJOCO_GL=egl` renders raw video
frames on the GPU (about 1 ms per frame instead of about 200 ms on CPU). EGL and software (osmesa) rendering give slightly different
pixels and the backend is recorded in every video manifest, so use one rendering backend for a whole benchmark. TD-MPC2 uses CUDA by default
when it is available; the other controllers default to CPU.

**Image** `kangaroo-natural-rl:0.3.0`: Python 3.10, torch 2.7.1 (cu126), stable-baselines3 and sb3-contrib 2.7.1, tensordict 0.8.3, PyAV 13.1.0
(FFV1), ffmpeg, on top of the simulator image (numpy 1.26.4, gym 0.25.2, mujoco 3.4.0).

### Running the whole benchmark

One command runs the whole programme: every phase, with the calibration locks and the P3/P4 selections between them taken from real results.
It starts as many runs at once as the machine allows (one process per run, `CPU cores / threads` of them, bounded by memory; 2 threads per run
and the GPU for the learning when a CUDA device is visible, 4 threads per run on CPU), shows one overall display, and can be stopped and
started again at any time: finished work is skipped and a run that was interrupted is replaced.

```bash
python -m natural_rl run-all                                   # everything: 3 actions x 8 states x all phases
python -m natural_rl run-all --dry-run                         # the plan, the totals and the resources it would use; runs nothing
python -m natural_rl run-all --actions torque --states v3      # one cell of the grid
python -m natural_rl run-all --phases P0,P1,P2                 # some phases (P0 first: it fixes the reward coefficients)
python -m natural_rl run-all --workers 8 --device cuda --threads 2
python -m natural_rl run-all --status                          # what is finished and what is left; starts nothing
```

The overall line shows the stage, finished runs, finished transitions of the whole programme, how many runs are running, the aggregate speed
and the time left (an estimate: cells differ in speed; P3 and P4 are counted for one cell until the best cell is chosen from P2). The rows
below it show the runs in progress (controller, reward, `action/state`, seed, share trained, current step: `train`, `video verify`, `eval`).
The output of each run goes to `results/logs/<run-id>.log`. When a run fails, `run-all` finishes the stage, names the run and its log, and stops; run the
same command again to continue.

**Continuing after an interruption.** Stop it with Ctrl-C, `docker stop`, a crash or a power cut, then continue with the lines below. Paste them one at a time.

First, see what is finished and what is left (this starts nothing):

```bash
python -m natural_rl run-all --status
```

It prints, for P0, P1 and P2, how many runs are done, need only their evaluation, are unfinished or have not started, and ends with the exact command to continue with.

Then continue by pasting the same command you started with:

```bash
python -m natural_rl run-all
```

If you started it with options, paste it with the same options so it continues the same part of the programme, for example:

```bash
python -m natural_rl run-all --actions torque --states v3
```

Nothing else is needed. On Ctrl-C or `docker stop`, `run-all` first terminates the running jobs and prints this continue command. When it starts again:

* a run whose videos are verified and whose evaluations are present is skipped;
* a run that finished training but is missing its verification or evaluation gets only the missing lines;
* a run that was interrupted or failed starts again from step 0: training cannot resume from a checkpoint, so its partial folder is replaced (finished runs are never touched);
* the reward and HKS-scaler locks that are already written to `configs/` are kept, and the P3/P4 selection is recomputed from the same validation tables, so it gives the same cells and controllers.

The progress shown counts finished runs and the trained share of every running one, so it continues from where the results stand.

The stages, in order:

| Step | What it is | What it produces |
| --- | --- | --- |
| P0 | The original experiment re-trained: DecQN with the original reward `r0_original`, joint-velocity actions and the original 90-value state (`--contract hist_v1`), 3 seeds x 1M transitions. | Checkpoints and verified raw videos. Its trained policies are the pilot data that fixes the reward coefficients. |
| Lock rewards | Rolls the P0 policies out on the training tasks and fixes the coefficients of the calibrated reward terms (section 4). | `configs/rewards/reward_v1.yaml`, status `locked`. Profiles with calibrated terms (every profile except r0 and r1) refuse to train before. |
| P1 | Pilot of the benchmark over **every action x state cell**: ten controllers x {`r1_hold`, `r7_balanced`} x 3 seeds x 200k transitions. | A checkpoint and video per 50k, and one validation evaluation of the final checkpoint of each run. It shows that every controller trains, saves, reloads and renders under every action and state. |
| Lock scalers | Fits the HKS scalers once, pooled over all cells and backends (section 5), from the successful validation episodes of P1. | `configs/metrics/hks_scalers_v1.yaml`, status `locked`. If P1 has too few successes, the registered escalation fits them from the P2 validation data instead. |
| P2 | The main comparison over every cell: ten controllers x {`r1_hold`, `r7_balanced`} x 5 seeds x 1M transitions. | Per run: 21 checkpoints with videos, validation on every checkpoint, the test evaluation of the validation-selected checkpoint, and the summary table. |
| Select | Reads P2 validation tables only: the best action x state cell (highest mean validation R-HKS of its `r7_balanced` runs) and, within it, the three best non-DecQN controllers. | The cell and controllers P3 and P4 use (`catalog programme --phase P3 --select` prints the same choice). |
| P3 | The reward-term study in the chosen cell: DecQN and the three selected controllers x the ten remaining profiles x 5 seeds x 1M transitions. | Same outputs as P2, for every profile. |
| P4 | The cost-budget study in the chosen cell: SAC-PID-Lagrangian x `r7_balanced` x {moderate, strict} budget x fresh seeds 5-9 x 1M transitions. | Same outputs as P2, plus the budget and multiplier traces. |

**Phase by phase.** `run-all` is the same programme as these lines, which you can also run in any order or in several shells:

```bash
runall() { grep '^python' | while IFS= read -r l; do
  echo; echo ">>> $l"; eval "$l" < /dev/null || { echo "STOPPED at: $l"; break; }
done; }
```

`runall` executes the lines one after another and stops at the first failing line. Every list ends in `--todo`, which leaves out what `results/` already holds
(runs whose manifest says `complete` get no `train` line; runs whose videos and evaluations are all present are left out completely). To redo a run, run its
`train` line yourself: starting `train` with an existing run ID replaces that run completely (its folder is deleted and it starts again from step 0; training does not
resume from a checkpoint). There is no clean-git-tree check; the manifest records both commits and whether the tree had uncommitted changes.
`catalog programme` accepts `--actions` and `--states` (the cells of P1 and P2) and `--cells ACTIONS:STATES` (the cells of P3 and P4), e.g. `--actions torque --states v3`
or `--cells torque:v3`.

**P0.** Train and verify the three historical runs.

```bash
python -m natural_rl catalog programme --phase P0 --with-eval --todo | runall
```

**Lock rewards.** Then commit the file so the locked coefficients are versioned.

```bash
python -m natural_rl calibrate rewards --pilot-runs decqn-r0hist-s00,decqn-r0hist-s01,decqn-r0hist-s02
git add configs/rewards/reward_v1.yaml && git commit -m "Lock reward_v1 from the P0 pilot"
```

**P1.** 1,440 runs (60 per cell). Each line set: train, verify every video, evaluate the final checkpoint on the validation tasks.

```bash
python -m natural_rl catalog programme --phase P1 --with-eval --todo | runall
```

**Lock scalers.** Uses the final validation evaluation of every P1 run (all cells, both backends pooled), then commit the file.

```bash
python -m natural_rl calibrate scalers --runs "$(python -m natural_rl catalog programme --phase P1 --commands | grep -o -- '--run-id [^ ]*' | cut -d' ' -f2 | paste -sd,)"
git add configs/metrics/hks_scalers_v1.yaml && git commit -m "Lock HKS scalers"
```

The command needs at least 200 successful, quality-valid validation episodes and a spread in every metric; otherwise it stops with a blocking
error and writes nothing. In that case run P2 first and fit the scalers from the P2 runs instead (the same command with
`--phase P2` in the run list); the registered escalation uses validation data only, never test data.

**P2.** 2,400 runs (100 per cell). Each line set: train, verify, validate every checkpoint, test the validation-selected checkpoint, summarise.

```bash
python -m natural_rl catalog programme --phase P2 --with-eval --todo | runall
```

**P3.** 200 runs. `--select` first reads the P2 validation tables and prints the best action x state cell and, in it, the three controllers; the commands
are printed for DecQN plus those three, in that cell.

```bash
python -m natural_rl catalog programme --phase P3 --select --with-eval --todo | runall
```

**P4.** 10 runs, in the selected cell (`--select` picks it from the P2 tables; `--cells ACTIONS:STATES` names it without selection).

```bash
python -m natural_rl catalog programme --phase P4 --select --with-eval --todo | runall
```

**Scores after the scalers are locked.** Evaluations store the raw descriptors of every episode, so HKS and R-HKS can be computed at any time.
`analyse` scores every stored evaluation of a run with the locked scalers; run it again for runs evaluated before the lock:

```bash
python -m natural_rl catalog programme --phase P2 --commands | grep -o -- '--run-id [^ ]*' | cut -d' ' -f2 | while read -r id; do python -m natural_rl analyse --run-id "$id"; done
```

Open several container shells (`docker compose run --rm rl bash`) to run different phases or cells at the same time, but never the same run ID twice.
The per-controller commands (a single run, `play`, `video record`, `export`) are in section 3; every controller accepts `--action velocity|position|torque` and
`--obs v1|v2|v3`.

---

## 2. The problem: state, action, transition, reward

The controlled chain has 7 joints (5 arm joints and the 2 wrist joints of the RH8D hand). The *reach point* `p(q) ∈ ℝ³` is the hand's
manipulation link in the robot base frame (`x` forward, `y` left, `z` up, metres). A goal `g ∈ ℝ³` is a point above the table.

**State.** The joint angles and the goal, `s_t=(q_t, g)` with `q_t ∈ ℝ⁷` (rad). Each joint is normalised to `[-1,1]` over its range and
every normalised value `x` is encoded with a sinusoidal positional encoding of `L=4` frequencies:

$$
\tilde q_j=2\,\frac{q_j-q_j^{\min}}{q_j^{\max}-q_j^{\min}}-1
$$

$$
\phi(x)=\big[\,x,\ \sin(2^{0}\pi x),\cos(2^{0}\pi x),\ \dots,\ \sin(2^{3}\pi x),\cos(2^{3}\pi x)\,\big]
$$

The goal is part of the state so a single policy serves every goal. Beyond the joint angles and the goal, the state can carry more information: what the arm just did, where
the hand is, how fast it moves, how close the arm is to its limits and to the table. Each piece of information is a named **component**; a **state** is a set of components,
concatenated in the fixed order of the table below, so every controller sees an identical, documented feature map. Because the benchmark asks which state helps produce natural movement,
the states differ only by the information they add, and `--obs` selects one.

| Component | What it is | Size |
| --- | --- | --- |
| `q` | joint angles, normalised to `[-1,1]` over each joint range and encoded with `φ` (`L=4`) | `7·9=63` |
| `goal` | goal position, normalised over the goal region and encoded with `φ` | `3·9=27` |
| `prev_cmd` | the previous command `u_(t-1)` (what the policy did one decision ago) | 7 |
| `dwell` | `min(c_t,5)/5` and `h_t`: how long the hand has stayed within 2 cm of the goal, and whether a 5-step hold has been completed | 2 |
| `qd` | the executed joint velocity `q̇_t/v_max`, clipped to `[-1,1]` | 7 |
| `ee` | the hand position `p(q)` from forward kinematics (robot base frame), normalised over the goal region and encoded with `φ` | `3·9=27` |
| `goal_rel` | the goal error `(g-p)/0.5 m`, clipped to `[-1,1]`, and the distance `min(‖g-p‖/0.5 m, 1)` | `3+1=4` |
| `ee_vel` | the hand velocity `(p_t-p_(t-1))/Δt`, divided by 1 m/s and clipped to `[-1,1]` | 3 |
| `margin` | per joint, the normalised distance to the nearest joint limit, `0` at a limit and `1` at mid-range | 7 |
| `table` | the hand height above the table top and its horizontal distance from the table's near edge, each divided by 0.5 m and clipped to `[-1,1]` | 2 |
| `time` | the fraction of the episode that has elapsed, `t/H` | 1 |

With `p_t=p(q_t)` the hand position (forward kinematics, metres), the derived components are

$$
e_t=g-p_t
$$

$$
\tilde e_t=\mathrm{clip}\!\Big(\frac{e_t}{0.5\,\mathrm m},-1,1\Big),\qquad \tilde d_t=\min\!\Big(\frac{\lVert e_t\rVert}{0.5\,\mathrm m},1\Big)
$$

$$
\tilde{\dot p}_t=\mathrm{clip}\!\Big(\frac{p_t-p_{t-1}}{\Delta t\cdot 1\,\mathrm m/s},-1,1\Big)
$$

$$
m_j=\frac{2\,\min\!\big(q_j-q_j^{\min},\ q_j^{\max}-q_j\big)}{q_j^{\max}-q_j^{\min}}
$$

The registered states are named sets of these components:

| State | Components | Size | What it adds |
| --- | --- | --- | --- |
| `v1` | `q`, `goal` | 90 | the original input: joint angles and goal only |
| `v2` | `v1` + `prev_cmd`, `dwell` | 99 | the policy's own history and the hold state; makes the hold-aware rewards Markov |
| `v3` | `v2` + `qd` | 106 | the joint velocity (the dynamic state of a torque-controlled arm) |
| `v4` | `v2` + `ee`, `goal_rel` | 130 | task space: where the hand is and how far it is from the goal, without computing forward kinematics itself |
| `v5` | `v4` + `ee_vel` | 133 | the hand's own velocity, the quantity the movement metrics score |
| `v6` | `v2` + `margin`, `table` | 108 | the safety geometry: joint-limit margins and the table |
| `v7` | `goal`, `prev_cmd`, `dwell`, `ee`, `goal_rel`, `ee_vel`, `table` | 72 | hand-centric: no joint angles at all, only what the hand does in space |
| `v8` | every component | 150 | everything, as an upper bound on the information |

`--obs` also accepts any comma-separated list of components (`--obs q,goal,ee,ee_vel`), in which case the manifest records the list. The matched default is `v2` for velocity and position
actions and `v3` for torque; the benchmark runs every action with every state. The components that depend on the goal (`goal` and `goal_rel`) are recomputed when a replayed
sample gets a relabelled goal, and `dwell` is recomputed from step 0; every other component follows the stored trajectory, and a test checks that the replay rebuilds every state exactly as the
environment showed it. `v7` is partially observed: the joint configuration of a 7-joint arm is not determined by the hand position.

**Action.** Every controller outputs a command `u ∈ [-1,1]⁷` for one decision of `Δt=0.05` s (20 Hz):

* *Discrete controllers* choose one bin per joint: `u_j=b_(k_j)` with `b=(-1,-0.3,-0.1,0,0.1,0.3,1)` and `k_j ∈ {0,…,6}`,
  i.e. `7⁷=823,543` joint actions.
* *Continuous controllers* output `a ∈ [-1,1]⁷` with `u=clip(a,-1,1)`.

What `u` means is the **action mode** (`--action`):

| Mode | The command is | Transition |
| --- | --- | --- |
| `velocity` | joint velocity `v=v_max·u`, `v_max=1` rad/s | kinematic: `q̂=clip(q+Δt·v)` |
| `position` | joint position step: target `q+u·v_max·Δt` | the same kinematic transition as velocity (an exact equivalence; the mode is registered separately so the equivalence is measured, and its physics evaluation issues pure position targets without velocity feed-forward) |
| `torque` | joint torque `τ=τ_max⊙u` (`τ_max` from `configs/env/dynamics_v1.yaml`) | arm dynamics, below |

Everywhere in the code the command is carried as `v_cmd=u·v_max`, so the effort and command-change reward terms, the previous-command input
and the unsafe-event cost are the same formulas for every mode (for torque they act on the normalised torque `u` and its change).

**Transition (velocity and position).** The commanded step is integrated for one decision and clipped to the joint limits; if the arm at the clipped pose
penetrates the table or the robot, the whole move is rejected:

$$
\hat q=\mathrm{clip}\!\big(q+\Delta t\,v,\ q^{\min},q^{\max}\big)
$$

$$
q'=\begin{cases}q & \text{if the arm at }\hat q\text{ collides (move rejected, collision flag }c=1)\\ \hat q & \text{otherwise.}\end{cases}
$$

**Transition (torque).** The arm model of the scene (joint inertia, damping and armature from the MuJoCo model) is integrated for ten steps of 5 ms per decision,
with every joint outside the controlled arm locked. Gravity and Coriolis forces are compensated, so `u=0` holds the pose, and each joint speed is limited to `v_max` like
in the other modes:

$$
M(q)\,\ddot q=\tau_{\max}\odot u-D\,\dot q
$$

$$
\lvert\dot q_j\rvert\le v_{\max}
$$

After the decision the same two safety rules as above are applied to the executed result: joints beyond their limits are clipped and stopped (a proposed limit
violation), and a whole-arm collision rejects the move, leaving the arm at the previous pose with zero velocity (a proposed collision). The scene's arm joints are
heavily damped, so the distal joints respond to torque almost like a velocity command with a short lag (`q̇≈τ/D`), and the two proximal joints show real inertia. The torque
limits are `τ_max=(12, 12, 6, 6, 5.5, 0.25, 0.25)` N·m, the torque that reaches the speed limit in 0.1 s against the joint damping (the scene's motors allow 43 N·m).

The goal never changes within an episode. Collisions are tested with MuJoCo on a filtered copy of the compiled model (right arm against the
table and the rest of the robot). Joint clipping and collision rejection are always active, under every reward profile and action mode.

**Episode and objective.** An episode has `H=100` decisions (5 s). There are no terminal states: the time limit is a *truncation*, so
bootstrapping continues from the true final state. All controllers maximise the discounted return with `γ=0.95`:

$$
J(\pi)=\mathbb E_\pi\Big[\sum_{t=0}^{H-1}\gamma^{t}\,r_t\Big]
$$

**Hindsight goal relabelling** (used by every off-policy controller). An episode that misses its goal still shows how to reach every point the
hand passed through. With probability 0.8 a sampled transition gets the goal replaced by a position reached later in the same episode:

$$
\tilde g=p(q_{t'})
$$

$$
t'\sim\mathcal U\lbrace t,\dots,H-1\rbrace
$$

$$
\tilde r_i=r(\,\cdot\,;\tilde g\,)\ \text{recomputed}
$$

This is valid because the transition `(q',c)` does not depend on the goal; only the reward does, so rewards are never stored. Because the
rewards and observations contain *history*, relabelling recomputes them from step 0 of the episode for the new goal:

$$
d_t=\lVert p_{t}-\tilde g\rVert
$$

$$
c_t=(c_{t-1}+1)\,\mathbb 1[d_t\le 0.02\,\mathrm m]
$$

$$
h_t=\mathbb 1\big[\max_{s\le t}c_s\ge 5\big]
$$

and the `n`-step return used by the value-based methods is `R_t(m) = Σ_(i=0..m-1) γ^i r̃_(t+i)` with `m=min(n,H-t)`, bootstrapping
with `γ^m`.

---

## 3. The ten controllers

All ten act on the problem above: they observe the chosen state (`v1` to `v8`, section 2), output a normalised command `u` for seven joints (the *Action* column: 7 bins per joint or a
continuous box) that the chosen action mode turns into a joint velocity, a joint position step or a joint torque, and are trained and evaluated on identical task manifests, actuator
limits, safety rules and evaluation code. The controller itself is unchanged by the action mode and state; only the input size of its networks follows the state (72 to 150 values).

| ID | Family | Action | Learns | Exploration | Goal relabelling |
| --- | --- | --- | --- | --- | --- |
| `decqn` | value-based, factored | 7 bins x 7 joints | decoupled Q | per-joint `ε`-greedy | yes |
| `growing_q` | value-based, factored | 2→3→5→7 bins x 7 joints | decoupled Q with masking | per-joint `ε`-greedy on active bins | yes |
| `ppo_md` | on-policy actor-critic | 7 bins x 7 joints | categorical policy | policy sampling | no |
| `ppo_box` | on-policy actor-critic | Box `[-1,1]⁷` | Gaussian policy | policy sampling | no |
| `sac` | off-policy actor-critic | Box | max-entropy policy | stochastic policy | yes |
| `td3` | off-policy actor-critic | Box | deterministic policy | Gaussian noise | yes |
| `tqc` | off-policy actor-critic | Box | distributional critics | stochastic policy | yes |
| `crossq` | off-policy actor-critic | Box | batch-normalised critics | stochastic policy | yes |
| `tdmpc2` | model-based | Box | latent world model + planning | planner noise | yes |
| `sac_pidlag` | constrained off-policy | Box | SAC + cost critic | stochastic policy | yes |

Contrasts that can be interpreted: DecQN vs Growing Q vs PPO-MD (discrete stacks); PPO-MD vs PPO-Box (action representation inside PPO,
identical settings otherwise); the continuous family PPO-Box / SAC / TD3 / TQC / CrossQ / TD-MPC2; SAC vs SAC-PID-Lagrangian (cost-budget
training). Controllers are policy *stacks*, not single-factor changes. Registered hyperparameters live in `configs/models/<id>.yaml`.
### How to run each part

Every controller is run with the same commands, in five batches (each controller's section below lists the batches for `r7_balanced`, seed 0; the run ID names the run everywhere). A run is
one controller, one reward profile, one seed, one action and one state. All commands are `python -m natural_rl <command>` inside the container.

| Part | Command | What it does |
| --- | --- | --- |
| 1. Train | `train --model M --reward R --steps N --seed S --run-id ID ...` | trains one run; saves a checkpoint at step 0, every `--checkpoint-every` transitions and at the end, and with `--record-every-checkpoint` a verified raw video of each; replaces an existing run with the same ID |
| 2. List checkpoints | `checkpoints list --run-id ID` | prints each saved step with its kind, SHA256 prefix, file integrity and video status |
| 3. Play | `play --run-id ID --checkpoint-step K --task-id T` | replays one exact saved checkpoint on one task, headless, and writes a trace of the movement |
| 4. Record a video | `video record --run-id ID --checkpoint-step K --suite checkpoint_video_v1` | renders the raw video of one checkpoint (never overwrites a verified one; it reports whether a re-render is pixel-identical) |
| 5. Fill missing videos | `video backfill --run-id ID --suite checkpoint_video_v1` | renders only the videos that are missing or failed |
| 6. Verify videos | `video verify --run-id ID --all-checkpoints` | decodes every video with two decoders and compares every frame with its manifest |
| 7. Evaluate | `eval --run-id ID --suite SUITE ...` | scores checkpoints on a task suite and writes the episode, action and trajectory tables; validation first (`val_v1 --all-checkpoints`), then the held-out test (`test_v1`) |
| 8. Summarise | `analyse --run-id ID` | one summary table per run, scored with the locked HKS scalers |
| 9. Plot | `plot --run-id ID --kind all` | the entry point for the per-run figures |
| 10. Export | `export --run-id ID --kind trajectories --suite SUITE` | the executed trajectories as CSV and parquet |
| 11. Locate | `results locate --run-id ID` | prints the run's status and where each of its files is |

The parts are run in five batches: batch 1 is part 1 (train), batch 2 parts 2 and 3 (inspect and replay), batch 3 parts 4 to 6 (videos), batch 4 part 7 (evaluate) and batch 5 parts 8 to 11 (summarise and export).
A batch can be run on its own as long as the earlier ones have produced what it needs (batch 4 needs batch 1; the test evaluation needs the validation evaluation before it).

**`train` settings** (`python -m natural_rl train --help` prints the same list):

| Setting | Default | Values and meaning |
| --- | --- | --- |
| `--model` | required | `decqn`, `growing_q`, `ppo_md`, `ppo_box`, `sac`, `td3`, `tqc`, `crossq`, `tdmpc2`, `sac_pidlag` |
| `--reward` | required | `r0_original`, `r1_hold`, `r2_time`, `r3_path`, `r4_speed`, `r5_smooth`, `r6_rom`, `r7_balanced`, `r8_drop_path`, `r9_drop_speed`, `r10_drop_smooth`, `r11_drop_rom` (section 4; profiles with calibrated terms need the locked register) |
| `--steps` | required | training length in single-environment transitions (a multiple of the controller's `n_envs`) |
| `--seed` | `0` | seeds numpy, torch and the environments |
| `--run-id` | required | names the run; the folder is `results/runs/<backend>/<model>/<reward>/<run-id>/` |
| `--action` | `velocity` | `velocity`, `position` or `torque` (section 2) |
| `--obs` | `v2` (`v3` for torque) | `v1` to `v8` or a comma-separated component list (section 2) |
| `--suite` | `train_v1` | the training task manifest |
| `--checkpoint-every` | `50000` | transitions between checkpoints (a multiple of `n_envs`); step 0 and the final step are always saved |
| `--record-every-checkpoint` | off | record and verify a raw video of every checkpoint |
| `--strict-video` | off | abort at the first failed video instead of marking the run `incomplete` at the end |
| `--cost-budget-id` | none | `budget_moderate_v1` or `budget_strict_v1`; required by `sac_pidlag` only |
| `--contract` | `bench_v1` | `hist_v1` re-runs the frozen original DecQN (P0: DecQN, `r0_original`, velocity, `v1`) |
| `--device` | `auto` | `auto`, `cpu` or `cuda` (`auto`: CUDA for TD-MPC2 when available, CPU otherwise) |
| `--threads` | `4` | CPU threads of the process |
| `--set` | none | overrides of the controller settings, `hyper.key=value` (the tables below) |
| `--smoke` | off | a disposable run under `results/smoke/`, never part of the benchmark |

**Settings of the other parts.**

| Command | Setting | Meaning |
| --- | --- | --- |
| `eval` | `--suite` | `train_v1` (500 tasks), `val_v1` (100), `test_v1` (200) or `ood_v1` (100) |
| `eval` | `--manifest` | for `test_v1` and `ood_v1`, the registered manifest path (`configs/eval/test_v1.json`); any other file is refused |
| `eval` | `--checkpoint-step K` / `--all-checkpoints` / `--final` | one exact saved step, every saved step, or the last one; with none of them `val_v1` uses the last step and `test_v1`/`ood_v1` use the validation-selected checkpoint |
| `eval` | `--backend` | `kinematic`, `dynamics` or `physics`; default: the backend the run was trained on (section 6) |
| `play` | `--checkpoint-step K --task-id T` | the exact saved step and a task ID such as `val_001` |
| `video record` | `--checkpoint-step K`, `--suite`, `--force` | which step; the video task definition; `--force` records anew even if a verified video exists |
| `video backfill`, `video verify` | `--all-checkpoints`, `--suite` | all saved steps; the video task definition |
| `export` | `--kind`, `--suite` | `trajectories`; the suite whose evaluated trajectories are exported |
| `calibrate rewards` | `--pilot-runs`, `--steps`, `--dry-run` | the P0 runs; `final` or `all` checkpoints; print without writing |
| `calibrate scalers` | `--runs`, `--backend` | the runs with a `val_v1` evaluation; force one backend (default: each run's own, all pooled) |

Examples:

```bash
python -m natural_rl train --model sac --reward r7_balanced --suite train_v1 --steps 1000000 --seed 0 --run-id sac-r7-tor-o5-s00 --action torque --obs v5 --checkpoint-every 50000 --record-every-checkpoint --device cuda
python -m natural_rl train --model ppo_box --reward r1_hold --suite train_v1 --steps 4000 --seed 0 --run-id try-ppo --checkpoint-every 2000 --smoke --set hyper.n_steps=100 hyper.batch_size=160
python -m natural_rl eval --run-id sac-r7-tor-o5-s00 --suite val_v1 --all-checkpoints
python -m natural_rl eval --run-id sac-r7-tor-o5-s00 --suite test_v1 --manifest configs/eval/test_v1.json
```

The first line is a full run in the torque / `v5` cell on the GPU; the second a few-second smoke test with a changed setting; the last two validate every checkpoint and then test the
validation-selected one. Each controller block below lists that controller's own settings, and the commands in the order above.

### decqn: DecQN

*Seyde et al., Solving Continuous Control via Q-learning, ICLR 2023.* https://arxiv.org/abs/2210.12566

**What it does.** A value-based method for many action dimensions. A full joint action has `7⁷` combinations, far too many to evaluate, so the
Q-function is *decoupled*: the network outputs one utility per joint and per bin and the value of a joint action is their mean. The best joint
action is then simply the best bin of every joint, found with `7×7` evaluations and exactly equal to the arg-max over all `7⁷` actions.

**Network.** state (99 for `obs_v2`) → 256 → 256 (LayerNorm, ReLU) → 49 utilities `U_j(s,k)`.

$$
Q_\theta(s,a)=\frac1J\sum_{j=1}^{J}U_j(s,a_j;\theta)
$$

$$
a^*_j=\arg\max_k U_j(s,k;\theta)
$$

**Learning.** Double-Q with `n=3`-step returns from the relabelled replay, Huber loss on the joint action actually taken, Polyak-averaged target
network (`τ=0.005`), gradient clipping at 10, Adam `3×10⁻⁴`, batch 256, targets clipped to the reward bounds divided by `1-γ`:

$$
y=\sum_{i=0}^{m-1}\gamma^{i}\tilde r_{t+i}+\gamma^{m}\,Q_{\theta^-}\!\big(s_{t+m},\hat a\big)
$$

$$
\hat a_j=\arg\max_k U_j(s_{t+m},k;\theta)
$$

$$
\mathcal L(\theta)=\mathrm{Huber}\!\big(y-Q_\theta(s_t,a_t)\big)
$$

**Exploration.** Per joint, `ε` falls linearly from 1 to 0.02 over the first 15 % of training; an exploring joint repeats one random
bin for 1-10 decisions so random motion travels instead of jittering. 16 environments run in parallel, with 4 gradient updates per 16 transitions
and 10,000 random warm-up transitions.

**Settings.** Registered in `configs/models/decqn.yaml`; the values of one run are saved in its `config_resolved.yaml`. Change any value for a single run with `--set hyper.<key>=<value>` (several at once are allowed, for example `--set hyper.lr=1e-4 hyper.batch=512`). The run settings that every controller shares (`--action`, `--obs`, `--reward`, `--steps`, `--seed`, `--device` ...) are listed in the table at the start of this section.

| Setting | Registered value | Meaning |
| --- | --- | --- |
| `n_envs` | `16` | environments stepped in parallel; `--steps` and `--checkpoint-every` must be multiples of it |
| `hyper.hidden` | `256` | units per hidden layer |
| `hyper.depth` | `2` | hidden layers |
| `hyper.gamma` | `0.95` | discount factor `γ` |
| `hyper.lr` | `0.0003` | learning rate (Adam) |
| `hyper.batch` | `256` | minibatch size per update |
| `hyper.tau` | `0.005` | Polyak coefficient of the target network |
| `hyper.grad_clip` | `10.0` | gradient-norm clip |
| `hyper.n_step` | `3` | return length `n` (relabelled replay) |
| `hyper.replay` | `1000000` | replay capacity in transitions |
| `hyper.warmup` | `10000` | random-action transitions before learning starts |
| `hyper.updates_per_step` | `4` | gradient updates per vector step (one transition from each environment) |
| `hyper.her_ratio` | `0.8` | share of replayed samples whose goal is relabelled (hindsight) |
| `hyper.eps_start` | `1.0` | initial exploration rate `ε` |
| `hyper.eps_end` | `0.02` | final exploration rate |
| `hyper.eps_fraction` | `0.15` | share of training over which `ε` falls linearly |
| `hyper.max_repeat` | `10` | an exploring joint repeats one random bin for 1 to this many decisions |

* Discrete action: seven joints x seven bins, so no continuous-action option applies; it runs on CPU unless `--device cuda` is given.
* P0 (`--contract hist_v1`) runs the frozen original agent instead and ignores `--set`.

**Batch 1: train.** The long step. It trains for 1,000,000 transitions on the 500 training tasks, saves a checkpoint at step 0, every 50,000 transitions and at the end (21 in all), and records and verifies a lossless raw video of each (`--record-every-checkpoint`), with a progress bar showing the model and the run. Everything goes to `results/runs/<backend>/<model>/<reward>/<run_id>/`. Running it again with the same `--run-id` replaces the run.

```bash
python -m natural_rl train --model decqn --reward r7_balanced --suite train_v1 --steps 1000000 --seed 0 --run-id decqn-r7-s00 --checkpoint-every 50000 --record-every-checkpoint
```

**Batch 2: look at what was saved.** `checkpoints list` prints every saved step with its file hash and video status. `play` replays one exact saved checkpoint (step 150000) on the validation task `val_001`, headless, and writes the movement as a trace under `play/`.

```bash
python -m natural_rl checkpoints list --run-id decqn-r7-s00
python -m natural_rl play --run-id decqn-r7-s00 --checkpoint-step 150000 --task-id val_001
```

**Batch 3: videos.** `video record` renders the raw video of one checkpoint again (a verified video is never overwritten; it reports whether the new render is pixel-identical). `video backfill` renders only the videos that are missing or failed. `video verify` decodes every video with two independent decoders and compares each frame with its manifest.

```bash
python -m natural_rl video record --run-id decqn-r7-s00 --checkpoint-step 150000 --suite checkpoint_video_v1
python -m natural_rl video backfill --run-id decqn-r7-s00 --suite checkpoint_video_v1
python -m natural_rl video verify --run-id decqn-r7-s00 --all-checkpoints
```

**Batch 4: evaluate.** The first line scores every checkpoint on the 100 validation tasks; it is what picks the checkpoint to test. The second scores that validation-selected checkpoint on the 200 held-out test tasks (the `--manifest` argument pins the registered task file; any other file is refused). Both write the episode, action and trajectory tables under `eval/`.

```bash
python -m natural_rl eval --run-id decqn-r7-s00 --suite val_v1 --all-checkpoints
python -m natural_rl eval --run-id decqn-r7-s00 --suite test_v1 --manifest configs/eval/test_v1.json
```

**Batch 5: summarise and export.** `analyse` writes one summary table for the run (scored with the locked HKS scalers), `plot` is the entry point for the per-run figures, `export` writes the executed test-task trajectories as CSV and parquet, and `results locate` prints the run's status and where each of its files is.

```bash
python -m natural_rl analyse --run-id decqn-r7-s00
python -m natural_rl plot --run-id decqn-r7-s00 --kind all
python -m natural_rl export --run-id decqn-r7-s00 --kind trajectories --suite test_v1
python -m natural_rl results locate --run-id decqn-r7-s00
```

### growing_q: Growing Q-Networks

*Seyde et al., Growing Q-networks: Solving continuous control tasks with adaptive control resolution, L4DC 2024.* https://proceedings.mlr.press/v242/seyde24a.html

**What it does.** DecQN that starts with a very coarse action set and refines it during training. Coarse control (bang-bang) explores quickly and
is easy to learn; fine control is needed later for precision. The network always has all seven bins, but only an *active subset*
`A_g ⊆ {0,…,6}` may be chosen: masking is applied to the greedy action, to the epsilon-greedy exploration and to the maximum
in the Double-Q target:

$$
\hat a_j=\arg\max_{k\in\mathcal A_g}U_j(s_{t+m},k;\theta)
$$

$$
y=\sum_{i<m}\gamma^{i}\tilde r_{t+i}+\gamma^{m}\,\frac1J\sum_{j}U_j(s_{t+m},\hat a_j;\theta^-)
$$

**Schedule.** The paper's linear schedule on nested subsets of the seven historical levels, growing at 20 %, 40 % and 60 % of training:

| Stage | Active velocities (rad/s) | Bins |
| --- | --- | --- |
| 0 | `{-1,+1}` | 2 |
| 1 | `{-1,0,+1}` | 3 |
| 2 | `{-1,-0.3,0,0.3,1}` | 5 |
| 3 | all seven | 7 |

The active set is saved inside every checkpoint (a replay acts at the resolution the checkpoint had) and the active resolution is logged at
every decision. All other settings equal DecQN.

**Settings.** Registered in `configs/models/growing_q.yaml`; the values of one run are saved in its `config_resolved.yaml`. Change any value for a single run with `--set hyper.<key>=<value>` (several at once are allowed, for example `--set hyper.lr=1e-4 hyper.batch=512`). The run settings that every controller shares (`--action`, `--obs`, `--reward`, `--steps`, `--seed`, `--device` ...) are listed in the table at the start of this section.

| Setting | Registered value | Meaning |
| --- | --- | --- |
| `n_envs` | `16` | environments stepped in parallel; `--steps` and `--checkpoint-every` must be multiples of it |
| `hyper.hidden` | `256` | units per hidden layer |
| `hyper.depth` | `2` | hidden layers |
| `hyper.gamma` | `0.95` | discount factor `γ` |
| `hyper.lr` | `0.0003` | learning rate (Adam) |
| `hyper.batch` | `256` | minibatch size per update |
| `hyper.tau` | `0.005` | Polyak coefficient of the target network |
| `hyper.grad_clip` | `10.0` | gradient-norm clip |
| `hyper.n_step` | `3` | return length `n` (relabelled replay) |
| `hyper.replay` | `1000000` | replay capacity in transitions |
| `hyper.warmup` | `10000` | random-action transitions before learning starts |
| `hyper.updates_per_step` | `4` | gradient updates per vector step (one transition from each environment) |
| `hyper.her_ratio` | `0.8` | share of replayed samples whose goal is relabelled (hindsight) |
| `hyper.eps_start` | `1.0` | initial exploration rate `ε` |
| `hyper.eps_end` | `0.02` | final exploration rate |
| `hyper.eps_fraction` | `0.15` | share of training over which `ε` falls linearly |
| `hyper.max_repeat` | `10` | an exploring joint repeats one random bin for 1 to this many decisions |
| `hyper.growth_schedule` | `linear` | how the bin set grows (linear: fixed fractions of training) |
| `hyper.growth_fractions` | `[0.2, 0.4, 0.6]` | training shares at which the active bins grow 2→3, 3→5, 5→7 |

* The active bins per stage are `{±1}`-based nested subsets of the seven levels (2, 3, 5, then all 7); `growth_fractions` changes when they grow.

**Batch 1: train.** The long step. It trains for 1,000,000 transitions on the 500 training tasks, saves a checkpoint at step 0, every 50,000 transitions and at the end (21 in all), and records and verifies a lossless raw video of each (`--record-every-checkpoint`), with a progress bar showing the model and the run. Everything goes to `results/runs/<backend>/<model>/<reward>/<run_id>/`. Running it again with the same `--run-id` replaces the run.

```bash
python -m natural_rl train --model growing_q --reward r7_balanced --suite train_v1 --steps 1000000 --seed 0 --run-id growing_q-r7-s00 --checkpoint-every 50000 --record-every-checkpoint
```

**Batch 2: look at what was saved.** `checkpoints list` prints every saved step with its file hash and video status. `play` replays one exact saved checkpoint (step 150000) on the validation task `val_001`, headless, and writes the movement as a trace under `play/`.

```bash
python -m natural_rl checkpoints list --run-id growing_q-r7-s00
python -m natural_rl play --run-id growing_q-r7-s00 --checkpoint-step 150000 --task-id val_001
```

**Batch 3: videos.** `video record` renders the raw video of one checkpoint again (a verified video is never overwritten; it reports whether the new render is pixel-identical). `video backfill` renders only the videos that are missing or failed. `video verify` decodes every video with two independent decoders and compares each frame with its manifest.

```bash
python -m natural_rl video record --run-id growing_q-r7-s00 --checkpoint-step 150000 --suite checkpoint_video_v1
python -m natural_rl video backfill --run-id growing_q-r7-s00 --suite checkpoint_video_v1
python -m natural_rl video verify --run-id growing_q-r7-s00 --all-checkpoints
```

**Batch 4: evaluate.** The first line scores every checkpoint on the 100 validation tasks; it is what picks the checkpoint to test. The second scores that validation-selected checkpoint on the 200 held-out test tasks (the `--manifest` argument pins the registered task file; any other file is refused). Both write the episode, action and trajectory tables under `eval/`.

```bash
python -m natural_rl eval --run-id growing_q-r7-s00 --suite val_v1 --all-checkpoints
python -m natural_rl eval --run-id growing_q-r7-s00 --suite test_v1 --manifest configs/eval/test_v1.json
```

**Batch 5: summarise and export.** `analyse` writes one summary table for the run (scored with the locked HKS scalers), `plot` is the entry point for the per-run figures, `export` writes the executed test-task trajectories as CSV and parquet, and `results locate` prints the run's status and where each of its files is.

```bash
python -m natural_rl analyse --run-id growing_q-r7-s00
python -m natural_rl plot --run-id growing_q-r7-s00 --kind all
python -m natural_rl export --run-id growing_q-r7-s00 --kind trajectories --suite test_v1
python -m natural_rl results locate --run-id growing_q-r7-s00
```

### ppo_md: PPO with MultiDiscrete actions

*Schulman et al., Proximal Policy Optimization Algorithms, 2017.* https://arxiv.org/abs/1707.06347

**What it does.** An on-policy actor-critic. It collects fresh rollouts with the current policy and improves it with a clipped surrogate objective that
keeps each update close to the policy that gathered the data. No replay buffer, so no goal relabelling.

**Policy.** Seven independent categorical distributions over the 7 bins (one per joint), `π_θ(a∣ s)=Π_jCat(a_j;softmaxℓ_j(s))`,
and a separate value network `V_ψ(s)`; both are MLPs with two layers of 256.

$$
\mathcal L^{\text{CLIP}}=\mathbb E_t\Big[\min\big(\rho_t\hat A_t,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\,\hat A_t\big)\Big]
$$

$$
\rho_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}
$$

$$
\hat A_t=\sum_{l\ge0}(\gamma\lambda)^{l}\delta_{t+l}
$$

$$
\delta_t=r_t+\gamma V_\psi(s_{t+1})-V_\psi(s_t)
$$

$$
\mathcal L^{V}=\big(V_\psi(s_t)-\hat R_t\big)^2
$$

**Settings.** `ε=0.2`, `λ=0.95`, 16 environments x 256 steps = 4096 transitions per update, 10 epochs, minibatch 256, entropy
coefficient 0. At the 100-decision time limit the value of the final observation is bootstrapped (truncation, not termination).

**Settings.** Registered in `configs/models/ppo_md.yaml`; the values of one run are saved in its `config_resolved.yaml`. Change any value for a single run with `--set hyper.<key>=<value>` (several at once are allowed, for example `--set hyper.lr=1e-4 hyper.batch=512`). The run settings that every controller shares (`--action`, `--obs`, `--reward`, `--steps`, `--seed`, `--device` ...) are listed in the table at the start of this section.

| Setting | Registered value | Meaning |
| --- | --- | --- |
| `n_envs` | `16` | environments stepped in parallel; `--steps` and `--checkpoint-every` must be multiples of it |
| `hyper.gamma` | `0.95` | discount factor `γ` |
| `hyper.gae_lambda` | `0.95` | GAE `λ` |
| `hyper.lr` | `0.0003` | learning rate (Adam) |
| `hyper.n_steps` | `256` | rollout length per environment between updates |
| `hyper.batch_size` | `256` | minibatch size per update |
| `hyper.n_epochs` | `10` | optimisation epochs per rollout |
| `hyper.clip_range` | `0.2` | PPO clip range |
| `hyper.ent_coef` | `0.0` | entropy coefficient (`auto`: tuned to a target entropy) |
| `hyper.net_arch` | `{pi: [256, 256], vf: [256, 256]}` | hidden layer widths of the networks |

* One update uses `n_steps x n_envs = 4096` transitions; there is no replay and no goal relabelling.
* `ppo_md` and `ppo_box` share every value, so changing one should change both.

**Batch 1: train.** The long step. It trains for 1,000,000 transitions on the 500 training tasks, saves a checkpoint at step 0, every 50,000 transitions and at the end (21 in all), and records and verifies a lossless raw video of each (`--record-every-checkpoint`), with a progress bar showing the model and the run. Everything goes to `results/runs/<backend>/<model>/<reward>/<run_id>/`. Running it again with the same `--run-id` replaces the run.

```bash
python -m natural_rl train --model ppo_md --reward r7_balanced --suite train_v1 --steps 1000000 --seed 0 --run-id ppo_md-r7-s00 --checkpoint-every 50000 --record-every-checkpoint
```

**Batch 2: look at what was saved.** `checkpoints list` prints every saved step with its file hash and video status. `play` replays one exact saved checkpoint (step 150000) on the validation task `val_001`, headless, and writes the movement as a trace under `play/`.

```bash
python -m natural_rl checkpoints list --run-id ppo_md-r7-s00
python -m natural_rl play --run-id ppo_md-r7-s00 --checkpoint-step 150000 --task-id val_001
```

**Batch 3: videos.** `video record` renders the raw video of one checkpoint again (a verified video is never overwritten; it reports whether the new render is pixel-identical). `video backfill` renders only the videos that are missing or failed. `video verify` decodes every video with two independent decoders and compares each frame with its manifest.

```bash
python -m natural_rl video record --run-id ppo_md-r7-s00 --checkpoint-step 150000 --suite checkpoint_video_v1
python -m natural_rl video backfill --run-id ppo_md-r7-s00 --suite checkpoint_video_v1
python -m natural_rl video verify --run-id ppo_md-r7-s00 --all-checkpoints
```

**Batch 4: evaluate.** The first line scores every checkpoint on the 100 validation tasks; it is what picks the checkpoint to test. The second scores that validation-selected checkpoint on the 200 held-out test tasks (the `--manifest` argument pins the registered task file; any other file is refused). Both write the episode, action and trajectory tables under `eval/`.

```bash
python -m natural_rl eval --run-id ppo_md-r7-s00 --suite val_v1 --all-checkpoints
python -m natural_rl eval --run-id ppo_md-r7-s00 --suite test_v1 --manifest configs/eval/test_v1.json
```

**Batch 5: summarise and export.** `analyse` writes one summary table for the run (scored with the locked HKS scalers), `plot` is the entry point for the per-run figures, `export` writes the executed test-task trajectories as CSV and parquet, and `results locate` prints the run's status and where each of its files is.

```bash
python -m natural_rl analyse --run-id ppo_md-r7-s00
python -m natural_rl plot --run-id ppo_md-r7-s00 --kind all
python -m natural_rl export --run-id ppo_md-r7-s00 --kind trajectories --suite test_v1
python -m natural_rl results locate --run-id ppo_md-r7-s00
```

### ppo_box: PPO with continuous actions

*Schulman et al., Proximal Policy Optimization Algorithms, 2017.* https://arxiv.org/abs/1707.06347

**What it does.** The same algorithm with the same network sizes, optimiser and rollout budget as `ppo_md`; only the action distribution differs.
The policy is a diagonal Gaussian over the seven normalised velocities,

$$
a\sim\mathcal N\big(\mu_\theta(s),\mathrm{diag}\sigma^2\big)
$$

$$
v=v_{\max}\mathrm{clip}(a,-1,1)
$$

with a state-independent learned log-standard-deviation. Comparing it with `ppo_md` isolates the effect of the action representation inside the
same method.

**Settings.** Registered in `configs/models/ppo_box.yaml`; the values of one run are saved in its `config_resolved.yaml`. Change any value for a single run with `--set hyper.<key>=<value>` (several at once are allowed, for example `--set hyper.lr=1e-4 hyper.batch=512`). The run settings that every controller shares (`--action`, `--obs`, `--reward`, `--steps`, `--seed`, `--device` ...) are listed in the table at the start of this section.

| Setting | Registered value | Meaning |
| --- | --- | --- |
| `n_envs` | `16` | environments stepped in parallel; `--steps` and `--checkpoint-every` must be multiples of it |
| `hyper.gamma` | `0.95` | discount factor `γ` |
| `hyper.gae_lambda` | `0.95` | GAE `λ` |
| `hyper.lr` | `0.0003` | learning rate (Adam) |
| `hyper.n_steps` | `256` | rollout length per environment between updates |
| `hyper.batch_size` | `256` | minibatch size per update |
| `hyper.n_epochs` | `10` | optimisation epochs per rollout |
| `hyper.clip_range` | `0.2` | PPO clip range |
| `hyper.ent_coef` | `0.0` | entropy coefficient (`auto`: tuned to a target entropy) |
| `hyper.net_arch` | `{pi: [256, 256], vf: [256, 256]}` | hidden layer widths of the networks |

* Identical to `ppo_md` except for the Gaussian action distribution.

**Batch 1: train.** The long step. It trains for 1,000,000 transitions on the 500 training tasks, saves a checkpoint at step 0, every 50,000 transitions and at the end (21 in all), and records and verifies a lossless raw video of each (`--record-every-checkpoint`), with a progress bar showing the model and the run. Everything goes to `results/runs/<backend>/<model>/<reward>/<run_id>/`. Running it again with the same `--run-id` replaces the run.

```bash
python -m natural_rl train --model ppo_box --reward r7_balanced --suite train_v1 --steps 1000000 --seed 0 --run-id ppo_box-r7-s00 --checkpoint-every 50000 --record-every-checkpoint
```

**Batch 2: look at what was saved.** `checkpoints list` prints every saved step with its file hash and video status. `play` replays one exact saved checkpoint (step 150000) on the validation task `val_001`, headless, and writes the movement as a trace under `play/`.

```bash
python -m natural_rl checkpoints list --run-id ppo_box-r7-s00
python -m natural_rl play --run-id ppo_box-r7-s00 --checkpoint-step 150000 --task-id val_001
```

**Batch 3: videos.** `video record` renders the raw video of one checkpoint again (a verified video is never overwritten; it reports whether the new render is pixel-identical). `video backfill` renders only the videos that are missing or failed. `video verify` decodes every video with two independent decoders and compares each frame with its manifest.

```bash
python -m natural_rl video record --run-id ppo_box-r7-s00 --checkpoint-step 150000 --suite checkpoint_video_v1
python -m natural_rl video backfill --run-id ppo_box-r7-s00 --suite checkpoint_video_v1
python -m natural_rl video verify --run-id ppo_box-r7-s00 --all-checkpoints
```

**Batch 4: evaluate.** The first line scores every checkpoint on the 100 validation tasks; it is what picks the checkpoint to test. The second scores that validation-selected checkpoint on the 200 held-out test tasks (the `--manifest` argument pins the registered task file; any other file is refused). Both write the episode, action and trajectory tables under `eval/`.

```bash
python -m natural_rl eval --run-id ppo_box-r7-s00 --suite val_v1 --all-checkpoints
python -m natural_rl eval --run-id ppo_box-r7-s00 --suite test_v1 --manifest configs/eval/test_v1.json
```

**Batch 5: summarise and export.** `analyse` writes one summary table for the run (scored with the locked HKS scalers), `plot` is the entry point for the per-run figures, `export` writes the executed test-task trajectories as CSV and parquet, and `results locate` prints the run's status and where each of its files is.

```bash
python -m natural_rl analyse --run-id ppo_box-r7-s00
python -m natural_rl plot --run-id ppo_box-r7-s00 --kind all
python -m natural_rl export --run-id ppo_box-r7-s00 --kind trajectories --suite test_v1
python -m natural_rl results locate --run-id ppo_box-r7-s00
```

### sac: Soft Actor-Critic

*Haarnoja et al., Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor, ICML 2018.* https://proceedings.mlr.press/v80/haarnoja18b.html

**What it does.** An off-policy actor-critic that maximises return *and* policy entropy, which keeps exploration alive and makes training stable.
The actor is a squashed Gaussian; two critics `Q_φ1, Q_φ2` (2 x 256 MLPs) with Polyak-averaged targets counter over-estimation.

$$
J(\pi)=\mathbb E\Big[\sum_t\gamma^{t}\big(r_t+\alpha\,\mathcal H(\pi(\cdot\mid s_t))\big)\Big]
$$

$$
y=\tilde r+\gamma\Big(\min_{i}Q_{\bar\phi_i}(s',a')-\alpha\log\pi(a'\mid s')\Big),\ a'\sim\pi(\cdot\mid s')
$$

$$
\mathcal L_Q=\sum_i\big(Q_{\phi_i}(s,a)-y\big)^2
$$

$$
\mathcal L_\pi=\mathbb E\big[\alpha\log\pi(a\mid s)-\min_iQ_{\phi_i}(s,a)\big]
$$

$$
\mathcal L_\alpha=-\alpha\,\big(\log\pi(a\mid s)+\bar{\mathcal H}\big),\ \ \bar{\mathcal H}=-\dim\mathcal A=-7
$$

The temperature `α` is tuned automatically toward the target entropy. One gradient update per environment transition (update-to-data ratio 1),
10,000 random warm-up transitions, batch 256, learning rate `3×10⁻⁴`, `τ=0.005`, relabelled replay. Evaluation and videos use the
deterministic mean action.

**Settings.** Registered in `configs/models/sac.yaml`; the values of one run are saved in its `config_resolved.yaml`. Change any value for a single run with `--set hyper.<key>=<value>` (several at once are allowed, for example `--set hyper.lr=1e-4 hyper.batch=512`). The run settings that every controller shares (`--action`, `--obs`, `--reward`, `--steps`, `--seed`, `--device` ...) are listed in the table at the start of this section.

| Setting | Registered value | Meaning |
| --- | --- | --- |
| `n_envs` | `16` | environments stepped in parallel; `--steps` and `--checkpoint-every` must be multiples of it |
| `hyper.gamma` | `0.95` | discount factor `γ` |
| `hyper.lr` | `0.0003` | learning rate (Adam) |
| `hyper.batch` | `256` | minibatch size per update |
| `hyper.tau` | `0.005` | Polyak coefficient of the target network |
| `hyper.learning_starts` | `10000` | random-action transitions before learning starts |
| `hyper.replay` | `1000000` | replay capacity in transitions |
| `hyper.her_ratio` | `0.8` | share of replayed samples whose goal is relabelled (hindsight) |
| `hyper.n_step` | `1` | return length `n` (relabelled replay) |
| `hyper.utd` | `1` | gradient updates per environment transition |
| `hyper.net_arch` | `[256, 256]` | hidden layer widths of the networks |
| `hyper.ent_coef` | `auto` | entropy coefficient (`auto`: tuned to a target entropy) |

* `learning_starts` must be at least `n_envs x 100` so that one complete episode is in the replay before learning.

**Batch 1: train.** The long step. It trains for 1,000,000 transitions on the 500 training tasks, saves a checkpoint at step 0, every 50,000 transitions and at the end (21 in all), and records and verifies a lossless raw video of each (`--record-every-checkpoint`), with a progress bar showing the model and the run. Everything goes to `results/runs/<backend>/<model>/<reward>/<run_id>/`. Running it again with the same `--run-id` replaces the run.

```bash
python -m natural_rl train --model sac --reward r7_balanced --suite train_v1 --steps 1000000 --seed 0 --run-id sac-r7-s00 --checkpoint-every 50000 --record-every-checkpoint
```

**Batch 2: look at what was saved.** `checkpoints list` prints every saved step with its file hash and video status. `play` replays one exact saved checkpoint (step 150000) on the validation task `val_001`, headless, and writes the movement as a trace under `play/`.

```bash
python -m natural_rl checkpoints list --run-id sac-r7-s00
python -m natural_rl play --run-id sac-r7-s00 --checkpoint-step 150000 --task-id val_001
```

**Batch 3: videos.** `video record` renders the raw video of one checkpoint again (a verified video is never overwritten; it reports whether the new render is pixel-identical). `video backfill` renders only the videos that are missing or failed. `video verify` decodes every video with two independent decoders and compares each frame with its manifest.

```bash
python -m natural_rl video record --run-id sac-r7-s00 --checkpoint-step 150000 --suite checkpoint_video_v1
python -m natural_rl video backfill --run-id sac-r7-s00 --suite checkpoint_video_v1
python -m natural_rl video verify --run-id sac-r7-s00 --all-checkpoints
```

**Batch 4: evaluate.** The first line scores every checkpoint on the 100 validation tasks; it is what picks the checkpoint to test. The second scores that validation-selected checkpoint on the 200 held-out test tasks (the `--manifest` argument pins the registered task file; any other file is refused). Both write the episode, action and trajectory tables under `eval/`.

```bash
python -m natural_rl eval --run-id sac-r7-s00 --suite val_v1 --all-checkpoints
python -m natural_rl eval --run-id sac-r7-s00 --suite test_v1 --manifest configs/eval/test_v1.json
```

**Batch 5: summarise and export.** `analyse` writes one summary table for the run (scored with the locked HKS scalers), `plot` is the entry point for the per-run figures, `export` writes the executed test-task trajectories as CSV and parquet, and `results locate` prints the run's status and where each of its files is.

```bash
python -m natural_rl analyse --run-id sac-r7-s00
python -m natural_rl plot --run-id sac-r7-s00 --kind all
python -m natural_rl export --run-id sac-r7-s00 --kind trajectories --suite test_v1
python -m natural_rl results locate --run-id sac-r7-s00
```

### td3: Twin Delayed DDPG

*Fujimoto, van Hoof and Meger, Addressing Function Approximation Error in Actor-Critic Methods, ICML 2018.* https://proceedings.mlr.press/v80/fujimoto18a.html

**What it does.** An off-policy actor-critic with a *deterministic* actor `μ_ψ(s)` and three devices against value over-estimation: twin critics
(take the minimum), target-policy smoothing, and delayed actor updates.

$$
y=\tilde r+\gamma\min_{i}Q_{\bar\phi_i}\!\big(s',\ \mathrm{clip}(\mu_{\bar\psi}(s')+\epsilon,-1,1)\big)
$$

$$
\epsilon=\mathrm{clip}\!\big(\mathcal N(0,0.2^{2}),-0.5,0.5\big)
$$

$$
\mathcal L_Q=\sum_i\big(Q_{\phi_i}(s,a)-y\big)^2
$$

$$
\mathcal L_\mu=-\,\mathbb E\big[Q_{\phi_1}(s,\mu_\psi(s))\big]\ \text{every 2 critic updates.}
$$

Exploration noise `N(0,0.1²)` is added to the actor's action during training only. Other settings as SAC.

**Settings.** Registered in `configs/models/td3.yaml`; the values of one run are saved in its `config_resolved.yaml`. Change any value for a single run with `--set hyper.<key>=<value>` (several at once are allowed, for example `--set hyper.lr=1e-4 hyper.batch=512`). The run settings that every controller shares (`--action`, `--obs`, `--reward`, `--steps`, `--seed`, `--device` ...) are listed in the table at the start of this section.

| Setting | Registered value | Meaning |
| --- | --- | --- |
| `n_envs` | `16` | environments stepped in parallel; `--steps` and `--checkpoint-every` must be multiples of it |
| `hyper.gamma` | `0.95` | discount factor `γ` |
| `hyper.lr` | `0.0003` | learning rate (Adam) |
| `hyper.batch` | `256` | minibatch size per update |
| `hyper.tau` | `0.005` | Polyak coefficient of the target network |
| `hyper.learning_starts` | `10000` | random-action transitions before learning starts |
| `hyper.replay` | `1000000` | replay capacity in transitions |
| `hyper.her_ratio` | `0.8` | share of replayed samples whose goal is relabelled (hindsight) |
| `hyper.n_step` | `1` | return length `n` (relabelled replay) |
| `hyper.utd` | `1` | gradient updates per environment transition |
| `hyper.net_arch` | `[256, 256]` | hidden layer widths of the networks |
| `hyper.policy_delay` | `2` | actor update every this many critic updates |
| `hyper.target_policy_noise` | `0.2` | std of the noise added to target actions |
| `hyper.target_noise_clip` | `0.5` | clip of that noise |
| `hyper.exploration_noise_sigma` | `0.1` | std of the Gaussian exploration noise |

* The deterministic policy acts without noise at evaluation; `exploration_noise_sigma` applies in training only.

**Batch 1: train.** The long step. It trains for 1,000,000 transitions on the 500 training tasks, saves a checkpoint at step 0, every 50,000 transitions and at the end (21 in all), and records and verifies a lossless raw video of each (`--record-every-checkpoint`), with a progress bar showing the model and the run. Everything goes to `results/runs/<backend>/<model>/<reward>/<run_id>/`. Running it again with the same `--run-id` replaces the run.

```bash
python -m natural_rl train --model td3 --reward r7_balanced --suite train_v1 --steps 1000000 --seed 0 --run-id td3-r7-s00 --checkpoint-every 50000 --record-every-checkpoint
```

**Batch 2: look at what was saved.** `checkpoints list` prints every saved step with its file hash and video status. `play` replays one exact saved checkpoint (step 150000) on the validation task `val_001`, headless, and writes the movement as a trace under `play/`.

```bash
python -m natural_rl checkpoints list --run-id td3-r7-s00
python -m natural_rl play --run-id td3-r7-s00 --checkpoint-step 150000 --task-id val_001
```

**Batch 3: videos.** `video record` renders the raw video of one checkpoint again (a verified video is never overwritten; it reports whether the new render is pixel-identical). `video backfill` renders only the videos that are missing or failed. `video verify` decodes every video with two independent decoders and compares each frame with its manifest.

```bash
python -m natural_rl video record --run-id td3-r7-s00 --checkpoint-step 150000 --suite checkpoint_video_v1
python -m natural_rl video backfill --run-id td3-r7-s00 --suite checkpoint_video_v1
python -m natural_rl video verify --run-id td3-r7-s00 --all-checkpoints
```

**Batch 4: evaluate.** The first line scores every checkpoint on the 100 validation tasks; it is what picks the checkpoint to test. The second scores that validation-selected checkpoint on the 200 held-out test tasks (the `--manifest` argument pins the registered task file; any other file is refused). Both write the episode, action and trajectory tables under `eval/`.

```bash
python -m natural_rl eval --run-id td3-r7-s00 --suite val_v1 --all-checkpoints
python -m natural_rl eval --run-id td3-r7-s00 --suite test_v1 --manifest configs/eval/test_v1.json
```

**Batch 5: summarise and export.** `analyse` writes one summary table for the run (scored with the locked HKS scalers), `plot` is the entry point for the per-run figures, `export` writes the executed test-task trajectories as CSV and parquet, and `results locate` prints the run's status and where each of its files is.

```bash
python -m natural_rl analyse --run-id td3-r7-s00
python -m natural_rl plot --run-id td3-r7-s00 --kind all
python -m natural_rl export --run-id td3-r7-s00 --kind trajectories --suite test_v1
python -m natural_rl results locate --run-id td3-r7-s00
```

### tqc: Truncated Quantile Critics

*Kuznetsov et al., Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics, ICML 2020.* https://proceedings.mlr.press/v119/kuznetsov20a.html

**What it does.** SAC with *distributional* critics. Each of 5 critics predicts `N=25` quantiles of the return distribution. To control over-estimation
the largest atoms are discarded before forming the target: all `5×25` next-state atoms are pooled and sorted, and the top `5×2=10` are dropped.

$$
\mathcal Z=\mathrm{sort}\big\lbrace z^{n,m}_{\bar\phi}(s',a')\big\rbrace_{n\le5,\,m\le25}
$$

$$
y_i=\tilde r+\gamma\big(z_{(i)}-\alpha\log\pi(a'\mid s')\big),\ \ i=1,\dots,115
$$

$$
\mathcal L=\sum_{n,i,m}\rho^{H}_{\tau_m}\!\big(y_i-z^{n,m}_\phi(s,a)\big)
$$

$$
\mathcal L_\pi=\mathbb E\Big[\alpha\log\pi(a\mid s)-\tfrac1{5\cdot25}\sum_{n,m}z^{n,m}_\phi(s,a)\Big]
$$

where `ρ_τ^H` is the quantile Huber loss and `τ_m` the quantile midpoints. Temperature, update ratio and relabelled replay as in SAC.

**Settings.** Registered in `configs/models/tqc.yaml`; the values of one run are saved in its `config_resolved.yaml`. Change any value for a single run with `--set hyper.<key>=<value>` (several at once are allowed, for example `--set hyper.lr=1e-4 hyper.batch=512`). The run settings that every controller shares (`--action`, `--obs`, `--reward`, `--steps`, `--seed`, `--device` ...) are listed in the table at the start of this section.

| Setting | Registered value | Meaning |
| --- | --- | --- |
| `n_envs` | `16` | environments stepped in parallel; `--steps` and `--checkpoint-every` must be multiples of it |
| `hyper.gamma` | `0.95` | discount factor `γ` |
| `hyper.lr` | `0.0003` | learning rate (Adam) |
| `hyper.batch` | `256` | minibatch size per update |
| `hyper.tau` | `0.005` | Polyak coefficient of the target network |
| `hyper.learning_starts` | `10000` | random-action transitions before learning starts |
| `hyper.replay` | `1000000` | replay capacity in transitions |
| `hyper.her_ratio` | `0.8` | share of replayed samples whose goal is relabelled (hindsight) |
| `hyper.n_step` | `1` | return length `n` (relabelled replay) |
| `hyper.utd` | `1` | gradient updates per environment transition |
| `hyper.net_arch` | `[256, 256]` | hidden layer widths of the networks |
| `hyper.ent_coef` | `auto` | entropy coefficient (`auto`: tuned to a target entropy) |
| `hyper.n_critics` | `5` | number of critic networks |
| `hyper.n_quantiles` | `25` | quantiles per critic |
| `hyper.top_quantiles_to_drop_per_net` | `2` | largest quantiles dropped per critic (overestimation control) |

* `n_quantiles x n_critics - top_quantiles_to_drop_per_net x n_critics` quantiles remain for the target.

**Batch 1: train.** The long step. It trains for 1,000,000 transitions on the 500 training tasks, saves a checkpoint at step 0, every 50,000 transitions and at the end (21 in all), and records and verifies a lossless raw video of each (`--record-every-checkpoint`), with a progress bar showing the model and the run. Everything goes to `results/runs/<backend>/<model>/<reward>/<run_id>/`. Running it again with the same `--run-id` replaces the run.

```bash
python -m natural_rl train --model tqc --reward r7_balanced --suite train_v1 --steps 1000000 --seed 0 --run-id tqc-r7-s00 --checkpoint-every 50000 --record-every-checkpoint
```

**Batch 2: look at what was saved.** `checkpoints list` prints every saved step with its file hash and video status. `play` replays one exact saved checkpoint (step 150000) on the validation task `val_001`, headless, and writes the movement as a trace under `play/`.

```bash
python -m natural_rl checkpoints list --run-id tqc-r7-s00
python -m natural_rl play --run-id tqc-r7-s00 --checkpoint-step 150000 --task-id val_001
```

**Batch 3: videos.** `video record` renders the raw video of one checkpoint again (a verified video is never overwritten; it reports whether the new render is pixel-identical). `video backfill` renders only the videos that are missing or failed. `video verify` decodes every video with two independent decoders and compares each frame with its manifest.

```bash
python -m natural_rl video record --run-id tqc-r7-s00 --checkpoint-step 150000 --suite checkpoint_video_v1
python -m natural_rl video backfill --run-id tqc-r7-s00 --suite checkpoint_video_v1
python -m natural_rl video verify --run-id tqc-r7-s00 --all-checkpoints
```

**Batch 4: evaluate.** The first line scores every checkpoint on the 100 validation tasks; it is what picks the checkpoint to test. The second scores that validation-selected checkpoint on the 200 held-out test tasks (the `--manifest` argument pins the registered task file; any other file is refused). Both write the episode, action and trajectory tables under `eval/`.

```bash
python -m natural_rl eval --run-id tqc-r7-s00 --suite val_v1 --all-checkpoints
python -m natural_rl eval --run-id tqc-r7-s00 --suite test_v1 --manifest configs/eval/test_v1.json
```

**Batch 5: summarise and export.** `analyse` writes one summary table for the run (scored with the locked HKS scalers), `plot` is the entry point for the per-run figures, `export` writes the executed test-task trajectories as CSV and parquet, and `results locate` prints the run's status and where each of its files is.

```bash
python -m natural_rl analyse --run-id tqc-r7-s00
python -m natural_rl plot --run-id tqc-r7-s00 --kind all
python -m natural_rl export --run-id tqc-r7-s00 --kind trajectories --suite test_v1
python -m natural_rl results locate --run-id tqc-r7-s00
```

### crossq: CrossQ

*Bhatt et al., CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity, ICLR 2024.* https://proceedings.iclr.cc/paper_files/paper/2024/hash/f381114cf5aba4e45552869863deaaa7-Abstract-Conference.html

**What it does.** A SAC-style learner that removes the target networks. Stability comes from adding batch normalisation to the critics and evaluating
`(s,a)` and `(s',a')` in a *single joint forward pass*, so the normalisation statistics mix on-policy and off-policy samples:

$$
\big[Q(s,a),\,Q(s',a')\big]=Q_\phi\big([\,(s,a);(s',a')\,]\big)
$$

$$
y=\tilde r+\gamma\Big(\mathrm{sg}\big[\min_iQ_i(s',a')\big]-\alpha\log\pi(a'\mid s')\Big)
$$

and the critic loss is the squared error to `y` with gradients only through `Q(s,a)`. The actor maximises the same entropy-regularised value. All
library defaults of the sb3-contrib implementation (wide critics, policy delay, learning rate, batch-norm momentum) are used and recorded in
`config_resolved.yaml`; replay, relabelling and update ratio follow the other off-policy methods.

**Settings.** Registered in `configs/models/crossq.yaml`; the values of one run are saved in its `config_resolved.yaml`. Change any value for a single run with `--set hyper.<key>=<value>` (several at once are allowed, for example `--set hyper.lr=1e-4 hyper.batch=512`). The run settings that every controller shares (`--action`, `--obs`, `--reward`, `--steps`, `--seed`, `--device` ...) are listed in the table at the start of this section.

| Setting | Registered value | Meaning |
| --- | --- | --- |
| `n_envs` | `16` | environments stepped in parallel; `--steps` and `--checkpoint-every` must be multiples of it |
| `hyper.gamma` | `0.95` | discount factor `γ` |
| `hyper.batch` | `256` | minibatch size per update |
| `hyper.learning_starts` | `10000` | random-action transitions before learning starts |
| `hyper.replay` | `1000000` | replay capacity in transitions |
| `hyper.her_ratio` | `0.8` | share of replayed samples whose goal is relabelled (hindsight) |
| `hyper.n_step` | `1` | return length `n` (relabelled replay) |
| `hyper.utd` | `1` | gradient updates per environment transition |
| `hyper.library_defaults` | true | learning rate, `τ`, network sizes, policy delay and batch-norm momentum are the sb3-contrib CrossQ defaults |

* Has no target networks; the values listed as library defaults are read from the installed sb3-contrib and recorded in `config_resolved.yaml`.

**Batch 1: train.** The long step. It trains for 1,000,000 transitions on the 500 training tasks, saves a checkpoint at step 0, every 50,000 transitions and at the end (21 in all), and records and verifies a lossless raw video of each (`--record-every-checkpoint`), with a progress bar showing the model and the run. Everything goes to `results/runs/<backend>/<model>/<reward>/<run_id>/`. Running it again with the same `--run-id` replaces the run.

```bash
python -m natural_rl train --model crossq --reward r7_balanced --suite train_v1 --steps 1000000 --seed 0 --run-id crossq-r7-s00 --checkpoint-every 50000 --record-every-checkpoint
```

**Batch 2: look at what was saved.** `checkpoints list` prints every saved step with its file hash and video status. `play` replays one exact saved checkpoint (step 150000) on the validation task `val_001`, headless, and writes the movement as a trace under `play/`.

```bash
python -m natural_rl checkpoints list --run-id crossq-r7-s00
python -m natural_rl play --run-id crossq-r7-s00 --checkpoint-step 150000 --task-id val_001
```

**Batch 3: videos.** `video record` renders the raw video of one checkpoint again (a verified video is never overwritten; it reports whether the new render is pixel-identical). `video backfill` renders only the videos that are missing or failed. `video verify` decodes every video with two independent decoders and compares each frame with its manifest.

```bash
python -m natural_rl video record --run-id crossq-r7-s00 --checkpoint-step 150000 --suite checkpoint_video_v1
python -m natural_rl video backfill --run-id crossq-r7-s00 --suite checkpoint_video_v1
python -m natural_rl video verify --run-id crossq-r7-s00 --all-checkpoints
```

**Batch 4: evaluate.** The first line scores every checkpoint on the 100 validation tasks; it is what picks the checkpoint to test. The second scores that validation-selected checkpoint on the 200 held-out test tasks (the `--manifest` argument pins the registered task file; any other file is refused). Both write the episode, action and trajectory tables under `eval/`.

```bash
python -m natural_rl eval --run-id crossq-r7-s00 --suite val_v1 --all-checkpoints
python -m natural_rl eval --run-id crossq-r7-s00 --suite test_v1 --manifest configs/eval/test_v1.json
```

**Batch 5: summarise and export.** `analyse` writes one summary table for the run (scored with the locked HKS scalers), `plot` is the entry point for the per-run figures, `export` writes the executed test-task trajectories as CSV and parquet, and `results locate` prints the run's status and where each of its files is.

```bash
python -m natural_rl analyse --run-id crossq-r7-s00
python -m natural_rl plot --run-id crossq-r7-s00 --kind all
python -m natural_rl export --run-id crossq-r7-s00 --kind trajectories --suite test_v1
python -m natural_rl results locate --run-id crossq-r7-s00
```

### tdmpc2: TD-MPC2

*Hansen, Su and Wang, TD-MPC2: Scalable, Robust World Models for Continuous Control, ICLR 2024.* https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html

**What it does.** A model-based method: it learns a *latent world model* from experience and chooses each action by *planning* inside it, instead of
reading it from a policy network. Components (smallest official preset, 1M parameters):

| Component | Maps | Role |
| --- | --- | --- |
| encoder `h_θ` | `s ↦ z` | embeds the observation in a normalised latent space |
| dynamics `d_θ` | `(z,a) ↦ z'` | predicts the next latent |
| reward head `R_θ` | `(z,a) ↦ r̂` | discrete regression over 101 bins |
| value heads `Q_θ` | `(z,a) ↦ q̂` | discrete regression, ensemble with target copies |
| policy prior `π_θ` | `z ↦ a` | proposes actions to the planner, trained by maximum entropy |

**Model learning.** Windows of `H_p=3` consecutive steps come from the relabelled replay (each window with its own relabelled goal, with dwell and
rewards recomputed over the window). With `z_0=h_θ(s_0)`, `z_(t+1)=d_θ(z_t,a_t)` and `ρ=0.5`:

$$
\mathcal L=\sum_{t=0}^{H_p-1}\rho^{t}\Big(c_{1}\,\big\Vert z_{t+1}-\mathrm{sg}\,h_\theta(s_{t+1})\big\Vert_2^2
+c_{2}\mathrm{CE}\!\big(\hat r_t,r_t\big)+c_{3}\mathrm{CE}\!\big(\hat q_t,\mathrm{sg}\,y_t\big)\Big)
$$

$$
y_t=r_t+\gamma\,\bar Q_\theta\big(z_{t+1},\pi_\theta(z_{t+1})\big)
$$

$$
c_1=20,\ c_2=c_3=0.1
$$

**Planning (MPPI).** At every decision the agent samples `N=512` action sequences of length `H_p=3` (24 of them from the policy prior), scores each in the
latent model, keeps the 64 best and refits a Gaussian over sequences, for 6 iterations, then executes the first action:

$$
\hat G(a_{0:H_p-1})=\sum_{t=0}^{H_p-1}\gamma^{t}R_\theta(z_t,a_t)+\gamma^{H_p}\,Q_\theta\big(z_{H_p},\pi_\theta(z_{H_p})\big)
$$

$$
w_i\propto\exp\!\big(\tau(\hat G_i-\max\hat G)\big),\ \tau=0.5
$$

One environment is used (the planner handles one observation at a time), with 1,000 random seed transitions, a pretraining pass of 1,000 updates, then one
update per transition. The planner keeps one state per environment at evaluation. Planning samples among elite sequences even at evaluation time, so a
checkpoint replays exactly only for the same device and seed (the tools seed it; `NATURAL_RL_POLICY_DEVICE` selects the device, otherwise CUDA if available). The
official code is vendored unchanged apart from device and compile handling, listed in the docstring of `natural_rl/vendor/tdmpc2/__init__.py`.

**Settings.** Registered in `configs/models/tdmpc2.yaml`; the values of one run are saved in its `config_resolved.yaml`. Change any value for a single run with `--set hyper.<key>=<value>` (several at once are allowed, for example `--set hyper.lr=1e-4 hyper.batch=512`). The run settings that every controller shares (`--action`, `--obs`, `--reward`, `--steps`, `--seed`, `--device` ...) are listed in the table at the start of this section.

| Setting | Registered value | Meaning |
| --- | --- | --- |
| `n_envs` | `1` | environments stepped in parallel; `--steps` and `--checkpoint-every` must be multiples of it |
| `hyper.model_size` | `1` | official TD-MPC2 preset (1: about 1M parameters, 2 Q heads) |
| `hyper.batch_size` | `256` | minibatch size per update |
| `hyper.reward_coef` | `0.1` | weight of the reward-prediction loss |
| `hyper.value_coef` | `0.1` | weight of the value loss |
| `hyper.consistency_coef` | `20` | weight of the latent-consistency loss |
| `hyper.rho` | `0.5` | decay of the loss weight along the horizon |
| `hyper.lr` | `0.0003` | learning rate (Adam) |
| `hyper.enc_lr_scale` | `0.3` | learning-rate factor of the encoder |
| `hyper.grad_clip_norm` | `20` | gradient-norm clip |
| `hyper.tau` | `0.01` | Polyak coefficient of the target network |
| `hyper.discount_denom` | `5` | episode-length scale of the discount rule |
| `hyper.discount_min` | `0.95` | lower discount bound |
| `hyper.discount_max` | `0.95` | upper discount bound (equal bounds: the task discount 0.95) |
| `hyper.iterations` | `6` | MPPI iterations per planning step |
| `hyper.num_samples` | `512` | action sequences sampled per iteration |
| `hyper.num_elites` | `64` | best sequences used to refit the sampler |
| `hyper.num_pi_trajs` | `24` | sequences proposed by the learned policy |
| `hyper.horizon` | `3` | planning horizon in decisions |
| `hyper.min_std` | `0.05` | lower bound of the planner std |
| `hyper.max_std` | `2` | upper bound of the planner std |
| `hyper.temperature` | `0.5` | MPPI elite weighting temperature |
| `hyper.log_std_min` | `-10` | lower bound of the policy log-std |
| `hyper.log_std_max` | `2` | upper bound of the policy log-std |
| `hyper.entropy_coef` | `0.0001` | policy entropy weight |
| `hyper.num_bins` | `101` | bins of the discrete reward/value regression |
| `hyper.vmin` | `-10` | lower end of the value support |
| `hyper.vmax` | `10` | upper end of the value support |
| `hyper.num_channels` | `32` | channels of the pixel encoder (not used for state input) |
| `hyper.dropout` | `0.01` | dropout in the Q network |
| `hyper.simnorm_dim` | `8` | SimNorm group size of the latent |
| `hyper.seed_steps` | `1000` | random-action transitions, then one pretraining pass of this many updates |
| `hyper.replay` | `1000000` | replay capacity in transitions |
| `hyper.her_ratio` | `0.8` | share of replayed samples whose goal is relabelled (hindsight) |

* Uses one environment (`n_envs: 1`), so `--steps` need not be a multiple of 16; it uses CUDA by default when available (`--device cpu` forces CPU) and replays are deterministic only for the same device and seed.
* Planning cost per decision is set by `iterations`, `num_samples` and `horizon`; lower them for a quick test.

**Batch 1: train.** The long step. It trains for 1,000,000 transitions on the 500 training tasks, saves a checkpoint at step 0, every 50,000 transitions and at the end (21 in all), and records and verifies a lossless raw video of each (`--record-every-checkpoint`), with a progress bar showing the model and the run. Everything goes to `results/runs/<backend>/<model>/<reward>/<run_id>/`. Running it again with the same `--run-id` replaces the run.

```bash
python -m natural_rl train --model tdmpc2 --reward r7_balanced --suite train_v1 --steps 1000000 --seed 0 --run-id tdmpc2-r7-s00 --checkpoint-every 50000 --record-every-checkpoint
```

**Batch 2: look at what was saved.** `checkpoints list` prints every saved step with its file hash and video status. `play` replays one exact saved checkpoint (step 150000) on the validation task `val_001`, headless, and writes the movement as a trace under `play/`.

```bash
python -m natural_rl checkpoints list --run-id tdmpc2-r7-s00
python -m natural_rl play --run-id tdmpc2-r7-s00 --checkpoint-step 150000 --task-id val_001
```

**Batch 3: videos.** `video record` renders the raw video of one checkpoint again (a verified video is never overwritten; it reports whether the new render is pixel-identical). `video backfill` renders only the videos that are missing or failed. `video verify` decodes every video with two independent decoders and compares each frame with its manifest.

```bash
python -m natural_rl video record --run-id tdmpc2-r7-s00 --checkpoint-step 150000 --suite checkpoint_video_v1
python -m natural_rl video backfill --run-id tdmpc2-r7-s00 --suite checkpoint_video_v1
python -m natural_rl video verify --run-id tdmpc2-r7-s00 --all-checkpoints
```

**Batch 4: evaluate.** The first line scores every checkpoint on the 100 validation tasks; it is what picks the checkpoint to test. The second scores that validation-selected checkpoint on the 200 held-out test tasks (the `--manifest` argument pins the registered task file; any other file is refused). Both write the episode, action and trajectory tables under `eval/`.

```bash
python -m natural_rl eval --run-id tdmpc2-r7-s00 --suite val_v1 --all-checkpoints
python -m natural_rl eval --run-id tdmpc2-r7-s00 --suite test_v1 --manifest configs/eval/test_v1.json
```

**Batch 5: summarise and export.** `analyse` writes one summary table for the run (scored with the locked HKS scalers), `plot` is the entry point for the per-run figures, `export` writes the executed test-task trajectories as CSV and parquet, and `results locate` prints the run's status and where each of its files is.

```bash
python -m natural_rl analyse --run-id tdmpc2-r7-s00
python -m natural_rl plot --run-id tdmpc2-r7-s00 --kind all
python -m natural_rl export --run-id tdmpc2-r7-s00 --kind trajectories --suite test_v1
python -m natural_rl results locate --run-id tdmpc2-r7-s00
```

### sac_pidlag: SAC with PID-Lagrangian cost budgets

*SAC: Haarnoja et al., ICML 2018. PID Lagrangian: Stooke, Achiam and Abbeel, Responsive Safety in Reinforcement Learning by PID Lagrangian Methods, ICML 2020.*
https://proceedings.mlr.press/v80/haarnoja18b.html · https://proceedings.mlr.press/v119/stooke20a.html

**What it does.** SAC that also tries to keep the *expected number of unsafe proposals* below a budget. An unsafe proposal is a collision attempt or a
pre-clip joint-limit attempt (counted on the proposed motion, so rejected or clipped commands are never hidden). This is an adaptation of the two methods and is not
attributed to either paper alone.

$$
\max_\pi\ J_r(\pi)\quad\text{s.t.}\quad J_c(\pi)=\mathbb E\Big[\sum_{t=0}^{H-1}c_t\Big]\le d
$$

$$
c_t=\mathbb 1[\text{collision attempt}]+\mathbb 1[\text{limit attempt}]
$$

with `d=0.50` (`budget_moderate_v1`) or `d=0.10` (`budget_strict_v1`) events per episode. A second pair of critics learns the discounted cost,
`y_c=c+γ Q̄_c(s',a')`, and the actor maximises the normalised Lagrangian:

$$
\mathcal L_\pi=\mathbb E\Big[\alpha\log\pi(a\mid s)-\frac{\min_iQ_{r,i}(s,a)-\lambda\,\bar Q_c(s,a)}{1+\lambda}\Big]
$$

Here `Q̄_c` is the mean of the two cost critics (of the two target cost critics inside `y_c`).

The multiplier follows a PID rule on a smoothed measurement `Ĵ_c` of the undiscounted episodic cost (exponential average, `α_ema=0.05`), updated
after every block of finished episodes:

$$
e_k=\hat J_{c,k}-d
$$

$$
I_{k+1}=\max(0,I_k+e_k)
$$

$$
D_k=\max(0,\hat J_{c,k}-\hat J_{c,k-1})
$$

$$
\lambda_{k+1}=\mathrm{clip}\big(K_Pe_k+K_II_{k+1}+K_DD_k,\ 0,\ 100\big)
$$

with `K_P=0.1`, `K_I=0.01`, `K_D=0.01`. The budget bounds the *expected* cost; it is not a per-episode or zero-event guarantee, and joint clipping and collision
rejection stay active regardless. Raw unsafe counts, the multiplier trace and the budget-violation frequency are logged. Needs `--cost-budget-id`.

**Settings.** Registered in `configs/models/sac_pidlag.yaml`; the values of one run are saved in its `config_resolved.yaml`. Change any value for a single run with `--set hyper.<key>=<value>` (several at once are allowed, for example `--set hyper.lr=1e-4 hyper.batch=512`). The run settings that every controller shares (`--action`, `--obs`, `--reward`, `--steps`, `--seed`, `--device` ...) are listed in the table at the start of this section.

| Setting | Registered value | Meaning |
| --- | --- | --- |
| `n_envs` | `16` | environments stepped in parallel; `--steps` and `--checkpoint-every` must be multiples of it |
| `hyper.gamma` | `0.95` | discount factor `γ` |
| `hyper.lr` | `0.0003` | learning rate (Adam) |
| `hyper.batch` | `256` | minibatch size per update |
| `hyper.tau` | `0.005` | Polyak coefficient of the target network |
| `hyper.learning_starts` | `10000` | random-action transitions before learning starts |
| `hyper.replay` | `1000000` | replay capacity in transitions |
| `hyper.her_ratio` | `0.8` | share of replayed samples whose goal is relabelled (hindsight) |
| `hyper.n_step` | `1` | return length `n` (relabelled replay) |
| `hyper.utd` | `1` | gradient updates per environment transition |
| `hyper.net_arch` | `[256, 256]` | hidden layer widths of the networks |
| `hyper.ent_coef` | `auto` | entropy coefficient (`auto`: tuned to a target entropy) |

* Needs `--cost-budget-id budget_moderate_v1` (0.50 expected unsafe events per 100-decision episode) or `budget_strict_v1` (0.10); the PID gains `kp=0.1, ki=0.01, kd=0.01`, `lambda_max=100` and the smoothing `ema_alpha=0.05` of the cost estimate are in `configs/costs/budgets_v1.yaml`.

**Batch 1: train.** The long step. It trains for 1,000,000 transitions on the 500 training tasks, saves a checkpoint at step 0, every 50,000 transitions and at the end (21 in all), and records and verifies a lossless raw video of each (`--record-every-checkpoint`), with a progress bar showing the model and the run. Everything goes to `results/runs/<backend>/<model>/<reward>/<run_id>/`. Running it again with the same `--run-id` replaces the run.

```bash
python -m natural_rl train --model sac_pidlag --reward r7_balanced --suite train_v1 --steps 1000000 --seed 0 --run-id sac_pidlag-r7-s00 --checkpoint-every 50000 --record-every-checkpoint --cost-budget-id budget_moderate_v1
```

**Batch 2: look at what was saved.** `checkpoints list` prints every saved step with its file hash and video status. `play` replays one exact saved checkpoint (step 150000) on the validation task `val_001`, headless, and writes the movement as a trace under `play/`.

```bash
python -m natural_rl checkpoints list --run-id sac_pidlag-r7-s00
python -m natural_rl play --run-id sac_pidlag-r7-s00 --checkpoint-step 150000 --task-id val_001
```

**Batch 3: videos.** `video record` renders the raw video of one checkpoint again (a verified video is never overwritten; it reports whether the new render is pixel-identical). `video backfill` renders only the videos that are missing or failed. `video verify` decodes every video with two independent decoders and compares each frame with its manifest.

```bash
python -m natural_rl video record --run-id sac_pidlag-r7-s00 --checkpoint-step 150000 --suite checkpoint_video_v1
python -m natural_rl video backfill --run-id sac_pidlag-r7-s00 --suite checkpoint_video_v1
python -m natural_rl video verify --run-id sac_pidlag-r7-s00 --all-checkpoints
```

**Batch 4: evaluate.** The first line scores every checkpoint on the 100 validation tasks; it is what picks the checkpoint to test. The second scores that validation-selected checkpoint on the 200 held-out test tasks (the `--manifest` argument pins the registered task file; any other file is refused). Both write the episode, action and trajectory tables under `eval/`.

```bash
python -m natural_rl eval --run-id sac_pidlag-r7-s00 --suite val_v1 --all-checkpoints
python -m natural_rl eval --run-id sac_pidlag-r7-s00 --suite test_v1 --manifest configs/eval/test_v1.json
```

**Batch 5: summarise and export.** `analyse` writes one summary table for the run (scored with the locked HKS scalers), `plot` is the entry point for the per-run figures, `export` writes the executed test-task trajectories as CSV and parquet, and `results locate` prints the run's status and where each of its files is.

```bash
python -m natural_rl analyse --run-id sac_pidlag-r7-s00
python -m natural_rl plot --run-id sac_pidlag-r7-s00 --kind all
python -m natural_rl export --run-id sac_pidlag-r7-s00 --kind trajectories --suite test_v1
python -m natural_rl results locate --run-id sac_pidlag-r7-s00
```

---

## 4. Reward profiles and cost budgets

$$
r=r_0+\sum_i\lambda_i f_i
$$

Raw terms `f_i`, weights `λ_i`, weighted contributions and totals are logged at every decision. `d_t` is the distance (m) from the accepted end effector to the goal.

| Term | Definition |
| --- | --- |
| `r_0` | `-min(d/0.5,1)+exp(-d²/(2·0.02²))-c-0.01 mean((v/v_max)²)` (the historical reward, exactly) |
| hold | `0.2·1[d ≤ 0.02]+0.8·1[c_t ≥ 5]` (fixed, not calibrated) |
| time | `-Δt/5 s` per decision before the first 5-step hold |
| path | `-‖p_(t+1)-p_t‖/0.5 m` (executed end-effector motion) |
| speed | `-(Δt/5 s) mean(u²)` with `u=v/v_max` (command-effort proxy, not energy; for torque the normalised torque) |
| accel | `-mean(((u_t-u_(t-1))/2)²)` (change of the normalised command; needs the previous command) |
| rom | `-mean(max(0,(0.1-m_j)/0.1)²)`, `m_j` = normalised distance to the nearest limit (range exposure, not stress) |

Overlines are means over the seven joints. The twelve profiles (`configs/rewards/reward_v1.yaml`):

| Profile | Terms | Question |
| --- | --- | --- |
| `r0_original` | r0 | does the historical baseline reproduce? |
| `r1_hold` | r0 + hold | does explicit dwell improve terminal holding? |
| `r2_time` `r3_path` `r4_speed` `r5_smooth` `r6_rom` | r1 + one term | does that term help? (single additions, compared with r1) |
| `r7_balanced` | r1 + path + speed + accel + rom | how do combined terms behave? |
| `r8_drop_path` `r9_drop_speed` `r10_drop_smooth` `r11_drop_rom` | r7 minus one term | is that term needed in the combination? (drop-ones, compared with r7) |

Additions and drop-ones are separate estimands and are never merged into one ranking.

**Coefficients are calibrated, never hand-picked.** `python -m natural_rl calibrate rewards --pilot-runs RUN1,RUN2` rolls the given pilot runs out on the
`train_v1` tasks only and sets

$$
\lambda_i=\min\!\Big(10,\ \frac{0.10\cdot\mathrm{median}\lvert R_0\rvert}{\max\!\big(10^{-4},\,P_{75}\lvert F_i\rvert\big)}\Big)
$$

with `R_0` the episodic r0 return and `F_i` the episodic unweighted sum of term `i`. The values, the data SHA256 and the code hash are written into the
register, which is then locked and immutable (a rule change means a new benchmark id). Profiles that contain a calibrated term refuse to train until
the register is locked. `--dry-run` prints the table without writing. The coefficients are fitted once, on the historical velocity-control pilot, and used for
every action mode: the speed and command-change terms act on the normalised command `u`, so they have the same scale for velocity, position steps and torque fractions.

**Cost budgets** (`configs/costs/budgets_v1.yaml`) apply to `sac_pidlag`: the cost per decision is the number of collision attempts plus pre-clip joint-limit
attempts, so a joint that is already at its limit and is commanded further counts at every decision. Budgets: 0.50 and 0.10 expected events per 100-decision episode.

---

## 5. Metrics, HKS and statistics

All metrics use **executed, accepted end-effector positions** `p_k` at 20 samples/s (sample `k` is the state after `k` decisions, `k=0` the initial state), never
commands, and success is computed independently of the training reward.

**Outcomes.** With `d_k=‖p_k-g‖` and tolerance 2 cm: *first entry* is the first `k ≥ 1` with `d_k ≤ 0.02`; *any hold* is any 5 consecutive accepted positions
inside; the *terminal hold* (the primary success variable) is

$$
\text{terminal hold}=\mathbb 1\big[\,d_k\le0.02\ \ \text{for the last 5 samples}\,\big]
$$

Also stored: final error, executed path length, duration, goal departures after first entry, unsafe proposals per episode.

**Movement segment.** Scoring uses onset to offset only: onset is the first sample whose speed reaches 5 % of the episode's peak; offset is the *last* sample at or above 5 % of
peak speed, capped before the terminal 5-sample window. The steady hold therefore never inflates smoothness, pauses and a late second sub-movement stay inside the
segment, and dithering inside the tolerance sphere is scored. Velocity and jerk come from a consistent Savitzky-Golay filter (window 7, order 3) applied to the
whole episode before slicing; raw finite differences are kept as a sensitivity.

**Descriptors.** With `T` the segment duration, `L=Σ‖p_(k+1)-p_k‖` its path length and `v(t)` the speed:

$$
\eta=\frac{\lVert g-p_0\rVert}{L_{0\to\text{offset}}}
$$

$$
\mathrm{DJ}=\frac{T^{5}}{L^{2}}\int_0^{T}\big\Vert p^{(3)}(t)\big\Vert^{2}dt\ \ (\text{reported as }\log(1+\mathrm{DJ}))
$$

$$
\mathrm{SPARC}=-\int_{0}^{f_c}\sqrt{\Big(\frac1{f_c}\Big)^{2}+\Big(\frac{d\hat V(f)}{df}\Big)^{2}}\,df
$$

$$
\hat V(f)=\frac{V(f)}{V(0)}
$$

where `V` is the Fourier magnitude of the speed profile (zero-padded) and the cutoff is fixed at `f_c ≤ 8` Hz (the code refuses cutoffs at or above the 10 Hz Nyquist
frequency). Less negative SPARC is smoother. The timing descriptors compare the unit-mean speed `v̂(s)=v/v̄` on normalised time `s ∈ [0,1]` with the
analytic minimum-jerk profile (a metric reference only, never a controller):

$$
x(\tau)=10\tau^{3}-15\tau^{4}+6\tau^{5}
$$

$$
v_{\mathrm{mj}}(s)=30\,s^{2}(1-s)^{2}
$$

$$
\mathrm{RMSE}=\sqrt{\tfrac1n\sum_i\big(\hat v(s_i)-v_{\mathrm{mj}}(s_i)\big)^2}
$$

Further descriptors: number of speed peaks (10 % prominence), pause fraction (share of segment samples below 1 % of peak speed), joint-reversal rate (executed joint
velocity sign changes per second, motion below `10⁻⁴` rad ignored). Short, zero-displacement, non-finite or non-held episodes receive explicit quality flags and never
an artificially perfect score. Because the window drops the first and last ~6 % of a movement, an ideal minimum-jerk reach has `RMSE ≈ 0.14` and
`DJ ≈ 200` through this exact pipeline (the textbook 720 is for the full movement); these are constant floors for every policy.

**HKS (0-100).** Each descriptor is scaled to `[0,1]` with frozen anchors `(q_lo,q_hi)`,

$$
b_\uparrow(x)=\mathrm{clip}\!\Big(\frac{x-q_{lo}}{q_{hi}-q_{lo}},0,1\Big)
$$

$$
b_\downarrow(x)=1-b_\uparrow(x)
$$

and four equally weighted domains are averaged:

$$
D_1=\tfrac12\,b_\uparrow(\mathrm{SPARC})+\tfrac12\,b_\downarrow\big(\log(1+\mathrm{DJ})\big) \quad \text{smoothness}
$$

$$
D_2=\tfrac12\,b_\downarrow(\mathrm{RMSE})+\tfrac12\,b_\downarrow\big(\max(\text{peaks}-1,0)\big) \quad \text{timing}
$$

$$
D_3=b_\uparrow(\eta) \quad \text{spatial (direct-reach tasks only)}
$$

$$
D_4=\tfrac12\,b_\downarrow(\text{pause fraction})+\tfrac12\,b_\downarrow(\text{reversal rate}) \quad \text{continuity}
$$

$$
\mathrm{HKS}=25\,(D_1+D_2+D_3+D_4)
$$

The lower anchor `q_lo` is the 5th percentile; the upper anchor `q_hi` is the first of the 95th percentile, the 99th percentile and the maximum that exceeds `q_lo` (several
descriptors are exactly 0 for clean reaches). The rung used is recorded per metric, and a metric that never varies is a blocking error and is never silently replaced by zero.
Anchors are fitted once, pooled over all runs given (every action, state and backend, so that HKS and R-HKS sit on the same scale for every cell), on successful, quality-valid **validation**
episodes only (`calibrate scalers --runs RUN1,RUN2`, at least 200 episodes), and locked before the cell selection and any held-out ranking.

**Reliability-adjusted score.** Failures and unscorable successes contribute zero and stay in the denominator:

$$
R\text{-}\mathrm{HKS}=\frac1{N_{\text{tasks}}}\sum_{i}\ \text{terminal hold}_i\cdot\mathrm{HKS}_i
$$

**Statistics.** Frames are never replicates; the resampling units are held-out task IDs and training seeds (crossed bootstrap, 2,000 resamples, seed 20261121, 95 % percentile
intervals). Leaderboards share tasks across models and report rank probabilities. A reward effect is a within-policy paired difference on matched (seed, task) cells,

$$
\Delta=\overline{\text{R-HKS}_{\text{treatment}}-\text{R-HKS}_{\text{baseline}}}
$$

$$
q_{(i)}=\min_{j\ge i}\ \frac{m}{j}\,p_{(j)}\quad\text{(Benjamini-Hochberg over the registered family of }m\text{ contrasts)}
$$

and missing cells (crashes, timeouts) are counted and never imputed. A high point estimate is not called a winner unless the intervals and rank probabilities resolve it.

---

## 6. Tasks and evaluation

Immutable manifests `configs/eval/{train,val,test,ood}_v1.json` are generated deterministically (seed 20261121, content SHA256 recorded, `manifests verify` re-checks them):

| Split | Tasks | Purpose |
| --- | --- | --- |
| `train_v1` | 500 | reach configurations for learning |
| `val_v1` | 100 | checkpoint and coefficient calibration; `val_001` is the canonical video task |
| `test_v1` | 200 | direct-reach tasks for the full HKS ranking |
| `ood_v1` | 100 | 50 obstacle-detour tasks (the straight line crosses the scene's existing table edge) and 50 near-joint-limit tasks |

Direct-reach tasks start collision-free with the hand at least 3 cm above the table, have reach distance 0.10-0.50 m, are feasible for a continuous Jacobian controller (used only
as a generation filter, never trained or ranked), and no two tasks share a start and goal within 2 cm. Each task fixes the start pose, goal, scene and a 100-decision budget.
The robot base sits 0.25 m behind the table edge (Kangaroo_Libero's default is 0.55 m) through a generated settings override; Kangaroo_Libero itself is unmodified.

**Three backends, kept in separate tables.** The *fast kinematic* backend (exact transition: forward kinematics plus the filtered collision test) trains and evaluates velocity and position
runs. The *dynamics* backend (`natural_rl/env/dynamics.py`, about 1.5 ms per decision) integrates the arm dynamics for torque runs; `eval` uses the backend a run was trained on unless
`--backend` says otherwise. The *physics* backend (`eval ... --backend physics`) executes the same decisions in Kangaroo_Libero (100 Hz PD control, 5 ticks per decision, measured hand
positions) as an evaluation stratum for velocity and position runs (it takes position targets only, so torque runs are refused there). The same task IDs and geometry serve all of them, and
results live in separate tables labelled with their backend. HKS and R-HKS use one pooled scaler fit for the kinematic and dynamics backends, so scores are comparable across action modes.

**Which checkpoint is tested.** The validation-selected one: highest `val_v1` terminal-hold rate on the run's training backend, ties broken by fewer unsafe events, then the later step.
Run `eval --suite val_v1 --all-checkpoints` first; `eval --suite test_v1` refuses to choose from test results. `--final` or `--checkpoint-step N` override it.

**Testbed facts.** The simulator enforces joint *position* limits only: there is no joint-velocity limit, so 1 rad/s is a benchmark contract inherited from the original DecQN. From the
home pose the hand hangs below the table top, so straight start-to-goal lines would hit the table edge; direct-reach tasks start above the table. No energy quantity is reported; torque is a *command* of the dynamics backend, not a measured joint load.

---

## 7. Training programme

The programme is organised by **cell** = (action, state): 3 actions (`velocity`, `position`, `torque`) x 8 states (`v1` to `v8`) = 24 cells. P1 and P2 cover
every cell, because the point of the benchmark is to find out which state and action make the movement most human-like; P3 and P4 then study the reward terms and the cost
budgets in the best cell chosen from P2 validation data.

| Phase | Programme | Runs | Videos | Transitions |
| --- | --- | --- | --- | --- |
| P0 | historical DecQN `r0_original` (`--contract hist_v1`, velocity, obs_v1), 3 seeds x 1M | 3 | 63 | 3M |
| P1 | 24 cells x 10 controllers x {r1, r7} x 3 seeds x 200k (60 runs per cell) | 1,440 | 7,200 | 288M |
| P2 | 24 cells x 10 controllers x {r1, r7} x 5 seeds x 1M (100 runs per cell) | 2,400 | 50,400 | 2,400M |
| P3 | best cell: DecQN + 3 selected controllers x 10 other profiles x 5 seeds x 1M | 200 | 4,200 | 200M |
| P4 | best cell: SAC-PID-Lagrangian x r7 x {moderate, strict} x seeds 5-9 x 1M | 10 | 210 | 10M |
| **Total** | | **4,053** | **62,073** | **2,901M** |

For a single cell (for example `--actions velocity --states v2`) the programme is the original 373 runs, 6,873 videos and 325M transitions. `--actions` and `--states` choose the cells
(`run-all --dry-run` prints the totals and the storage for any choice before anything starts): the three original states (`--states v1,v2,v3`) give 9 cells, 1,653 runs and 1,221M transitions.

The best cell is the one with the highest mean validation R-HKS over its P2 `r7_balanced` runs (each at its validation-selected checkpoint; ties: lower unsafe rate), and P3's three
controllers are the highest validation terminal-hold rates on `r7_balanced` in that cell (ties: lower unsafe rate); both come from `catalog programme --phase P3 --select` and read validation
tables only. `--steps` counts single-environment transitions, not gradient updates or wall time, and must be a multiple of the environment count (16; TD-MPC2 uses 1).

Run IDs are `<model>-<reward>[-<budget>][-<action>][-<state>]-s<seed>[-p1]`: the velocity action carries no action tag, `position` is `pos`, `torque` is `tor`; `obs_v2` carries no state tag,
`v1` is `o1`, `v3` is `o3`, and so on up to `v8` as `o8`. Examples: `decqn-r1-s00` (velocity, obs_v2), `sac-r7-tor-o3-s02` (torque, obs_v3), `ppo_box-r1-pos-o1-s04-p1`, `sac_pidlag-r7-bstrict-tor-o3-s05`; P0 is
`decqn-r0hist-s00`. A raw video is about 17 MB, so the 62,073 videos need roughly 1.06 TB; `train` stops with an explicit disk alert if free space is below its estimate and never deletes or
downsamples anything.

---

## 8. Checkpoints, videos and results

**Checkpoints** are saved atomically (SHA256 recorded in `checkpoint_index.csv`) at step 0, every `--checkpoint-every` transitions, at the final step even if off cadence, and whenever you
`touch <run_dir>/MANUAL_CHECKPOINT`. `--checkpoint-step N` always means exactly that saved step, never "the latest".

**Every checkpoint gets a raw video** (`--record-every-checkpoint`). The checkpoint is reloaded from disk into a separate evaluation environment, the validation task `val_001` is replayed
deterministically (seed 20261121), and the accepted robot state is rendered at `t=0` and after every 0.05 s decision (101 frames for 100 decisions) into `raw.mkv`: lossless FFV1, 640x480,
fixed camera `canonical_v1`, no overlays or goal marker, rejected and still frames kept. Frames are encoded as `bgr0` (this FFmpeg's FFV1 has no 8-bit planar RGB), a reversible channel reorder.
`preview.mp4` is a lossy viewing copy and never the record. `raw.mkv` plays in VLC or `ffplay` (FFV1 support varies by player; use `preview.mp4` if yours cannot).

* A run is `complete` only when every checkpoint has a verified video; a failed capture marks it `incomplete`, prints `[ERROR]` and exits non-zero (`--strict-video` aborts at once).
* `video verify` decodes every file with two independent decoders and compares each frame's SHA256 with the manifest.
* `video record` on a step that already has a verified video never overwrites it: it re-renders in memory and reports whether the pixels are identical.
* `video backfill` regenerates only missing or failed videos.
* `play` runs one exact saved checkpoint headless and writes a synchronised trace.

**Progress display.** `train` shows which model and run is executing and how far it is, redrawn on one line in a terminal:

```
sac | sac-r7-s00 | r7_balanced seed 0 | ██████░░░░░░░░░░░░░░░░░░  26.5% 265,000/1,000,000 | ckpt 5/21 | 1,230 tr/s | 0:03:35 elapsed, ETA 0:10:12
```

It also announces each checkpoint and video above the bar (`recording video of step ...` appears while one is captured). Speed is measured on training
time only, and the ETA adds the expected time of the videos still to come. When output is a file or pipe (`docker compose run -T`, `| tee`) it prints one
plain `[progress]` line about every 5 %.

**Determinism.** `PYTHONHASHSEED=0`, numpy and torch seeded from `--seed`, CPU threads from `--threads` (default 4). Replay of a checkpoint is deterministic for every controller
(TD-MPC2: for a fixed device and seed). Smoke runs (`--smoke`) go to `results/smoke/`.

**Results layout.** Units are metres, seconds, radians; positions are in the `robot0_base_link` frame.

```
results/runs/<backend>/<model>/<reward>/<run_id>/        # backend: kinematic (velocity, position) or dynamics (torque)
    manifest.json  config_resolved.yaml  train_history.parquet  train_episodes.parquet  progress.json
    checkpoint_index.csv  video_index.csv
    checkpoints/step_000150000/{manifest.json, policy_state/}
    videos/step_000150000/val_001/{raw.mkv, preview.mp4, video_manifest.json, rollout.parquet}
    eval/<backend>/<suite>/step_XXXXXXXXX/{episode_metrics, action_events, trajectories}.parquet
    analysis/summary.csv   exports/trajectories/*.{csv,parquet}   play/
results/logs/<run_id>.log                                  # the output of each run started by run-all
```

The run manifest records the action mode, the state (`obs_version`), the backend, both git commits and whether the tree had uncommitted changes, the image fingerprint, library versions, and config and manifest hashes. `episode_metrics` has one row per task
(terminal hold, first entry, final error, path, duration, every descriptor, D1-D4, HKS, R-HKS, quality flag); `action_events` one row per decision (commanded and executed velocity, proposed
collision, rejected collision, clip proposal, raw and weighted reward terms, cost); `trajectories` one row per frame (joint angles, end-effector position, goal, velocity).

---

## 9. Tests

`python -m pytest -q` inside the container (`-m "not slow"` skips the tests that render real videos; tests marked as GPU-only skip themselves where there is no GPU). `tests/bench/` checks, against analytic references:

* **Metrics and scoring:** terminal-hold timing, minimum-jerk reference values, flags for short/zero/non-finite signals, the Nyquist guard, HKS scaling and zero-for-failure R-HKS, the scaler fallback ladder.
* **Statistics:** Benjamini-Hochberg against a hand example, paired-bootstrap alignment, rank probabilities, determinism.
* **Rewards and replay:** the historical r0 to 1e-12, dwell and first-hold logic, agreement of the stepwise and whole-history implementations, and that every relabelled replay sample equals an independent live rollout.
* **Controllers:** DecQN parity with the historical agent, Growing Q masking, TD-MPC2 windows and seeded replay, and each of the ten controllers training, checkpointing at exact transition counts, reloading and obeying its action contract on CPU and on CUDA.
* **Video and run integrity:** bit-exact lossless roundtrip, detection of corruption and of wrong, missing or lossy frames, deterministic re-render, validation-only checkpoint selection, calibration immutability, the programme totals (373 runs, 6,873 videos, 325M transitions for one cell; 4,053 runs, 62,073 videos, 2,901M transitions for the whole grid), the physics stratum, export, and that every command in this README parses with the real CLI.
* **Actions and states:** the eight state sizes (90, 99, 106, 130, 133, 108, 72, 150), the value and range of every new component, custom component lists, the replay rebuilding every state exactly as the environment showed it, the matched defaults, the exact equivalence of velocity and position steps on the kinematic backend, the torque dynamics (zero torque holds the pose to 2 mrad over 100 decisions, the speed limit, direction of motion, rejection leaves the arm at rest, determinism), the replay rebuilding `obs_v3` exactly, every controller family training and reloading under other actions and states, and the torque backend's videos and evaluation tables.
* **Run-all:** the resource plan (cores, memory, GPU), the order of the stages and calibration locks, the deferred scaler lock, stopping on a failed stage, parallel execution of command lines, and the overall progress arithmetic.
* **Historical parity:** the frozen original training loop under the benchmark checkpoint/video contract, whose converted policy picks the same actions as the native agent.

---

## 10. Repository layout

| Path | What it is |
| --- | --- |
| `natural_rl/` | the benchmark: `cli.py`, `train.py`, `hist.py`, `eval.py`, `rollout.py`, `replay.py`, `costs.py`, `selection.py`, `programme.py`, `stats.py`, `checkpoints.py`, `render.py`, `video.py`, `checkpoint_video.py`; `orchestrate.py` (`run-all`); `env/` (contract, fast, dynamics and physics environments, task manifests); `rewards/`; `metrics/` (outcomes, kinematics, HKS); `models/` (the ten controllers); `vendor/tdmpc2/` |
| `configs/` | `models/`, `rewards/`, `costs/`, `metrics/`, `render/`, `env/` (torque dynamics), `eval/` (manifests and the video task); the original `reach*.yaml` and `slice2.yaml` |
| `tests/` | `bench/` (benchmark) and the original suite |
| `compose.yaml`, `compose.gpu.yaml`, `.env.example`, `docker/` | the `rl` service, the CDI GPU override, the pinned image (`Dockerfile.natural`; `Dockerfile` is the original image) |
| `kangaroo_rl/`, `run_rl_docker.bash` | the original experiment and its launcher |
| `sim/Kangaroo_Libero` | the simulator, a pinned git submodule |
| `dgx_spark_slurm.sbatch`, `slurm/` (submit scripts, `progress.py`, `image_progress.py`), `docker/Dockerfile.natural-arm64`, `docker/build_arm64.bash` | running on the DGX Sparks: one Slurm job, the chained pipeline, the progress bar, the ARM64 image and its builder (section 12) |
| `pal-assets/` | PAL's robot model copied out of the simulator image, mounted at run time; git-ignored, never committed (section 12) |
| `results/`, `data/` | bind-mounted outputs (git-ignored) |

---

## 11. The original experiment

`kangaroo_rl` trains the goal-conditioned DecQN of section 3 on a fast model of the arm and evaluates it in the simulator. The model is the same compiled MuJoCo model as the simulator
(forward kinematics agrees to `10⁻⁹` m and the collision test agrees with full contact detection), but a model step costs a fraction of a simulator decision, so millions of transitions are
cheap. Goals are collision-free reach points over the table (a pool of 8,508 points thinned to one per 2 cm voxel); starts are 50 % where the previous episode ended (a retargeting
situation), 25 % home with noise, 25 % a random collision-free pose.

**Executing a decision in the simulator.** Kangaroo_Libero accepts only absolute joint-position targets at 100 Hz, so one decision is five ticks of a linearly interpolated reference
with a velocity feed-forward `τ_j=k_d/k_p` (none on the forearm roll, whose inertia is too small for it, nor on the two wrist joints, which are position actuators without a velocity term):

$$
q_{\text{ref}}(m)=q_0+\tfrac m5\,(q_{\text{cmd}}-q_0)
$$

$$
m=1..5
$$

$$
q_{\text{target}}=\mathrm{clip}\big(q_{\text{ref}}(m)+\tau_j\,v\big)
$$

Two observation modes exist: A (the policy sees the commanded positions) and B (the measured positions). Error is always measured on the simulated hand.

```bash
./run_rl_docker.bash --build --headless -- python -m pytest -q                 # builds kangaroo-rl:0.1.0 (simulator image + torch 2.5.1)
python -m kangaroo_rl.train --config configs/reach.yaml --seed 0              # runs/reach/seed0/{config.yaml,run.json,metrics.csv,train.csv,checkpoints/}
python -m kangaroo_rl.evaluate --run runs/reach/seed0 --backend kinematic     # every checkpoint on 256 held-out goals
python -m kangaroo_rl.evaluate --run runs/reach/seed0 --backend libero --modes A B --goals 64
python -m kangaroo_rl.evaluate --config configs/reach.yaml --baseline dls --backend kinematic   # Jacobian reference controllers
python -m kangaroo_rl.dp_check --run runs/slice2/seed0                        # learned Q vs exact value iteration on a 2-joint lattice
```

`configs/reach_raw.yaml` (raw angles instead of the sinusoidal encoding) and `configs/reach_noher.yaml` (no relabelling) are comparison configs; `configs/slice2.yaml` is the 2-joint
lattice on which the Bellman equation can be solved exactly by value iteration.

---

## 12. Running on the DGX Sparks (Slurm)

The lab cluster (three NVIDIA DGX Sparks, 20 ARM cores, one GB10 GPU and about 118 GB of RAM each; see the
[LabWiki](https://shu-sit-lab.github.io/LabWiki/getting-started/)) runs the benchmark as Slurm jobs in Enroot/Pyxis containers. There is no Docker on the
Sparks, and the Sparks are **ARM64**, so the x86 image of sections 1 and 10 cannot run there.

**The ARM64 image.** `kangaroo-natural-rl:0.3.0` is built on the simulator image, which is amd64-only (it sits on PAL Robotics' ROS image).
`docker/Dockerfile.natural-arm64` rebuilds the same pinned Python stack on plain Ubuntu 22.04 (the same pins; PyTorch 2.7.1 with the cu128 wheels, because the
GB10 is Blackwell). `natural_rl` never imports ROS, so nothing is lost. Build it on a machine with Docker, not on a Spark, and push it:

```bash
git submodule update --init
docker login -u aung9htet
./docker/build_arm64.bash --no-push     # build only, under QEMU on x86 (slow; registers ARM64 emulation with tonistiigi/binfmt)
./docker/build_arm64.bash               # build and push aung9htet/kangaroo-natural-rl:0.3.0-arm64
```

**The PAL robot model is not in the image.** The model (`/opt/pal/alum/share/kangaroo_mujoco`, 68 MB, and the derived `/opt/kangaroo-assets`) belongs to PAL Robotics
and comes from the private simulator image. Copy it out once into `pal-assets/` (git-ignored and docker-ignored) and the jobs mount it read-only:

```bash
cid=$(docker create --platform linux/amd64 aung9htet/kangaroo-libero:0.1.0-ubuntu-22.04)
docker cp $cid:/opt/pal/alum/share/kangaroo_mujoco pal-assets/kangaroo_mujoco
docker cp $cid:/opt/kangaroo-assets pal-assets/kangaroo-assets && docker rm $cid
```

**Setting up on the Spark.** The shared home is mounted on every node, so the checkout and the image file only need to be put there once.

```bash
rsync -a --exclude=.git ./ aung9htet@<spark-c3d8>:Kangaroo-RL-Control/      # code, sim/, slurm/, pal-assets/ (no git needed on the Spark)
mkdir -p ~/Kangaroo-RL-Control/logs ~/containers
# import the image once into a shared squashfs; jobs use it automatically, instead of re-importing 8 GB each
sbatch -A hpc -p gpu --time=02:00:00 --wrap='enroot import -o $HOME/containers/kangaroo-natural-rl-arm64.sqsh docker://aung9htet/kangaroo-natural-rl:0.3.0-arm64'
```

**One job.** `dgx_spark_slurm.sbatch` is one job on one Spark (a whole node: 20 cores, 110 GB, the GPU). The Slurm `logs/` folder must exist before
you submit.

```bash
sbatch dgx_spark_slurm.sbatch                                                            # the whole programme on one Spark
sbatch dgx_spark_slurm.sbatch 'python -u -m natural_rl run-all --phases P1 --actions torque --states v1,v2 --no-lock-scalers'
IMAGE='docker.io#aung9htet/kangaroo-natural-rl:0.3.0-arm64' sbatch dgx_spark_slurm.sbatch   # another image or a .sqsh path
```

The job mounts the checkout at `/workspace/Kangaroo-RL-Control` and the PAL files at their expected paths, and runs the quoted command (default
`run-all --device cuda --threads 2`). The `spark-c3d8` manager is excluded (`#SBATCH --exclude`) so that it stays free. Pyxis does not run the image ENTRYPOINT;
the environment (`PATH`, `KANGAROO_REPO`, `PYTHONPATH`, ...) is baked into the image instead.

**Several Sparks.** `run-all` is a single-node program, so a programme over several Sparks is split into chained jobs. The two global locks (reward
coefficients after P0, HKS scalers pooled over every cell) need every cell, so they are separate steps, and the split `run-all` jobs use
`--no-lock-scalers` so that no job fits the scalers from part of the cells.

```bash
P0_JOB=<id> SQSH_JOB=<id> WALL=7-00:00:00 ./slurm/submit_all_sparks.bash   # P0 -> 12 x P1 -> scaler lock -> 12 x P2 -> (scaler lock) -> P3 | P4
./slurm/submit_all_sparks.bash --dry-run                                   # print the sbatch lines only
```

P1 and P2 are cut into 12 chunks each (one action x two states, torque first), so Slurm starts a chunk on whichever Spark is free and the nodes finish together.
(`slurm/submit_two_sparks.bash` is the earlier version with two fixed halves.) If a job fails or times out, the jobs behind it stay blocked: read its `.err` in `logs/`,
fix the cause and submit that stage again; finished runs are skipped and an interrupted run starts again from step 0.

**Progress.** After `ssh`, with no container needed:

```bash
python3 slurm/progress.py --watch    # bars for transitions and runs, the speed, the time left, the running jobs; redraws in place every 10 s
python3 slurm/progress.py            # once
squeue -u $USER                      # the full queue (the display lists only running jobs and counts the rest);  scancel -u $USER  stops everything
tail -f logs/nrl-<stage>-<jobid>.out # run-all's own display for one stage; each run's log is in results/logs/<run-id>.log
```

`--watch` redraws on the normal screen (scrolling works) and needs a window taller than its frame, about 12 lines.

The bars count `results/runs/*/*/*/<run-id>/manifest.json` (finished) and `progress.json` (live step) against the size of the programme (4,053 runs,
2,901,000,000 transitions with P3 and P4 counted for one cell). Nothing moves while a job is still importing its image (the first lines of the `.err` file).

**Image download progress.** The first job on a node, or the shared-image import, downloads the 4.1 GB image from Docker Hub, which can take hours on the
Sparks' link. Enroot prints no percentage in a batch job, so `slurm/image_progress.py` reads the layer files that Enroot writes on the node
(`/shared/enroot/cache/user-<uid>/<layer-digest>.*`, node-local) through `srun --overlap` and compares them with the layer sizes in `slurm/image_layers.json`:

```bash
python3 slurm/image_progress.py <jobid> --watch   # one bar per layer and the total; <jobid> is the import job (or a job that is still importing)
```

`slurm/image_layers.json` belongs to the pushed image: regenerate it after pushing a new one (`docker buildx imagetools inspect <image>@<arm64-digest> --raw`,
digest -> size of every layer).

**Known limits.** The ARM64 image is a rebuild, not the validated amd64 one: run `python -m natural_rl verify-env --suite train_v1` in it before a long run.
The first job on a node imports the image, which can take many minutes on the shared link. EGL rendering (`MUJOCO_GL=egl`) is used for raw videos; if it fails in
Enroot, use `osmesa` for the whole benchmark.
