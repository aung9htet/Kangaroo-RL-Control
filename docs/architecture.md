# Architecture: how the code is put together

The mathematics is in [`../TECHNICAL.md`](../TECHNICAL.md); this page is about the code. One rule shapes it:
**the MDP is defined once** (`kangaroo_rl/mdp.py`) and used, unchanged, by the fast model, the simulator
adapter, the replay buffer (hindsight rewards), the exact dynamic programming and the reference controllers.

## 1. Data flow

```
          configs/*.yaml ──► experiment.py ──► Scene (scene.py) ◄── Kangaroo_Libero make_env (once)
                                  │               │  compiled MuJoCo model, 7 joints, limits, home,
                                  │               │  reach point, table, collision-filtered copy
                                  ▼               ▼
        mdp.py: VelocityBins · TransitionRule · Reward · GoalPool · StartSampler
              │                         │                              │
   training   ▼                         ▼ evaluation                   ▼ checks
   envs/kinematic.py             envs/libero.py                    dp.py (2-joint lattice)
   16 MjData copies,             KangarooEnv, 5 ticks/decision,    value iteration ──► dp_check.py
   exact T, collision            feed-forward, modes A/B,          baselines.py (Jacobian, random)
        │                        bowl appears under the goal
        ▼                                │
   agent/replay.py (raw q, a, q', c, p'; HER at sample time)
        │                                │
        ▼                                ▼
   agent/decqn.py ◄── agent/network.py   evaluation.py ──► metrics.py
   (Double-Q target, Huber, Polyak,       (model: home / chain; simulator: object goals)
    explorer)
        │
        ▼
   train.py ──► runs/<name>/seed<k>/{metrics.csv, train.csv, run.json, checkpoints/}
   evaluate.py, dp_check.py, retarget_demo.py ──► <run>/eval, <run>/dp, docs/results
   scripts/make_figures.py ──► docs/figures, docs/results/figures, docs/results/data
```

## 2. Modules

| Module | Responsibility | Key objects |
| --- | --- | --- |
| `scene.py` | builds the LIBERO scene through Kangaroo_Libero (with the settings override), settles the home pose, reads the 7 joints, limits, actuator kinds, reach point, base frame and table; makes the kinematic copy whose collision bits test only right arm against table/body | `build_scene`, `Scene`, `ArmKinematics` (FK, collision test, Jacobian) |
| `encoding.py` | sin/cos positional encoding (NumPy and torch, identical) | `positional_encoding`, `positional_encoding_torch` |
| `mdp.py` | the MDP | `VelocityBins`, `TransitionRule`, `step_model`, `Reward`, `GoalRegion`, `GoalPool`, `StartSampler` |
| `envs/kinematic.py` | n synchronous copies of the model, explicit starts and goals, optional active-joint subset | `KinematicReachEnv`, `Step` |
| `envs/libero.py` | decisions executed by KangarooEnv; object placement; modes A/B | `LiberoReachEnv`, `PhysicsStep` |
| `agent/network.py` | encoding inside the network, trunk, 7 × 7 utilities | `DecQNetwork` |
| `agent/decqn.py` | factored Double-Q target, Huber loss, Polyak target, greedy action, explorer | `DecQNAgent`, `Explorer`, `joint_q` |
| `agent/replay.py` | episodes of raw transitions, hindsight goals and reward recomputation at sample time | `EpisodeReplay` |
| `experiment.py` | config → scene, bins, reward, goal pool, network settings | `make_experiment`, `Experiment` |
| `evaluation.py`, `metrics.py` | model and simulator evaluation, object goals, measures | `KinematicEvaluator`, `physics_run`, `object_goals`, `summarise` |
| `dp.py`, `dp_check.py` | 2-joint lattice, transition tables, value iteration, comparison with the agent | `build_lattice`, `value_iteration`, `compare` |
| `baselines.py` | random and damped-least-squares controllers | `RandomPolicy`, `DLSPolicy` |
| `train.py`, `evaluate.py`, `retarget_demo.py` | command-line entry points | |
| `runinfo.py` | commits, image ID, versions, settings, written into every run | `run_metadata` |
| `viz/` | plot style, diagrams, offscreen renderer with goal/path overlays | |

## 3. Choices that matter

- **Training runs on the model, evaluation on the simulator.** The model is the same compiled MuJoCo model
  as the simulator (FK agrees to 1e-9 m, the collision test agrees with full contact detection), but a step
  is forward kinematics plus a filtered collision test, about 0.1-0.2 ms. A simulator decision is 5 physics
  ticks plus robosuite bookkeeping, about 25 ms. Training in the simulator would be roughly 100 times slower.
- **Raw values in the replay buffer.** The encoding is part of the network. Hindsight relabelling needs the
  achieved reach point and the rejection flag, so those are stored; rewards are not.
- **Synchronous episodes.** All copies start and end together, and an episode is a fixed block of 100 steps.
  Hindsight goals are then a simple index into the same block.
- **No changes to Kangaroo_Libero.** The placement override is a generated settings file
  (`.runtime/simulation-<hash>.yaml`, via `KANGAROO_CONFIG`). Objects are moved through the MuJoCo state.
  Commands go through `KangarooEnv.step`, so the simulator's own validation and controller are used.
- **Caches.** Goal pools, object goals and the lattice are cached under `.runtime/cache/`, keyed by a hash
  of the scene and the settings, so concurrent runs and later evaluations use identical sets.
