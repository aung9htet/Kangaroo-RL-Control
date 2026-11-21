"""The common action and observation contract shared by all ten controllers.

Every controller emits a normalised command u in [-1, 1]^7 (a Box action, or a MultiDiscrete bin index mapped to a
fraction of the maximum). What u MEANS is the action mode, one of three:
  * velocity : joint velocity u * v_max (v_max = 1 rad/s, a benchmark contract: the simulator itself enforces only joint
               position limits); q' = clip(q + dt * v_max * u)
  * position : joint position step, target q + u * v_max * dt. In the fast kinematic backend this is the same transition
               as velocity (an exact equivalence, kept as a registered mode so it is measured, and used as the
               physics-evaluation command without velocity feed-forward)
  * torque   : joint torque u * tau_max (gravity-compensated arm dynamics, `env/dynamics.py`), joint speed limited to v_max
  * MultiDiscrete(7,...,7) : bin index k_j -> LEVELS[k_j]   (the historical bins, unchanged)
  * Box[-1, 1]^7           : a_j -> clip(a_j, -1, 1)
Everywhere in the code the command is carried as `u * v_max` (named v_cmd), so the effort and command-change reward
terms, the previous-command observation and the cost accounting are identical for every mode.
Observation: a state is a list of named components, concatenated in a fixed order, so every algorithm sees an identical, documented
feature map (README, section 2). The components are q, goal, prev_cmd, dwell, qd, ee, goal_rel, ee_vel, margin, table and time;
the registered states v1..v8 are named sets of them (`STATE_PRESETS`), and `--obs q,goal,ee` builds any other set. Features that
depend on the goal are recomputed for relabelled goals by the replay buffer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kangaroo_rl.encoding import encoded_size, normalise, positional_encoding

LEVELS = (-1.0, -0.3, -0.1, 0.0, 0.1, 0.3, 1.0)
N_BINS = len(LEVELS)
ACTION_MODES = ('velocity', 'position', 'torque')
COMPONENTS = ('q', 'goal', 'prev_cmd', 'dwell', 'qd', 'ee', 'goal_rel', 'ee_vel', 'margin', 'table', 'time')   # canonical order
_HISTORY = ('prev_cmd', 'dwell')
STATE_PRESETS = {
    'v1': ('q', 'goal'),                                                          # joint angles and goal (the original input)
    'v2': ('q', 'goal') + _HISTORY,                                               # + previous command and hold state
    'v3': ('q', 'goal') + _HISTORY + ('qd',),                                     # + joint velocity
    'v4': ('q', 'goal') + _HISTORY + ('ee', 'goal_rel'),                          # + forward-kinematics hand position and goal error
    'v5': ('q', 'goal') + _HISTORY + ('ee', 'goal_rel', 'ee_vel'),                # + hand velocity
    'v6': ('q', 'goal') + _HISTORY + ('margin', 'table'),                         # + joint-limit margins and table geometry
    'v7': ('goal',) + _HISTORY + ('ee', 'goal_rel', 'ee_vel', 'table'),           # hand-centric: no joint angles at all
    'v8': COMPONENTS,                                                             # everything
}
OBS_VERSIONS = tuple(STATE_PRESETS)


def parse_state(version: str) -> tuple:
    """Components of a state: a registered id (v1..v8) or a comma-separated list of component names, in canonical order."""
    if version in STATE_PRESETS:
        return STATE_PRESETS[version]
    names = [n for n in str(version).split(',') if n]
    unknown = [n for n in names if n not in COMPONENTS]
    if not names or unknown:
        raise ValueError(f'unknown state {version!r}: use one of {list(STATE_PRESETS)} or a comma-separated list of {list(COMPONENTS)}')
    return tuple(c for c in COMPONENTS if c in names)
DEFAULT_OBS = {'velocity': 'v2', 'position': 'v2', 'torque': 'v3'}      # matched state of each action when --obs is not given


def backend_of(action_mode: str) -> str:
    """Torque runs on the arm-dynamics backend; velocity and position on the fast kinematic backend."""
    if action_mode not in ACTION_MODES:
        raise ValueError(f'unknown action mode {action_mode!r}; modes: {ACTION_MODES}')
    return 'dynamics' if action_mode == 'torque' else 'kinematic'


def multidiscrete_to_velocity(actions, v_max: float = 1.0) -> np.ndarray:
    """Bin indices (..., J) -> commanded velocities (..., J) in rad/s."""
    actions = np.asarray(actions)
    if not np.issubdtype(actions.dtype, np.integer):
        raise TypeError('MultiDiscrete actions must be integer bin indices')
    if actions.min(initial=0) < 0 or actions.max(initial=0) >= N_BINS:
        raise ValueError(f'bin index outside 0..{N_BINS - 1}')
    return v_max * np.asarray(LEVELS)[actions]


def box_to_velocity(actions, v_max: float = 1.0) -> np.ndarray:
    """Normalised Box actions (..., J) -> commanded velocities (..., J) in rad/s; |v| <= v_max always."""
    return v_max * np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)


def box_was_clipped(actions) -> np.ndarray:
    """Per-decision flag: did the policy ask for more than the actuator limit (logged, separate from joint clipping)."""
    return (np.abs(np.asarray(actions)) > 1.0).any(axis=-1)


@dataclass(frozen=True)
class ObsSpec:
    version: str
    q_lower: np.ndarray
    q_upper: np.ndarray
    goal_lower: np.ndarray
    goal_upper: np.ndarray
    n_joints: int = 7
    joint_freqs: int = 4
    goal_freqs: int = 4
    hold_steps: int = 5
    v_max: float = 1.0
    action_mode: str = 'velocity'
    table_top: float | None = None        # height of the table top in the robot base frame (m)
    table_edge: float | None = None       # x of the table's near edge in the base frame (m)
    horizon: int = 100
    dt: float = 0.05
    distance_scale: float = 0.5           # m: goal_rel and table features saturate beyond this distance
    ee_speed_scale: float = 1.0           # m/s: hand velocity is divided by this and clipped to [-1, 1]

    def __post_init__(self):
        if self.action_mode not in ACTION_MODES:
            raise ValueError(f'unknown action mode {self.action_mode!r}')
        parse_state(self.version)

    @classmethod
    def for_scene(cls, version: str, scene, pool, action_mode: str = 'velocity', **kwargs) -> 'ObsSpec':
        """The spec of a scene: joint and goal ranges, and the table geometry that the `table` component needs."""
        return cls(version, scene.lower, scene.upper, pool.lower, pool.upper, action_mode=action_mode,
                   table_top=float(scene.table_top_base), table_edge=float(scene.table_edge_base), **kwargs)

    @property
    def components(self) -> tuple:
        return parse_state(self.version)

    @property
    def backend(self) -> str:
        return backend_of(self.action_mode)

    def sizes(self) -> dict:
        j, g = self.n_joints, 3
        return {'q': encoded_size(j, self.joint_freqs), 'goal': encoded_size(g, self.goal_freqs), 'prev_cmd': j, 'dwell': 2, 'qd': j,
                'ee': encoded_size(g, self.goal_freqs), 'goal_rel': 4, 'ee_vel': 3, 'margin': j, 'table': 2, 'time': 1}

    @property
    def dim(self) -> int:
        sizes = self.sizes()
        return sum(sizes[c] for c in self.components)

    def encode(self, q, goal, v_prev=None, count=None, held_once=None, qd=None, p=None, p_prev=None, step=None) -> np.ndarray:
        """Flat float32 observation(s) of the components of this state.

        q (..., J) and goal (..., 3) are always given. v_prev (..., J), count and held_once (...) feed prev_cmd and dwell; qd (..., J) the
        executed joint velocity; p (..., 3) the hand position (forward kinematics) and p_prev the position one decision earlier (ee, goal_rel,
        ee_vel, table); step the number of decisions already taken in the episode (time).
        """
        q = np.asarray(q, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        def need(value, name, comp):
            if value is None:
                raise ValueError(f'state {self.version!r} needs {name} for the {comp!r} component')
            return value
        parts = []
        for comp in self.components:
            if comp == 'q':
                parts.append(positional_encoding(normalise(q, self.q_lower, self.q_upper), self.joint_freqs))
            elif comp == 'goal':
                parts.append(positional_encoding(normalise(goal, self.goal_lower, self.goal_upper), self.goal_freqs))
            elif comp == 'prev_cmd':
                parts.append(np.asarray(need(v_prev, 'v_prev', comp), dtype=np.float64) / self.v_max)
            elif comp == 'dwell':
                count_, held_ = need(count, 'count', comp), need(held_once, 'held_once', comp)
                parts += [(np.minimum(np.asarray(count_), self.hold_steps) / self.hold_steps)[..., None],
                          np.asarray(held_, dtype=np.float64)[..., None]]
            elif comp == 'qd':
                parts.append(np.clip(np.asarray(need(qd, 'qd', comp), dtype=np.float64) / self.v_max, -1.0, 1.0))
            elif comp == 'ee':
                parts.append(positional_encoding(normalise(np.asarray(need(p, 'p', comp), dtype=np.float64), self.goal_lower, self.goal_upper),
                                                 self.goal_freqs))
            elif comp == 'goal_rel':
                delta = goal - np.asarray(need(p, 'p', comp), dtype=np.float64)
                parts += [np.clip(delta / self.distance_scale, -1.0, 1.0),
                          np.minimum(np.linalg.norm(delta, axis=-1) / self.distance_scale, 1.0)[..., None]]
            elif comp == 'ee_vel':
                move = np.asarray(need(p, 'p', comp), dtype=np.float64) - np.asarray(need(p_prev, 'p_prev', comp), dtype=np.float64)
                parts.append(np.clip(move / (self.dt * self.ee_speed_scale), -1.0, 1.0))
            elif comp == 'margin':
                parts.append(2.0 * np.minimum(q - self.q_lower, self.q_upper - q) / (self.q_upper - self.q_lower))
            elif comp == 'table':
                if self.table_top is None or self.table_edge is None:
                    raise ValueError("the 'table' component needs the table geometry: build the spec with ObsSpec.for_scene")
                hand = np.asarray(need(p, 'p', comp), dtype=np.float64)
                parts.append(np.stack([np.clip((hand[..., 2] - self.table_top) / self.distance_scale, -1.0, 1.0),
                                       np.clip((hand[..., 0] - self.table_edge) / self.distance_scale, -1.0, 1.0)], axis=-1))
            elif comp == 'time':
                done = np.broadcast_to(np.asarray(need(step, 'step', comp), dtype=np.float64), q.shape[:-1])
                parts.append(np.clip(done / self.horizon, 0.0, 1.0)[..., None])
        return np.concatenate(parts, axis=-1).astype(np.float32)
