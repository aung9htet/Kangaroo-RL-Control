"""Immutable, disjoint task manifests: train_v1 (500), val_v1 (100), test_v1 (200), ood_v1 (100).

Every task fixes its start pose q0 (7 joint angles), the start reach point p0, the goal g (reach-point target in
the robot base frame, metres), the scene and the time budget (100 decisions of 0.05 s). Tasks are generated
deterministically from one seed and written to configs/eval/<split>_v1.json with a SHA256 of their canonical
content. Changing a rule or a seed means a new manifest ID, never an edit of these files.

Direct-reach tasks (train/val/test): collision-free start with the hand at least 3 cm above the table top (so a
straight line to a goal above the table clears the top surface), reach distance in [0.10, 0.50] m, and a goal
that a continuous Jacobian controller reaches within 1 cm (a feasibility FILTER only: that controller is never
trained, ranked or plotted). OOD tasks: 50 obstacle-detour tasks (start below the table top, goal above it, so
the straight line crosses the table edge that already exists in the scene) and 50 near-joint-limit tasks.
Near-duplicates are rejected across and within splits: start reach points within 2 cm AND goals within 2 cm.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SPLITS = ('train', 'val', 'test', 'ood')
COUNTS = {'train': 500, 'val': 100, 'test': 200, 'ood': 100}
SEED = 20261121
MANIFEST_VERSION = 'v1'
RULES = {
    'reach_distance_m': [0.10, 0.50],
    'start_min_height_above_table_m': 0.03,
    'near_duplicate_radius_m': 0.02,
    'oracle_tolerance_m': 0.01,
    'near_limit_fraction_of_range': 0.05,
    'horizon_decisions': 100,
    'dt_s': 0.05,
    'ood': {'detour': 50, 'near_limit': 50},
}


def canonical_sha(tasks: list) -> str:
    """SHA256 of the tasks in canonical form (sorted keys, floats rounded to 9 decimals)."""
    def clean(x):
        if isinstance(x, float):
            return round(x, 9)
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, dict):
            return {k: clean(v) for k, v in sorted(x.items())}
        return x
    return hashlib.sha256(json.dumps(clean(tasks), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _Generator:
    def __init__(self, exp, seed):
        from kangaroo_rl.baselines import DLSPolicy
        from kangaroo_rl.envs.kinematic import KinematicReachEnv
        from kangaroo_rl.scene import ArmKinematics

        self.exp, self.scene, self.seed = exp, exp.scene, seed
        self.kin = ArmKinematics(self.scene)
        self.top, self.edge = self.scene.table_top_base, self.scene.table_edge_base
        self.oracle = DLSPolicy(self.scene, exp.bins, quantise=False)
        self.env = KinematicReachEnv(self.scene, 32, exp.bins, exp.reward, exp.horizon, True)
        self.accepted_p0, self.accepted_goal = [], []

    # ---------------------------------------------------------------- sampling
    def _free_pose(self, rng, predicate, max_tries=2000):
        for _ in range(max_tries):
            q = rng.uniform(self.scene.lower, self.scene.upper)
            point, collided = self.kin.evaluate(q)
            if not collided and predicate(q, point):
                return q, point
        raise RuntimeError('no admissible start pose found; the start rule is too strict for this scene')

    def _near_limit_pose(self, rng, max_tries=4000):
        span = self.scene.upper - self.scene.lower
        for _ in range(max_tries):
            q = rng.uniform(self.scene.lower, self.scene.upper)
            j = rng.integers(len(q))
            side = rng.random() < 0.5
            offset = rng.uniform(0.0, RULES['near_limit_fraction_of_range']) * span[j]
            q[j] = self.scene.lower[j] + offset if side else self.scene.upper[j] - offset
            point, collided = self.kin.evaluate(q)
            if not collided and point[2] >= self.top + RULES['start_min_height_above_table_m']:
                return q, point
        raise RuntimeError('no near-limit start found')

    # ---------------------------------------------------------------- rules
    def _duplicate(self, p0, goal):
        r = RULES['near_duplicate_radius_m']
        for a, b in zip(self.accepted_p0, self.accepted_goal):
            if np.linalg.norm(a - p0) < r and np.linalg.norm(b - goal) < r:
                return True
        return False

    def _feasible(self, q0s, goals, waypoint=True):
        """Continuous Jacobian controller (a feasibility filter, never a benchmark policy) reaches <= 1 cm?"""
        self.oracle.waypoint = waypoint
        ok = np.zeros(len(q0s), dtype=bool)
        n = self.env.n
        for begin in range(0, len(q0s), n):
            rows = np.arange(begin, min(begin + n, len(q0s)))
            pad = np.r_[rows, np.full(n - len(rows), rows[-1])]
            self.env.reset_to(np.asarray(q0s)[pad], np.asarray(goals)[pad])
            q, g = self.env.observe()
            distance = None
            for _ in range(self.exp.horizon):
                step, _ = self.env.step(self.oracle(q, g))
                q, distance = step.q_next, step.distance
            ok[rows] = distance[:len(rows)] <= RULES['oracle_tolerance_m']
        return ok

    # ---------------------------------------------------------------- one split
    def _candidate(self, kind, rng):
        """One raw candidate (q0, p0, goal) of the given kind, or None if it breaks a cheap geometric rule."""
        pool = self.exp.pool
        if kind == 'direct':
            q0, p0 = self._free_pose(rng, lambda q, p: p[2] >= self.top + RULES['start_min_height_above_table_m'])
            goal = pool.sample(rng, 1)[0]
        elif kind == 'near_limit':
            q0, p0 = self._near_limit_pose(rng)
            goal = pool.sample(rng, 1)[0]
        elif kind == 'detour':
            q0 = np.clip(self.scene.home + rng.normal(0, 0.05, self.scene.n_joints), self.scene.lower, self.scene.upper)
            p0, collided = self.kin.evaluate(q0)
            goal = pool.sample(rng, 1)[0]
            # hand below the table top, goal above it and beyond the near edge: the straight line crosses the edge
            if collided or p0[2] >= self.top or not (goal[0] > self.edge and goal[2] > self.top):
                return None
        else:
            raise ValueError(kind)
        return q0, p0, goal

    def _generate_kind(self, split, kind, quota, rng, first_index):
        lo, hi = RULES['reach_distance_m']
        tasks = []
        while len(tasks) < quota:
            candidates = []
            for _ in range(64):
                made = self._candidate(kind, rng)
                if made is None:
                    continue
                q0, p0, goal = made
                reach = float(np.linalg.norm(goal - p0))
                if lo <= reach <= hi and not self._duplicate(p0, goal):
                    candidates.append((q0, p0, goal, reach))
            if not candidates:
                continue
            feasible = self._feasible([c[0] for c in candidates], [c[2] for c in candidates])
            for (q0, p0, goal, reach), ok in zip(candidates, feasible):
                if len(tasks) >= quota or not ok or self._duplicate(p0, goal):
                    continue
                self.accepted_p0.append(p0)
                self.accepted_goal.append(goal)
                tasks.append({'task_id': f'{split}_{first_index + len(tasks):03d}', 'split': split, 'category': kind,
                              'q0': [float(v) for v in q0], 'p0': [float(v) for v in p0],
                              'goal': [float(v) for v in goal], 'reach_distance_m': reach,
                              'horizon_decisions': RULES['horizon_decisions']})
        return tasks

    def generate(self, split, n, rng):
        plan = [('direct', n)] if split != 'ood' else [('detour', RULES['ood']['detour']),
                                                       ('near_limit', RULES['ood']['near_limit'])]
        tasks = []
        for kind, quota in plan:
            tasks += self._generate_kind(split, kind, quota, rng, len(tasks))
        if len(tasks) != n:
            raise RuntimeError(f'{split}: planned {len(tasks)} tasks, expected {n}')
        return tasks


def build_manifests(config: dict, out_dir, seed: int = SEED, splits=SPLITS) -> dict:
    """Generate and write every split. `config` is the parsed configs/reach.yaml (scene and goal definitions)."""
    from kangaroo_rl.experiment import make_experiment

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exp = make_experiment(config)
    gen = _Generator(exp, seed)
    code_sha = _file_sha(Path(__file__))
    written = {}
    for index, split in enumerate(SPLITS):
        if split not in splits:
            continue
        rng = np.random.default_rng([seed, index])
        tasks = gen.generate(split, COUNTS[split], rng)
        doc = {'manifest_id': f'{split}_{MANIFEST_VERSION}', 'split': split, 'generator_seed': seed,
               'generator_code_sha256': code_sha, 'scene_signature': exp.scene.signature(),
               'scene_settings': {k: exp.scene.settings[k] for k in sorted(exp.scene.settings)},
               'coordinate_frame': 'robot0_base_link (x forward, y left, z up), metres',
               'joint_order': list(exp.scene.joint_names), 'rules': RULES, 'n_tasks': len(tasks),
               'tasks_sha256': canonical_sha(tasks), 'tasks': tasks}
        path = out_dir / f'{split}_{MANIFEST_VERSION}.json'
        path.write_text(json.dumps(doc, indent=1) + '\n')
        written[split] = path
    return written


def load_manifest(path) -> dict:
    doc = json.loads(Path(path).read_text())
    if canonical_sha(doc['tasks']) != doc['tasks_sha256']:
        raise ValueError(f'{path}: tasks do not match their recorded SHA256 (manifest was edited)')
    return doc


def task_arrays(doc: dict) -> dict:
    tasks = doc['tasks']
    return {'task_id': [t['task_id'] for t in tasks], 'q0': np.array([t['q0'] for t in tasks]),
            'p0': np.array([t['p0'] for t in tasks]), 'goal': np.array([t['goal'] for t in tasks]),
            'category': [t['category'] for t in tasks]}


def find_task(doc: dict, task_id: str) -> dict:
    for task in doc['tasks']:
        if task['task_id'] == task_id:
            return task
    raise KeyError(f'task {task_id!r} not in manifest {doc["manifest_id"]}')


def verify_manifests(directory, expected_counts=COUNTS) -> list:
    """Return a list of problems (empty = all manifests intact, disjoint and rule-conforming)."""
    directory = Path(directory)
    problems, p0s, goals, ids = [], [], [], set()
    r = RULES['near_duplicate_radius_m']
    lo, hi = RULES['reach_distance_m']
    for split in SPLITS:
        path = directory / f'{split}_{MANIFEST_VERSION}.json'
        if not path.exists():
            problems.append(f'missing manifest {path}')
            continue
        try:
            doc = load_manifest(path)
        except ValueError as error:
            problems.append(str(error))
            continue
        if doc['n_tasks'] != expected_counts[split] or len(doc['tasks']) != expected_counts[split]:
            problems.append(f'{split}: {len(doc["tasks"])} tasks, expected {expected_counts[split]}')
        for t in doc['tasks']:
            if t['task_id'] in ids:
                problems.append(f'duplicate task id {t["task_id"]}')
            ids.add(t['task_id'])
            if not lo <= t['reach_distance_m'] <= hi:
                problems.append(f'{t["task_id"]}: reach distance {t["reach_distance_m"]:.3f} outside [{lo}, {hi}]')
            p0s.append(np.array(t['p0']))
            goals.append(np.array(t['goal']))
    if len(p0s) > 1:
        p0s, goals = np.array(p0s), np.array(goals)
        for i in range(len(p0s)):
            close = (np.linalg.norm(p0s[i + 1:] - p0s[i], axis=1) < r) & (np.linalg.norm(goals[i + 1:] - goals[i], axis=1) < r)
            if close.any():
                problems.append(f'near-duplicate tasks: index {i} and {i + 1 + int(np.argmax(close))}')
    return problems
