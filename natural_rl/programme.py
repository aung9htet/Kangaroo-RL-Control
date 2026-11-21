"""The locked training programme (protocol section 09): phases, seeds, run ids, commands and totals.

Every run belongs to a CELL = (action mode, state): the action says what the policy output means (velocity, position step,
torque) and the state what it sees (obs_v1, obs_v2, obs_v3). P1 and P2 cover every cell (the screening and the main
comparison of "which action and state give the most human-like movement"); P3 and P4 run on the best cell(s) chosen from P2
validation data (`catalog programme --phase P3 --select`).

Pure arithmetic over registered facts; it contains no runtime or result. `catalog programme` prints it and every
registered `train` command, so the whole programme can be launched run by run from the container shell.
"""
from __future__ import annotations

from .checkpoints import scheduled_steps
from .models.registry import MODEL_IDS

ANCHORS = ('r1_hold', 'r7_balanced')
ALL_PROFILES = ('r0_original', 'r1_hold', 'r2_time', 'r3_path', 'r4_speed', 'r5_smooth', 'r6_rom', 'r7_balanced',
                'r8_drop_path', 'r9_drop_speed', 'r10_drop_smooth', 'r11_drop_rom')
P3_PROFILES = tuple(p for p in ALL_PROFILES if p not in ANCHORS)       # the 10 additional profiles
BUDGETS = {'budget_moderate_v1': 'bmod', 'budget_strict_v1': 'bstrict'}
CHECKPOINT_EVERY = 50_000
ACTIONS = ('velocity', 'position', 'torque')
STATES = ('v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8')
LEGACY_CELL = ('velocity', 'v2')                      # the cell whose run ids carry no tag (the first runs of the programme)
ALL_CELLS = (LEGACY_CELL,) + tuple((a, s) for a in ACTIONS for s in STATES if (a, s) != LEGACY_CELL)   # the legacy cell runs first
_ACTION_TAG = {'velocity': '', 'position': 'pos', 'torque': 'tor'}
_STATE_TAG = {'v1': 'o1', 'v2': '', 'v3': 'o3', 'v4': 'o4', 'v5': 'o5', 'v6': 'o6', 'v7': 'o7', 'v8': 'o8'}


def cell_tag(action: str, state: str) -> str:
    return '-'.join(t for t in (_ACTION_TAG[action], _STATE_TAG[state]) if t)


def parse_cells(actions=None, states=None) -> tuple:
    """Cells from comma-separated (or sequence) action and state filters; None means all of them."""
    split = lambda x: [i for i in (x.split(',') if isinstance(x, str) else (x or [])) if i]
    a, st = split(actions) or list(ACTIONS), split(states) or list(STATES)
    bad = [x for x in a if x not in ACTIONS] + [x for x in st if x not in STATES]
    if bad:
        raise ValueError(f'unknown action or state {bad}; actions {ACTIONS}, states {STATES}')
    return tuple(sorted(((x, y) for x in a for y in st), key=ALL_CELLS.index))


def reward_number(profile: str) -> str:
    return profile.split('_')[0]                                       # 'r7_balanced' -> 'r7'


def run_id(model, profile, seed, tag='', budget=None, hist=False, cell=LEGACY_CELL) -> str:
    reward = 'r0hist' if hist else reward_number(profile)
    middle = f'-{BUDGETS[budget]}' if budget else ''
    ctag = cell_tag(*cell)
    return f'{model}-{reward}{middle}{("-" + ctag) if ctag else ""}-s{seed:02d}{("-" + tag) if tag else ""}'


def phase_runs(phase: str, policies=None, cells=None) -> list:
    """Registered runs of a phase as dicts {run_id, model, reward, seed, steps, budget, contract, phase, action, state}.

    `cells` are (action, state) pairs. P0 is always the historical velocity / obs_v1 reproduction. P1 and P2 default to every
    cell; P3 and P4 default to the legacy (velocity, obs_v2) cell and are normally given the cell(s) chosen from P2.
    """
    runs = []
    def add(model, profile, seed, steps, tag='', budget=None, hist=False, cell=LEGACY_CELL):
        runs.append({'phase': phase, 'run_id': run_id(model, profile, seed, tag, budget, hist, cell), 'model': model, 'reward': profile,
                     'seed': seed, 'steps': steps, 'budget': budget, 'contract': 'hist_v1' if hist else 'bench_v1',
                     'action': 'velocity' if hist else cell[0], 'state': 'v1' if hist else cell[1]})
    budget_of = lambda model: 'budget_moderate_v1' if model == 'sac_pidlag' else None
    if phase == 'P0':
        for seed in range(3):
            add('decqn', 'r0_original', seed, 1_000_000, hist=True)
    elif phase in ('P1', 'P2'):
        for cell in (cells or ALL_CELLS):
            for model in MODEL_IDS:
                for profile in ANCHORS:
                    for seed in range(3 if phase == 'P1' else 5):
                        add(model, profile, seed, 200_000 if phase == 'P1' else 1_000_000, tag='p1' if phase == 'P1' else '',
                            budget=budget_of(model), cell=cell)
    elif phase == 'P3':
        if not policies or len(policies) != 4 or policies[0] != 'decqn':
            raise ValueError('P3 needs --policies decqn,A,B,C: DecQN plus the three non-DecQN controllers selected on '
                             'validation terminal-hold rate (unsafe-proposal fraction breaks ties), before any test result')
        for cell in (cells or (LEGACY_CELL,)):
            for model in policies:
                for profile in P3_PROFILES:
                    for seed in range(5):
                        add(model, profile, seed, 1_000_000, budget=budget_of(model), cell=cell)
    elif phase == 'P4':
        for cell in (cells or (LEGACY_CELL,)):
            for budget in BUDGETS:
                for seed in range(5, 10):
                    add('sac_pidlag', 'r7_balanced', seed, 1_000_000, budget=budget, cell=cell)
    else:
        raise KeyError(f'unknown phase {phase!r}; phases are P0 P1 P2 P3 P4')
    return runs


def train_command(run: dict, suite='train_v1') -> str:
    cmd = (f'python -m natural_rl train --model {run["model"]} --reward {run["reward"]} --suite {suite} --steps {run["steps"]} '
           f'--seed {run["seed"]} --run-id {run["run_id"]} --checkpoint-every {CHECKPOINT_EVERY} --record-every-checkpoint')
    if run['budget']:
        cmd += f' --cost-budget-id {run["budget"]}'
    if run['contract'] == 'hist_v1':
        cmd += ' --contract hist_v1'
    else:
        if run['action'] != 'velocity':
            cmd += f' --action {run["action"]}'
        if run['state'] != {'velocity': 'v2', 'position': 'v2', 'torque': 'v3'}[run['action']]:
            cmd += f' --obs {run["state"]}'
    return cmd


def totals(runs: list) -> dict:
    return {'runs': len(runs), 'transitions': sum(r['steps'] for r in runs),
            'videos': sum(len(scheduled_steps(r['steps'], CHECKPOINT_EVERY)) for r in runs)}


def programme_totals(p3_policies=('decqn', 'ppo_md', 'sac', 'td3'), cells=None, later_cells=None) -> dict:
    """Per-phase and grand totals. The P3 controllers are placeholders here: totals do not depend on which three.
    `cells` are the P1/P2 cells (default all); `later_cells` the P3/P4 cells (default one cell)."""
    per = {p: totals(phase_runs(p, p3_policies if p == 'P3' else None, later_cells if p in ('P3', 'P4') else cells))
           for p in ('P0', 'P1', 'P2', 'P3', 'P4')}
    per['TOTAL'] = {k: sum(v[k] for v in per.values()) for k in ('runs', 'transitions', 'videos')}
    return per


def run_lines(run: dict, with_eval: bool = True) -> list:
    """The ordered lines to run for one registered run: train (which also records every checkpoint video), then verify and evaluate.

    P0 only trains and verifies (its rollouts calibrate the reward coefficients). P1 adds the final-checkpoint validation evaluation
    that the HKS scalers are fitted from. P2-P4 add validation on every checkpoint, the test evaluation of the validation-selected
    checkpoint and the summary.
    """
    rid = run['run_id']
    lines = [train_command(run)]
    if not with_eval:
        return lines
    lines.append(f'python -m natural_rl video verify --run-id {rid} --all-checkpoints')
    if run['phase'] == 'P1':
        lines.append(f'python -m natural_rl eval --run-id {rid} --suite val_v1 --final')
    elif run['phase'] in ('P2', 'P3', 'P4'):
        lines += [f'python -m natural_rl eval --run-id {rid} --suite val_v1 --all-checkpoints',
                  f'python -m natural_rl eval --run-id {rid} --suite test_v1 --manifest configs/eval/test_v1.json',
                  f'python -m natural_rl analyse --run-id {rid}']
    return lines


def pending_lines(run: dict, runs_root, with_eval: bool = True) -> list:
    """`run_lines` minus what the results directory already holds (for `catalog programme --todo`).

    A run whose manifest says `complete` needs no `train` line; a run whose expected outputs all exist (every video `ok`, the
    validation evaluation of P1, the test evaluation and summary of P2-P4) needs nothing. Runs that are absent or unfinished keep
    every line, so `train` stops loudly on an unfinished folder instead of this function guessing what to do with it.
    """
    import csv
    import json
    from pathlib import Path
    lines = run_lines(run, with_eval)
    matches = list(Path(runs_root).glob(f'*/*/*/{run["run_id"]}'))
    if len(matches) != 1 or not (matches[0] / 'manifest.json').exists():
        return lines
    out = matches[0]
    if json.loads((out / 'manifest.json').read_text()).get('status') != 'complete':
        return lines
    lines = lines[1:]
    if not with_eval:
        return []
    try:
        videos_ok = all(r['status'] == 'ok' for r in csv.DictReader(open(out / 'video_index.csv')))
    except OSError:
        videos_ok = False
    evaluated = {'P0': True,
                 'P1': any((out / 'eval').glob('*/val_v1/step_*/episode_metrics.parquet')),
                 }.get(run['phase'], any((out / 'eval').glob('*/test_v1/step_*/episode_metrics.parquet')) and (out / 'analysis' / 'summary.csv').exists())
    return [] if videos_ok and evaluated else lines
