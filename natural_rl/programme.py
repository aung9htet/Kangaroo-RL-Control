"""The locked training programme (protocol section 09): phases, seeds, run ids, commands and totals.

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


def reward_number(profile: str) -> str:
    return profile.split('_')[0]                                       # 'r7_balanced' -> 'r7'


def run_id(model, profile, seed, tag='', budget=None, hist=False) -> str:
    reward = 'r0hist' if hist else reward_number(profile)
    middle = f'-{BUDGETS[budget]}' if budget else ''
    return f'{model}-{reward}{middle}-s{seed:02d}{("-" + tag) if tag else ""}'


def phase_runs(phase: str, policies=None) -> list:
    """Registered runs of a phase as dicts {run_id, model, reward, seed, steps, budget, contract, phase}."""
    runs = []
    def add(model, profile, seed, steps, tag='', budget=None, hist=False):
        runs.append({'phase': phase, 'run_id': run_id(model, profile, seed, tag, budget, hist), 'model': model, 'reward': profile,
                     'seed': seed, 'steps': steps, 'budget': budget, 'contract': 'hist_v1' if hist else 'bench_v1'})
    if phase == 'P0':
        for seed in range(3):
            add('decqn', 'r0_original', seed, 1_000_000, hist=True)
    elif phase == 'P1':
        for model in MODEL_IDS:
            for profile in ANCHORS:
                for seed in range(3):
                    add(model, profile, seed, 200_000, tag='p1', budget='budget_moderate_v1' if model == 'sac_pidlag' else None)
    elif phase == 'P2':
        for model in MODEL_IDS:
            for profile in ANCHORS:
                for seed in range(5):
                    add(model, profile, seed, 1_000_000, budget='budget_moderate_v1' if model == 'sac_pidlag' else None)
    elif phase == 'P3':
        if not policies or len(policies) != 4 or policies[0] != 'decqn':
            raise ValueError('P3 needs --policies decqn,A,B,C: DecQN plus the three non-DecQN controllers selected on '
                             'validation terminal-hold rate (unsafe-proposal fraction breaks ties), before any test result')
        for model in policies:
            for profile in P3_PROFILES:
                for seed in range(5):
                    add(model, profile, seed, 1_000_000, budget='budget_moderate_v1' if model == 'sac_pidlag' else None)
    elif phase == 'P4':
        for budget in BUDGETS:
            for seed in range(5, 10):
                add('sac_pidlag', 'r7_balanced', seed, 1_000_000, budget=budget)
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
    return cmd


def totals(runs: list) -> dict:
    return {'runs': len(runs), 'transitions': sum(r['steps'] for r in runs),
            'videos': sum(len(scheduled_steps(r['steps'], CHECKPOINT_EVERY)) for r in runs)}


def programme_totals(p3_policies=('decqn', 'ppo_md', 'sac', 'td3')) -> dict:
    """Per-phase and grand totals. The P3 controllers are placeholders here: totals do not depend on which three."""
    per = {p: totals(phase_runs(p, p3_policies if p == 'P3' else None)) for p in ('P0', 'P1', 'P2', 'P3', 'P4')}
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
