"""Registered selections made from P2 VALIDATION tables only (test results can neither be read nor influence a choice; a test checks this).

Cell selection (P3 and P4 run on it): every (action, state) cell is scored by the mean validation R-HKS of its P2 `r7_balanced`
runs at their validation-selected checkpoints (R-HKS = terminal hold x human-likeness, so it needs the locked HKS scalers); ties are
broken by the lower unsafe-event rate per decision. The best `top` cells are chosen.

P3 controller selection: DecQN plus the three non-DecQN controllers with the highest VALIDATION terminal-hold rate.

Ranking is on `r7_balanced`, averaged over the five P2 seeds, each seed at its own validation-selected checkpoint; ties
are broken by the lower unsafe-event rate per decision. Only `val_v1` evaluation tables on the training (kinematic)
backend are read; test results can neither be read nor influence the choice (a test checks this).
"""
from __future__ import annotations

import pandas as pd

from .checkpoints import step_name
from .eval import select_checkpoint
from .models.registry import MODEL_IDS
from .programme import LEGACY_CELL, run_id
from .registry import find_run


def _val_frame(run, backend):
    step = select_checkpoint(run, backend, 'val_v1')
    return pd.read_parquet(run / 'eval' / backend / 'val_v1' / step_name(step) / 'episode_metrics.parquet')


def best_cells(cells, models=MODEL_IDS, seeds=range(5), top: int = 1, horizon: int = 100):
    """(chosen cells, table): rank (action, state) cells by mean validation R-HKS of their P2 r7_balanced runs."""
    from .env.contract import backend_of
    from .eval import load_scalers, score_frame
    rows, missing = [], []
    for cell in cells:
        backend = backend_of(cell[0])
        scalers = load_scalers(backend)
        if scalers is None:
            raise RuntimeError(f'cell selection needs R-HKS, so the HKS scalers for the {backend} backend must be locked first '
                               '(python -m natural_rl calibrate scalers ...)')
        scores, unsafe, holds = [], [], []
        for model in models:
            for seed in seeds:
                rid = run_id(model, 'r7_balanced', seed, budget='budget_moderate_v1' if model == 'sac_pidlag' else None, cell=cell)
                try:
                    df = score_frame(_val_frame(find_run(rid), backend), scalers)
                except FileNotFoundError as error:
                    missing.append(f'{rid}: {error}')
                    continue
                scores.append(float(df['R_HKS'].mean()))
                holds.append(float(df['terminal_hold'].mean()))
                unsafe.append(float(df['unsafe_events'].mean()) / horizon)
        if scores:
            rows.append({'action': cell[0], 'state': cell[1], 'val_R_HKS': sum(scores) / len(scores),
                         'val_terminal_hold_rate': sum(holds) / len(holds), 'unsafe_events_per_decision': sum(unsafe) / len(unsafe),
                         'n_runs': len(scores)})
    if missing and not rows:
        raise FileNotFoundError('cell selection needs P2 r7_balanced runs with a validation evaluation; missing:\n  ' + '\n  '.join(missing[:20]))
    table = pd.DataFrame(rows).sort_values(['val_R_HKS', 'unsafe_events_per_decision'], ascending=[False, True], ignore_index=True)
    chosen = [(r.action, r.state) for r in table.head(top).itertuples()]
    return chosen, table


def p3_selection(models=MODEL_IDS, seeds=range(5), horizon: int = 100, cell=LEGACY_CELL):
    """(['decqn', A, B, C], table). Raises with the list of missing runs/evaluations if the P2 data is incomplete."""
    from .env.contract import backend_of
    backend = backend_of(cell[0])
    rows, missing = [], []
    for model in models:
        if model == 'decqn':
            continue
        holds, unsafe = [], []
        for seed in seeds:
            rid = run_id(model, 'r7_balanced', seed, budget='budget_moderate_v1' if model == 'sac_pidlag' else None, cell=cell)
            try:
                df = _val_frame(find_run(rid), backend)
            except FileNotFoundError as error:
                missing.append(f'{rid}: {error}')
                continue
            holds.append(float(df['terminal_hold'].mean()))
            unsafe.append(float(df['unsafe_events'].mean()) / horizon)
        if len(holds) == len(list(seeds)):
            rows.append({'model': model, 'val_terminal_hold_rate': sum(holds) / len(holds),
                         'unsafe_events_per_decision': sum(unsafe) / len(unsafe), 'n_seeds': len(holds)})
    if missing:
        raise FileNotFoundError('P3 selection needs every P2 r7_balanced run with a validation evaluation; missing:\n  ' + '\n  '.join(missing))
    table = pd.DataFrame(rows).sort_values(['val_terminal_hold_rate', 'unsafe_events_per_decision'], ascending=[False, True],
                                           ignore_index=True)
    return ['decqn'] + table['model'].head(3).tolist(), table
