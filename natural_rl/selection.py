"""Registered P3 selection: DecQN plus the three non-DecQN controllers with the highest VALIDATION terminal-hold rate.

Ranking is on `r7_balanced`, averaged over the five P2 seeds, each seed at its own validation-selected checkpoint; ties
are broken by the lower unsafe-event rate per decision. Only `val_v1` evaluation tables on the training (kinematic)
backend are read; test results can neither be read nor influence the choice (a test checks this).
"""
from __future__ import annotations

import pandas as pd

from .checkpoints import step_name
from .eval import select_checkpoint
from .models.registry import MODEL_IDS
from .programme import run_id
from .registry import find_run


def p3_selection(models=MODEL_IDS, seeds=range(5), horizon: int = 100):
    """(['decqn', A, B, C], table). Raises with the list of missing runs/evaluations if the P2 data is incomplete."""
    rows, missing = [], []
    for model in models:
        if model == 'decqn':
            continue
        holds, unsafe = [], []
        for seed in seeds:
            rid = run_id(model, 'r7_balanced', seed, budget='budget_moderate_v1' if model == 'sac_pidlag' else None)
            try:
                run = find_run(rid)
                step = select_checkpoint(run, 'kinematic', 'val_v1')
                df = pd.read_parquet(run / 'eval' / 'kinematic' / 'val_v1' / step_name(step) / 'episode_metrics.parquet')
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
