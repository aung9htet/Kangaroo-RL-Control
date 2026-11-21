"""Growing Q-Networks (Seyde et al., L4DC 2024) under the benchmark contract: DecQN with a growing active action set.

Following the paper, the network holds the full discretisation from the start; the active sub-space A_g is enforced
by action MASKING in epsilon-greedy sampling and in the Double-Q target max (Eq. 3). The paper's simple linear
schedule is used: resolution doubles at fixed fractions of training. Registered adaptation (documented in the
README): the nested sub-spaces are subsets of the seven historical levels, {-1,+1} -> +0 -> +-0.3 -> all seven
(2, 3, 5, 7 bins), growing at 20 %, 40 % and 60 % of training (every 1/(N+1) with N = 4 sub-spaces). The paper's
adaptive return-threshold schedule is not used: it needs evaluation returns during training, which would add an
evaluation budget the other controllers do not receive. The active resolution is part of each checkpoint and is
logged for every decision (`policy.info()['active_bins']`).
"""
from __future__ import annotations

from . import decqn
from .decqn import DecQNPolicy  # noqa: F401  (GrowingQ checkpoints are DecQNPolicy objects carrying their active mask)


class GrowingQPolicy(DecQNPolicy):
    model_id = 'growing_q'


def train(ctx) -> None:
    decqn.train(ctx, growing=True)
