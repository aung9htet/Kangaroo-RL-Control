"""Vendored official TD-MPC2 agent (MIT licence, see LICENSE).

Source: https://github.com/nicklashansen/tdmpc2 at commit 332f19e4aba4bffe2dc736f45ea7f7bdc585903a. Vendored: tdmpc2/tdmpc2.py
(as agent.py) and tdmpc2/common/{__init__,init,layers,math,scale,world_model}.py. Not vendored: trainer, environments, logger, parser
and the torchrl replay buffer (replaced by natural_rl's goal-relabelling replay).

The only changes, so the agent runs under natural_rl on CPU or GPU (nothing in the algorithm is altered):
1. agent.py: `from common ...` -> `from .common ...`; torch.device('cuda:0') -> torch.device(cfg.device); the multitask discount
   device='cuda:0' -> device=self.device.
2. common/scale.py: the two buffers use torch.device(cfg.device) instead of 'cuda:0'.
3. common/world_model.py: `from common import layers, math, init` -> `from . import layers, math, init`.
4. agent.py: torch.compiler.cudagraph_mark_step_begin() in update() is only called when cfg.compile is true (it imports the inductor
   stack, which is unnecessary and fails on this image's numpy; natural_rl runs with compile: false).
5. agent.py: both Adam optimizers use capturable=(self.device.type == 'cuda') instead of True (CUDA-graph capture is unsupported,
   and rejected by torch 2.7, on CPU).
"""
from .agent import TDMPC2  # noqa: F401
