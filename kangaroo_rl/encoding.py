"""Positional encoding of joint angles and goal coordinates.

A value x normalised to [-1, 1] is encoded as

    phi(x) = [x, sin(2^0 pi x), cos(2^0 pi x), ..., sin(2^(L-1) pi x), cos(2^(L-1) pi x)],

the sinusoidal positional encoding of Transformers / NeRF, applied to every coordinate separately.
L = 0 gives the raw normalised value. Each coordinate therefore becomes 2L + 1 numbers.
"""
from __future__ import annotations

import numpy as np


def normalise(x, lower, upper):
    """Map [lower, upper] onto [-1, 1]."""
    return 2.0 * (np.asarray(x) - lower) / (upper - lower) - 1.0


def encoded_size(n_values: int, n_freq: int) -> int:
    return n_values * (2 * n_freq + 1)


def positional_encoding(x, n_freq: int):
    """NumPy encoding of x (..., D) already normalised to [-1, 1]; returns (..., D * (2L + 1))."""
    x = np.asarray(x, dtype=np.float64)
    features = [x]
    for k in range(n_freq):
        angle = (2.0 ** k) * np.pi * x
        features += [np.sin(angle), np.cos(angle)]
    return np.concatenate(features, axis=-1)


def positional_encoding_torch(x, n_freq: int):
    """The same encoding for torch tensors (used inside the Q-network)."""
    import torch

    features = [x]
    for k in range(n_freq):
        angle = (2.0 ** k) * np.pi * x
        features += [torch.sin(angle), torch.cos(angle)]
    return torch.cat(features, dim=-1)
