"""The twelve registered reward profiles: r = r0 + sum_i lambda_i f_i, with raw and weighted terms exposed.

Coefficients lambda_i come only from the locked calibration in configs/rewards/reward_v1.yaml. A profile that
contains a calibrated term refuses to build until the register is locked, so no profile can be trained with a
hand-picked or missing weight. Calibration itself uses the raw terms and never needs the weights.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .terms import TERM_NAMES, RewardConfig, term_bounds

CALIBRATED = ('time', 'path', 'speed', 'accel', 'rom')


class RewardNotLocked(RuntimeError):
    """Raised when a profile with calibrated terms is requested before `calibrate rewards` locked the register."""


@dataclass(frozen=True)
class RewardProfile:
    name: str
    terms: tuple                  # registered term names, always starting with 'r0'
    weights: dict                 # term -> lambda (1.0 for r0 and hold, calibrated lambda otherwise)
    config: RewardConfig

    def combine(self, raw: dict) -> tuple:
        """(total reward, weighted terms dict) from raw terms; the weighted dict holds only this profile's terms."""
        weighted = {t: self.weights[t] * raw[t] for t in self.terms}
        total = sum(weighted.values())
        return total, weighted

    def bounds(self, n_joints: int = 7) -> tuple:
        """(r_min, r_max) over all states and actions, for clipping bootstrapped targets to bounds / (1 - gamma)."""
        b = term_bounds(self.config, n_joints)
        low = sum(min(self.weights[t] * b[t][0], self.weights[t] * b[t][1]) for t in self.terms)
        high = sum(max(self.weights[t] * b[t][0], self.weights[t] * b[t][1]) for t in self.terms)
        return float(low), float(high)


def profile_names(cfg: dict) -> list:
    return list(cfg['profiles'])


def load_profile(cfg: dict, name: str, allow_unlocked: bool = False) -> RewardProfile:
    """Build a profile from the parsed reward_v1.yaml.

    allow_unlocked=True is used only by calibration and tests: calibrated terms get weight 1.0 so that raw terms
    can be inspected. It must never be used for training.
    """
    if name not in cfg['profiles']:
        raise KeyError(f'unknown reward profile {name!r}; registered: {profile_names(cfg)}')
    terms = tuple(cfg['profiles'][name]['terms'])
    unknown = [t for t in terms if t not in TERM_NAMES]
    if unknown:
        raise KeyError(f'profile {name} names unregistered terms {unknown}')
    weights = {'r0': 1.0, 'hold': 1.0}
    locked = cfg.get('status') == 'locked'
    for term in terms:
        if term in CALIBRATED:
            lam = cfg['calibration']['lambda'].get(term)
            if not locked or lam is None:
                if not allow_unlocked:
                    raise RewardNotLocked(
                        f'profile {name} uses calibrated term {term!r} but configs/rewards/reward_v1.yaml is '
                        f'status={cfg.get("status")!r}; run `python -m natural_rl calibrate rewards` first')
                lam = 1.0
            weights[term] = float(lam)
    return RewardProfile(name, terms, weights, RewardConfig.from_yaml(cfg))


def zero_unused(raw: dict, profile: RewardProfile) -> dict:
    """Raw terms restricted to a profile (others are ignored by combine, kept here for logging completeness)."""
    return {t: np.asarray(raw[t]) for t in profile.terms}
