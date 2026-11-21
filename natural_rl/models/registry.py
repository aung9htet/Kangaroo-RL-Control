"""Model registry: the ten mandatory configurations. An unimplemented model fails loudly; nothing is substituted."""
from __future__ import annotations

from importlib import import_module

MODEL_IDS = ('decqn', 'growing_q', 'ppo_md', 'ppo_box', 'sac', 'td3', 'tqc', 'crossq', 'tdmpc2', 'sac_pidlag')
# model id -> (module, policy class name); modules are imported lazily so one missing dependency never hides the rest
_IMPLEMENTATIONS = {
    'decqn': ('natural_rl.models.decqn', 'DecQNPolicy'),
    'growing_q': ('natural_rl.models.growing_q', 'GrowingQPolicy'),
    'ppo_md': ('natural_rl.models.sb3_models', 'PPOMDPolicy'),
    'ppo_box': ('natural_rl.models.sb3_models', 'PPOBoxPolicy'),
    'sac': ('natural_rl.models.sb3_models', 'SACPolicy'),
    'td3': ('natural_rl.models.sb3_models', 'TD3Policy'),
    'tqc': ('natural_rl.models.sb3_models', 'TQCPolicy'),
    'crossq': ('natural_rl.models.sb3_models', 'CrossQPolicy'),
    'sac_pidlag': ('natural_rl.models.sac_pidlag', 'SACPIDLagPolicy'),
    'tdmpc2': ('natural_rl.models.tdmpc2', 'TDMPC2Policy'),
}


class ModelNotImplemented(NotImplementedError):
    pass


def _module(model_id: str):
    if model_id not in MODEL_IDS:
        raise KeyError(f'unknown model {model_id!r}; the registered roster is {list(MODEL_IDS)}')
    if model_id not in _IMPLEMENTATIONS:
        raise ModelNotImplemented(f'model {model_id!r} is registered but not implemented yet (milestone M4); '
                                  'the roster is fixed and no other model is substituted for it')
    return import_module(_IMPLEMENTATIONS[model_id][0])


def ensure_implemented(model_id: str) -> None:
    """Raise ModelNotImplemented (or KeyError for an unknown id) before any config or scene is touched."""
    _module(model_id)


def load_policy(model_id: str, directory, spec):
    """Load the policy saved in a checkpoint's policy_state directory."""
    module = _module(model_id)
    return getattr(module, _IMPLEMENTATIONS[model_id][1]).load(directory, spec)


def train_model(model_id: str, ctx) -> None:
    _module(model_id).train(ctx)


def implemented() -> list:
    return [m for m in MODEL_IDS if m in _IMPLEMENTATIONS]
