"""Agents: additive (DecQN) or sequential (autoregressive) factorisation of the joint action."""


def make_agent(kind, network, **kwargs):
    if kind == 'additive':
        from .decqn import DecQNAgent
        return DecQNAgent(network, **kwargs)
    if kind == 'sequential':
        from .seqq import SequentialQAgent
        return SequentialQAgent(network, **kwargs)
    raise ValueError(f'Unknown agent kind {kind!r}')
