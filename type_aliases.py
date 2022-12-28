from typing import NamedTuple, Tuple

import torch as th
from stable_baselines3.common.type_aliases import TensorDict


class RecurrentRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor 
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    lstm_states: Tuple[th.Tensor,th.Tensor]
    episode_starts: th.Tensor
    next_observations: th.Tensor
    episode_ends: th.Tensor
    mask: th.Tensor


class RecurrentDictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    lstm_states: Tuple[th.Tensor,th.Tensor]
    episode_starts: th.Tensor
    next_observations: TensorDict
    episode_ends: th.Tensor
    mask: th.Tensor
