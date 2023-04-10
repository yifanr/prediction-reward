from gym .spaces import Discrete
import torch as th

def action_to_onehot(actions: th.Tensor, action_space: Discrete):
    return th.nn.functional.one_hot(actions, action_space.n)