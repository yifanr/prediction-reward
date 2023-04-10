import gym
from gym import spaces
import torch

class RewardEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, base_env: gym.Env, hidden_size: int=64):
    super(RewardEnv, self).__init__()

    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = base_env.action_space
    # Example for using image as input:
    self.observation_space = base_env.observation_space
    self.base_env = base_env
    self.hidden_size = hidden_size
    self.hidden_state = torch.zeros[hidden_size]
    self.last_feature = None

  def step(self, action):
    self.hidden_state = action[0:self.hidden_size]
    observation, reward, done, info = self.base_env.step(action)
    info["hidden_state"] = self.hidden_state
    return observation, reward, done, info

  def reset(self):
    return self.base_env.reset()

  def render(self, mode='human'):
    self.base_env.render()

  def close (self):
    self.base_env.close()