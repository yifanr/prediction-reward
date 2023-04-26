import gym_envs
import gym
import numpy as np
from gym import spaces

class SimpleDistractorWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.timestep = 0
        self._distractor = 0
        self.observation_space = self.observation_space = spaces.Box(0, env.size - 1, shape=(4,), dtype=int)
    
    def step(self, action):
        # self.timestep += 1
        observation, reward, terminated, info = self.env.step(action)

        if (action == 4):
            self._distractor = np.random.rand()
            # print(self.timestep)
        
        observation = np.append(observation, self._distractor)
        
        return observation, reward, terminated, info
    
    def reset(self):
        observation = self.env.reset()

        return np.append(observation, self._distractor)
    
env = gym.make('gym_envs/GridWorld-v0')
wrapped_env = SimpleDistractorWrapper(env)