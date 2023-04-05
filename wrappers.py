import gym

class NoReward(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        return ob, 0.0, done, info
    
class AtariWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self):
        obs = self.env.reset()[0]
        return obs
    
    def step(self, action):
        ob, rew, done, truncated, info = self.env.step(action)
        done = truncated or done
        return ob, rew, done, info