from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common import env_util
# from stable_baselines3 import PPO
from sb3_contrib import ppo_recurrent
from recurrent_policy import RecurrentActorCriticPolicy
from recurrent_ppo import RecurrentPPO
from ppo import PPO
from policies import ActorCriticPolicy
import gym
import os

models_dir = "models/mc_minmax"
model_path = f"{models_dir}/400000.zip"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
# env = make_atari_env("PongNoFrameskip-v4", n_envs=4, seed=0)
# Frame-stacking with 4 frames
# env = VecFrameStack(env, n_stack=4)

episodes = 100

env = env_util.make_vec_env("MountainCar-v0", n_envs=1)

# env = gym.make("Pong-v4")

model = PPO.load(model_path, env=env)

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

env.close()