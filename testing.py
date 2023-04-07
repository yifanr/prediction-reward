from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common import env_util
from stable_baselines3 import PPO
from sb3_contrib import ppo_recurrent
from recurrent_policy import RecurrentActorCriticPolicy
from recurrent_ppo import RecurrentPPO
import gym
import os

models_dir = "models/Lunar_RPPO"
logs_dir = "temp"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
# env = make_atari_env("PongNoFrameskip-v4", n_envs=4, seed=0)
# Frame-stacking with 4 frames
# env = VecFrameStack(env, n_stack=4)

env = env_util.make_vec_env("LunarLander-v2", n_envs=1)

# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir)

model = RecurrentPPO(RecurrentActorCriticPolicy, env, verbose=1, vf_coef=0.0001, tensorboard_log=logs_dir)

TIMESTEPS = 25000

i = 0
while True:
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="Lunar_RPPO")
    i += 1
    model.save(f"{models_dir}/{TIMESTEPS*i}")


obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()