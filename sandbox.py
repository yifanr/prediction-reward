from recurrent_ppo import RecurrentPPO
from stable_baselines3.common import env_util
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import A2C, PPO
from gym import envs

from rnn_policy import RnnPolicy
from test_policy import CustomActorCriticPolicy
from recurrent_policy import RecurrentActorCriticPolicy
from sb3_contrib import ppo_recurrent
# from mujoco_py import GlfwContext

# GlfwContext(offscreen=True)  # Create a window to init GLFW.

# Custom MLP policy of three layers of size 128 each for the actor and 2 layers of 32 for the critic,
# with a nature_cnn feature extractor

print(envs.registry.all())
# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
# env = env_util.make_vec_env("LunarLander-v2", n_envs=4)
env = env_util.make_vec_env("HalfCheetah-v2", n_envs=4)
# # Frame-stacking with 4 frames
# env = VecFrameStack(env, n_stack=4)

# model = PPO(RnnPolicy, env, verbose=1)
model = RecurrentPPO(RecurrentActorCriticPolicy, env, verbose=1, vf_coef=0.0001)
# model = ppo_recurrent.RecurrentPPO("MlpLstmPolicy", env, verbose=1)
model.learn(total_timesteps=int(100000))
obs = env.reset()
#model = A2C.load("A2C_breakout") #uncomment to load saved model
model.save("A2C_breakout")
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()