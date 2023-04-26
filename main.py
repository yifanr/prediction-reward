from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common import env_util
# from stable_baselines3 import PPO
from sb3_contrib import ppo_recurrent
from recurrent_policy import RecurrentActorCriticPolicy
from recurrent_ppo import RecurrentPPO
from ppo import PPO
from policies import ActorCriticPolicy, ActorCriticCnnPolicy
import gym
from losses import minmax_loss, simple_loss, gaussian_loss
import os
import gym_envs

def start_experiment(**args):

    models_dir = "models/mc_minmax"
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

    env = env_util.make_vec_env("MountainCar-v0", n_envs=1)
    # env = gym.make("Pong-v4")
    # env = gym.make('gym_envs/GridWorld-v0', size=20)
    print(env.reset())

    # model = PPO(ActorCriticPolicy, env, verbose=1, tensorboard_log=logs_dir)
    model = PPO(ActorCriticPolicy, env, verbose=1, tensorboard_log=logs_dir)

    # model = RecurrentPPO(RecurrentActorCriticPolicy, env, verbose=1, vf_coef=0.0001, tensorboard_log=logs_dir)

    TIMESTEPS = 100000

    i = 0
    while True:
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="mc_minmax")
        i += 1
        model.save(f"{models_dir}/{TIMESTEPS*i}")


    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--exp_name', type=str, default='')

    parser.add_argument('--cnn', type=bool, default=False)

    parser.add_argument('--icm', type=bool, default=False)

    parser.add_argument('--env', help='environment ID', default='MountainCar-v0',
                        type=str)
    
    parser.add_argument('--reward', type=str, default="none",
                        choices=["none", "minmax", "gaussian", "error"])
    
    parser.add_argument('--max-timesteps', help='maximum number of timesteps', default=4500, type=int)


    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--num_timesteps', type=int, default=int(1e8))



    args = parser.parse_args()

    start_experiment(**args.__dict__)
