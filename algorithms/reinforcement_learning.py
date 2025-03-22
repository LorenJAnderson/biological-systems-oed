from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO

from environments.diffusion1d import Diffusion1DEnv
from environments.source2d import Source2DEnv
from environments.lotka_volterra import LotkaVolterraEnv

ENV_CLASS = Diffusion1DEnv
ENV_NAME = 'diffusion1d'
REW_FUNC = 'kl'
NUM_TIME_STEPS = 10_000
SAVE_FREQ = 1_000
SAVE_PATH = '../data/ppo_logs/' + ENV_NAME + '_' + REW_FUNC + '/models'
TENSORBOARD_PATH = '../data/ppo_logs/' + ENV_NAME + '_' + REW_FUNC + '/'


def rl_train() -> None:
    """Trains and saves a PPO agent to maximize reward function."""
    env = ENV_CLASS(rew_func=REW_FUNC)
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=SAVE_PATH,
        name_prefix='ppo_model'
    )
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=128,
        batch_size=128,
        tensorboard_log=TENSORBOARD_PATH
    )
    model.learn(NUM_TIME_STEPS, callback=checkpoint_callback)


if __name__ == '__main__':
    rl_train()
