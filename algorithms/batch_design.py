import pickle
from multiprocessing import Pool

from environments.diffusion1d import Diffusion1DEnv
from environments.source2d import Source2DEnv
from environments.lotka_volterra import LotkaVolterraEnv

ENV_CLASS = Diffusion1DEnv
ENV_NAME = 'diffusion1d'
REW_FUNC = 'kl'
NUM_EPISODES = 10_000
NUM_CPUS = 14
SAVE_PATH = '../data/' + ENV_NAME + '_batch_' + REW_FUNC + '.pkl'


def batch_pair_experiments(actions: tuple[int, int]) -> tuple:
    """Conducts many batch design experiments for a single action pair and
    returns list of episode rewards with action pair."""
    act1, act2 = actions
    env = ENV_CLASS(rew_func=REW_FUNC)
    rewards = []
    for _ in range(NUM_EPISODES):
        obs, _ = env.reset()
        done = False
        while not done:
            action = act1 if env.time_step == 0 else act2
            obs, rew, term, trunc, info = env.step(action)
            done = term or trunc
            if done:
                rewards.append(rew)
    return rewards, act1, act2


def batch_design() -> None:
    """Conducts many batch design experiments for all unordered pairs of
    designs and saves reward data."""
    num_acts = ENV_CLASS().action_space.n
    all_act_pairs = []
    for act1 in range(num_acts):
        for act2 in range(act1, num_acts):
            all_act_pairs.append((act1, act2))
    with Pool(NUM_CPUS) as p:
        results = p.map(batch_pair_experiments, all_act_pairs)
        with open(SAVE_PATH, 'wb') as f:
            pickle.dump(results, f)

if __name__ == '__main__':
    batch_design()
