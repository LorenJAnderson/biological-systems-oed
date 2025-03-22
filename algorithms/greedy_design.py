import copy
import random
import pickle
from multiprocessing import Pool

import gymnasium as gym
import numpy as np

from environments.diffusion1d import Diffusion1DEnv
from environments.source2d import Source2DEnv
from environments.lotka_volterra import LotkaVolterraEnv

ENV_CLASS = Diffusion1DEnv
ENV_NAME = 'diffusion1d'
REW_FUNC = 'kl'
NUM_EXPS = 10_000
NUM_PARAM_SAMPLES = 1_000
NUM_CPUS = 14
SAVE_PATH = '../data/' + ENV_NAME + '_greedy_' + REW_FUNC + '.pkl'


def greedy_experiment(idx: int) -> tuple[float, list[int]]:
    """Conducts single greedy experiment and returns reward & actions."""
    env = ENV_CLASS(rew_func=REW_FUNC)
    obs, _ = env.reset()
    done = False
    rew = None
    acts = []
    while not done:
        if env.time_step == 0:
            action = np.random.randint(ENV_CLASS().action_space.n)
        else:
            action = det_greedy_action(env)
        acts.append(action)
        obs, rew, term, trunc, info = env.step(action)
        done = term or trunc
    return rew, acts


def det_greedy_action(env: gym.Env) -> int:
    """Determines the greedy action at the current time step.
    Parameter probabilities need to be converted from log probabilities in
    certain environments before sampling parameters."""
    grid_points = copy.deepcopy(env.grid_points)
    grid_vals = copy.deepcopy(env.grid_vals)
    if ENV_NAME in ['source2d', 'lotka_volterra']:
        grid_vals -= np.max(grid_vals)
        grid_vals = np.exp(grid_vals)
        grid_vals /= np.sum(grid_vals)
    sample_params = random.choices(population=grid_points,
                                   weights=grid_vals,
                                   k=NUM_PARAM_SAMPLES)
    temp_env = ENV_CLASS(rew_func=REW_FUNC)
    action_scores = []

    for action in list(range(ENV_CLASS().action_space.n)):
        act_rew_sum = 0
        for param in sample_params:
            if ENV_NAME == 'diffusion1d':
                true_param = np.array([param])
            else:
                true_param = param
            time_step = copy.deepcopy(env.time_step)
            physical_state = copy.deepcopy(env.physical_state)
            grid_points = copy.deepcopy(env.grid_points)
            grid_vals = copy.deepcopy(env.grid_vals)
            designs = copy.deepcopy(env.designs)
            measurements = copy.deepcopy(env.measurements)
            temp_env.greedy_reset(true_param,
                                  time_step,
                                  physical_state,
                                  grid_points,
                                  grid_vals,
                                  designs,
                                  measurements)

            _, rew, _, _, _ = temp_env.step(action)
            act_rew_sum += rew
        action_scores.append((act_rew_sum / NUM_PARAM_SAMPLES, action))
    action_scores.sort(reverse=True)
    return action_scores[0][1]


def greedy_design() -> None:
    """Conducts many greedy design experiments and saves reward &
    action data."""
    with Pool(NUM_CPUS) as p:
        results = p.map(greedy_experiment, list(range(NUM_EXPS)))
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    greedy_design()
