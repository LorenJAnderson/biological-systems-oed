import copy

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Source2DEnv(gym.Env):
    """Environment from https://ieeexplore.ieee.org/document/1369649 with
    some minor parameter changes."""
    def __init__(self,
                 background_coeff: float=0.1,
                 maximum_coeff: float=1e-4,
                 movement_coeff: float=0.1,
                 param_low: float=-4.0,
                 param_high: float=4.0,
                 max_experiments: int=2,
                 stdv: float=0.1,
                 granularity: int=50,
                 rew_func: str='kl') -> None:
        self.background_coeff = background_coeff
        self.maximum_coeff = maximum_coeff
        self.movement_coeff = movement_coeff
        self.param_low, self.param_high = param_low, param_high
        self.max_exps = max_experiments
        self.stdv = stdv
        self.gran = granularity
        self.rew_func = rew_func

        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(9,))

        self.true_param = None
        self.time_step = None
        self.physical_state = None
        self.grid_points = None
        self.grid_vals = None
        self.designs = None
        self.measurements = None

    def greedy_reset(self,
                     true_param: np.ndarray,
                     time_step: int,
                     physical_state: np.ndarray,
                     grid_points: np.ndarray,
                     grid_vals: np.ndarray,
                     designs: list[int],
                     measurements: list[float]) -> None:
        """Resets environment to given fields. Used for greedy design."""
        self.true_param = true_param
        self.time_step = time_step
        self.physical_state = physical_state
        self.grid_points = grid_points
        self.grid_vals = grid_vals
        self.designs = designs
        self.measurements = measurements

    def reset(self, seed: int=None, options: dict=None) \
            -> tuple[np.ndarray, dict]:
        """Resets environment, returns gymnasium standard signals."""
        np.random.seed(seed)
        self.true_param = np.random.normal(0, 1, 2)
        self.time_step = 0
        self.physical_state = np.array([0.0, 0.0])
        self.reset_grid()
        self.designs = [4]
        self.measurements = []
        self.take_measurement()
        return self.get_obs(), {}

    def reset_grid(self) -> None:
        """Resets belief grid to uniform distribution. Uses log probabilities
        to prevent underflow."""
        grid_1d = np.linspace(self.param_low, self.param_high, self.gran)
        self.grid_points = np.array([(x_val, y_val) for x_val in grid_1d
                                     for y_val in grid_1d])
        self.grid_vals = np.log(np.ones(self.gran ** 2) / (self.gran ** 2))

    def step(self, action: int) -> \
            tuple[np.ndarray, float, bool, bool, dict]:
        """Runs one time step of environment and returns gymnasium standard
        signals."""
        self.time_step += 1
        self.designs.append(action)
        self.physical_state += np.array([(action // 3) - 1, (action % 3) - 1])
        self.take_measurement()
        term, trunc = False, False
        reward = 0
        if self.time_step == self.max_exps:
            term = True
            reward = self.det_rew()
        obs = self.get_obs()
        info = {}
        return obs, reward, term, trunc, info

    def get_obs(self) -> np.ndarray:
        """Returns concatenated observation detailing experiment number,
        all designs, and all measurements."""
        experiments_obs = np.zeros(self.max_exps + 1)
        experiments_obs[self.time_step] = 1
        designs_obs = np.zeros(self.max_exps + 1)
        for idx, design in enumerate(self.designs):
            designs_obs[idx] = design
        measurements_obs = np.zeros(self.max_exps+1)
        for idx, measurement in enumerate(self.measurements):
            measurements_obs[idx] = measurement
        return np.concatenate([experiments_obs, designs_obs, measurements_obs])

    def take_measurement(self) -> None:
        """Takes a measurement, stores measurement, and updates belief grid."""
        new_source = np.array([self.true_param])
        true_val = self.forward(new_source, self.physical_state).item()
        measurement = np.random.normal(true_val, self.stdv)
        self.measurements.append(measurement)
        self.update_belief_grid()

    def forward(self, source: np.ndarray, location: np.ndarray) -> np.ndarray:
        """Returns concentrations at specified locations for given
        parameters."""
        distances = np.sum((location - source) ** 2, axis=1)
        return self.background_coeff + (1.0 / (self.maximum_coeff + distances))

    def update_belief_grid(self) -> None:
        """Updates belief grid based on measurement log likelihood."""
        true_vals = self.forward(self.grid_points, self.physical_state)
        log_like_factor = -1/2 * (((true_vals - self.measurements[-1])/
                                   self.stdv) ** 2)
        self.grid_vals += log_like_factor

    def det_rew(self) -> float:
        """Determines reward using reward function field."""
        if self.rew_func == 'max_proximity':
            return self.max_proximity_rew()
        elif self.rew_func == 'max_forward':
            return self.max_forward_rew()
        else:
            return self.kl_rew()

    def kl_rew(self) -> float:
        """Returns KL-Divergence from initial prior to current belief state.
        Negligible probabilities are discarded."""
        prior = 1.0 / (self.gran ** 2)
        grid_vals = self.convert_log_probs(copy.deepcopy(self.grid_vals))
        return sum([val * np.log(val/prior) for val in grid_vals
                    if val > 1e-128])

    def max_proximity_rew(self) -> float:
        """Returns distance error between MAP estimate of unknown parameters
        and true parameter values."""
        grid_vals_and_points = list(zip(self.grid_vals, self.grid_points))
        grid_vals_and_points.sort(reverse=True, key=lambda x: x[0])
        max_a_posteriori = np.array([grid_vals_and_points[0][1]])
        return -np.sum((max_a_posteriori - self.true_param) ** 2)

    def max_forward_rew(self) -> float:
        """Returns forward error in model using MAP estimate and
        pre-specified inputs. Median in errors is used due to outliers."""
        grid_vals_and_points = list(zip(self.grid_vals, self.grid_points))
        grid_vals_and_points.sort(reverse=True, key=lambda x: x[0])
        max_a_posteriori = np.array([grid_vals_and_points[0][1]])
        grid_1d = np.linspace(self.param_low, self.param_high, self.gran)
        test_locs = np.array([(x_val, y_val) for x_val in grid_1d
                              for y_val in grid_1d])
        test_forward = self.forward(max_a_posteriori, test_locs)
        true_forward = self.forward(self.true_param, test_locs)
        errors = -np.abs(test_forward - true_forward)
        return np.median(errors).item()

    @staticmethod
    def convert_log_probs(grid_vals: np.ndarray) -> np.ndarray:
        """Converts parameter log probabilities to normalized probabilities.
        Initially subtracting the maximum log probability prevents underflow
        for at least one parameter."""
        grid_vals -= np.max(grid_vals)
        grid_vals = np.exp(grid_vals)
        grid_vals /= np.sum(grid_vals)
        return grid_vals

    def plot_belief_grid(self) -> None:
        """Plots belief grid."""
        grid_vals = self.convert_log_probs(copy.deepcopy(self.grid_vals))
        grid_2d = np.zeros((self.gran, self.gran))
        for x in range(self.gran):
            for y in range(self.gran):
                grid_2d[x][y] = grid_vals[x + y*self.gran]
        grid_2d = np.flip(grid_2d, axis=1)
        sns.heatmap(grid_2d, cbar_kws={'label': 'Probability'})
        plt.xticks(ticks=np.linspace(0, self.gran-1, 5),
                   labels=['-4', '-2', '0', '2', '4 '])
        plt.yticks(ticks=np.linspace(0, self.gran-1, 5),
                   labels=['4', '2', '0', '-2', '-4 '])
        plt.show()
