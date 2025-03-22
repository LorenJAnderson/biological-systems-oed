import copy

import gymnasium as gym
import numpy as np
from numba import njit


class LotkaVolterraEnv(gym.Env):
    """Predator-prey or Lotka-Volterra environment."""
    def __init__(self,
                 param_low: float=0.5,
                 param_high: float=1.0,
                 max_experiments: int=2,
                 stdv: float=0.1,
                 granularity: int=10,
                 rew_func='kl') -> None:
        self.param_low, self.param_high = param_low, param_high
        self.max_exps = max_experiments
        self.stdv = stdv
        self.gran = granularity
        self.rew_func = rew_func
        self.init_pop_x = 8.0
        self.init_pop_y = 4.0

        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(12,))

        self.a_param = None
        self.b_param = None
        self.d_param = None
        self.g_param = None
        self.time_step = None
        self.physical_state = None
        self.grid_points = None
        self.grid_vals = None
        self.designs = None
        self.measurements = None

    def greedy_reset(self,
                     true_params: np.ndarray,
                     time_step: int,
                     physical_state: np.ndarray,
                     grid_points: np.ndarray,
                     grid_vals: np.ndarray,
                     designs: list[int],
                     measurements: list[float]) -> None:
        """Resets environment to given fields. Used for greedy design."""
        a_param, b_param, d_param, g_param = true_params
        self.a_param = a_param
        self.b_param = b_param
        self.d_param = d_param
        self.g_param = g_param
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
        self.a_param = np.random.uniform(self.param_low, self.param_high)
        self.b_param = np.random.uniform(self.param_low, self.param_high)
        self.d_param = np.random.uniform(self.param_low, self.param_high)
        self.g_param = np.random.uniform(self.param_low, self.param_high)
        self.time_step = 0
        self.physical_state = np.array([1.0])
        self.reset_grid()
        self.designs = [0]
        self.measurements = []
        self.take_measurement()
        return self.get_obs(), {}

    def reset_grid(self) -> None:
        """Resets belief grid to uniform distribution. Uses log probabilities
        to prevent underflow."""
        self.grid_points = []
        for _, a in enumerate(np.linspace(
                self.param_low, self.param_high, self.gran)):
            for _, b in enumerate(np.linspace(
                    self.param_low, self.param_high, self.gran)):
                for _, d in enumerate(np.linspace(
                        self.param_low, self.param_high, self.gran)):
                    for _, g in enumerate(np.linspace(
                            self.param_low, self.param_high, self.gran)):
                        self.grid_points.append((a, b, d, g))
        self.grid_vals = np.log(np.ones(len(self.grid_points)) /
                                len(self.grid_points))

    def step(self, action: int) -> \
            tuple[np.ndarray, float, bool, bool, dict]:
        """Runs one time step of environment and returns gymnasium standard
        signals."""
        self.time_step += 1
        self.designs.append(action)
        taken_action = max(action + 1, self.physical_state.item())
        self.physical_state = np.array([taken_action])
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
        measurements_obs = np.zeros(2*(self.max_exps+1))
        for i, obs in enumerate(self.measurements):
            measurements_obs[i] = obs
        return np.concatenate([experiments_obs, designs_obs, measurements_obs])

    def take_measurement(self) -> None:
        """Takes a measurement, stores measurement, and updates belief grid."""
        a_arr = np.array([[self.a_param]])
        b_arr = np.array([[self.b_param]])
        d_arr = np.array([[self.d_param]])
        g_arr = np.array([[self.g_param]])

        pops = np.array([[self.init_pop_x, self.init_pop_y]])
        true_val = self.forward_rk4(self.lv_derivative, pops,
                                    self.physical_state[0],
                                    a_arr, b_arr, d_arr, g_arr)
        measured_val = np.random.normal(true_val[0], self.stdv, 2)
        self.measurements.append(measured_val[0])
        self.measurements.append(measured_val[1])
        self.update_belief_grid()

    def update_belief_grid(self) -> None:
        """Updates belief grid based on measurement log likelihood."""
        alphas = np.array([[a] for a, b, d, g in self.grid_points])
        betas = np.array([[b] for a, b, d, g in self.grid_points])
        deltas = np.array([[d] for a, b, d, g in self.grid_points])
        gammas = np.array([[g] for a, b, d, g in self.grid_points])

        pops = np.array([[self.init_pop_x, self.init_pop_y]
                        for _ in range(len(self.grid_vals))])
        true_vals = self.forward_rk4(self.lv_derivative, pops,
                                     self.physical_state[0],
                                     alphas, betas, deltas, gammas)
        last_measurement = np.array([self.measurements[-2],
                                     self.measurements[-1]])
        log_like_factor = -1/2 * (((true_vals - last_measurement)/
                            self.stdv) ** 2)
        self.grid_vals += np.sum(log_like_factor, axis=1)

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
        prior = 1.0 / len(self.grid_vals)
        grid_vals = self.convert_log_probs(copy.deepcopy(self.grid_vals))
        return sum([val * np.log(val/prior) for val in grid_vals
                    if val > 1e-128])

    def max_proximity_rew(self) -> float:
        """Returns distance error between MAP estimate of unknown parameters
        and true parameter values."""
        grid_vals_and_points = list(zip(self.grid_vals, self.grid_points))
        grid_vals_and_points.sort(reverse=True)
        max_a_posteriori = np.array(grid_vals_and_points[0][1])
        true_params = np.array([self.a_param, self.b_param,
                                self.d_param,self.g_param])
        return -np.sum((max_a_posteriori - true_params) ** 2).item()

    def max_forward_rew(self) -> float:
        """Returns forward error in model using MAP estimate and
        pre-specified inputs."""
        grid_vals_and_points = list(zip(self.grid_vals, self.grid_points))
        grid_vals_and_points.sort(reverse=True)
        a_map, b_map, d_map, g_map = grid_vals_and_points[0][1]
        pops_1 = np.array([[self.init_pop_x, self.init_pop_y]])
        pops_2 = np.array([[self.init_pop_x, self.init_pop_y]])
        return self.score_max_forward(self.lv_derivative, pops_1, pops_2, 1.0,
            np.array([[self.a_param]]), np.array([[self.b_param]]),
            np.array([[self.d_param]]), np.array([[self.g_param]]),
            np.array([[a_map]]), np.array([[b_map]]),
            np.array([[d_map]]), np.array([[g_map]]))

    @staticmethod
    @njit()
    def lv_derivative(X, t, alpha, beta, delta, gamma):
        """Returns equation derivatives for given parameters and
        populations."""
        coeff = np.concatenate((beta, -1 * gamma), axis=1)
        const = np.concatenate((alpha, -1 * delta), axis=1)
        return (const - coeff * np.flip(X)) * X

    @staticmethod
    @njit()
    def forward_rk4(func, pops, time_len, a, b, d, g):
        """Computes populations for given parameters, initial populations,
        and length of time."""
        integration_granularity = 1_000
        ts = np.linspace(0., time_len, integration_granularity)
        dt = ts[1] - ts[0]
        for i in range(len(ts) - 1):
            k1 = func(pops, ts[i], a, b, d, g)
            k2 = func(pops + dt / 2. * k1, ts[i] + dt / 2., a, b, d, g)
            k3 = func(pops + dt / 2. * k2, ts[i] + dt / 2., a, b, d, g)
            k4 = func(pops + dt * k3, ts[i] + dt, a, b, d, g)
            pops = pops + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
        return pops

    @staticmethod
    @njit()
    def score_max_forward(func, pops1, pops2, time_len, a1, b1, d1, g1,
                          a2, b2, d2, g2):
        """Computes intermediate populations for given parameters, initial
        populations, and length of time. Returns MSE between intermediate
        populations with true and given (MAP) parameters."""
        integration_granularity = 100
        ts = np.linspace(0., time_len, integration_granularity)
        dt = ts[1] - ts[0]
        error_total = 0
        for i in range(len(ts) - 1):
            k1 = func(pops1, ts[i], a1, b1, d1, g1)
            k2 = func(pops1 + dt / 2. * k1, ts[i] + dt / 2., a1, b1, d1, g1)
            k3 = func(pops1 + dt / 2. * k2, ts[i] + dt / 2., a1, b1, d1, g1)
            k4 = func(pops1 + dt * k3, ts[i] + dt, a1, b1, d1, g1)
            pops1 = pops1 + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)

            k11 = func(pops2, ts[i], a2, b2, d2, g2)
            k21 = func(pops2 + dt / 2. * k11, ts[i] + dt / 2., a2, b2, d2, g2)
            k31 = func(pops2 + dt / 2. * k21, ts[i] + dt / 2., a2, b2, d2, g2)
            k41 = func(pops2 + dt * k31, ts[i] + dt, a2, b2, d2, g2)
            pops2 = pops2 + dt / 6. * (k11 + 2. * k21 + 2. * k31 + k41)
            error_total -= np.sum(np.power(pops1-pops2, 2))
        return error_total / integration_granularity

    @staticmethod
    def convert_log_probs(grid_vals: np.ndarray) -> np.ndarray:
        """Converts parameter log probabilities to normalized probabilities.
        Initially subtracting the maximum log probability prevents underflow
        for at least one parameter."""
        grid_vals -= np.max(grid_vals)
        grid_vals = np.exp(grid_vals)
        grid_vals /= np.sum(grid_vals)
        return grid_vals
