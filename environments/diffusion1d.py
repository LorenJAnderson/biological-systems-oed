import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


class Diffusion1DEnv(gym.Env):
    """Environment taken from section 4.2 of https://arxiv.org/abs/1604.08320
    with some minor parameter changes."""
    def __init__(self,
                 diffusion_coeff: float=0.1,
                 operation_cost: float=0.1,
                 movement_cost_coeff: float=0.1,
                 concentration_strength: float=30.0,
                 param_low: float=-5.0,
                 param_high: float=5.0,
                 max_experiments: int=2,
                 stdv: float=2.0,
                 granularity: int=100,
                 rew_func: str='kl') -> None:
        self.diffusion_coeff = diffusion_coeff
        self.operation_cost = operation_cost
        self.movement_coeff = movement_cost_coeff
        self.concentration_stren = concentration_strength
        self.param_low, self.param_high = param_low, param_high
        self.max_exps = max_experiments
        self.stdv = stdv
        self.gran = granularity
        self.rew_func = rew_func

        self.action_space = gym.spaces.Discrete(11)
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
        self.true_param = np.random.uniform(self.param_low, self.param_high, 1)
        self.time_step = 0
        self.physical_state = np.array([0.0])
        self.reset_grid()
        self.designs = [5]
        self.measurements = []
        self.take_measurement()
        return self.get_obs(), {}

    def reset_grid(self) -> None:
        """Resets belief grid to uniform distribution."""
        self.grid_points = np.linspace(self.param_low, self.param_high,
                                       self.gran)
        self.grid_vals = np.ones(self.gran) / self.gran

    def step(self, action: int) -> \
            tuple[np.ndarray, float, bool, bool, dict]:
        """Runs one time step of environment and returns gymnasium standard
        signals."""
        self.time_step += 1
        self.designs.append(action)
        self.physical_state += np.array([-2.5 + 0.5 * action])
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
        experiments_obs = np.zeros(self.max_exps+1)
        experiments_obs[self.time_step] = 1
        designs_obs = np.zeros(self.max_exps+1)
        for idx, design in enumerate(self.designs):
            designs_obs[idx] = design
        measurements_obs = np.zeros(self.max_exps+1)
        for idx, measurement in enumerate(self.measurements):
            measurements_obs[idx] = measurement
        return np.concatenate([experiments_obs, designs_obs, measurements_obs])


    def take_measurement(self) -> None:
        """Takes a measurement, stores measurement, and updates belief grid."""
        true_val = self.forward(self.true_param, self.physical_state).item()
        measurement = np.random.normal(true_val, self.stdv)
        self.measurements.append(measurement)
        self.update_belief_grid()

    def forward(self, param: np.ndarray, loc: np.ndarray) -> np.ndarray:
        """Returns concentrations at specified locations for given
        parameters."""
        factor_1_num = self.concentration_stren
        factor_1_denom = np.sqrt(2 * np.pi) * np.sqrt(1.2 + 4 *
                         self.diffusion_coeff * self.time_step)
        factor_1 = factor_1_num / factor_1_denom
        factor_2_num = (param - loc) ** 2
        factor_2_denom = 2 * (1.2 + 4 * self.diffusion_coeff * self.time_step)
        factor_2 = np.exp(-factor_2_num / factor_2_denom)
        return factor_1 * factor_2

    def update_belief_grid(self) -> None:
        """Updates belief grid based on measurement likelihood."""
        true_vals = self.forward(self.grid_points, self.physical_state)
        like_factor_1 = 1 / (self.stdv * np.sqrt(2 * np.pi))
        like_factor_2 = np.exp(-1/2 * ((self.measurements[-1] -
                                        true_vals)/self.stdv)**2)
        self.grid_vals = like_factor_1 * like_factor_2 * self.grid_vals
        self.grid_vals /= np.sum(self.grid_vals)

    def det_rew(self) -> float:
        """Determines reward using reward function field."""
        if self.rew_func == 'max_proximity':
            return self.max_proximity_rew()
        elif self.rew_func == 'max_forward':
            return self.max_forward_rew()
        else:
            return self.kl_rew()

    def kl_rew(self) -> float:
        """Returns KL divergence from initial prior to current belief state."""
        prior = np.ones(self.gran) / self.gran
        return np.sum(self.grid_vals * np.log(self.grid_vals/prior))

    def max_proximity_rew(self) -> float:
        """Returns distance error between MAP estimate of unknown parameter
        and true parameter value."""
        grid_vals_and_points = list(zip(self.grid_vals, self.grid_points))
        grid_vals_and_points.sort(reverse=True)
        max_a_posteriori = np.array([grid_vals_and_points[0][1]])
        return (-(max_a_posteriori - self.true_param) ** 2).item()

    def max_forward_rew(self) -> float:
        """Returns forward error in model using MAP estimate and
        pre-specified inputs."""
        grid_vals_and_points = list(zip(self.grid_vals, self.grid_points))
        grid_vals_and_points.sort(reverse=True)
        max_a_posteriori = np.array([grid_vals_and_points[0][1]])
        test_locs = np.linspace(self.param_low, self.param_high, self.gran)
        test_forward = self.forward(max_a_posteriori, test_locs)
        true_forward = self.forward(self.true_param, test_locs)
        errors = -(test_forward - true_forward) ** 2
        return np.mean(errors).item()

    def plot_belief_grid(self) -> None:
        """Plots belief grid."""
        plt.plot(self.grid_vals)
        plt.xticks(ticks=np.linspace(0, self.gran-1, 6),
                   labels=['-5', '-3', '-1', '1', '3 ', '5'])
        plt.ylabel('Probability')
        plt.show()
