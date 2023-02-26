import numpy as np

class KArmedBandit:
    """K-armed bandit environment.

    The agent has to choose one of K arms to pull at each timestep.
    The reward is randomly drawn from a normal distribution
    with mean `q_star[a]` and standard deviation `reward_std`,
    where `q_star` is the true mean reward for arm `a`.
    The goal is to learn to select the arm that yields the highest expected reward.

    Args:
        num_arms: An integer specifying the number of arms in the bandit.
        max_steps: An integer specifying the maximum number of timesteps for an episode.
        reward_mean: A float specifying the mean of the true reward distribution for each arm.
        reward_std: A float specifying the standard deviation of the true reward distribution for each arm.

    Attributes:
        action_space: A numpy array containing the available actions (i.e., the arms).
        num_arms: An integer specifying the number of arms in the bandit.
        max_steps: An integer specifying the maximum number of timesteps for an episode.
        reward_mean: A float specifying the mean of the true reward distribution for each arm.
        reward_std: A float specifying the standard deviation of the true reward distribution for each arm.
        q_star: A numpy array containing the true mean rewards for each arm.
        timestep: An integer specifying the current timestep.
    """

    def __init__(self, num_arms: int, max_steps: int, reward_mean: float = 0.0, reward_std: float = 1.0) -> None:
        self.action_space = np.arange(num_arms)
        self.observation_space = np.arange(num_arms)
        self.num_arms = num_arms
        self.max_steps = max_steps
        self.reward_mean = reward_mean
        self.reward_std = reward_std
        self.q_star = np.random.normal(loc=self.reward_mean, scale=self.reward_std, size=self.num_arms)
        self.timestep = 0

    def reset(self) -> tuple:
        """Reset the environment to its initial state.

        Returns:
            A tuple containing the current timestep, the initial reward (always 0), a boolean indicating whether
            the episode is over, a boolean indicating whether the episode was truncated, and an empty dictionary of
            additional information.
        """
        self.timestep = 0
        return 0

    def step(self, action: int) -> tuple:
        """Take a step in the environment by choosing an arm and receiving a reward.

        Args:
            action: An integer specifying the chosen arm.

        Returns:
            A tuple containing the next timestep, the received reward, a boolean indicating whether the episode is
            over, a boolean indicating whether the episode was truncated, and an empty dictionary of additional
            information.
        """
        if action not in self.action_space:
            raise ValueError(f"Invalid action {action}")

        reward = np.random.normal(loc=self.q_star[action], scale=self.reward_std)
        self.timestep += 1

        if self.timestep >= self.max_steps:
            terminal = True
        else:
            terminal = False

        return action, reward, terminal, False, {}

