import numpy as np

class QLearningAgent:
    """
    A table-based Q-Learning agent.

    Attributes:
        n_states (int): The number of states in the environment.
        n_actions (int): The number of actions in the environment.
        alpha (float): The learning rate.
        gamma (float): The discount factor.
        epsilon (float): The exploration rate.
        Q (numpy.ndarray): The Q-table.
    """

    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.5, gamma: float = 0.9, epsilon: float = 0.1) -> None:
        """
        Initializes the Q-Learning agent.

        Args:
            n_states (int): The number of states in the environment.
            n_actions (int): The number of actions in the environment.
            alpha (float, optional): The learning rate. Defaults to 0.5.
            gamma (float, optional): The discount factor. Defaults to 0.9.
            epsilon (float, optional): The exploration rate. Defaults to 0.1.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))

    def choose_action(self, state: int) -> int:
        """
        Chooses an action based on the current state and the exploration rate.

        Args:
            state (int): The current state.

        Returns:
            int: The chosen action.
        """
        if np.random.random() < self.epsilon:
            # With probability epsilon, choose a random action
            return np.random.choice(self.n_actions)
        else:
            # Otherwise, choose the action with the highest Q-value
            return np.argmax(self.Q[state])

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """
        Updates the Q-table based on the current state, action, reward, and next state.

        Args:
            state (int): The current state.
            action (int): The chosen action.
            reward (float): The received reward.
            next_state (int): The next state.
        """
        # Calculate the TD error
        td_error = reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, action]
        # Update the Q-value for the current state and action
        self.Q[state, action] += self.alpha * td_error

    def set_epsilon(self, epsilon: float) -> None:
        """
        Sets the exploration rate to the given value.

        Args:
            epsilon (float): The exploration rate.
        """
        self.epsilon = epsilon
