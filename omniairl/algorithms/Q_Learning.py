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

    def __init__(
            self, 
            n_states: int, 
            n_actions: int, 
            alpha: float = 0.5, 
            gamma: float = 0.9, 
            epsilon: float = 0.1, 
            update_method: str='average', 
            use_q_noise: bool=False,
            epsilon_annealing: bool=False,
            ) -> None:
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
        self.epsilon = epsilon if not epsilon_annealing else 1.0
        self.init_epsilon = 1.0
        self.Q = np.zeros((n_states, n_actions))
        self.update_method=update_method
        self.counts=0
        self.use_q_noise=use_q_noise

    def choose_action(self, state: int) -> int:
        """
        Chooses an action based on the current state and the exploration rate.

        Args:
            state (int): The current state.

        Returns:
            int: The chosen action.
        """
        # Add Gaussian noise to Q-table
        if self.use_q_noise:
            self.Q += np.random.normal(0, 0.01, size=(self.n_states, self.n_actions))
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
        self.counts+=1
        # Update the Q-value for the current state and action
        if self.update_method=='average':
            self.Q[state, action] += (reward-self.Q[state, action])/self.counts
        elif self.update_method=='td':
            self.Q[state, action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, action])
        else: 
            self.Q[state, action] += (reward-self.Q[state, action])*self.alpha

    def epsilon_annealing(self, epoch: int, end_epoch: int) -> None:
        """
        Sets the exploration rate to the given value.

        Args:
            epsilon (float): The exploration rate.
        """
        self.epsilon = self.init_epsilon*min((1-epoch/end_epoch), 1)+0.1
