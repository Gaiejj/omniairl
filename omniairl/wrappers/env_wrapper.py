import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from omniairl.envs.env_registry import env_registry
import numpy as np
from typing import Any

class EnvWrapper:
    """Wrapper class for RL environment.

    The EnvWrapper class wraps an environment from the `env_registry` dictionary and provides methods for running a
    single epoch of the environment with an agent, and for choosing actions and updating the agent's model.

    Args:
        env: A string specifying the key of the environment to wrap.

    Attributes:
        env: An instance of the environment being wrapped.
        episode_rewards: A list containing the total reward obtained in each episode.
    """
    def __init__(self, env: str, env_cfgs: dict) -> None:
        self.env = env_registry.get(env)(**env_cfgs)
        self.action_space=self.env.action_space
        self.observation_space=self.env.observation_space
        self.episode_rewards = []
    
    def run_epoch(self, agent: Any, epoch: int) -> float:
        """Run a single epoch of the environment with an agent.

        Args:
            agent: An instance of the agent to use for interacting with the environment.
            num_episodes: An integer specifying the number of episodes to run.

        Returns:
            The average reward obtained over the specified number of episodes.
        """
        episode_rewards = []
        
        state = self.env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)
            agent.update(state, action, reward, next_state)
            episode_reward += reward
            state = next_state
            
        episode_rewards.append(episode_reward)
        print(f"Epoch {epoch+1}: reward={episode_reward}")
        
        self.episode_rewards.extend(episode_rewards)
        return np.mean(episode_rewards)



