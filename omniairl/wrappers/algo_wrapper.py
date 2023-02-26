import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from omniairl.utils.yaml_utils import get_default_yaml
from omniairl.algorithms.algo_registry import algo_registry


class AlgoWrapper:
    """Algo Wrapper for algo."""

    def __init__(self, algo, algo_cfgs=None):
        self.algo = algo
        self.evaluator = None
        print(algo_registry.get(self.algo))
        self.agent=algo_registry.get(self.algo)(**algo_cfgs)
        self._init_checks()

    def _init_checks(self):
        """Init checks."""

    def choose_action(self, state: int) -> int:
        """Choose action"""
        return self.agent.choose_action(state)
    
    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """Update"""
        self.agent.update(state, action, reward, next_state)

    def evaluate(self, num_episodes: int = 10, horizon: int = 1000, cost_criteria: float = 1.0):
        """Agent Evaluation."""
        assert self.evaluator is not None, 'Please run learn() first!'
        self.evaluator.evaluate(num_episodes, horizon, cost_criteria)

    # pylint: disable-next=too-many-arguments
    def render(
        self,
        num_episode: int = 0,
        horizon: int = 1000,
        seed: int = None,
        play=True,
        save_replay_path: str = None,
    ):
        """Render the environment."""
        assert self.evaluator is not None, 'Please run learn() first!'
        self.evaluator.render(num_episode, horizon, seed, play, save_replay_path)