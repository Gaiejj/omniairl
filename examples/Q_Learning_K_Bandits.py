import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from omniairl.utils.yaml_utils import get_default_yaml
from omniairl.wrappers.algo_wrapper import AlgoWrapper
from omniairl.wrappers.env_wrapper import EnvWrapper

env_cfgs, algo_cfgs=get_default_yaml('Q_Learning', 'K_Armed_Bandit')
env=EnvWrapper('K_Armed_Bandit', env_cfgs=env_cfgs)
n_states=env.observation_space.shape[0]
n_actions=env.action_space.shape[0]
algo_cfgs['n_states']=int(n_states)
algo_cfgs['n_actions']=int(n_actions)
agent=AlgoWrapper('Q_Learning', algo_cfgs=algo_cfgs)
Epochs=100

for epoch in range(Epochs):
    env.run_epoch(agent=agent, epoch=epoch)
