import os
import sys
from matplotlib import pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from omniairl.utils.yaml_utils import get_default_yaml
from omniairl.wrappers.algo_wrapper import AlgoWrapper
from omniairl.wrappers.env_wrapper import EnvWrapper
import numpy as np

K=5

env_cfgs, algo_cfgs=get_default_yaml('Q_Learning', 'K_Armed_Bandit')
env=EnvWrapper('K_Armed_Bandit', env_cfgs=env_cfgs)
n_states=env.observation_space.shape[0]
n_actions=env.action_space.shape[0]
algo_cfgs['n_states']=int(n_states)
algo_cfgs['n_actions']=int(n_actions)
agent=AlgoWrapper('Q_Learning', algo_cfgs=algo_cfgs)
Epochs=10000
rewards=[]

for epoch in range(Epochs):
    res=env.run_epoch(agent=agent, epoch=epoch)
    agent.agent.epsilon_annealing(epoch, Epochs/2)
    rewards.append(res)


# 计算平均值和方差
avg_reward = np.mean(rewards)
std_reward = np.std(rewards)

# 绘制每个epoch的reward
plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(rewards)), rewards)
plt.title('Reward per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Reward')
plt.show()

# 计算滑动平均
rolling_avg_rewards = np.convolve(rewards, np.ones(K)/K, mode='valid')

# 绘制滑动平均曲线
plt.figure(figsize=(10, 5))
plt.plot(np.arange(K-1, len(rewards)), rolling_avg_rewards)
plt.title(f'Rolling Average Reward ({K} epochs window)')
plt.xlabel('Epoch')
plt.ylabel('Rolling Average Reward')
plt.show()


