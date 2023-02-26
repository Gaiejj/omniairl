import os
import sys
from matplotlib import pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import necessary modules and packages
from omniairl.utils.yaml_utils import get_default_yaml
from omniairl.wrappers.algo_wrapper import AlgoWrapper
from omniairl.wrappers.env_wrapper import EnvWrapper
import numpy as np

# Load default configuration for environment and algorithm
env_cfgs, algo_cfgs = get_default_yaml('Q_Learning', 'K_Armed_Bandit')

# Create environment wrapper object
env = EnvWrapper('K_Armed_Bandit', env_cfgs=env_cfgs)

# Get number of states and actions from the environment
n_states = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]

# Update algorithm configuration with number of states and actions
algo_cfgs['n_states'] = int(n_states)
algo_cfgs['n_actions'] = int(n_actions)

# Create algorithm wrapper object
agent = AlgoWrapper('Q_Learning', algo_cfgs=algo_cfgs)

# Set number of epochs and initialize rewards list
Epochs = 10000
rewards = []

# Run training loop for specified number of epochs
for epoch in range(Epochs):
    # Run an epoch of the environment with the agent and get the results
    res = env.run_epoch(agent=agent, epoch=epoch)
    # Update agent's epsilon value for epsilon-greedy exploration strategy
    agent.agent.epsilon_annealing(epoch, Epochs/2)
    # Append the result to rewards list
    rewards.append(res)

# Compute mean and standard deviation of rewards
avg_reward = np.mean(rewards)
std_reward = np.std(rewards)

# Create a figure with four subplots
fig, axs = plt.subplots(2, 2)

# Define a list of window sizes for rolling average
K = [1, 5, 10, 20]

# Plot rolling average rewards for different window sizes
for idx, k in enumerate(K):
    # Compute rolling average rewards using convolution with ones kernel
    rolling_avg_rewards = np.convolve(rewards, np.ones(k)/k, mode='valid')
    # Plot rolling average rewards in the corresponding subplot
    x = idx % 2
    y = idx // 2
    axs[x, y].plot(rolling_avg_rewards, label=f'k={k}')
    axs[x, y].set_title(f'Average Reward, Window size {k}')
    axs[x, y].set_xlabel('Epochs')
    axs[x, y].set_ylabel('Rewards')

# Adjust the spacing between subplots
fig.subplots_adjust(hspace=0.5)

# Show the figure
plt.show()

