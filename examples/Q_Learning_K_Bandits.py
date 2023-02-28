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

total_rewards=[]
total_actions=[]
for _ in range(200):
    # Create environment wrapper object
    env = EnvWrapper('K_Armed_Bandit', env_cfgs=env_cfgs)

    # Get number of states and actions from the environment
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    # Update algorithm configuration with number of states and actions
    algo_cfgs['n_states'] = int(n_states)
    algo_cfgs['n_actions'] = int(n_actions)
    algo_cfgs['num_agents']=env_cfgs['num_envs']

    # Create algorithm wrapper object
    agent = AlgoWrapper('Q_Learning', algo_cfgs=algo_cfgs)

    # Set number of epochs and initialize rewards list
    Epochs = 1000
    rewards = []
    actions = []

    # Run training loop for specified number of epochs
    for epoch in range(Epochs):
        # Run an epoch of the environment with the agent and get the results
        res = env.run_epoch(agent=agent, epoch=epoch)
        target_action=np.max(env.env.q_star)
        current_action=np.max(agent.agent.Q)
        action_value=float(current_action/(target_action+0.000001))
        # Update agent's epsilon value for epsilon-greedy exploration strategy
        if algo_cfgs.get('epsilon_annealing'):
            agent.agent.epsilon_annealing(epoch, Epochs/10)
        # Append the result to rewards list
        rewards.append(res)
        actions.append(action_value)
    total_rewards.append(rewards)
    total_actions.append(actions)
    
total_rewards=np.array(total_rewards)
total_actions=np.array(total_actions)
mean_rewards=np.mean(total_rewards,axis=0)
mean_actions=np.mean(total_actions,axis=0)
rewards=mean_rewards
actions=mean_actions
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
    axs[x, y].set_xlabel('Steps')
    axs[x, y].set_ylabel('Rewards')

# Adjust the spacing between subplots
fig.subplots_adjust(hspace=0.5)

plt.show()
fig, axs = plt.subplots(2, 2)
K=[1,1,1,1]
for idx, k in enumerate(K):
    # Compute rolling average rewards using convolution with ones kernel
    rolling_avg_actions = np.convolve(actions, np.ones(k)/k, mode='valid')
    # Plot rolling average rewards in the corresponding subplot
    x = idx % 2
    y = idx // 2
    axs[x, y].plot(rolling_avg_actions, label=f'k={k}')
    axs[x, y].set_title(f'Average Action Values, Window size {k}')
    axs[x, y].set_xlabel('Steps')
    axs[x, y].set_ylabel('Action Values')

# Adjust the spacing between subplots
fig.subplots_adjust(hspace=0.5)

# Show the figure
plt.show()