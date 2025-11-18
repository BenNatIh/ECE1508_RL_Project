import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from model import ActorCriticNetwork
from ppo_agent import Agent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Creating environment
env = gym.make("FetchReachDense-v4")

state_dim = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
action_dim = env.action_space.shape[0]

gamma = 0.99
n_episodes = 500

# Grid search hyperparameters
lr_list = [1e-4, 5e-4, 1e-3]
batch_size_list = [1024, 2048]
clip_ratio_list = [0.1, 0.2]
entropy_coef_list = [0.0, 0.01]
epochs_list = [5, 10]

results = []
for lr in lr_list:
	for batch_size in batch_size_list:
		for clip_ratio in clip_ratio_list:
			for entropy_coef in entropy_coef_list:
				for epochs in epochs_list:
					print(f"\n=== Running: lr={lr}, batch_size={batch_size}, clip_ratio={clip_ratio}, entropy_coef={entropy_coef}, epochs={epochs} ===")
					env = gym.make("FetchReachDense-v4")
					actor_model = ActorCriticNetwork(state_dim, action_dim).to(device)
					critic_model = ActorCriticNetwork(state_dim, 1).to(device)
					actor_optimizer = torch.optim.Adam(actor_model.parameters(), lr=lr)
					critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=lr)
					agent = Agent(env, state_dim, action_dim, gamma=gamma, actor_model=actor_model, critic_model=critic_model, actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer, device=device, entropy_coef=entropy_coef)
					agent.clip_ratio = clip_ratio
					avg_reward, avg_dist = agent.evaluate_policy(n_eval_episodes=10)
					print(f"Initial eval: avg_reward={avg_reward:.3f}, avg_dist={avg_dist}")
					agent.train(n_episodes, batch_size, epochs, plot=False)
					avg_reward, avg_dist = agent.evaluate_policy(n_eval_episodes=10)
					print(f"Final eval: avg_reward={avg_reward:.3f}, avg_dist={avg_dist}")
					results.append({
						'lr': lr,
						'batch_size': batch_size,
						'clip_ratio': clip_ratio,
						'entropy_coef': entropy_coef,
						'epochs': epochs,
						'avg_reward': avg_reward,
						'avg_dist': avg_dist
					})
					env.close()

print("\n=== Grid Search Results ===")
for res in results:
	print(res)