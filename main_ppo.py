import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from model import ActorCriticNetwork
from ppo_agent import Agent
from noisy_wrapper import NoisyObservationWrapper
import matplotlib.pyplot as plt
import time
import json


train = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

# Environment selection: 'reach'
TASK = 'reach'

if TASK == 'reach':
	env_name = "FetchReach-v4"
	gamma = 1.0
elif TASK == 'reach-dense':
	env_name = "FetchReachDense-v4"
	gamma = 0.99

# Toggle noisy observations for training
USE_NOISY_OBS = False
noise = 0.05  #standard deviation of Gaussian noise to add

#Toggle Hindsight Experience Replay
USE_HER = True
# Creating environment (may be wrapped for noisy observations)
env = gym.make(env_name)
if USE_NOISY_OBS:
	env = NoisyObservationWrapper(env, noise_std={'observation': noise, 'achieved_goal': noise}, keep_clean=True, seed=42)

#temp env to check state_dim and action_dim
state_dim = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
action_dim = env.action_space.shape[0]
env.close()

n_episodes = 2500

lr = 3e-4
BS = 2048
clip_ratio = 0.2
entropy_coef = 0.01
epochs = 10

if train:
	results = []

	# Multi-run support: run training multiple times with different seeds and collect histories
	def run_one(seed, use_noisy=USE_NOISY_OBS):
		np.random.seed(seed)
		torch.manual_seed(seed)
		# create a fresh env per run
		train_env = gym.make(env_name)
		if use_noisy:
			train_env = NoisyObservationWrapper(train_env, noise_std={'observation': noise, 'achieved_goal': noise}, keep_clean=True, seed=seed)

		actor_model = ActorCriticNetwork(state_dim, action_dim).to(device)
		critic_model = ActorCriticNetwork(state_dim, 1).to(device)
		actor_optimizer = torch.optim.Adam(actor_model.parameters(), lr=lr)
		critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=3e-4)
		# number of HER samples to get from trajectories
		her_k = 4
		agent = Agent(train_env, state_dim, action_dim, gamma=gamma, actor_model=actor_model, critic_model=critic_model, actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer, device=device, entropy_coef=entropy_coef, her_k=her_k, use_her=USE_HER)
		agent.clip_ratio = clip_ratio

		# evaluate initial on clean env
		eval_env = gym.make(env_name)
		init_r, init_len, init_success = agent.evaluate_policy(n_eval_episodes=5, env_override=eval_env)

		# train
		agent.train(n_episodes, BS, epochs, plot=False)

		# evaluate final on clean env
		final_r, final_len, final_success = agent.evaluate_policy(n_eval_episodes=5, env_override=eval_env)

		# save weights per-seed
		torch.save(actor_model.state_dict(), f'actor_final_seed{seed}.pth')
		torch.save(critic_model.state_dict(), f'critic_final_seed{seed}.pth')

		# collect history
		history = {
			'seed': seed,
			'init_reward': init_r,
			# number of timesteps to reach goal
			'init_dist': init_len,
			'init_success_rate': init_success,
			'final_reward': final_r,
			# episode length
			'final_dist': final_len,
			'final_success_rate': final_success,
			'eval_rewards': agent.history.get('eval_rewards', []).copy(),
			'eval_episodes': agent.history.get('eval_episodes', []).copy(),
			'eval_success_rate': agent.history.get('eval_success_rate', []).copy()
		}

		train_env.close()
		eval_env.close()
		return history


	N_RUNS = 1 #number of seeds to run
	base_seed = 100
	all_histories = []
	for i in range(N_RUNS):
		seed = base_seed + i
		print(f"\n=== Run seed={seed} ===")
		hist = run_one(seed, use_noisy=USE_NOISY_OBS)
		all_histories.append(hist)
		results.append({
			'seed': seed,
			'final_reward': hist['final_reward'],
			'final_dist': hist['final_dist']
		})

	#save evaluation metrics to JSON for later plotting/combination
	eval_output = []
	for h in all_histories:
		eval_output.append({
			'seed': h.get('seed'),
			'init_reward': h.get('init_reward'),
			'final_reward': h.get('final_reward'),
			'final_dist': h.get('final_dist'),
			'eval_episodes': h.get('eval_episodes', []),
			'eval_rewards': h.get('eval_rewards', []),
			'eval_success_rate': h.get('eval_success_rate', [])
		})

	with open('eval_metrics_multiple_runs.json', 'w') as jf:
		json.dump(eval_output, jf, indent=2)

	print('Saved evaluation metrics to eval_metrics_multiple_runs.json')

	#plot eval rewards during training
	plt.figure(figsize=(10, 6))
	for h in all_histories:
		if 'eval_rewards' in h and h['eval_rewards']:
			plt.plot(h['eval_episodes'], h['eval_rewards'], marker='o', label=f"seed={h['seed']}", linewidth=2, alpha=0.8, markersize=4)
	plt.title('Evaluation Rewards During Training', fontsize=14)
	plt.xlabel('Episode', fontsize=12)
	plt.ylabel('Average Rewards/timesteps', fontsize=12)
	plt.legend(fontsize=10)
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig('eval_rewards.png', dpi=150)
	print('Saved eval rewards plot to eval_rewards.png')

	# Render using the last trained seed's actor (if available)
	last_seed = base_seed + N_RUNS - 1
	actor_path = f'actor_final_seed{last_seed}.pth'
else:
	# For rendering, load the last trained actor
	actor_path = f'actor_final_seed{100}.pth'
try:
	test_actor = ActorCriticNetwork(state_dim, action_dim).to(device)
	test_actor.load_state_dict(torch.load(actor_path, map_location=device))
	test_actor.eval()

	test_env = gym.make(env_name, render_mode='human')
	n_test_episodes = 5
	for ep in range(n_test_episodes):
		state_dict, _ = test_env.reset()
		state = np.concatenate([state_dict['observation'], state_dict['desired_goal']])
		done = False
		total_r = 0.0
		while not done:
			state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
			with torch.no_grad():
				action_mu = test_actor(state_tensor)
			action = torch.tanh(action_mu).cpu().numpy().squeeze(0)
			action = np.clip(action, -1.0, 1.0)
			next_state_dict, reward, terminated, truncated, _ = test_env.step(action)
			total_r += float(reward)  # Cast reward to float
			done = terminated or truncated
			state = np.concatenate([next_state_dict['observation'], next_state_dict['desired_goal']])
			# explicit render call in case env doesn't auto-render on step
			try:
				test_env.render()
			except Exception:
				pass
		print(f'Test Episode {ep+1}: Total Reward = {total_r:.3f}')
	test_env.close()
except Exception as e_load:
	print(f'Could not load actor for rendering ({actor_path}):', e_load)



