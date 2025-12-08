
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import matplotlib.pyplot as plt

# Register environments
gym.register_envs(gymnasium_robotics)

# DENSE REWARDS
env_id = "FetchReachDense-v4"  

# Observation noise level (std of Gaussian noise added to 'observation')
OBS_NOISE_STD = 0.1 

# Observation noise wrapper
# Adds Gaussian noise to the 'observation' part of the FetchReach dict observation.
# 'achieved_goal' and 'desired_goal' are left unchanged (clean reward signal).
class NoisyObservationWrapper(gym.ObservationWrapper):
   
    def __init__(self, env, obs_noise_std=0.05):
        super().__init__(env)
        self.obs_noise_std = obs_noise_std

    def observation(self, obs):
        # obs is a dict with keys: 'observation', 'achieved_goal', 'desired_goal'
        noisy_obs = obs.copy()

        # add random noise with specified std to 'observation'
        noisy_obs["observation"] = obs["observation"] + np.random.normal(
            loc=0.0,
            scale=self.obs_noise_std,
            size=obs["observation"].shape,
        )

        # achieved_goal and desired_goal are unchanged
        return noisy_obs


# Evaluation function: tracks average stepstane to reach goal explicitly
# Evaluate success rate, average final distance to goal, average # of steps to reach goal (over successful episodes)
def evaluate_td3(model, env, n_episodes=100):
    
    successes = []
    distances = []
    episode_rewards = []
    episode_lengths = []      # total steps per episode 
    steps_to_goal_list = []   # None if never reached goal in that episode

    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0
        steps_to_goal = None   # will be set when we first hit success
        last_distance = None

        while not done:
            # Deterministic policy during evaluation (no exploration noise)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1

            # Current distance (using clean achieved_goal / desired_goal)
            achieved = obs["achieved_goal"]
            desired = obs["desired_goal"]
            dist = np.linalg.norm(achieved - desired)
            last_distance = dist

            # Check for success at this step
            if "is_success" in info:
                is_success_step = bool(info["is_success"])
            else:
                is_success_step = dist < 0.05  # threshold definition for a successful episode

            # Record the first step where success happens
            if is_success_step and (steps_to_goal is None):
                steps_to_goal = steps

        # End of episode
        episode_lengths.append(steps)
        distances.append(last_distance if last_distance is not None else np.nan)
        steps_to_goal_list.append(steps_to_goal)
        episode_rewards.append(episode_reward)

        # Episode-level success flag (to record if goal was ever reached)
        successes.append(steps_to_goal is not None)

    successes = np.array(successes, dtype=bool)
    success_rate = successes.mean() * 100.0 if len(successes) > 0 else 0.0
    avg_distance = float(np.nanmean(distances)) if distances else np.nan

    # Average steps to goal (calculated using successful episodes only)
    # If there were no successes, report the max episode length
    successful_steps = [s for s in steps_to_goal_list if s is not None]
    if len(successful_steps) > 0:
        avg_steps_to_goal = float(np.mean(successful_steps))
    else:
        # No successes at all during this evaluation
        avg_steps_to_goal = float(max(episode_lengths)) if episode_lengths else np.nan

    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Avg Distance to Goal: {avg_distance:.4f} m")
    print(f"Avg Steps to Goal (successes): {avg_steps_to_goal:.2f}")
    print(f"Best 10% (distance): {np.percentile(distances, 10):.4f} m")
    print(f"Worst 10% (distance): {np.percentile(distances, 90):.4f} m")

    return success_rate, avg_distance, avg_steps_to_goal, distances

# random seet settings
seeds = [100, 101, 102, 103, 104]

eval_interval = 750       # env steps between evaluations
total_steps   = 15_000    # total env steps per seed
n_intervals   = total_steps // eval_interval  

# Store per-seed results
all_eval_timesteps   = {}
all_eval_successes   = {}
all_eval_avg_steps   = {}   # avg steps to goal
all_eval_avg_dists   = {}
final_distances_per_seed = {}

# Main loop over seeds
for seed in seeds:
    print(f"  Training TD3 (DENSE + noisy obs) with seed {seed}")

    # Base env (no modification)
    base_env = gym.make(env_id, max_episode_steps=50)

    # Wrap environment with observation noise to simulate noisy sensors
    env = NoisyObservationWrapper(base_env, obs_noise_std=OBS_NOISE_STD)
    env.reset(seed=seed)

    print(f"DENSE REWARDS + NOISY OBS: {env_id}, obs_noise_std={OBS_NOISE_STD}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Add action noise for exploration
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    # Initialize TD3 
    model = TD3(
        "MultiInputPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        tensorboard_log="./td3_fetchreach_dense_noisy_tensorboard/",
        
        # Key hyperparameters for dense rewards
        learning_starts=5_000,    
        buffer_size=500_000,
        batch_size=256,
        gamma=0.98,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        policy_delay=2,
        
        # Network architecture
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])),
        
        seed=seed
    )

    # Per-seed evaluation traces
    eval_timesteps   = []
    eval_successes   = []
    eval_avg_steps   = []  
    eval_avg_dists   = []

    current_total = 0 

    # Incremental training + evaluation 
    for _ in range(n_intervals):
        print(f"\n=== [Seed {seed}] Training from {current_total} to {current_total + eval_interval} steps (dense + noisy obs) ===")

        # Train for exactly eval_interval environment steps
        model.learn(
            total_timesteps=eval_interval,
            reset_num_timesteps=False,
            log_interval=10,
            progress_bar=False,
        )

        current_total += eval_interval

        # Evaluate on 10 episodes
        success_rate, avg_dist, avg_steps_to_goal, _ = evaluate_td3(
            model, env, n_episodes=10
        )

        eval_timesteps.append(current_total)
        eval_successes.append(success_rate)
        eval_avg_steps.append(avg_steps_to_goal)
        eval_avg_dists.append(avg_dist)

        print(
            f"[Seed {seed}] Steps: {current_total} | "
            f"Success Rate: {success_rate:.1f}% | "
            f"Avg Steps to Goal: {avg_steps_to_goal:.2f} | "
            f"Avg Dist: {avg_dist:.4f} m"
        )

    # Final, more thorough evaluation after training
    print(f"\n=== [Seed {seed}] Final evaluation over 100 episodes (dense + noisy obs) ===")
    success_rate_dense, avg_dist_dense, avg_steps_dense, final_distances_dense = evaluate_td3(
        model, env, n_episodes=100
    )

    # SB3 built-in evaluation 
    mean_reward_dense, std_reward_dense = evaluate_policy(
        model,
        env,
        n_eval_episodes=100,
        deterministic=True
    )
    print(f"\n[Seed {seed}] SB3 evaluate_policy (dense + noisy obs): {mean_reward_dense:.2f} ± {std_reward_dense:.2f}")

    # Save model for this seed
    model.save(f"td3_dense_noisy_obs_seed{seed}")

    # Store results for plotting after all seeds are done
    all_eval_timesteps[seed]   = eval_timesteps
    all_eval_successes[seed]   = eval_successes
    all_eval_avg_steps[seed]   = eval_avg_steps
    all_eval_avg_dists[seed]   = eval_avg_dists
    final_distances_per_seed[seed] = final_distances_dense

    # Close env for this seed
    env.close()

# Plot 1: Learning curve – avg steps to goal vs env steps (all seeds)
plt.figure(figsize=(10, 4))
for seed in seeds:
    plt.plot(
        all_eval_timesteps[seed],
        all_eval_avg_steps[seed],
        marker="o",
        label=f"seed={seed}"
    )
plt.xlabel("Environment steps")
plt.ylabel("Average steps to goal (10-episode eval)")
plt.title(f"TD3 (Dense + noisy obs, σ={OBS_NOISE_STD}) – Avg Steps to Goal vs Steps")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("td3_dense_noisy0.1_learning_avg_steps_multiseed.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

# Plot 2: Learning curve – success rate vs env steps (all seeds)
plt.figure(figsize=(10, 4))
for seed in seeds:
    plt.plot(
        all_eval_timesteps[seed],
        all_eval_successes[seed],
        marker="o",
        label=f"seed={seed}"
    )
plt.xlabel("Environment steps")
plt.ylabel("Success rate (%)")
plt.title(f"TD3 (Dense + noisy obs, σ={OBS_NOISE_STD}) – Success Rate vs Steps (multi-seed)")
plt.ylim(0, 105)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("td3_dense_noisy0.1_learning_success_multiseed.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

# Plot 3: Final distance distribution (all seeds)
plt.figure(figsize=(10, 4))

# Left: overlay histograms per seed
plt.subplot(1, 2, 1)
for seed in seeds:
    plt.hist(
        final_distances_per_seed[seed],
        bins=20,
        alpha=0.4,
        edgecolor='black',
        label=f"seed={seed}"
    )
plt.axvline(0.05, color='red', linestyle='--', label='Success threshold (0.05m)')
plt.xlabel('Distance to Goal (m)')
plt.ylabel('Frequency')
plt.title(f'Final Distance Distribution (Dense + noisy obs, σ={OBS_NOISE_STD}, multi-seed)')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

# Right: distance per episode (100 eval) for each seed
plt.subplot(1, 2, 2)
for seed in seeds:
    plt.plot(final_distances_per_seed[seed], label=f"seed={seed}")
plt.axhline(0.05, color='red', linestyle='--', label='Success threshold')
plt.xlabel('Episode')
plt.ylabel('Distance to Goal (m)')
plt.title('Distance per Episode (100 eval, Dense + noisy obs, multi-seed)')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("td3_dense_noisy0.1_sb3_multiseed.png", dpi=150, bbox_inches='tight')
plt.show()
plt.close()

print("\n TD3 Dense + noisy observation multi-seed training complete!")
print(" Learning curves saved as 'td3_dense_noisy0.1_learning_avg_steps_multiseed.png' "
      "and 'td3_dense_noisy0.1_learning_success_multiseed.png'")
print(" Final distance plots saved as 'td3_dense_noisy0.1_sb3_multiseed.png'")

