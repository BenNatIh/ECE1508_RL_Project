
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import matplotlib.pyplot as plt

# Register environments
gym.register_envs(gymnasium_robotics)

env_id = "FetchReach-v4"   # Sparse rewards version


# Observation noise wrapper
# Adds Gaussian noise to the 'observation' part of the FetchReach dict observation.
# 'achieved_goal' and 'desired_goal' are left unchanged (clean reward signal).
class NoisyObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, obs_noise_std=0.05):
        super().__init__(env)
        self.obs_noise_std = obs_noise_std

    def observation(self, obs):
        noisy_obs = obs.copy()
        noisy_obs["observation"] = obs["observation"] + np.random.normal(
            loc=0.0, 
            scale=self.obs_noise_std, 
            size=obs["observation"].shape
        )
        return noisy_obs


# Evaluate function (sparse version)
# Evaluates success rate, average final distance, average episode reward
def evaluate_td3(model, env, n_episodes=100):
    
    successes = []
    distances = []
    episode_rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        achieved = obs["achieved_goal"]
        desired = obs["desired_goal"]
        dist = np.linalg.norm(achieved - desired)
        distances.append(dist)

        if "is_success" in info:
            successes.append(info["is_success"])
        else:
            successes.append(dist < 0.05)

        episode_rewards.append(episode_reward)

    success_rate = np.mean(successes) * 100.0
    avg_distance = np.mean(distances)
    avg_reward = np.mean(episode_rewards)
    return success_rate, avg_distance, avg_reward, distances


# noise level and random seed settings
noise_levels = [0.01, 0.05, 0.1]     
seeds        = [100, 101, 102, 103, 104]

eval_interval = 5_000
total_steps   = 100_000
n_intervals   = total_steps // eval_interval  


# Outer Loop (for each noise level)
for noise_std in noise_levels:
    print(f" Running experiment for obs_noise_std = {noise_std}")

    # storage for multi-seed curves
    all_eval_timesteps   = {}
    all_eval_successes   = {}
    all_eval_avg_rewards = {}
    all_eval_avg_dists   = {}
    final_distances_per_seed = {}

    # store final eval stats per seed for this noise level
    final_success_rates_per_seed = []
    final_avg_rewards_per_seed   = []

    # Loop over seeds
    for seed in seeds:
        print(f"  Training TD3 (sparse + noisy) seed={seed}, σ={noise_std}")

        # Base env
        base_env = gym.make(env_id, max_episode_steps=50)

        # Wrap env with noise
        env = NoisyObservationWrapper(base_env, obs_noise_std=noise_std)
        env.reset(seed=seed)

        print(f"SPARSE REWARDS + NOISY OBS: {env_id}, obs_noise_std={noise_std}")

        # Add action noise
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )

        # TD3 model 
        model = TD3(
            "MultiInputPolicy",
            env,
            action_noise=action_noise,
            verbose=0,  # keep SB3 logs quiet
            learning_starts=10_000,
            buffer_size=1_000_000,
            batch_size=512,
            gamma=0.99,
            tau=0.005,
            train_freq=1,
            gradient_steps=2,
            policy_delay=2,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])),
            seed=seed
        )

        # Seed-level logs
        eval_timesteps   = []
        eval_successes   = []
        eval_avg_rewards = []
        eval_avg_dists   = []

        # Incremental training + evaluation
        for i in range(n_intervals):
            step_start = i * eval_interval
            step_end   = (i + 1) * eval_interval

            # Train exactly eval_interval environment steps
            model.learn(
                total_timesteps=eval_interval,
                reset_num_timesteps=False,
                progress_bar=False,
            )

            # Evaluate 20 episodes
            success, avg_dist, avg_reward, _ = evaluate_td3(model, env, n_episodes=20)

            current_steps = step_end
            eval_timesteps.append(current_steps)
            eval_successes.append(success)
            eval_avg_rewards.append(avg_reward)
            eval_avg_dists.append(avg_dist)

        
        # Final, more thorough evaluation after training
        print(f"\n[Seed {seed}, σ={noise_std}] Final 50-episode evaluation...")
        success_final, avg_dist_final, avg_reward_final, final_dists = evaluate_td3(
            model, env, n_episodes=50
        )

        print(
            f"[Seed {seed}, σ={noise_std}] FINAL EVAL | "
            f"Success: {success_final:.1f}% | "
            f"Avg Reward: {avg_reward_final:.2f} | "
            f"Avg Dist: {avg_dist_final:.4f} m"
        )


        # collect final metrics per seed for aggregation later
        final_success_rates_per_seed.append(success_final)
        final_avg_rewards_per_seed.append(avg_reward_final)

        final_distances_per_seed[seed] = final_dists

        # Save seed-specific results
        all_eval_timesteps[seed]   = eval_timesteps
        all_eval_successes[seed]   = eval_successes
        all_eval_avg_rewards[seed] = eval_avg_rewards
        all_eval_avg_dists[seed]   = eval_avg_dists

        env.close()

    # Aggregate final evaluation across seeds for this noise level 
    mean_final_success = np.mean(final_success_rates_per_seed)
    std_final_success  = np.std(final_success_rates_per_seed)

    mean_final_reward  = np.mean(final_avg_rewards_per_seed)
    std_final_reward   = np.std(final_avg_rewards_per_seed)

    print("\n====================================================")
    print(f" FINAL EVALUATION SUMMARY for σ={noise_std}")
    print("====================================================")
    print(
        f"Success rate across seeds (mean ± std): "
        f"{mean_final_success:.1f}% ± {std_final_success:.1f}%"
    )
    print(
        f"Avg episode reward across seeds (mean ± std): "
        f"{mean_final_reward:.2f} ± {std_final_reward:.2f}"
    )
    print(f"(Per-seed final success rates: {final_success_rates_per_seed})")
    print(f"(Per-seed final avg rewards:   {final_avg_rewards_per_seed})")

    # Make 3 plots for each noise level 
    # Plot 1: Avg Reward vs Steps
    plt.figure(figsize=(10, 4))
    for seed in seeds:
        plt.plot(
            all_eval_timesteps[seed],
            all_eval_avg_rewards[seed],
            marker="o",
            label=f"seed={seed}"
        )
    plt.xlabel("Environment steps")
    plt.ylabel("Avg episode reward")
    plt.title(f"TD3 Sparse + Noisy Obs (σ={noise_std}) – Avg Reward vs Steps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"td3_sparse_noisy_reward_{noise_std}v2.png", dpi=150)
    plt.close()

    # Plot 2: Success Rate vs Steps
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
    plt.title(f"TD3 Sparse + Noisy Obs (σ={noise_std}) – Success Rate vs Steps")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"td3_sparse_noisy_success_{noise_std}v2.png", dpi=150)
    plt.close()

    # Plot 3: Final Distance Distribution 
    plt.figure(figsize=(10, 4))

    # left graph: histogram per seed
    plt.subplot(1, 2, 1)
    for seed in seeds:
        plt.hist(
            final_distances_per_seed[seed],
            bins=20,
            alpha=0.4,
            edgecolor="black",
            label=f"seed={seed}"
        )
    plt.axvline(0.05, color="red", linestyle="--", label="Success threshold")
    plt.title(f"Final Distance Dist. (σ={noise_std})")
    plt.xlabel("Distance to goal")
    plt.ylabel("Frequency")
    plt.legend(fontsize=7)
    plt.grid(True, alpha=0.3)

    # right graph: Distance-per-episode 
    plt.subplot(1, 2, 2)
    for seed in seeds:
        plt.plot(final_distances_per_seed[seed], label=f"seed={seed}")
    plt.axhline(0.05, color="red", linestyle="--")
    plt.title(f"100-Episode Distances (σ={noise_std})")
    plt.xlabel("Episode")
    plt.ylabel("Distance")
    plt.legend(fontsize=7)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"td3_sparse_noisy_finaldist_{noise_std}v2.png", dpi=150)
    plt.close()

    print(f"\nFinished noise level σ={noise_std}. 3 plots saved.")


