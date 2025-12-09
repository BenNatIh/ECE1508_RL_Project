
import json
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC

RESULTS_DIR = Path("results")
MODELS_DIR = RESULTS_DIR / "models"
JSON_DIR = RESULTS_DIR / "json"
PLOTS_DIR = RESULTS_DIR / "plots"

for p in [RESULTS_DIR, MODELS_DIR, JSON_DIR, PLOTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

gym.register_envs(gymnasium_robotics)


class NoisyObservationWrapper(gym.ObservationWrapper):

    def __init__(self, env, noise_std: float = 0.0):
        super().__init__(env)
        self.noise_std = float(noise_std)

    def observation(self, obs):
        if self.noise_std <= 0.0:
            return obs

        noisy_obs = obs.copy()
        for key in ["observation", "achieved_goal"]:
            if key in noisy_obs:
                arr = np.asarray(noisy_obs[key], dtype=np.float32)
                arr = arr + np.random.normal(0.0, self.noise_std, size=arr.shape)
                noisy_obs[key] = arr
        return noisy_obs


def make_env(
    env_id: str,
    noise_std: float,
    max_episode_steps: int = 50,
    seed: int | None = None,
):
    env = gym.make(env_id, max_episode_steps=max_episode_steps)
    env = NoisyObservationWrapper(env, noise_std=noise_std)

    if seed is not None:
        env.reset(seed=seed)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)

    return env

def evaluate_sac(
    model,
    env,
    n_episodes: int = 100,
    success_threshold: float = 0.05,
    max_episode_steps: int = 50,
    reward_type: str = "dense",
):
    successes = []
    distances = []
    episode_rewards = []
    steps_to_success = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        t = 0
        first_success_step = None

        while not done and t < max_episode_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            t += 1

            achieved = obs["achieved_goal"]
            desired = obs["desired_goal"]
            dist = np.linalg.norm(achieved - desired)

            if first_success_step is None and dist < success_threshold:
                first_success_step = t

        final_distance = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
        distances.append(final_distance)

        if "is_success" in info:
            successes.append(info["is_success"])
        else:
            successes.append(final_distance < success_threshold)

        episode_rewards.append(episode_reward)

        if first_success_step is None:
            steps_to_success.append(max_episode_steps)
        else:
            steps_to_success.append(first_success_step)

    success_rate = float(np.mean(successes) * 100.0)
    avg_distance = float(np.mean(distances))
    avg_reward = float(np.mean(episode_rewards))
    avg_steps_to_success = float(np.mean(steps_to_success))

    if reward_type == "dense":
        eval_metric = avg_steps_to_success
        metric_name = "avg_steps_to_success"
    else:  # sparse
        eval_metric = avg_reward
        metric_name = "avg_reward"

    print("------------------------------------------------------")
    print(f"Eval ({reward_type}): success_rate      = {success_rate:.1f}%")
    print(f"Avg Distance to Goal:  {avg_distance:.4f} m")
    print(f"Avg Episode Reward:    {avg_reward:.2f}")
    print(f"Avg Steps to Success:  {avg_steps_to_success:.2f}")
    print(f"Main eval metric ({metric_name}): {eval_metric:.4f}")
    print("------------------------------------------------------")

    return {
        "success_rate": success_rate,
        "avg_distance": avg_distance,
        "avg_reward": avg_reward,
        "avg_steps_to_success": avg_steps_to_success,
        "eval_metric": float(eval_metric),
        "metric_name": metric_name,
    }


# ================================================================
# One training run for a given (reward_type, noise_std, seed)
# ================================================================

def train_sac_single_run(
    env_id: str,
    reward_type: str,
    noise_std: float,
    seed: int,
    total_steps: int = 100_000,
    eval_fraction: float = 0.05,
    max_episode_steps: int = 50,
    n_eval_episodes: int = 20,
):
    """
    Train SAC on a single (env_id, reward_type, noise_std, seed) configuration.

    Returns:
        eval_results: list of dicts with timesteps and metrics
    """
    print(
        f"\n=== Starting run: reward_type={reward_type}, env_id={env_id}, "
        f"noise_std={noise_std}, seed={seed} ==="
    )

    env = make_env(
        env_id=env_id,
        noise_std=noise_std,
        max_episode_steps=max_episode_steps,
        seed=seed,
    )

    model = SAC(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="./sac_fetchreach_tensorboard/",
        learning_starts=5_000,
        buffer_size=500_000,
        batch_size=256,
        gamma=0.98,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_entropy="auto",
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=seed,
    )

    eval_interval = int(total_steps * eval_fraction)
    assert eval_interval > 0, "eval_interval must be positive"

    eval_results = []
    timesteps_done = 0

    while timesteps_done < total_steps:
        steps_to_train = min(eval_interval, total_steps - timesteps_done)
        print(
            f"\n--- Training from {timesteps_done} "
            f"to {timesteps_done + steps_to_train} steps ---"
        )

        model.learn(
            total_timesteps=steps_to_train,
            reset_num_timesteps=False,
            progress_bar=False,
            log_interval=10,
        )
        timesteps_done += steps_to_train

        metrics = evaluate_sac(
            model,
            env,
            n_episodes=n_eval_episodes,
            max_episode_steps=max_episode_steps,
            reward_type=reward_type,
        )

        eval_results.append(
            {
                "timesteps": int(timesteps_done),
                "eval_metric": metrics["eval_metric"],
                "metric_name": metrics["metric_name"],
                "success_rate": metrics["success_rate"],
                "avg_distance": metrics["avg_distance"],
                "avg_reward": metrics["avg_reward"],
                "avg_steps_to_success": metrics["avg_steps_to_success"],
            }
        )

        print(
            f"Eval @ {timesteps_done} steps "
            f"({reward_type}, noise={noise_std}, seed={seed}): "
            f"{metrics['metric_name']} = {metrics['eval_metric']:.4f}"
        )

    # ---------------- Save model ----------------
    noise_tag = f"{noise_std}".replace(".", "_")
    model_dir = MODELS_DIR / f"sac_{reward_type}" / f"noise{noise_tag}"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"seed{seed}.zip"
    model.save(str(model_path))
    print(f"Saved model to {model_path}")

    # ---------------- Save eval JSON ----------------
    json_dir = JSON_DIR / f"sac_{reward_type}" / f"noise{noise_tag}"
    json_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "algo": "SAC",
        "env_id": env_id,
        "reward_type": reward_type,
        "noise_std": float(noise_std),
        "seed": int(seed),
        "total_steps": int(total_steps),
        "eval_interval": int(eval_interval),
        "eval_results": eval_results,
    }

    json_path = json_dir / f"seed{seed}.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved eval data to {json_path}")

    env.close()

    return eval_results


# ================================================================
# Plotting for one (reward_type, noise_std) with multi seeds
# ================================================================

def plot_multi_seed_curves(
    reward_type: str,
    noise_std: float,
    seeds: list[int],
):
    """
    Load JSONs for a given (reward_type, noise_std, seeds) and plot
    all runs on the same figure.
    """
    noise_tag = f"{noise_std}".replace(".", "_")
    json_dir = JSON_DIR / f"sac_{reward_type}" / f"noise{noise_tag}"

    curves = []

    for seed in seeds:
        json_path = json_dir / f"seed{seed}.json"
        if not json_path.exists():
            print(f"Warning: {json_path} not found, skipping seed {seed}")
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        eval_results = data["eval_results"]
        timesteps = [er["timesteps"] for er in eval_results]
        metrics = [er["eval_metric"] for er in eval_results]
        metric_name = (
            eval_results[0]["metric_name"]
            if eval_results
            else ("avg_steps_to_success" if reward_type == "dense" else "avg_reward")
        )

        curves.append(
            {
                "seed": seed,
                "timesteps": timesteps,
                "metrics": metrics,
                "metric_name": metric_name,
            }
        )

    if not curves:
        print(
            f"No data found for reward_type={reward_type}, "
            f"noise={noise_std}, skipping plot."
        )
        return

    plt.figure(figsize=(8, 5))
    for c in curves:
        plt.plot(
            c["timesteps"],
            c["metrics"],
            marker="o",
            linestyle="-",
            label=f"seed {c['seed']}",
        )

    plt.xlabel("Environment steps")
    if reward_type == "dense":
        plt.ylabel("Avg steps to success (dist < 0.05 m)")
        title_metric = "Avg Steps to Success"
    else:
        plt.ylabel("Avg episode reward")
        title_metric = "Avg Reward"

    noise_str = f"{noise_std}" if noise_std > 0 else "0 (no noise)"
    plt.title(
        f"SAC on FetchReach ({reward_type}, noise={noise_str})\n"
        f"{title_metric} vs Steps"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_dir = PLOTS_DIR / f"sac_{reward_type}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_path = plot_dir / f"sac_{reward_type}_noise{noise_tag}_multi_seed.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {plot_path}")

def run_all_experiments():
    seeds = [100, 101, 102, 103, 104]
    total_steps = 100_000
    eval_fraction = 0.05
    max_episode_steps = 50
    n_eval_episodes = 20

    configs = {
        "dense": "FetchReachDense-v4",
        "sparse": "FetchReach-v4",
    }

    noise_levels = [0.0, 0.01, 0.05, 0.1]

    for reward_type, env_id in configs.items():
        for noise_std in noise_levels:
            print("\n====================================================")
            print(
                f"Running config: reward_type={reward_type}, "
                f"env_id={env_id}, noise_std={noise_std}"
            )
            print("====================================================")

            for seed in seeds:
                train_sac_single_run(
                    env_id=env_id,
                    reward_type=reward_type,
                    noise_std=noise_std,
                    seed=seed,
                    total_steps=total_steps,
                    eval_fraction=eval_fraction,
                    max_episode_steps=max_episode_steps,
                    n_eval_episodes=n_eval_episodes,
                )

            plot_multi_seed_curves(
                reward_type=reward_type,
                noise_std=noise_std,
                seeds=seeds,
            )


if __name__ == "__main__":
    run_all_experiments()
