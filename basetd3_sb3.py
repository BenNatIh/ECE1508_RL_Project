# td3_fetchreach_dense_v4.py
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import matplotlib.pyplot as plt

# register environments
gym.register_envs(gymnasium_robotics)


env_id = "FetchReachDense-v4"

# Make environment
env = gym.make(env_id, max_episode_steps=50)
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

# add Gaussian noise to the deterministic policy output (i.e the action) to encourage exploration during training
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# initialize TD3 model
model = TD3(
    "MultiInputPolicy",
    env,
    action_noise=action_noise,
    verbose=1,
    tensorboard_log="./td3_fetchreach_dense_tensorboard/",
    
    # Key hyperparameters for dense rewards
    learning_starts=5_000,      # no gradient updates until X transitions are stored in the replay buffer
    buffer_size=500_000,        # replay buffer can hold X transitions
    batch_size=256,             # each gradient update samples a batch of X transitions
    gamma=0.98,                 # discount factor
    tau=0.005,                  # Polyak averaging step size for target network updates
    train_freq=1,
    gradient_steps=1,           # after every environment step, do one gradient update
    policy_delay=2,             # TD3's delayed policy updates
    
    # Network architecture (actor and critic networks both use 2 hidden layers of size 256)
    policy_kwargs=dict(net_arch=[256, 256]),
    
    seed=42
)

# Train the model (300k environment steps)
model.learn(total_timesteps=100_000, log_interval=10, progress_bar=True)

# Save the model
model.save("basetd3_sb3")

# Evaluation function
def evaluate_td3(model, env, n_episodes=100):
    """Evaluate success rate, average distance, and episode rewards."""
    successes = []
    distances = []
    episode_rewards = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            # Deterministic actions = no exploration noise during eval
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        # Compute final distance from last observation
        achieved = obs["achieved_goal"]
        desired = obs["desired_goal"]
        final_distance = np.linalg.norm(achieved - desired)
        distances.append(final_distance)
        
        # Get success flag if provided, else infer from distance
        if "is_success" in info:
            successes.append(info["is_success"])
        else:
            successes.append(final_distance < 0.05)  # Fetch success threshold
        
        episode_rewards.append(episode_reward)
    
    success_rate = np.mean(successes) * 100
    avg_distance = np.mean(distances)
    avg_reward = np.mean(episode_rewards)
    
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Avg Distance to Goal: {avg_distance:.4f} m")
    print(f"Avg Episode Reward: {avg_reward:.2f}")
    print(f"Best 10%: {np.percentile(distances, 10):.4f} m")
    print(f"Worst 10%: {np.percentile(distances, 90):.4f} m")
    
    return success_rate, avg_distance, avg_reward, distances




# Run evaluation
success_rate, avg_dist, avg_reward, final_distances = evaluate_td3(model, env, 100)

# stable-baseline3 built-in evaluation function
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)
print(f"\nSB3 evaluate_policy: {mean_reward:.2f} ± {std_reward:.2f}")

# close environment
env.close()

# Optional: Plot distance distribution
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(final_distances, bins=20, alpha=0.7, edgecolor='black')
plt.axvline(0.05, color='red', linestyle='--', label='Success threshold (0.05m)')
plt.xlabel('Distance to Goal (m)')
plt.ylabel('Frequency')
plt.title('Final Distance Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(final_distances)
plt.axhline(0.05, color='red', linestyle='--', label='Success threshold')
plt.xlabel('Episode')
plt.ylabel('Distance to Goal (m)')
plt.title('Distance per Episode (100 eval)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("basetd3_sb3.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n TD3 training complete! Model saved as 'basetd3_sb3.zip'")
print(" Evaluation plots saved as 'basetd3_sb3.png'")