import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import RescaleAction
import numpy as np

def make_fetchreach(seed=0):
    env = gym.make("FetchReach-v4",render_mode="human")
    env.reset(seed=seed)
    return env

def run_baseline(episodes=50, k_p=5.0, seed=0):
    env = make_fetchreach(seed=seed)
    succ, dists = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            pos = obs["observation"][:3]  
            goal = obs["desired_goal"]
            err = (goal - pos)
            delta_xyz = np.clip(k_p * err, -1.0, 1.0)
            action = np.array([delta_xyz[0], delta_xyz[1], delta_xyz[2], 0.0], dtype=np.float32)
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        dists.append(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
        succ.append(info.get("is_success", 0.0))
    env.close()
    return np.mean(succ), float(np.mean(dists))

print(run_baseline())

