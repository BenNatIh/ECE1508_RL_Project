import gymnasium as gym
import numpy as np


class NoisyObservationWrapper(gym.ObservationWrapper):
    """
    Adds Gaussian noise to specified keys of a dict observation space.

    - noise_std: dict mapping observation keys to std dev (float).
      e.g. {'observation': 0.01, 'achieved_goal': 0.005}
    - keep_clean: store original values in keys '<key>_clean'
    """
    def __init__(self, env, noise_std=None, keep_clean=True, seed=None):
        super().__init__(env)
        assert isinstance(self.observation_space, gym.spaces.Dict), "Wrapper expects a dict observation space"
        self.noise_std = noise_std or {}
        self.keep_clean = keep_clean
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        else:
            self.np_random = np.random

    def observation(self, obs):
        # obs is a dict with keys e.g. 'observation', 'achieved_goal', 'desired_goal'
        out = dict(obs)
        for key, std in self.noise_std.items():
            if key in obs and std is not None and std > 0.0:
                clean = np.array(obs[key], copy=True)
                noise = self.np_random.normal(loc=0.0, scale=std, size=clean.shape).astype(clean.dtype)
                noisy = clean + noise
                if self.keep_clean:
                    out[f"{key}_clean"] = clean
                out[key] = noisy
        return out
