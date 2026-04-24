"""
Thin Gymnasium wrapper around the existing HumanoidEnv.
Environment.py is NOT modified — this file just adapts the interface.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from Environment import create_environment


class HumanoidGymEnv(gym.Env):
    """Wraps HumanoidEnv so Stable-Baselines3 can use it."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, gui: bool = False):
        super().__init__()
        self.env = create_environment(gui=gui)

        # Derive spaces from a live observation
        obs, _ = self.env.get_obs()
        obs_dim = len(obs)
        act_dim = len(self.env.joint_ids)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset()          # returns np.array directly
        return obs, {}

    def step(self, action):
        obs, reward, done = self.env.step(action)
        terminated = done
        truncated  = False
        return obs, float(reward), terminated, truncated, {}

    def render(self):
        pass  # GUI is toggled at construction time

    def close(self):
        try:
            import pybullet as p
            p.disconnect()
        except Exception:
            pass