"""
Gymnasium wrapper for HumanoidEnv (PyBullet).
Environment.py is unchanged.
This wrapper optionally applies a residual humanoid controller.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from Environment import create_environment

try:
    from HumanoidController import HumanoidController
except Exception:
    HumanoidController = None


class HumanoidGymEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        gui: bool = False,
        motion_path: str = "humanoid3d_walk.txt",
        use_controller: bool = True
    ):
        super().__init__()

        self.env = create_environment(gui=gui, motion_path=motion_path)

        self.joint_dim = len(self.env.joint_ids)

        self.use_controller = use_controller and HumanoidController is not None
        self.controller = HumanoidController(self.joint_dim) if self.use_controller else None

        self.t = 0.0
        self.dt = 1.0 / 240.0

        obs, _ = self.env.get_obs()
        obs_dim = len(obs)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.joint_dim,),
            dtype=np.float32
        )

    # ------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0.0
        obs = self.env.reset()
        return obs, {}

    # ------------------------------------------------------------

    def step(self, action):

        action = np.asarray(action, dtype=np.float32)

        # Optional residual controller (correct design)
        if self.use_controller:
            obs, _ = self.env.get_obs()
            action = self.controller.step(obs, action)

        obs, reward, done = self.env.step(action)

        self.t += self.dt

        return obs, float(reward), done, False, {}

    # ------------------------------------------------------------

    def render(self):
        pass

    # ------------------------------------------------------------

    def close(self):
        try:
            import pybullet as p
            p.disconnect(self.env.cid)
        except Exception:
            pass