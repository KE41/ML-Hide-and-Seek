"""
BipedGymEnv.py
--------------
Gymnasium training environment for biped2d_pybullet.urdf.

Movement model
--------------
The biped2d URDF has a zero-mass "world" base that cannot be pushed by
external forces.  Instead we:
  • Rotate the base to a random target direction each episode.
  • Drive y_to_world with velocity control at a fixed target speed.
  • Hold z_to_y at 0 (height) and torso_to_z at 0 (upright).

The NN policy controls the six leg joints (hips, knees, ankles).  The reward
is based on the actual forward speed of the torso link.

Observation (25-D):
  torso quaternion       (4)
  torso height z         (1)
  torso linear velocity  (3)
  torso angular velocity (3)
  joint positions x6     (6)
  joint velocities x6    (6)
  target direction cos,sin (2)

Action (6-D):
  target joint velocities scaled to [-MAX_VEL, +MAX_VEL]
"""

import math
import os

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces


class BipedGymEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    CTRL_JOINTS = [
        "torso_to_rightleg",
        "torso_to_leftleg",
        "r_knee",
        "l_knee",
        "r_ankle",
        "l_ankle",
    ]

    JOINT_FORCE  = 600.0
    MAX_VEL      = 10.0
    TARGET_SPEED = 2.5     # m/s desired forward speed

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        if render_mode == "human":
            self.cid = p.connect(p.GUI)
        else:
            self.cid = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.8, physicsClientId=self.cid)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.cid)

        obs_dim = 4 + 1 + 3 + 3 + 6 + 6 + 2   # = 25
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        self._biped_id     = None
        self._joints       = {}
        self._ctrl_ids     = []
        self._torso_link   = 2   # child of torso_to_z joint
        self._target_dir   = np.array([1.0, 0.0])
        self._base_angle   = 0.0
        self._step_count   = 0
        self._prev_action  = np.zeros(6, dtype=np.float32)
        self._max_steps    = 600

        self._build_world()

    # ------------------------------------------------------------------
    # World setup
    # ------------------------------------------------------------------

    def _build_world(self):
        p.resetSimulation(physicsClientId=self.cid)
        p.setGravity(0, 0, -9.8, physicsClientId=self.cid)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.cid)
        p.loadURDF("plane.urdf", physicsClientId=self.cid)

        self._biped_id = p.loadURDF(
            os.path.join(pybullet_data.getDataPath(), "biped", "biped2d_pybullet.urdf"),
            basePosition=[0, 0, 0.6],
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=self.cid,
            useFixedBase=False,
        )

        self._joints = {}
        n = p.getNumJoints(self._biped_id, physicsClientId=self.cid)
        for j in range(n):
            info = p.getJointInfo(self._biped_id, j, physicsClientId=self.cid)
            self._joints[info[1].decode()] = (j, info[2])

        self._ctrl_ids = [
            self._joints[name][0]
            for name in self.CTRL_JOINTS
            if name in self._joints
        ]
        if "torso_to_z" in self._joints:
            self._torso_link = self._joints["torso_to_z"][0]

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self):
        ls = p.getLinkState(self._biped_id, self._torso_link,
                            computeLinkVelocity=1, physicsClientId=self.cid)
        torso_pos = ls[0]
        torso_orn = ls[1]
        lin_vel   = ls[6]
        ang_vel   = ls[7]

        j_pos, j_vel = [], []
        for jid in self._ctrl_ids:
            s = p.getJointState(self._biped_id, jid, physicsClientId=self.cid)
            j_pos.append(s[0])
            j_vel.append(s[1])

        obs = np.array(
            list(torso_orn) + [torso_pos[2]] + list(lin_vel) + list(ang_vel) +
            j_pos + j_vel + list(self._target_dir),
            dtype=np.float32,
        )
        return obs, torso_pos, torso_orn, lin_vel

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random target direction
        angle = np.random.uniform(0, 2 * math.pi)
        self._target_dir  = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        # θ such that local +Y maps to (vx, vy): θ = atan2(-vx, vy)
        self._base_angle  = math.atan2(-self._target_dir[0], self._target_dir[1])
        orn = p.getQuaternionFromEuler([0, 0, self._base_angle],
                                       physicsClientId=self.cid)

        p.resetBasePositionAndOrientation(
            self._biped_id, [0, 0, 0.6], orn,
            physicsClientId=self.cid,
        )
        p.resetBaseVelocity(
            self._biped_id, [0, 0, 0], [0, 0, 0],
            physicsClientId=self.cid,
        )
        for jid in self._ctrl_ids:
            p.resetJointState(self._biped_id, jid, 0.0, 0.0, physicsClientId=self.cid)
        for name in ("y_to_world", "z_to_y", "torso_to_z"):
            if name in self._joints:
                jid, _ = self._joints[name]
                p.resetJointState(self._biped_id, jid, 0.0, 0.0, physicsClientId=self.cid)

        # y_to_world: drive at target speed
        if "y_to_world" in self._joints:
            jid, _ = self._joints["y_to_world"]
            p.setJointMotorControl2(
                self._biped_id, jid,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=self.TARGET_SPEED, force=500,
                physicsClientId=self.cid,
            )
        # z_to_y: hold height
        if "z_to_y" in self._joints:
            jid, _ = self._joints["z_to_y"]
            p.setJointMotorControl2(
                self._biped_id, jid,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0, force=500,
                physicsClientId=self.cid,
            )
        # torso_to_z: hold upright
        if "torso_to_z" in self._joints:
            jid, _ = self._joints["torso_to_z"]
            p.setJointMotorControl2(
                self._biped_id, jid,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.0, force=300.0,
                physicsClientId=self.cid,
            )

        self._step_count  = 0
        self._prev_action = np.zeros(6, dtype=np.float32)

        obs, _, _, _ = self._get_obs()
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        # Apply NN leg joint velocity actions
        for i, jid in enumerate(self._ctrl_ids):
            vel = float(action[i]) * self.MAX_VEL
            p.setJointMotorControl2(
                self._biped_id, jid,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=vel,
                force=self.JOINT_FORCE,
                physicsClientId=self.cid,
            )

        # Keep y_to_world driving forward
        if "y_to_world" in self._joints:
            jid, _ = self._joints["y_to_world"]
            p.setJointMotorControl2(
                self._biped_id, jid,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=self.TARGET_SPEED, force=500,
                physicsClientId=self.cid,
            )

        for _ in range(4):
            p.stepSimulation(physicsClientId=self.cid)

        obs, torso_pos, torso_orn, lin_vel = self._get_obs()
        self._step_count += 1

        # Forward speed in the target direction
        fwd_vel = (
            lin_vel[0] * self._target_dir[0] +
            lin_vel[1] * self._target_dir[1]
        )
        r_velocity = float(np.clip(fwd_vel / self.TARGET_SPEED, 0.0, 1.0))

        # Upright reward (torso z-axis pointing up)
        rot = p.getMatrixFromQuaternion(torso_orn, physicsClientId=self.cid)
        upright  = float(rot[8])
        r_upright = max(upright, 0.0)

        r_height = float(np.clip(torso_pos[2] / 1.4, 0.0, 1.0))

        r_smooth = -0.05 * float(np.mean(np.abs(action - self._prev_action)))
        self._prev_action = action.copy()

        reward = 2.0 * r_velocity + 0.5 * r_upright + 0.3 * r_height + r_smooth + 0.1

        terminated = bool(torso_pos[2] < 0.5 or upright < -0.1)
        truncated  = self._step_count >= self._max_steps

        return obs, reward, terminated, truncated, {}

    def close(self):
        try:
            p.disconnect(physicsClientId=self.cid)
        except Exception:
            pass
