"""
Gymnasium wrapper for HumanoidEnv (PyBullet).
Environment.py is unchanged.

Motion imitation reward is baked directly into this wrapper so that
any trainer (SB3, custom loop, etc.) automatically gets the combined
reward: env_reward + imitation_reward.

Full joint reset is only applied when the humanoid actually falls
(terminated=True). On a timeout truncation the episode ends naturally
without teleporting or resetting joints mid-stride.
"""

import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p

from Environment import create_environment

try:
    from HumanoidController import HumanoidController
except Exception:
    HumanoidController = None


# ---------------------------------------------------------------------------
# Motion Imitation Reward
# ---------------------------------------------------------------------------

class MotionImitationReward:
    """
    Reads humanoid3d_walk.txt and computes a pose-matching reward each step
    by comparing actual joint angles to the interpolated reference frame.

    Uses the same exponential kernel as the DeepMimic paper:
        reward = exp(-2 * sum_of_squared_joint_errors)
    """

    def __init__(self, path: str, joint_ids: list, cid: int, humanoid_id: int):
        with open(path, "r") as f:
            data = json.load(f)

        frames = data["Frames"]
        self.frame_duration = frames[0][0]
        self.raw_frames     = np.array([frame[8:] for frame in frames], dtype=np.float32)
        self.num_frames     = len(self.raw_frames)
        self.total_duration = self.num_frames * self.frame_duration

        self.joint_ids  = joint_ids
        self.cid        = cid
        self.humanoid   = humanoid_id
        self.elapsed    = 0.0

        n_joints = len(joint_ids)
        refs = []
        for frame in self.raw_frames:
            frame_refs = []
            for i in range(n_joints):
                base = i * 4
                if base < len(frame):
                    frame_refs.append(float(frame[base]))
                else:
                    frame_refs.append(0.0)
            refs.append(frame_refs)

        self.joint_refs = np.array(refs, dtype=np.float32)
        print(f"[MotionImitation] {self.num_frames} frames | "
              f"{self.total_duration:.2f}s total | "
              f"{n_joints} joints tracked")

    def reset(self):
        self.elapsed = 0.0

    def step(self, dt: float) -> float:
        self.elapsed += dt

        t      = (self.elapsed % self.total_duration) / self.frame_duration
        idx_lo = int(t) % self.num_frames
        idx_hi = (idx_lo + 1) % self.num_frames
        alpha  = t - int(t)
        ref    = (1.0 - alpha) * self.joint_refs[idx_lo] + alpha * self.joint_refs[idx_hi]

        actual = np.array([
            p.getJointState(self.humanoid, j, physicsClientId=self.cid)[0]
            for j in self.joint_ids
        ], dtype=np.float32)

        error  = np.sum(np.square(actual - ref))
        return float(np.exp(-2.0 * error))


# ---------------------------------------------------------------------------
# Gymnasium Environment
# ---------------------------------------------------------------------------

class HumanoidGymEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    START_POS = [0, 0, 2.4]
    START_ORN = [0, 0, 0, 1]

    def __init__(
        self,
        gui: bool = False,
        motion_path: str = "humanoid3d_walk.txt",
        use_controller: bool = True,
        imitation_weight: float = 1.0,
        env_reward_weight: float = 1.0,
    ):
        super().__init__()

        self.env = create_environment(gui=gui, motion_path=motion_path)

        self.joint_dim = len(self.env.joint_ids)
        self.dt        = 1.0 / 240.0
        self.t         = 0.0
        self._fell     = False   # tracks whether last episode ended in a fall

        self.imitation_weight  = imitation_weight
        self.env_reward_weight = env_reward_weight

        self.use_controller = use_controller and HumanoidController is not None
        self.controller     = HumanoidController(self.joint_dim) if self.use_controller else None

        self.motion_reward = MotionImitationReward(
            path        = motion_path,
            joint_ids   = self.env.joint_ids,
            cid         = self.env.cid,
            humanoid_id = self.env.humanoid,
        )

        obs, _ = self.env.get_obs()
        obs_dim = len(obs)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.joint_dim,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Full joint reset — only called after a fall
    # ------------------------------------------------------------------

    def _reset_after_fall(self):
        """
        Full reset back to T-pose spawn.
        Only called when terminated=True (humanoid actually fell).
        Steps:
          1. Kill all motors so they don't resist repositioning
          2. Reset root body position, orientation and velocity
          3. Reset every joint angle and velocity to zero (T-pose)
          4. One sim step so PyBullet propagates the new state
        """
        cid      = self.env.cid
        humanoid = self.env.humanoid
        joints   = self.env.joint_ids

        # 1. Kill motors
        for j in joints:
            p.setJointMotorControl2(
                bodyUniqueId    = humanoid,
                jointIndex      = j,
                controlMode     = p.VELOCITY_CONTROL,
                targetVelocity  = 0.0,
                force           = 0.0,
                physicsClientId = cid,
            )

        # 2. Reset root body
        p.resetBasePositionAndOrientation(
            humanoid, self.START_POS, self.START_ORN, physicsClientId=cid
        )
        p.resetBaseVelocity(
            humanoid, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=cid
        )

        # 3. Reset every joint to zero angle and zero velocity
        for j in joints:
            p.resetJointState(
                humanoid,
                j,
                targetValue     = 0.0,
                targetVelocity  = 0.0,
                physicsClientId = cid,
            )

        # 4. Propagate new state
        p.stepSimulation(physicsClientId=cid)

    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.t = 0.0
        self.motion_reward.reset()

        if self._fell:
            # Humanoid fell last episode — do a full joint + body reset
            self._reset_after_fall()
            self._fell = False
        else:
            # Timeout truncation — env ended cleanly, no need to fix joints
            # Just let Environment.py's reset reposition the root body
            self.env.reset()

        obs, _ = self.env.get_obs()
        return np.array(obs, dtype=np.float32), {}

    # ------------------------------------------------------------------

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        if self.use_controller:
            obs, _ = self.env.get_obs()
            action = self.controller.step(obs, action)

        obs, env_reward, done = self.env.step(action)

        imitation_reward = self.motion_reward.step(self.dt)

        reward = (self.env_reward_weight * env_reward +
                  self.imitation_weight  * imitation_reward)

        self.t += self.dt

        if done:
            # Record that the next reset() should do a full joint reset
            self._fell = True

        return obs, float(reward), done, False, {}

    # ------------------------------------------------------------------

    def render(self):
        pass

    def close(self):
        try:
            p.disconnect(self.env.cid)
        except Exception:
            pass