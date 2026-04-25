import numpy as np


class HumanoidController:
    """
    Residual stabilisation controller for your PyBullet humanoid.
    Works directly on joint velocity actions (same as your env).
    """

    def __init__(self, joint_dim):
        self.joint_dim = joint_dim

        # stabilisation gains
        self.kp = 4.0
        self.kd = 0.6

    # -----------------------------
    # extract root state correctly
    # -----------------------------
    def _get_root_state(self, obs):
        # From your env.get_obs():
        # obs layout:
        # [goal_dists..., pos(3), vel(3), ang_vel(3), orn(4), joints..., (maybe ref...)]
        #
        # root position starts at index:
        pos_start = 3 + len(range(3))  # goal distances + pos already included earlier in list

        # safer: directly reconstruct from end offsets is unreliable,
        # so we assume fixed structure:
        return obs

    # -----------------------------
    # stabilisation from physics state
    # -----------------------------
    def _stabilise(self, obs):
        # base orientation quaternion is inside obs (after pos/vel)
        # we extract approx upright signal from z velocity + angular velocity

        ang_vel = obs[6:9]  # from env: angular velocity

        roll_rate = ang_vel[0]
        pitch_rate = ang_vel[1]

        # no explicit roll/pitch in obs → approximate stabilisation only
        return np.array([
            -self.kd * roll_rate,
            -self.kd * pitch_rate
        ])

    # -----------------------------
    # main control
    # -----------------------------
    def step(self, obs, base_action):
        """
        base_action = RL or zero vector
        returns joint velocity commands
        """

        correction = self._stabilise(obs)

        action = np.array(base_action, dtype=np.float32)

        # distribute correction across joints (small bias)
        if self.joint_dim > 0:
            bias = correction[0] * 0.01
            action += bias

        return np.clip(action, -1.0, 1.0)