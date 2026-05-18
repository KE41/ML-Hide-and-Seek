"""
HumanoidCPGWalker.py
--------------------
CPG locomotion controller for a second MJCF humanoid — used as the hider.

Loads mjcf/humanoid_symmetric_no_ground.xml (same skeleton and joint names
as the training humanoid but without the embedded floor plane, so the
game arena floor is the only floor in the scene).  Recoloured cyan so
the two humanoids are visually distinct.

Locomotion system
-----------------
A Central Pattern Generator (CPG) drives the joints with coupled
sinusoidal oscillators — one per leg, locked in antiphase:

    φ_right = ω·t
    φ_left  = ω·t + π

Joint targets derived from the oscillators:

    right_hip_y   =  A_hip  · sin(φ_r)           ← sagittal swing (main)
    left_hip_y    =  A_hip  · sin(φ_l)
    right_hip_x   =  A_lat  · sin(φ_r + π/2)     ← lateral stabilisation
    left_hip_x    =  A_lat  · sin(φ_l + π/2)
    right_knee    = -bias - A_knee·max(sin(φ_r + π/4), 0)   ← swing flex
    left_knee     = -bias - A_knee·max(sin(φ_l + π/4), 0)
    right_ankle_y = -A_ank  · sin(φ_r - π/4)     ← push-off
    left_ankle_y  = -A_ank  · sin(φ_l - π/4)
    right_shoulder1 = A_arm · sin(φ_l)            ← counter-swing
    left_shoulder1  = A_arm · sin(φ_r)

All joints use POSITION_CONTROL (vs. VELOCITY_CONTROL for the PPO seeker),
so the oscillator target is enforced stiffly — the robot does not fall.
Direction is controlled by applyExternalForce on the torso, same as
HumanoidNavigator.
"""

import math

import pybullet as p
import pybullet_data


class HumanoidCPGWalker:
    """
    CPG-driven MJCF humanoid hider.  Same model as the seeker; different
    locomotion algorithm (oscillator vs. PPO) and different colour (cyan).
    """

    CPG_OMEGA    = 2 * math.pi * 1.4   # rad/s  ≈ 1.4 Hz gait
    HIP_Y_AMP    = 0.40    # sagittal hip swing (rad)
    HIP_X_AMP    = 0.08    # lateral hip sway   (rad)
    KNEE_AMP     = 0.70    # knee flexion        (rad)  [knee range: -160°..-2°]
    ANKLE_Y_AMP  = 0.25    # ankle push-off      (rad)
    SHOULDER_AMP = 0.35    # arm counter-swing   (rad)

    JOINT_FORCE  = 150.0   # N — matches HumanoidEnv.MAX_FORCE
    MAX_VEL      = 10.0    # rad/s limit on position-control velocity

    STEER_FORCE  = 90.0    # N per (m/s) of game-AI target speed
    MAX_STEER    = 300.0   # N cap

    CYAN = [0.0, 0.8, 1.0, 1.0]

    def __init__(self, cid: int, start_pos: list):
        self.cid   = cid
        self.phase = 0.0

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)

        # Load the no-ground variant so we don't get a second floor plane
        bodies = p.loadMJCF(
            "mjcf/humanoid_symmetric_no_ground.xml",
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=cid,
        )
        self.all_bodies  = list(bodies)
        self.humanoid_id = bodies[-1]   # same convention as HumanoidEnv

        # Map joint names → indices on the root body
        self._joint_map = {}
        self._joint_ids = []
        n = p.getNumJoints(self.humanoid_id, physicsClientId=cid)
        for j in range(n):
            info = p.getJointInfo(self.humanoid_id, j, physicsClientId=cid)
            if info[2] != p.JOINT_FIXED:
                name = info[1].decode()
                self._joint_map[name] = j
                self._joint_ids.append(j)

        # Recolour every link of every sub-body cyan
        for body in self.all_bodies:
            nb = p.getNumJoints(body, physicsClientId=cid)
            for link in range(-1, nb):
                try:
                    p.changeVisualShape(
                        body, link,
                        rgbaColor       = self.CYAN,
                        physicsClientId = cid,
                    )
                except Exception:
                    pass

        # Place at game start position and hold neutral T-pose
        p.resetBasePositionAndOrientation(
            self.humanoid_id, start_pos, [0, 0, 0, 1], physicsClientId=cid
        )
        self._hold_neutral()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_joint(self, name: str, target: float):
        """Drive one named joint to a target angle via stiff position control."""
        if name not in self._joint_map:
            return
        p.setJointMotorControl2(
            self.humanoid_id, self._joint_map[name],
            controlMode    = p.POSITION_CONTROL,
            targetPosition = target,
            force          = self.JOINT_FORCE,
            maxVelocity    = self.MAX_VEL,
            physicsClientId = self.cid,
        )

    def _hold_neutral(self):
        """Reset all joints to 0 and hold there with stiff position control."""
        for j in self._joint_ids:
            p.resetJointState(
                self.humanoid_id, j, 0.0, 0.0, physicsClientId=self.cid
            )
            p.setJointMotorControl2(
                self.humanoid_id, j,
                controlMode    = p.POSITION_CONTROL,
                targetPosition = 0.0,
                force          = self.JOINT_FORCE,
                maxVelocity    = self.MAX_VEL,
                physicsClientId = self.cid,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_pos(self):
        """Return (x, y, z) of the humanoid base link."""
        pos, _ = p.getBasePositionAndOrientation(
            self.humanoid_id, physicsClientId=self.cid
        )
        return list(pos)

    def reset(self, position: list):
        """Teleport to a new arena position and reset to neutral stance."""
        # Kill all motors first so they don't resist the teleport
        for j in self._joint_ids:
            p.setJointMotorControl2(
                self.humanoid_id, j,
                controlMode    = p.VELOCITY_CONTROL,
                targetVelocity = 0.0,
                force          = 0.0,
                physicsClientId = self.cid,
            )
        p.resetBasePositionAndOrientation(
            self.humanoid_id, position, [0, 0, 0, 1], physicsClientId=self.cid
        )
        p.resetBaseVelocity(
            self.humanoid_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid
        )
        self._hold_neutral()
        self.phase = 0.0

    def step(self, target_vx: float, target_vy: float, dt: float):
        """
        One CPG decision step:
          1. Advance the oscillator phase (if moving).
          2. Compute and apply joint targets from the CPG equations.
          3. Apply a steering force toward the game AI's target direction.
        """
        speed = math.sqrt(target_vx * target_vx + target_vy * target_vy)

        if speed > 0.05:
            self.phase = (self.phase + self.CPG_OMEGA * dt) % (2 * math.pi)

        phi_r = self.phase
        phi_l = self.phase + math.pi   # left leg antiphase

        # ---- Sagittal hip swing (main forward/backward motion) ----
        self._set_joint('right_hip_y',  self.HIP_Y_AMP * math.sin(phi_r))
        self._set_joint('left_hip_y',   self.HIP_Y_AMP * math.sin(phi_l))

        # ---- Lateral hip sway (keeps CoM over stance foot) ----
        self._set_joint('right_hip_x',  self.HIP_X_AMP * math.sin(phi_r + math.pi / 2))
        self._set_joint('left_hip_x',   self.HIP_X_AMP * math.sin(phi_l + math.pi / 2))

        # ---- Knee flexion during swing phase only ----
        # Knee range is -160° to -2° so targets are always negative;
        # constant bias keeps the knee just slightly bent at all times.
        q_r_knee = -0.05 - self.KNEE_AMP * max(math.sin(phi_r + math.pi / 4), 0.0)
        q_l_knee = -0.05 - self.KNEE_AMP * max(math.sin(phi_l + math.pi / 4), 0.0)
        self._set_joint('right_knee', q_r_knee)
        self._set_joint('left_knee',  q_l_knee)

        # ---- Ankle push-off at end of stance ----
        self._set_joint('right_ankle_y', -self.ANKLE_Y_AMP * math.sin(phi_r - math.pi / 4))
        self._set_joint('left_ankle_y',  -self.ANKLE_Y_AMP * math.sin(phi_l - math.pi / 4))

        # ---- Arm counter-swing (right arm moves with left leg and vice versa) ----
        self._set_joint('right_shoulder1',  self.SHOULDER_AMP * math.sin(phi_l))
        self._set_joint('left_shoulder1',   self.SHOULDER_AMP * math.sin(phi_r))

        # ---- Slight forward lean for locomotion bias ----
        self._set_joint('abdomen_y', -0.10)

        # ---- Steering force toward game target ----
        if speed > 0.05:
            pos   = self.get_pos()
            scale = min(speed * self.STEER_FORCE, self.MAX_STEER)
            p.applyExternalForce(
                self.humanoid_id, -1,
                [target_vx / speed * scale,
                 target_vy / speed * scale,
                 0.0],
                pos,
                p.WORLD_FRAME,
                physicsClientId=self.cid,
            )
