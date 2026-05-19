"""
BipedNNWalker.py
----------------
Layered, physics-driven gait controller for a *massed* humanoid.

Why the rewrite
---------------
The previous version loaded biped/biped2d_pybullet.urdf.  That model has
a ZERO-MASS "world" base joined to the robot through prismatic
y_to_world / z_to_y joints — its own header file says external forces and
ground contact cannot move it.  So it could only ever be *slid* along the
floor by per-frame base teleporting: it looked like a thin stick gliding
on a rail, never walking, no matter how good the controller was.  You
cannot "give that model mass" — the masslessness is structural.

The fix is to drive a real humanoid that HAS mass and real legs, and let
PHYSICS move it (joint torques + ground contact + a steering force),
never teleporting the base during a step.  This mirrors the proven
control pattern in your own HumanoidCPGWalker.py (stiff position-control
legs + posture hold + steering force on the MJCF humanoid), wrapped in
the requested three-layer architecture:

      High level   ExplorationPlanner   raycast -> heading + speed
      Mid level    GaitGenerator        phase clock -> joint targets
                                         (proven CPG pattern, or IK)
      Low level    JointPDController     position control + force limit

Model
-----
Loads (first that exists):
    mjcf/humanoid_symmetric_no_ground.xml   <- your project's model
    mjcf/humanoid_symmetric.xml             <- stock pybullet_data
    biped/biped2d_pybullet.urdf             <- last-resort fallback only
The humanoid is a full 3-D massed body, so roll balance, lateral hip
sway and arm counter-swing are now ACTIVE (the DOFs exist), not inert.

API (unchanged -- HideSeekEnv.py keeps working)
-----------------------------------------------
    BipedNNWalker(cid, start_pos, colour=None, policy_path=None, ...)
    .biped_id            (= humanoid body id, for LOS / catch tests)
    .get_pos()           -> [x, y, z]   (real, physics-updated)
    .reset(position)     -> None        (only place the base is moved)
    .step(vx, vy, dt)    -> fell:bool   (external command; None => explore)
    .explore(dt)         -> fell:bool   (autonomous)
"""

import csv
import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import pybullet as p
import pybullet_data


# ===========================================================================
# Tunable parameters
# ===========================================================================

@dataclass
class GaitParams:
    # -- locomotion --
    walk_speed: float       = 1.6     # m/s target (drives steering force)
    max_turn_rate: float    = 1.2     # rad/s heading slew cap (gradual)
    step_frequency: float   = 1.7     # Hz gait cycles / second (snappy)
    step_length: float      = 0.14    # m  (IK mode only)
    step_height: float      = 0.08    # m  (IK mode only)

    gait_mode: str          = "cpg"   # "cpg" (proven, stable) or "ik"

    # how many physics substeps the game loop runs per walker.step()
    # call -- MUST match HideSeekEnv.SIM_STEPS_PER_DECISION so the
    # steering force is impulse-compensated correctly (see _locomote).
    substeps: int           = 8

    # -- CPG joint amplitudes (proven on the MJCF humanoid) --
    hip_y_amp: float        = 0.55    # sagittal hip swing  (rad) bigger stride
    hip_x_amp: float        = 0.10    # lateral hip sway     (rad)
    knee_amp: float         = 1.00    # knee swing flexion   (rad) clear lift
    knee_bias: float        = 0.08    # constant slight knee bend (rad)
    ankle_y_amp: float      = 0.30    # ankle push-off       (rad)
    shoulder_amp: float     = 0.45    # arm counter-swing    (rad) visible
    abdomen_lean: float     = 0.12    # forward trunk lean   (rad)

    # -- low-level position control --
    joint_force: float      = 240.0   # N.m -- legs must move body mass,
                                       # not just be dragged by the force
    joint_max_vel: float    = 12.0    # rad/s position-control speed cap
    kp: float               = 0.6     # position gain
    kd: float               = 0.9     # velocity gain

    # -- balance / stabilisation --
    balance_kp_pitch: float = 12.0
    balance_kd_pitch: float = 1.5
    balance_kp_roll: float  = 12.0
    balance_kd_roll: float  = 1.5

    # -- steering (direction via torso force, same as HumanoidCPGWalker) --
    steer_force: float      = 90.0    # N per (m/s) toward target
    max_steer: float        = 300.0   # N cap

    # -- exploration --
    ray_count: int          = 9
    ray_fov: float          = math.radians(140.0)
    ray_range: float        = 3.5
    obstacle_clear: float   = 1.4
    curiosity_period: float = 6.0
    stuck_window: float     = 4.0
    stuck_dist: float       = 0.30

    # -- fall handling --
    fall_height: float      = 0.55    # torso z below this => fallen
    fall_pitch: float       = 1.1     # |pitch|/|roll| above this => fallen
    auto_reset: bool        = True


# ===========================================================================
# Low level -- joint position controller with PD gains + force limit
# ===========================================================================

class JointPDController:
    """
    Stiff POSITION_CONTROL (an internal constrained PD loop) with a torque
    ceiling and velocity cap.  This is exactly the low-level scheme proven
    to keep the MJCF humanoid upright in HumanoidCPGWalker.py.
    """

    def __init__(self, body_id, cid, params):
        self.body = body_id
        self.cid  = cid
        self.prm  = params

    def read(self, joint_ids):
        q  = np.empty(len(joint_ids), dtype=np.float32)
        qd = np.empty(len(joint_ids), dtype=np.float32)
        for i, j in enumerate(joint_ids):
            s = p.getJointState(self.body, j, physicsClientId=self.cid)
            q[i], qd[i] = s[0], s[1]
        return q, qd

    def drive(self, joint_id, target_angle, max_force=None):
        p.setJointMotorControl2(
            self.body, joint_id,
            controlMode     = p.POSITION_CONTROL,
            targetPosition  = float(target_angle),
            positionGain    = self.prm.kp,
            velocityGain    = self.prm.kd,
            force           = float(self.prm.joint_force
                                    if max_force is None else max_force),
            maxVelocity     = self.prm.joint_max_vel,
            physicsClientId = self.cid,
        )


# ===========================================================================
# Mid level -- gait generator (phase clock -> joint targets)
# ===========================================================================

class GaitGenerator:
    """
    Continuous phase phi.  Right leg uses phi, left uses phi+pi (antiphase).

    mode "cpg" : coupled-oscillator joint targets -- the pattern proven to
                 keep this humanoid walking in HumanoidCPGWalker.py.
    mode "ik"  : cycloidal foot trajectory + closed-form 2-link inverse
                 kinematics (smooth, zero-velocity lift-off/touch-down).
    """

    def __init__(self, params, l_thigh, l_shank):
        self.prm   = params
        self.L1    = l_thigh
        self.L2    = l_shank
        self.safe_reach = 0.97 * (l_thigh + l_shank)
        self.phase = 0.0       # radians, [0, 2pi)
        self.fb_fwd = 0.0      # capture-point feedback (IK mode)
        self.fb_tilt = 0.0

    def reset(self):
        self.phase = 0.0
        self.fb_fwd = self.fb_tilt = 0.0

    # ---- closed-form planar leg IK (used by "ik" mode) -------------------
    def _leg_ik(self, fx, fd):
        L1, L2 = self.L1, self.L2
        D = math.hypot(fx, fd)
        D = min(max(D, abs(L1 - L2) + 1e-4), self.safe_reach)
        cosK = (L1 * L1 + L2 * L2 - D * D) / (2.0 * L1 * L2)
        K = math.acos(max(-1.0, min(1.0, cosK)))
        knee_flex = math.pi - K
        beta = math.atan2(fx, fd)
        cosA = (L1 * L1 + D * D - L2 * L2) / (2.0 * L1 * D)
        alpha = math.acos(max(-1.0, min(1.0, cosA)))
        hip = beta + alpha
        ankle = -(hip - knee_flex)
        return hip, knee_flex, ankle

    def _ik_targets(self, moving):
        prm = self.prm
        h0 = 0.92 * (self.L1 + self.L2)
        out = {}
        for phi, hipname, kneen, ankn, sh in (
            (self.phase, "right_hip_y", "right_knee",
             "right_ankle_y", "right_shoulder1"),
            (self.phase + math.pi, "left_hip_y", "left_knee",
             "left_ankle_y", "left_shoulder1"),
        ):
            ph = (phi % (2 * math.pi)) / (2 * math.pi)   # 0..1
            if ph < 0.5:                                  # stance
                fx = (0.25 - ph) * 2.0 * prm.step_length
                fd = h0
            else:                                         # swing
                s = (ph - 0.5) / 0.5
                fx = (-0.5 * prm.step_length + prm.step_length *
                      (s - math.sin(2 * math.pi * s) / (2 * math.pi)))
                fd = h0 - 0.5 * prm.step_height * (
                    1 - math.cos(2 * math.pi * s))
            if not moving:
                fx, fd = 0.0, h0
            fx += self.fb_fwd + self.fb_tilt
            mx = math.sqrt(max(self.safe_reach ** 2 - fd ** 2, 0.0))
            fx = max(-mx, min(mx, fx))
            hip, knee, ankle = self._leg_ik(fx, fd)
            out[hipname] = hip
            out[kneen]   = -(prm.knee_bias + knee)   # URDF knee is negative
            out[ankn]    = ankle
            out[sh]      = prm.shoulder_amp * math.sin(phi + math.pi)
        return out

    # ---- proven CPG joint pattern (default, stable) ---------------------
    def _cpg_targets(self, moving):
        prm = self.prm
        phi_r = self.phase
        phi_l = self.phase + math.pi
        out = {
            # sagittal hip swing -- main forward drive
            "right_hip_y":  prm.hip_y_amp * math.sin(phi_r),
            "left_hip_y":   prm.hip_y_amp * math.sin(phi_l),
            # lateral hip sway -- keeps CoM over the stance foot
            "right_hip_x":  prm.hip_x_amp * math.sin(phi_r + math.pi / 2),
            "left_hip_x":   prm.hip_x_amp * math.sin(phi_l + math.pi / 2),
            # knee flexes during swing only; slight constant bend always
            "right_knee": -prm.knee_bias - prm.knee_amp * max(
                math.sin(phi_r + math.pi / 4), 0.0),
            "left_knee":  -prm.knee_bias - prm.knee_amp * max(
                math.sin(phi_l + math.pi / 4), 0.0),
            # ankle push-off at end of stance
            "right_ankle_y": -prm.ankle_y_amp * math.sin(
                phi_r - math.pi / 4),
            "left_ankle_y":  -prm.ankle_y_amp * math.sin(
                phi_l - math.pi / 4),
            # arms counter-swing the opposite leg
            "right_shoulder1": prm.shoulder_amp * math.sin(phi_l),
            "left_shoulder1":  prm.shoulder_amp * math.sin(phi_r),
        }
        if not moving:
            for k in out:
                out[k] = -prm.knee_bias if "knee" in k else 0.0
        return out

    def step(self, dt, moving):
        if moving:
            self.phase = (self.phase +
                          2 * math.pi * self.prm.step_frequency * dt) \
                         % (2 * math.pi)
        if self.prm.gait_mode == "ik":
            return self._ik_targets(moving)
        return self._cpg_targets(moving)


# ===========================================================================
# High level -- exploration planner (raycast sensing -> heading)
# ===========================================================================

class ExplorationPlanner:

    def __init__(self, params):
        self.prm = params
        self.heading = 0.0
        self.target_heading = 0.0
        self._tc = 0.0
        self._ts = 0.0
        self._anchor = (0.0, 0.0)

    def reset(self, heading=0.0, pos=(0.0, 0.0)):
        self.heading = self.target_heading = heading
        self._tc = self._ts = 0.0
        self._anchor = (pos[0], pos[1])

    def sense(self, cid, origin, heading, sensor_distances=None):
        n, fov, rng = (self.prm.ray_count, self.prm.ray_fov,
                       self.prm.ray_range)
        angles = [heading - fov / 2 + fov * i / (n - 1) for i in range(n)]
        dirs = [(math.cos(a), math.sin(a)) for a in angles]
        if sensor_distances is not None:
            d = list(sensor_distances)[:n]
            d += [rng] * (n - len(d))
            return d, dirs
        z = max(origin[2], 0.6)
        froms = [[origin[0], origin[1], z]] * n
        tos = [[origin[0] + dx * rng, origin[1] + dy * rng, z]
               for dx, dy in dirs]
        hits = p.rayTestBatch(froms, tos, physicsClientId=cid)
        dists = [(h[2] * rng if h[0] >= 0 else rng) for h in hits]
        return dists, dirs

    def plan(self, dt, cid, origin, sensor_distances=None):
        self._tc += dt
        self._ts += dt
        dists, dirs = self.sense(cid, origin, self.heading,
                                 sensor_distances)
        n = len(dists)
        mid = n // 2
        cone = dists[max(0, mid - n // 6): mid + n // 6 + 1]
        blocked = (min(cone) < self.prm.obstacle_clear) if cone else False

        if blocked:
            best = int(np.argmax(dists))
            self.target_heading = math.atan2(dirs[best][1], dirs[best][0])
        elif self._tc > self.prm.curiosity_period:
            self.target_heading = self.heading + random.uniform(-1.2, 1.2)
            self._tc = 0.0

        if self._ts > self.prm.stuck_window:
            ax, ay = self._anchor
            if math.hypot(origin[0] - ax, origin[1] - ay) < \
                    self.prm.stuck_dist:
                self.target_heading = self.heading + random.choice(
                    (-1, 1)) * random.uniform(1.5, 2.5)
            self._anchor = (origin[0], origin[1])
            self._ts = 0.0

        err = math.atan2(math.sin(self.target_heading - self.heading),
                         math.cos(self.target_heading - self.heading))
        cap = self.prm.max_turn_rate * dt
        self.heading += max(-cap, min(cap, err))
        self.heading = math.atan2(math.sin(self.heading),
                                  math.cos(self.heading))
        return self.heading


# ===========================================================================
# Telemetry
# ===========================================================================

class _Logger:
    def __init__(self):
        self.rows = []

    def log(self, **kw):
        self.rows.append(kw)

    def dump(self, path):
        if not self.rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)
        print(f"[BipedNNWalker] telemetry -> {path} "
              f"({len(self.rows)} rows)")


# ===========================================================================
# Public walker
# ===========================================================================

class BipedNNWalker:

    BASE_JOINTS = ()                    # humanoid has a real free base
    SPAWN_Z   = 1.15                    # humanoid pelvis standing height
    MAP_LIMIT = 9.0

    MODEL_CANDIDATES = (
        ("mjcf", "mjcf/humanoid_symmetric_no_ground.xml"),
        ("mjcf", "mjcf/humanoid_symmetric.xml"),
        ("urdf", "biped/biped2d_pybullet.urdf"),     # massless fallback
    )

    def __init__(
        self,
        cid,
        start_pos,
        colour=None,
        policy_path=None,               # accepted for API compat (unused)
        params=None,
        log=False,
    ):
        self.cid = cid
        self.prm = params if params is not None else GaitParams()
        self._wx = float(start_pos[0])
        self._wy = float(start_pos[1])
        self._facing = 0.0
        self._fallen = False
        self._t = 0.0
        self.logger = _Logger() if log else None

        if colour is None:
            colour = [0.0, 0.8, 1.0, 1.0]

        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=cid)

        # ---- load the first available (massed) model ----
        self.all_bodies = []
        self.is_humanoid = False
        for kind, path in self.MODEL_CANDIDATES:
            try:
                if kind == "mjcf":
                    full = os.path.join(pybullet_data.getDataPath(), path)
                    if not (os.path.exists(full) or os.path.exists(path)):
                        continue
                    bodies = p.loadMJCF(
                        path, flags=p.URDF_USE_SELF_COLLISION,
                        physicsClientId=cid)
                    self.all_bodies = list(bodies)
                    self.biped_id = bodies[-1]
                    self.is_humanoid = True
                else:
                    self.biped_id = p.loadURDF(
                        os.path.join(pybullet_data.getDataPath(), path),
                        basePosition=[self._wx, self._wy, 0.48],
                        physicsClientId=cid, useFixedBase=False)
                    self.all_bodies = [self.biped_id]
                print(f"[BipedNNWalker] model: {path}")
                break
            except Exception as exc:
                print(f"[BipedNNWalker] {path} unavailable: {exc}")
        else:
            raise RuntimeError("No biped/humanoid model could be loaded")

        spawn_z = self.SPAWN_Z if self.is_humanoid else 0.48
        p.resetBasePositionAndOrientation(
            self.biped_id, [self._wx, self._wy, spawn_z],
            [0, 0, 0, 1], physicsClientId=cid)

        # ---- index joints + build role map ----
        self._joints = {}
        n = p.getNumJoints(self.biped_id, physicsClientId=cid)
        for j in range(n):
            info = p.getJointInfo(self.biped_id, j, physicsClientId=cid)
            if info[2] != p.JOINT_FIXED:
                self._joints[info[1].decode()] = j

        self._roles = self._build_role_map()

        # ---- colour every link of every sub-body ----
        for body in self.all_bodies:
            nb = p.getNumJoints(body, physicsClientId=cid)
            for link in range(-1, nb):
                try:
                    p.changeVisualShape(body, link, rgbaColor=colour,
                                        physicsClientId=cid)
                except Exception:
                    pass

        l1, l2 = self._measure_leg_geometry()

        self.pd = JointPDController(self.biped_id, cid, self.prm)
        self.gait = GaitGenerator(self.prm, l1, l2)
        self.planner = ExplorationPlanner(self.prm)
        self.planner.reset(self._facing, (self._wx, self._wy))

        self._hold_neutral()
        self._dbg_line = None
        print(f"[BipedNNWalker] layered controller ready "
              f"({'humanoid+mass' if self.is_humanoid else 'biped2d'}, "
              f"mode={self.prm.gait_mode}, "
              f"thigh={l1:.2f}m shank={l2:.2f}m).")

    # ------------------------------------------------------------------

    def _build_role_map(self):
        """Map abstract roles -> actual joint names for whichever model
        loaded.  Humanoid names first, biped2d names as fallback."""
        candidates = {
            "r_hip_y":  ("right_hip_y", "torso_to_rightleg"),
            "l_hip_y":  ("left_hip_y",  "torso_to_leftleg"),
            "r_hip_x":  ("right_hip_x",),
            "l_hip_x":  ("left_hip_x",),
            "r_knee":   ("right_knee", "r_knee"),
            "l_knee":   ("left_knee",  "l_knee"),
            "r_ankle":  ("right_ankle_y", "r_ankle"),
            "l_ankle":  ("left_ankle_y", "l_ankle"),
            "r_sh":     ("right_shoulder1",),
            "l_sh":     ("left_shoulder1",),
            "abdomen":  ("abdomen_y", "torso_to_z"),
        }
        roles = {}
        for role, names in candidates.items():
            for nm in names:
                if nm in self._joints:
                    roles[role] = nm
                    break
        return roles

    def _measure_leg_geometry(self):
        def jpos(role):
            nm = self._roles.get(role)
            if nm is None or nm not in self._joints:
                return None
            info = p.getJointInfo(self.biped_id, self._joints[nm],
                                  physicsClientId=self.cid)
            return np.array(info[14])
        try:
            knee = jpos("r_knee")
            ankle = jpos("r_ankle")
            l1 = float(np.linalg.norm(knee)) if knee is not None else 0.40
            l2 = float(np.linalg.norm(ankle)) if ankle is not None else 0.40
            l1 = l1 if 0.05 < l1 < 0.9 else 0.40
            l2 = l2 if 0.05 < l2 < 0.9 else 0.40
            return l1, l2
        except Exception:
            return 0.40, 0.40

    def _hold_neutral(self):
        for nm, jid in self._joints.items():
            try:
                p.resetJointState(self.biped_id, jid, 0.0, 0.0,
                                  physicsClientId=self.cid)
                p.setJointMotorControl2(
                    self.biped_id, jid, controlMode=p.POSITION_CONTROL,
                    targetPosition=0.0, force=self.prm.joint_force,
                    maxVelocity=self.prm.joint_max_vel,
                    physicsClientId=self.cid)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # State estimation
    # ------------------------------------------------------------------

    def _root_state(self):
        pos, orn = p.getBasePositionAndOrientation(
            self.biped_id, physicsClientId=self.cid)
        lin, ang = p.getBaseVelocity(self.biped_id,
                                     physicsClientId=self.cid)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        return (np.array(pos), roll, pitch,
                np.array(lin), np.array(ang))

    # ------------------------------------------------------------------
    # Balance -- torso pitch/roll PD via abdomen, capture-point feedback
    # ------------------------------------------------------------------

    def _balance(self, dt):
        pos, roll, pitch, lin, ang = self._root_state()
        self._cur_height = pos[2]
        self._cur_pitch = pitch
        self._cur_roll = roll

        # forward speed along facing -> capture-point term (IK mode)
        fwd_v = (lin[0] * math.cos(self._facing) +
                 lin[1] * math.sin(self._facing))
        self.gait.fb_fwd = float(np.clip(
            0.10 * (fwd_v - self.prm.walk_speed), -0.06, 0.06))
        self.gait.fb_tilt = float(np.clip(0.18 * pitch, -0.06, 0.06))

        # torso pitch PD -> abdomen lean correction (active: DOF exists)
        pitch_cmd = (-self.prm.balance_kp_pitch * pitch
                     - self.prm.balance_kd_pitch * ang[1]) * dt
        roll_cmd = (-self.prm.balance_kp_roll * roll
                    - self.prm.balance_kd_roll * ang[0]) * dt

        if "abdomen" in self._roles:
            base_lean = -self.prm.abdomen_lean
            self.pd.drive(self._joints[self._roles["abdomen"]],
                          base_lean + float(np.clip(pitch_cmd,
                                                    -0.3, 0.3)))
        if "abdomen_x" in self._joints:
            self.pd.drive(self._joints["abdomen_x"],
                          float(np.clip(roll_cmd, -0.3, 0.3)))

    def _check_fall(self):
        fell = (self._cur_height < self.prm.fall_height or
                abs(self._cur_pitch) > self.prm.fall_pitch or
                abs(getattr(self, "_cur_roll", 0.0)) > self.prm.fall_pitch)
        if fell:
            self._fallen = True
            if self.prm.auto_reset:
                self.reset([self._wx, self._wy, self.SPAWN_Z])
        return fell

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_pos(self):
        try:
            pos, _ = p.getBasePositionAndOrientation(
                self.biped_id, physicsClientId=self.cid)
            return [pos[0], pos[1], pos[2]]
        except Exception:
            return [self._wx, self._wy, self.SPAWN_Z]

    def reset(self, position):
        """The ONLY place the base is repositioned (spawn / after a fall)."""
        self._wx = float(position[0])
        self._wy = float(position[1])
        self._facing = 0.0
        self._fallen = False
        self._t = 0.0
        self.gait.reset()
        self.planner.reset(self._facing, (self._wx, self._wy))

        spawn_z = self.SPAWN_Z if self.is_humanoid else 0.48
        for jid in self._joints.values():
            p.setJointMotorControl2(
                self.biped_id, jid, controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0.0, force=0.0,
                physicsClientId=self.cid)
        p.resetBasePositionAndOrientation(
            self.biped_id, [self._wx, self._wy, spawn_z],
            [0, 0, 0, 1], physicsClientId=self.cid)
        p.resetBaseVelocity(self.biped_id, [0, 0, 0], [0, 0, 0],
                            physicsClientId=self.cid)
        for jid in self._joints.values():
            p.resetJointState(self.biped_id, jid, 0.0, 0.0,
                              physicsClientId=self.cid)
        self._hold_neutral()

    def explore(self, dt, sensor_distances=None):
        heading = self.planner.plan(dt, self.cid, self.get_pos(),
                                    sensor_distances)
        vx = math.cos(heading) * self.prm.walk_speed
        vy = math.sin(heading) * self.prm.walk_speed
        return self._locomote(vx, vy, dt, autonomous=True)

    def step(self, target_vx, target_vy, dt):
        if target_vx is None or target_vy is None:
            return self.explore(dt)
        return self._locomote(target_vx, target_vy, dt, autonomous=False)

    # ------------------------------------------------------------------
    # Core locomotion -- PHYSICS moves the body (no base teleport here)
    # ------------------------------------------------------------------

    def _locomote(self, target_vx, target_vy, dt, autonomous):
        self._t += dt
        speed = math.hypot(target_vx, target_vy)
        moving = speed > 0.05
        if moving:
            self._facing = math.atan2(target_vy, target_vx)

        # mid level: joint targets (refreshed by balance feedback)
        self._balance(dt)
        targets = self.gait.step(dt, moving)

        # low level: stiff position control on every mapped leg/arm joint
        for nm, ang in targets.items():
            if nm in self._joints:
                self.pd.drive(self._joints[nm], ang)

        # direction: steering force at the torso toward the target
        # heading -- same mechanism as HumanoidCPGWalker.  This + the leg
        # gait + ground contact moves the (massed) body through physics.
        # The base is NEVER teleported during a step.
        #
        # CRITICAL FIX -- substep duty-cycle compensation:
        # PyBullet CLEARS applyExternalForce after every stepSimulation().
        # The game loop calls walker.step() ONCE and then steps the sim
        # SUBSTEPS times, so a single applyExternalForce only acts on
        # 1 of N physics steps -- the robot got ~1/N of the intended
        # push and just shuffled in place ("barely moves").  Scaling the
        # force by N makes the net impulse over the decision interval
        # equal to a properly sustained force, so the body actually
        # translates.  (This is also why the CPG hider half-worked.)
        px, py, _ = self.get_pos()
        if moving:
            pos = [px, py, self.get_pos()[2]]
            scale = min(speed * self.prm.steer_force, self.prm.max_steer)
            scale *= self.prm.substeps          # duty-cycle compensation
            try:
                p.applyExternalForce(
                    self.biped_id, -1,
                    [target_vx / speed * scale,
                     target_vy / speed * scale, 0.0],
                    pos, p.WORLD_FRAME, physicsClientId=self.cid)
            except Exception:
                pass

        # soft arena keep-in (gentle inward nudge, not a teleport)
        if abs(px) > self.MAP_LIMIT or abs(py) > self.MAP_LIMIT:
            try:
                p.applyExternalForce(
                    self.biped_id, -1,
                    [-px * 40.0 * self.prm.substeps,
                     -py * 40.0 * self.prm.substeps, 0.0],
                    [px, py, self.get_pos()[2]], p.WORLD_FRAME,
                    physicsClientId=self.cid)
            except Exception:
                pass
        self._wx, self._wy = px, py

        fell = self._check_fall()

        if self.logger is not None:
            self.logger.log(
                t=round(self._t, 4),
                phase=round(self.gait.phase, 4),
                x=round(px, 4), y=round(py, 4),
                heading=round(self._facing, 4),
                pitch=round(getattr(self, "_cur_pitch", 0.0), 4),
                height=round(getattr(self, "_cur_height", 0.0), 4),
                speed=round(speed, 4), fell=int(fell),
                mode="auto" if autonomous else "cmd")
        return fell

    def draw_debug(self):
        o = self.get_pos()
        hx = o[0] + math.cos(self._facing) * 1.5
        hy = o[1] + math.sin(self._facing) * 1.5
        try:
            if self._dbg_line is None:
                self._dbg_line = p.addUserDebugLine(
                    o, [hx, hy, o[2]], [1, 1, 0], 2,
                    physicsClientId=self.cid)
            else:
                self._dbg_line = p.addUserDebugLine(
                    o, [hx, hy, o[2]], [1, 1, 0], 2,
                    replaceItemUniqueId=self._dbg_line,
                    physicsClientId=self.cid)
        except Exception:
            pass

    def save_log(self, path="biped_telemetry.csv"):
        if self.logger is not None:
            self.logger.dump(path)


# ===========================================================================
# Stand-alone autonomous-exploration demo
# ===========================================================================

def _demo():
    try:
        cid = p.connect(p.GUI)
        if cid < 0:
            raise RuntimeError
    except Exception:
        cid = p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                              physicsClientId=cid)
    p.setGravity(0, 0, -9.8, physicsClientId=cid)
    p.setTimeStep(1.0 / 240.0, physicsClientId=cid)
    p.loadURDF("plane.urdf", physicsClientId=cid)

    for (ox, oy) in ((3.0, 0.5), (-2.5, 2.0), (0.5, -3.0), (3.5, -2.5)):
        col = p.createCollisionShape(p.GEOM_BOX,
                                     halfExtents=[0.3, 0.3, 0.6],
                                     physicsClientId=cid)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.6],
                                  rgbaColor=[0.7, 0.2, 0.2, 1],
                                  physicsClientId=cid)
        p.createMultiBody(0, col, vis, [ox, oy, 0.6],
                          physicsClientId=cid)

    walker = BipedNNWalker(cid, [0, 0, BipedNNWalker.SPAWN_Z], log=True)

    dt, sub = 1.0 / 30.0, 8
    gui = p.getConnectionInfo(cid)["connectionMethod"] == p.GUI
    for k in range(int(30.0 / dt)):
        walker.explore(dt)
        for _ in range(sub):
            p.stepSimulation(physicsClientId=cid)
        if gui:
            walker.draw_debug()
            time.sleep(dt)
        if k % 30 == 0:
            x, y, z = walker.get_pos()
            print(f"  t={k*dt:5.1f}s pos=({x:+.2f},{y:+.2f},{z:.2f}) "
                  f"hd={math.degrees(walker._facing):+6.1f} deg")

    walker.save_log("biped_telemetry.csv")
    p.disconnect(physicsClientId=cid)


if __name__ == "__main__":
    _demo()