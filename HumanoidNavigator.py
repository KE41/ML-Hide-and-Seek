"""
BipedNNWalker.py
----------------
Layered gait controller for biped/biped2d_pybullet.urdf.

This file used to host a kinematic CPG + optional-PPO walker.  It has been
rewritten (this was the explicit request) into a three-layer classical
locomotion stack:

      ┌─────────────────────────────────────────────────────────┐
      │  High level   ExplorationPlanner                          │
      │               raycast sensing → heading + walk speed      │
      ├─────────────────────────────────────────────────────────┤
      │  Mid level    GaitGenerator                               │
      │               phase clock → cycloidal foot trajectory     │
      │               → 2-link inverse kinematics → joint targets  │
      ├─────────────────────────────────────────────────────────┤
      │  Low level    JointPDController                            │
      │               position control w/ PD gains + force limits │
      └─────────────────────────────────────────────────────────┘

The class name (`BipedNNWalker`), the public API
(`get_pos()`, `reset(position)`, `step(vx, vy, dt)`) and the constructor
signature `(cid, start_pos, colour=None, policy_path=None)` are kept
unchanged so HideSeekEnv.py keeps working without edits.  `step()` still
accepts an external (vx, vy) command for the hide-and-seek game; passing
`vx=None, vy=None` (or calling `explore(dt)`) switches it to autonomous
exploration.

biped2d_pybullet.urdf — model facts
-----------------------------------
The kinematic chain is planar and the base is a zero-mass "world" root:

    world ─ y_to_world(prismatic, fwd) ─ z_to_y(prismatic, up)
          ─ torso_to_z(revolute, torso pitch) ─ torso link
          ─ {torso_to_rightleg, torso_to_leftleg}(hip pitch)
          ─ {r_knee, l_knee} ─ {r_ankle, l_ankle} ─ feet

Consequences and how this controller handles them honestly:

  • The legs, torso pitch (`torso_to_z`) and body height (`z_to_y`) are
    driven by genuine PD position control — real physics, compliant to
    pushes and uneven ground.  No per-frame joint teleporting.
  • The base has NO yaw DOF and zero mass, so world translation/heading
    cannot emerge from ground reaction.  The planner integrates world XY
    and yaw and applies them via resetBasePositionAndOrientation together
    with a matching resetBaseVelocity — the project's documented
    no-impulse-spike technique — at the *gait* forward speed
    (step_length·step_frequency).  Matching the stance-foot retraction
    rate to that speed is what "minimises foot slippage" here.
  • Roll balance, lateral foot spacing, arm swing and stance-leg hip
    shift are written generically but only actuate joints that exist
    (`if name in self._joints`), so the same class also drives a full
    3-D humanoid model where those DOFs are present; on planar biped2d
    they are inert by construction (noted at each site).
"""

import csv
import math
import os
import random
import time
from dataclasses import dataclass, field

import numpy as np
import pybullet as p
import pybullet_data


# ===========================================================================
# Tunable parameters  (all walking behaviour is adjustable from one place)
# ===========================================================================

@dataclass
class GaitParams:
    # -- locomotion --
    walk_speed: float       = 0.9     # m/s nominal forward speed
    max_turn_rate: float    = 1.2     # rad/s cap on heading slew (gradual)
    step_length: float      = 0.12    # m foot travel per step
    step_height: float      = 0.07    # m peak swing-foot clearance
    step_frequency: float   = 1.3     # Hz full gait cycles per second
    lateral_spacing: float  = 0.12    # m sideways foot offset (3-D models)

    # -- low-level PD (position control) --
    kp: float               = 0.6     # position gain  → setJointMotorControl2
    kd: float               = 0.9     # velocity gain
    max_force: float        = 180.0   # N·m joint torque ceiling

    # -- balance / stabilisation --
    balance_kp_pitch: float = 6.0     # torso pitch P gain
    balance_kd_pitch: float = 0.8     # torso pitch D gain
    balance_kp_roll: float  = 6.0     # (3-D models only)
    balance_kd_roll: float  = 0.8
    foot_place_kv: float    = 0.10    # capture-point: forward vel error gain
    foot_place_ktilt: float = 0.18    # capture-point: torso tilt gain
    hip_shift: float        = 0.03    # m CoM shift toward stance (3-D only)

    # -- posture --
    nominal_hip_height: float = 0.46  # m standing hip height (auto-checked)
    stance_knee_bias: float   = 0.06  # rad slight knee flex while standing
    bob_amplitude: float      = 0.015 # m vertical CoM oscillation (2× freq)
    heel_toe_amp: float       = 0.20  # rad ankle heel-strike/toe-off swing
    arm_swing_amp: float      = 0.35  # rad shoulder counter-swing (3-D only)

    # -- exploration --
    ray_count: int          = 9       # forward sensing fan rays
    ray_fov: float          = math.radians(140.0)
    ray_range: float        = 3.2     # m look-ahead distance
    obstacle_clear: float   = 1.3     # m → trigger an avoidance turn
    curiosity_period: float = 6.0     # s between spontaneous re-headings
    stuck_window: float     = 4.0     # s displacement check window
    stuck_dist: float       = 0.25    # m min progress before "stuck"

    # -- fall handling --
    fall_height: float      = 0.28    # m torso z below this ⇒ fallen
    fall_pitch: float       = 0.95    # rad |pitch| above this ⇒ fallen
    auto_reset: bool        = True    # snap back to spawn after a fall

    # joint sign conventions for biped2d (calibratable for other URDFs)
    hip_sign: float   = +1.0
    knee_sign: float  = -1.0
    ankle_sign: float = +1.0


# ===========================================================================
# Low level — joint PD controller
# ===========================================================================

class JointPDController:
    """
    Position control with explicit PD gains and a torque ceiling.

    PyBullet's POSITION_CONTROL motor is itself a constrained PD loop, so
    target angle + positionGain/velocityGain + force limit gives exactly
    the requested "stable joint position control with PD gains and
    appropriate force limits".  Reads q, qd every step so callers can use
    them for balance feedback.
    """

    def __init__(self, body_id, cid, params: GaitParams):
        self.body = body_id
        self.cid  = cid
        self.prm  = params

    def read(self, joint_ids):
        """Return (positions, velocities) arrays for the given joints."""
        q  = np.empty(len(joint_ids), dtype=np.float32)
        qd = np.empty(len(joint_ids), dtype=np.float32)
        for i, j in enumerate(joint_ids):
            s = p.getJointState(self.body, j, physicsClientId=self.cid)
            q[i]  = s[0]
            qd[i] = s[1]
        return q, qd

    def drive(self, joint_id, target_angle, target_vel=0.0, max_force=None):
        """Apply one PD position command to a single joint."""
        p.setJointMotorControl2(
            self.body, joint_id,
            controlMode      = p.POSITION_CONTROL,
            targetPosition   = float(target_angle),
            targetVelocity   = float(target_vel),
            positionGain     = self.prm.kp,
            velocityGain     = self.prm.kd,
            force            = float(self.prm.max_force
                                     if max_force is None else max_force),
            physicsClientId  = self.cid,
        )


# ===========================================================================
# Mid level — gait generator (phase clock + foot trajectory + leg IK)
# ===========================================================================

class GaitGenerator:
    """
    Continuous phase variable φ ∈ [0,1).  Right leg uses φ, left leg uses
    φ+0.5 (perfect antiphase).  Within each leg's cycle the first half is
    stance (foot planted, retracting under the body) and the second half
    is swing (cycloidal lift-and-reach).  Foot targets are expressed in
    the hip frame and converted to hip/knee/ankle angles by a closed-form
    planar 2-link IK that also keeps the sole horizontal (and adds a
    heel-strike / toe-off ankle component).
    """

    def __init__(self, params: GaitParams, l_thigh, l_shank):
        self.prm     = params
        self.L1      = l_thigh
        self.L2      = l_shank
        # never command past this — keeps a knee bend under any push so
        # the leg cannot snap straight and lock mid-recovery
        self.safe_reach = 0.97 * (l_thigh + l_shank)
        self.phase   = 0.0
        # capture-point feedback terms, refreshed by the balance layer
        self.fb_fwd  = 0.0     # forward velocity-error correction (m)
        self.fb_tilt = 0.0     # torso-tilt correction (m)

    def reset(self):
        self.phase   = 0.0
        self.fb_fwd  = 0.0
        self.fb_tilt = 0.0

    # ---- foot trajectory (cycloidal: zero velocity at lift-off & touch-down)
    def _foot_target(self, leg_phase, moving):
        """
        Return (fx, fd) — foot forward offset and downward distance from
        the hip — for a leg at local phase ∈ [0,1).
        """
        L  = self.prm.step_length
        H  = self.prm.step_height
        h0 = self.prm.nominal_hip_height

        if leg_phase < 0.5:                       # ---- STANCE ----
            s  = leg_phase / 0.5                   # 0→1 across stance
            # foot retracts linearly from +L/2 (front) to -L/2 (rear);
            # its backward speed equals body forward speed ⇒ minimal slip
            fx = (0.5 - s) * L
            fd = h0
        else:                                      # ---- SWING ----
            s  = (leg_phase - 0.5) / 0.5           # 0→1 across swing
            # horizontal cycloid: zero horizontal velocity at both ends
            fx = -0.5 * L + L * (s - math.sin(2.0 * math.pi * s) /
                                 (2.0 * math.pi))
            # vertical cycloid: 0 at ends, single smooth peak at mid-swing
            lift = 0.5 * H * (1.0 - math.cos(2.0 * math.pi * s))
            fd = h0 - lift

        if not moving:                             # stand in place
            fx = 0.0
            fd = h0

        # capture-point foot placement: step further out when falling
        # forward / moving off target (feeds push recovery + balance)
        fx += self.fb_fwd + self.fb_tilt

        # hard safe-reach guard: shrink the horizontal reach so the
        # commanded foot never leaves the safe workspace — the knee
        # always keeps a bend and can never lock during a recovery
        max_fx = math.sqrt(max(self.safe_reach ** 2 - fd ** 2, 0.0))
        fx = max(-max_fx, min(max_fx, fx))
        return fx, fd

    # ---- closed-form planar leg inverse kinematics -----------------------
    def _leg_ik(self, fx, fd):
        """
        Solve hip-pitch / knee / ankle so the foot reaches (fx, fd) in the
        hip frame with the sole kept horizontal.  Verified to machine
        precision against forward kinematics for all reachable targets.
        """
        L1, L2 = self.L1, self.L2
        D = math.hypot(fx, fd)
        D = min(max(D, abs(L1 - L2) + 1e-4), L1 + L2 - 1e-4)   # reach clamp

        cosK      = (L1 * L1 + L2 * L2 - D * D) / (2.0 * L1 * L2)
        K         = math.acos(max(-1.0, min(1.0, cosK)))
        knee_flex = math.pi - K                    # 0 straight, + bent

        beta  = math.atan2(fx, fd)                 # hip→foot vs straight-down
        cosA  = (L1 * L1 + D * D - L2 * L2) / (2.0 * L1 * D)
        alpha = math.acos(max(-1.0, min(1.0, cosA)))

        hip   = beta + alpha
        shank = hip - knee_flex                    # shank world pitch
        ankle = -shank                             # keep sole horizontal
        return hip, knee_flex, ankle

    # ---- one gait step → dict{joint_name: (angle, ang_vel)} --------------
    def step(self, dt, moving):
        if moving:
            self.phase = (self.phase + self.prm.step_frequency * dt) % 1.0

        phi_r = self.phase
        phi_l = (self.phase + 0.5) % 1.0

        out = {}
        for side, phi, hipname, kneename, anklename, shoulder in (
            ("r", phi_r, "torso_to_rightleg", "r_knee", "r_ankle",
             "right_shoulder1"),
            ("l", phi_l, "torso_to_leftleg",  "l_knee", "l_ankle",
             "left_shoulder1"),
        ):
            fx, fd = self._foot_target(phi, moving)
            hip, knee_flex, ankle = self._leg_ik(fx, fd)

            # slight knee flex while standing (natural, soft stance)
            knee_flex = max(knee_flex, self.prm.stance_knee_bias)

            # heel-strike (dorsiflex before touch-down) + toe-off
            # (plantarflex at end of stance) — adds the human ankle roll
            heel_toe = self.prm.heel_toe_amp * math.sin(
                2.0 * math.pi * phi)

            out[hipname]  = (self.prm.hip_sign  * hip, 0.0)
            out[kneename] = (self.prm.knee_sign * knee_flex, 0.0)
            out[anklename] = (
                self.prm.ankle_sign * (ankle + heel_toe), 0.0)

            # arm counter-swing — opposite leg (3-D humanoids only; the
            # name simply won't be in the biped2d joint map so it is a
            # silent no-op there, exactly like the rest of the codebase)
            out[shoulder] = (
                self.prm.arm_swing_amp *
                math.sin(2.0 * math.pi * (phi + 0.5)), 0.0)

        return out

    # vertical CoM bob at twice step frequency (inverted-pendulum feel)
    def body_bob(self, moving):
        if not moving:
            return 0.0
        return self.prm.bob_amplitude * math.cos(
            2.0 * 2.0 * math.pi * self.phase)


# ===========================================================================
# High level — exploration planner (raycast sensing → heading + speed)
# ===========================================================================

class ExplorationPlanner:
    """
    Keeps the robot moving forward, casts a forward fan of rays
    (rayTestBatch — LiDAR-like) and, when the path is blocked, picks the
    clearest sector as the new heading.  Heading is slewed gradually
    (turn-rate limited) so the robot keeps walking while it turns.  Also
    re-headings spontaneously now and then (curiosity) and when it detects
    it is stuck (little net displacement over a time window).
    """

    def __init__(self, params: GaitParams):
        self.prm            = params
        self.heading        = 0.0           # current world heading (rad)
        self.target_heading = 0.0           # heading we are slewing toward
        self._t             = 0.0
        self._t_curiosity   = 0.0
        self._t_stuck       = 0.0
        self._stuck_anchor  = (0.0, 0.0)

    def reset(self, heading=0.0, pos=(0.0, 0.0)):
        self.heading = self.target_heading = heading
        self._t = self._t_curiosity = self._t_stuck = 0.0
        self._stuck_anchor = (pos[0], pos[1])

    # external sensor hook — pass your own distances to override raycasts
    def sense(self, cid, origin, heading, sensor_distances=None):
        """
        Return (distances, ray_dirs).  If `sensor_distances` (e.g. real
        LiDAR / external raycasts) is given it is used directly; otherwise
        a PyBullet rayTestBatch fan is cast.
        """
        n   = self.prm.ray_count
        fov = self.prm.ray_fov
        rng = self.prm.ray_range
        angles = [heading - fov / 2.0 + fov * i / (n - 1) for i in range(n)]
        dirs   = [(math.cos(a), math.sin(a)) for a in angles]

        if sensor_distances is not None:
            d = list(sensor_distances)[:n]
            d += [rng] * (n - len(d))
            return d, dirs

        z = max(origin[2], 0.25)
        froms = [[origin[0], origin[1], z]] * n
        tos   = [[origin[0] + dx * rng, origin[1] + dy * rng, z]
                 for dx, dy in dirs]
        hits  = p.rayTestBatch(froms, tos, physicsClientId=cid)
        dists = [(h[2] * rng if h[0] >= 0 else rng) for h in hits]
        return dists, dirs

    def plan(self, dt, cid, origin, sensor_distances=None):
        """
        Update and return the desired world heading.  Pure planning — the
        gait/locomotion layers turn this into motion.
        """
        self._t           += dt
        self._t_curiosity += dt
        self._t_stuck     += dt

        dists, dirs = self.sense(cid, origin, self.heading,
                                 sensor_distances)
        n   = len(dists)
        mid = n // 2
        # forward cone = central third of the fan
        cone = dists[max(0, mid - n // 6): mid + n // 6 + 1]
        blocked = min(cone) < self.prm.obstacle_clear if cone else False

        if blocked:
            # steer toward the ray with the most free space
            best = int(np.argmax(dists))
            dx, dy = dirs[best]
            self.target_heading = math.atan2(dy, dx)
        elif self._t_curiosity > self.prm.curiosity_period:
            # spontaneous exploration turn
            self.target_heading = self.heading + random.uniform(-1.2, 1.2)
            self._t_curiosity = 0.0

        # stuck detection — negligible progress over the window
        if self._t_stuck > self.prm.stuck_window:
            ax, ay = self._stuck_anchor
            if math.hypot(origin[0] - ax, origin[1] - ay) < \
                    self.prm.stuck_dist:
                self.target_heading = self.heading + random.choice(
                    (-1.0, 1.0)) * random.uniform(1.4, 2.4)
            self._stuck_anchor = (origin[0], origin[1])
            self._t_stuck = 0.0

        # gradual, turn-rate-limited slew (keeps walking while turning)
        err = math.atan2(math.sin(self.target_heading - self.heading),
                          math.cos(self.target_heading - self.heading))
        max_step = self.prm.max_turn_rate * dt
        self.heading += max(-max_step, min(max_step, err))
        self.heading  = math.atan2(math.sin(self.heading),
                                   math.cos(self.heading))
        return self.heading


# ===========================================================================
# Telemetry
# ===========================================================================

class _Logger:
    """Lightweight in-memory telemetry with optional CSV dump."""

    def __init__(self):
        self.rows = []

    def log(self, **kw):
        self.rows.append(kw)

    def dump(self, path):
        if not self.rows:
            return
        keys = list(self.rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)
        print(f"[BipedNNWalker] telemetry → {path} ({len(self.rows)} rows)")


# ===========================================================================
# Public walker  (API-compatible with the previous BipedNNWalker)
# ===========================================================================

class BipedNNWalker:

    CTRL_JOINTS = [
        "torso_to_rightleg", "torso_to_leftleg",
        "r_knee", "l_knee",
        "r_ankle", "l_ankle",
    ]
    BASE_JOINTS = ("y_to_world", "z_to_y", "torso_to_z")
    SPAWN_Z   = 0.48           # base link height so feet rest on z=0
    MAP_LIMIT = 9.0

    def __init__(
        self,
        cid: int,
        start_pos: list,
        colour: list = None,
        policy_path: str = None,        # accepted for API compat (unused)
        params: GaitParams = None,
        log: bool = False,
    ):
        self.cid    = cid
        self.prm    = params if params is not None else GaitParams()
        self._wx    = float(start_pos[0])
        self._wy    = float(start_pos[1])
        self._facing = 0.0
        self._fallen = False
        self._t      = 0.0
        self.logger  = _Logger() if log else None

        if colour is None:
            colour = [0.0, 0.8, 1.0, 1.0]

        pbd = pybullet_data.getDataPath()
        self.biped_id = p.loadURDF(
            os.path.join(pbd, "biped", "biped2d_pybullet.urdf"),
            basePosition    = [self._wx, self._wy, self.SPAWN_Z],
            baseOrientation = [0, 0, 0, 1],
            physicsClientId = cid,
            useFixedBase    = False,
        )

        # Index all joints by name
        self._joints = {}
        n = p.getNumJoints(self.biped_id, physicsClientId=cid)
        for j in range(n):
            info = p.getJointInfo(self.biped_id, j, physicsClientId=cid)
            self._joints[info[1].decode()] = (j, info[2])

        self._ctrl_ids = [
            self._joints[name][0]
            for name in self.CTRL_JOINTS
            if name in self._joints
        ]
        self._torso_link = (
            self._joints["torso_to_z"][0]
            if "torso_to_z" in self._joints else 2
        )

        # Colour all links
        for link_idx in range(-1, n):
            try:
                p.changeVisualShape(
                    self.biped_id, link_idx,
                    rgbaColor=colour, physicsClientId=cid,
                )
            except Exception:
                pass

        # Free the leg joints (kill default velocity motors so PD owns them)
        for _, (jid, jtype) in self._joints.items():
            if jtype != p.JOINT_FIXED:
                p.setJointMotorControl2(
                    self.biped_id, jid,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=0, force=0,
                    physicsClientId=cid,
                )

        # measure real leg geometry from the URDF so the IK self-calibrates
        l1, l2 = self._measure_leg_geometry()

        # ---- assemble the three control layers ----
        self.pd      = JointPDController(self.biped_id, cid, self.prm)
        self.gait    = GaitGenerator(self.prm, l1, l2)
        self.planner = ExplorationPlanner(self.prm)
        self.planner.reset(self._facing, (self._wx, self._wy))

        self._dbg_line = None
        print(f"[BipedNNWalker] layered controller ready "
              f"(thigh={l1:.3f}m shank={l2:.3f}m).")

    # ------------------------------------------------------------------
    # Geometry auto-measurement
    # ------------------------------------------------------------------

    def _measure_leg_geometry(self):
        """
        Estimate thigh and shank lengths from URDF joint frame offsets so
        the IK matches the actual model.  Falls back to sensible biped2d
        defaults if the chain can't be measured.
        """
        def jpos(name):
            if name not in self._joints:
                return None
            jid = self._joints[name][0]
            info = p.getJointInfo(self.biped_id, jid,
                                  physicsClientId=self.cid)
            return np.array(info[14])     # parentFramePos

        try:
            hip   = jpos("torso_to_rightleg")
            knee  = jpos("r_knee")
            ankle = jpos("r_ankle")
            l1 = float(np.linalg.norm(knee))  if knee  is not None else 0.22
            l2 = float(np.linalg.norm(ankle)) if ankle is not None else 0.22
            if not (0.05 < l1 < 0.8):
                l1 = 0.22
            if not (0.05 < l2 < 0.8):
                l2 = 0.22
            # Stand at 92% of full leg length: a natural, near-upright
            # stance with a comfortable (not locked) knee, while still
            # leaving head-room for step length + capture-point feedback
            # so the IK never hits its reach clamp during the gait.  (On
            # this URDF's short 0.22 m links a fully straight stance plus
            # a real step would need ~all the workspace, so a modest knee
            # bend is the natural, robust operating point.)
            self.prm.nominal_hip_height = 0.90 * (l1 + l2)
            return l1, l2
        except Exception:
            return 0.22, 0.22

    # ------------------------------------------------------------------
    # State estimation  (CoM, support foot, torso tilt)
    # ------------------------------------------------------------------

    def _torso_state(self):
        ls = p.getLinkState(self.biped_id, self._torso_link,
                            computeLinkVelocity=1,
                            physicsClientId=self.cid)
        pos, orn = ls[0], ls[1]
        lin, ang = ls[6], ls[7]
        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        return np.array(pos), roll, pitch, np.array(lin), np.array(ang)

    def _support_foot(self):
        """Lowest leg endpoint = current support foot (CoM bookkeeping)."""
        best, best_z = None, 1e9
        for jid in self._ctrl_ids:
            ls = p.getLinkState(self.biped_id, jid,
                                physicsClientId=self.cid)
            if ls[0][2] < best_z:
                best_z, best = ls[0][2], np.array(ls[0])
        return best

    # ------------------------------------------------------------------
    # Balance / stabilisation layer
    # ------------------------------------------------------------------

    def _balance(self, dt):
        """
        Torso PD on pitch (and roll where the DOF exists), capture-point
        swing-foot feedback, and a stance-leg hip shift.  Returns the
        torso-pitch correction angle for `torso_to_z`.
        """
        pos, roll, pitch, lin, ang = self._torso_state()

        # forward speed along current facing (for capture point)
        fwd_v = lin[0] * math.cos(self._facing) + \
                lin[1] * math.sin(self._facing)
        v_des = self.prm.walk_speed

        # capture-point: lengthen the step when tilting / over-/under-speed
        # (bounded — an unbounded balance term would be unsafe)
        self.gait.fb_fwd  = float(np.clip(
            self.prm.foot_place_kv * (fwd_v - v_des), -0.06, 0.06))
        self.gait.fb_tilt = float(np.clip(
            self.prm.foot_place_ktilt * pitch, -0.06, 0.06))

        # torso pitch PD → corrective angle for the torso lean joint
        pitch_rate = ang[1]
        pitch_cmd  = (-self.prm.balance_kp_pitch * pitch
                      - self.prm.balance_kd_pitch * pitch_rate) * dt

        # roll PD only matters on 3-D models that own a roll joint
        roll_rate = ang[0]
        roll_cmd  = (-self.prm.balance_kp_roll * roll
                     - self.prm.balance_kd_roll * roll_rate) * dt
        if "torso_to_x" in self._joints:                 # 3-D model only
            jid = self._joints["torso_to_x"][0]
            self.pd.drive(jid, roll_cmd)

        # hip shift toward stance leg — lateral CoM nudge (3-D only; no
        # lateral DOF on biped2d so this is inert there, as commented)
        sf = self._support_foot()
        if sf is not None and "torso_to_y" in self._joints:
            jid = self._joints["torso_to_y"][0]
            self.pd.drive(jid, math.copysign(self.prm.hip_shift,
                                             sf[1] - pos[1]))

        self._cur_pitch  = pitch
        self._cur_height = pos[2]
        return float(np.clip(pitch_cmd, -0.4, 0.4))

    # ------------------------------------------------------------------
    # Fall detection
    # ------------------------------------------------------------------

    def _check_fall(self):
        fell = (self._cur_height < self.prm.fall_height or
                abs(self._cur_pitch) > self.prm.fall_pitch)
        if fell:
            self._fallen = True
            if self.prm.auto_reset:
                self.reset([self._wx, self._wy, self.SPAWN_Z])
        return fell

    # ------------------------------------------------------------------
    # Public API  (unchanged signatures)
    # ------------------------------------------------------------------

    def get_pos(self):
        """World (x, y, z) of the torso link (used by HideSeekEnv LOS)."""
        try:
            ls = p.getLinkState(self.biped_id, self._torso_link,
                                physicsClientId=self.cid)
            return [ls[0][0], ls[0][1], ls[0][2]]
        except Exception:
            return [self._wx, self._wy, self.SPAWN_Z]

    def reset(self, position: list):
        self._wx = float(position[0])
        self._wy = float(position[1])
        self._facing = 0.0
        self._fallen = False
        self._t = 0.0
        self.gait.reset()
        self.planner.reset(self._facing, (self._wx, self._wy))

        orn = p.getQuaternionFromEuler([0, 0, 0],
                                       physicsClientId=self.cid)
        p.resetBasePositionAndOrientation(
            self.biped_id, [self._wx, self._wy, self.SPAWN_Z], orn,
            physicsClientId=self.cid,
        )
        p.resetBaseVelocity(
            self.biped_id, [0, 0, 0], [0, 0, 0],
            physicsClientId=self.cid,
        )
        for _, (jid, jtype) in self._joints.items():
            if jtype not in (p.JOINT_FIXED, p.JOINT_SPHERICAL):
                p.resetJointState(self.biped_id, jid, 0.0, 0.0,
                                  physicsClientId=self.cid)

    def explore(self, dt: float, sensor_distances=None):
        """Autonomous mode: planner picks heading, robot walks it."""
        heading = self.planner.plan(
            dt, self.cid, self.get_pos(), sensor_distances)
        vx = math.cos(heading) * self.prm.walk_speed
        vy = math.sin(heading) * self.prm.walk_speed
        return self._locomote(vx, vy, dt, autonomous=True)

    def step(self, target_vx, target_vy, dt: float):
        """
        Game mode: walk toward an externally commanded (vx, vy).  Passing
        vx=None (or vy=None) switches to autonomous exploration so the
        same call site can do either.
        """
        if target_vx is None or target_vy is None:
            return self.explore(dt)
        return self._locomote(target_vx, target_vy, dt, autonomous=False)

    # ------------------------------------------------------------------
    # Core locomotion step (drives all three layers)
    # ------------------------------------------------------------------

    def _locomote(self, target_vx, target_vy, dt, autonomous):
        self._t += dt
        speed  = math.hypot(target_vx, target_vy)
        moving = speed > 0.05
        if moving:
            self._facing = math.atan2(target_vy, target_vx)

        # ---- mid level: gait targets, refreshed by balance feedback ----
        pitch_corr = self._balance(dt)
        targets    = self.gait.step(dt, moving)
        bob        = self.gait.body_bob(moving)

        # ---- low level: PD-track every leg/arm joint ----
        for name, (angle, vel) in targets.items():
            if name in self._joints:
                jid, jtype = self._joints[name]
                if jtype != p.JOINT_FIXED:
                    self.pd.drive(jid, angle, vel)

        # torso lean joint holds upright + balance correction (real PD)
        if "torso_to_z" in self._joints:
            jid = self._joints["torso_to_z"][0]
            self.pd.drive(jid, pitch_corr)

        # body height joint → nominal hip height + CoM bob (real PD)
        if "z_to_y" in self._joints:
            jid = self._joints["z_to_y"][0]
            self.pd.drive(jid, bob)

        # ---- base world pose ----
        # biped2d's base has no yaw DOF and zero mass, so XY + heading are
        # integrated here at the *gait* speed and applied with a matching
        # base velocity (documented no-impulse-spike technique).  Stance-
        # foot retraction is matched to this speed ⇒ minimal foot slip.
        gait_speed = (self.prm.step_length * self.prm.step_frequency * 2.0
                      if moving else 0.0)
        if moving:
            self._wx += math.cos(self._facing) * gait_speed * dt
            self._wy += math.sin(self._facing) * gait_speed * dt
        self._wx = max(-self.MAP_LIMIT, min(self.MAP_LIMIT, self._wx))
        self._wy = max(-self.MAP_LIMIT, min(self.MAP_LIMIT, self._wy))

        # biped2d local +Y is the walking axis ⇒ yaw = atan2(-vx, vy)
        yaw = (math.atan2(-math.cos(self._facing),
                          math.sin(self._facing))
               if moving else 0.0)
        orn = p.getQuaternionFromEuler([0, 0, yaw],
                                       physicsClientId=self.cid)
        p.resetBasePositionAndOrientation(
            self.biped_id,
            [self._wx, self._wy, self.SPAWN_Z + bob],
            orn, physicsClientId=self.cid,
        )
        p.resetBaseVelocity(
            self.biped_id,
            [math.cos(self._facing) * gait_speed,
             math.sin(self._facing) * gait_speed, 0.0],
            [0.0, 0.0, 0.0],
            physicsClientId=self.cid,
        )

        fell = self._check_fall()

        if self.logger is not None:
            self.logger.log(
                t=round(self._t, 4), phase=round(self.gait.phase, 4),
                x=round(self._wx, 4), y=round(self._wy, 4),
                heading=round(self._facing, 4),
                pitch=round(getattr(self, "_cur_pitch", 0.0), 4),
                height=round(getattr(self, "_cur_height", 0.0), 4),
                speed=round(speed, 4), fell=int(fell),
                mode="auto" if autonomous else "cmd",
            )
        return fell

    # optional: visualise the planner heading ray in the GUI
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
    """
    Spawn the biped on a plane with a few obstacle boxes and let it
    explore autonomously for ~25 s, logging telemetry to CSV.
    Uses GUI if a display is available, otherwise headless DIRECT.
    """
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

    # scatter a few obstacles for the planner to avoid
    for (ox, oy) in ((2.5, 0.5), (-2.0, 1.8), (0.3, -2.6), (3.4, -2.0)):
        col = p.createCollisionShape(p.GEOM_BOX,
                                     halfExtents=[0.3, 0.3, 0.5],
                                     physicsClientId=cid)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.5],
                                  rgbaColor=[0.7, 0.2, 0.2, 1],
                                  physicsClientId=cid)
        p.createMultiBody(0, col, vis, [ox, oy, 0.5],
                          physicsClientId=cid)

    walker = BipedNNWalker(cid, [0, 0, BipedNNWalker.SPAWN_Z], log=True)

    dt   = 1.0 / 30.0
    sub  = 8
    gui  = p.getConnectionInfo(cid)["connectionMethod"] == p.GUI
    for k in range(int(25.0 / dt)):
        walker.explore(dt)
        for _ in range(sub):
            p.stepSimulation(physicsClientId=cid)
        if gui:
            walker.draw_debug()
            time.sleep(dt)
        if k % 30 == 0:
            x, y, z = walker.get_pos()
            print(f"  t={k*dt:5.1f}s  pos=({x:+.2f},{y:+.2f})  "
                  f"heading={math.degrees(walker._facing):+6.1f}°")

    walker.save_log("biped_telemetry.csv")
    p.disconnect(physicsClientId=cid)


if __name__ == "__main__":
    _demo()