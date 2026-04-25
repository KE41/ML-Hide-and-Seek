# Environment Setup and Creation - Pybullet
# Curriculum Learning version:
#   STAGE = 1 → just learn to stand upright (run until ep_rew_mean > 5.0)
#   STAGE = 2 → motion imitation walking (resume from stage 1 checkpoint)
# Change the STAGE constant below to switch between stages.

import pybullet as p
import pybullet_data
import numpy as np
import time
import json

# ---------------------------------------------------------------------------
# SET THIS TO SWITCH STAGES
# Stage 1: get the robot standing — run until ep_rew_mean consistently > 5.0
# Stage 2: motion imitation walking — resume from best stage 1 checkpoint
STAGE = 1
# ---------------------------------------------------------------------------


class MotionClip:
    """
    Loads a DeepMimic-format motion JSON file.
    Only used in Stage 2.
    """

    def __init__(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)

        frames = data["Frames"]
        self.frame_duration = frames[0][0]
        self.joint_frames   = np.array([frame[8:] for frame in frames], dtype=np.float32)
        self.num_frames     = len(self.joint_frames)
        self.total_duration = self.num_frames * self.frame_duration
        print(f"[MotionClip] Loaded {self.num_frames} frames, "
              f"{self.frame_duration:.4f}s/frame, "
              f"total {self.total_duration:.2f}s")

    def get_frame_at_time(self, elapsed: float) -> np.ndarray:
        """Return reference joints for elapsed time, looping automatically."""
        t      = (elapsed % self.total_duration) / self.frame_duration
        idx_lo = int(t) % self.num_frames
        idx_hi = (idx_lo + 1) % self.num_frames
        alpha  = t - int(t)
        return (1.0 - alpha) * self.joint_frames[idx_lo] + alpha * self.joint_frames[idx_hi]


# ---------------------------------------------------------------------------

def create_environment(gui, motion_path="humanoid3d_walk.txt"):
    cid = p.connect(p.GUI if gui else p.DIRECT)
    if cid < 0:
        raise RuntimeError("Failed to connect to PyBullet")

    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=cid)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)

    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    p.resetSimulation(physicsClientId=cid)
    p.setGravity(0, 0, -9.8, physicsClientId=cid)
    p.setTimeStep(1. / 240., physicsClientId=cid)

    if gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=5,
            cameraYaw=270,
            cameraPitch=-20,
            cameraTargetPosition=[0, 0, 1],
            physicsClientId=cid
        )

    # ---- Floor ----
    p.loadURDF("plane.urdf", physicsClientId=cid)

    # ---- Walls ----
    wall     = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 10, 2], physicsClientId=cid)
    tb_wall  = p.createCollisionShape(p.GEOM_BOX, halfExtents=[10, 0.4, 2], physicsClientId=cid)
    wall_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 10, 2],  rgbaColor=[1,0,0,1], physicsClientId=cid)
    tb_vis   = p.createVisualShape(p.GEOM_BOX, halfExtents=[10, 0.4, 2], rgbaColor=[0,0,1,1], physicsClientId=cid)

    p.createMultiBody(0, wall,    wall_vis, basePosition=[ 10,  0, 2], physicsClientId=cid)
    p.createMultiBody(0, wall,    wall_vis, basePosition=[-10,  0, 2], physicsClientId=cid)
    p.createMultiBody(0, tb_wall, tb_vis,   basePosition=[  0, 10, 2], physicsClientId=cid)
    p.createMultiBody(0, tb_wall, tb_vis,   basePosition=[  0,-10, 2], physicsClientId=cid)

    # 1 RED cube
    cube1_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.6], physicsClientId=cid)
    cube1_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.6], rgbaColor=[1, 0, 0, 1], physicsClientId=cid)
    p.createMultiBody(2.0, cube1_col, cube1_vis, basePosition=[4, 4, 1.8], physicsClientId=cid)

    # 2 BLUE cube
    cube2_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.6], physicsClientId=cid)
    cube2_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.6], rgbaColor=[0, 0, 1, 1], physicsClientId=cid)
    p.createMultiBody(2.0, cube2_col, cube2_vis, basePosition=[-4, -4, 1.8], physicsClientId=cid)

    # 3 GREEN cube
    cube3_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.6], physicsClientId=cid)
    cube3_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.6], rgbaColor=[0, 1, 0, 1], physicsClientId=cid)
    p.createMultiBody(2.0, cube3_col, cube3_vis, basePosition=[4, -4, 1.8], physicsClientId=cid)

    # 4 YELLOW cube
    cube4_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.6], physicsClientId=cid)
    cube4_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.6], rgbaColor=[1, 1, 0, 1], physicsClientId=cid)
    p.createMultiBody(2.0, cube4_col, cube4_vis, basePosition=[-4, 4, 1.8], physicsClientId=cid)

    # 5 CYAN rectangle
    rect5_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 1.2], physicsClientId=cid)
    rect5_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 1.2], rgbaColor=[0, 1, 1, 1], physicsClientId=cid)
    p.createMultiBody(3.0, rect5_col, rect5_vis, basePosition=[0, 6, 2.4], physicsClientId=cid)

    # 6 MAGENTA rectangle
    rect6_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 1.2], physicsClientId=cid)
    rect6_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 1.2], rgbaColor=[1, 0, 1, 1], physicsClientId=cid)
    p.createMultiBody(3.0, rect6_col, rect6_vis, basePosition=[0, -6, 2.4], physicsClientId=cid)

    # 7 ORANGE rectangle
    rect7_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 1.2], physicsClientId=cid)
    rect7_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 2.2], rgbaColor=[1, 0.5, 0, 1], physicsClientId=cid)
    p.createMultiBody(3.0, rect7_col, rect7_vis, basePosition=[6, 0, 2.4], physicsClientId=cid)

    # 8 PURPLE rectangle
    rect8_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 1.2], physicsClientId=cid)
    rect8_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 1.2], rgbaColor=[0.5, 0, 1, 1], physicsClientId=cid)
    p.createMultiBody(3.0, rect8_col, rect8_vis, basePosition=[-6, 0, 2.4], physicsClientId=cid)

    # Sphere corner markers (diagonal inside map)
    sphere_col = p.createCollisionShape(p.GEOM_SPHERE, radius=1.60, physicsClientId=cid)
    sphere_vis = p.createVisualShape(p.GEOM_SPHERE, radius=1.60, rgbaColor=[0.3, 0.3, 0.3, 2.8], physicsClientId=cid)
    z = 2.8
    p.createMultiBody(1.0, sphere_col, sphere_vis, basePosition=[6.2,  6.2, z], physicsClientId=cid)
    p.createMultiBody(1.0, sphere_col, sphere_vis, basePosition=[-6.2, 6.2, z], physicsClientId=cid)
    p.createMultiBody(1.0, sphere_col, sphere_vis, basePosition=[6.2, -6.2, z], physicsClientId=cid)
    p.createMultiBody(1.0, sphere_col, sphere_vis, basePosition=[-6.2,-6.2, z], physicsClientId=cid)
    p.createMultiBody(1.0, sphere_col, sphere_vis, basePosition=[3,    0,   z], physicsClientId=cid)

    world_body_count = p.getNumBodies(physicsClientId=cid)

    # Only load motion clip in stage 2 — not needed for standing
    motion = MotionClip(motion_path) if STAGE == 2 else None

    env = HumanoidEnv(cid, world_body_count, motion)
    env._load_humanoid()
    return env


# ---------------------------------------------------------------------------
class HumanoidEnv:

    START_POS = [0, 3.5, 1.2]   # raised to avoid floor clipping on spawn
    START_ORN = [0, 0, 0, 1]

    GOAL_POSITIONS = [
        [0,  0.0, 1.0],
        [0, -3.5, 1.0],
        [0, -7.0, 1.0],
    ]

    PHYSICS_HZ = 240
    SUB_STEPS  = 4
    MAX_FORCE  = 150
    VEL_SCALE  = 5.0

    # Stage 2 DeepMimic reward weights — must sum to 1.0
    W_POSE    = 0.65
    W_VEL     = 0.10
    W_END_EFF = 0.15
    W_ALIVE   = 0.10

    def __init__(self, cid, world_body_count, motion):
        self.cid              = cid
        self.world_body_count = world_body_count
        self.motion           = motion   # None in stage 1
        self.humanoid         = None
        self.all_bodies       = []
        self.joint_ids        = []
        self.start_time       = time.time()
        self.prev_pos         = np.array(self.START_POS, dtype=np.float64)
        self.current_goal_idx = 0
        self._step_count      = 0
        self._elapsed         = 0.0

    def _load_humanoid(self):
        for bid in self.all_bodies:
            try:
                p.removeBody(bid, physicsClientId=self.cid)
            except Exception:
                pass

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        humanoid_bodies = p.loadMJCF("mjcf/humanoid.xml",
                                     flags=p.URDF_USE_SELF_COLLISION,
                                     physicsClientId=self.cid)

        self.all_bodies = list(humanoid_bodies)
        self.humanoid   = humanoid_bodies[-1]

        self.joint_ids = []
        for i in range(p.getNumJoints(self.humanoid, physicsClientId=self.cid)):
            info = p.getJointInfo(self.humanoid, i, physicsClientId=self.cid)
            if info[2] != p.JOINT_FIXED:
                self.joint_ids.append(i)

        print(f"[HumanoidEnv] Stage {STAGE} | Controllable joints: {len(self.joint_ids)}")

        for bid in self.all_bodies:
            p.resetBasePositionAndOrientation(bid, self.START_POS, self.START_ORN, physicsClientId=self.cid)
            p.resetBaseVelocity(bid, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)

        for j in self.joint_ids:
            p.resetJointState(self.humanoid, j, targetValue=0.0, targetVelocity=0.0, physicsClientId=self.cid)

        # In stage 2, spawn in a valid walk pose
        if STAGE == 2 and self.motion is not None:
            self._apply_reference_pose(0.0)

        for _ in range(30):
            p.stepSimulation(physicsClientId=self.cid)

        for bid in self.all_bodies:
            p.resetBaseVelocity(bid, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)

    def _apply_reference_pose(self, elapsed: float):
        """Set all joints to match the reference motion at elapsed time. Stage 2 only."""
        if self.motion is None:
            return
        ref = self.motion.get_frame_at_time(elapsed)
        n   = min(len(self.joint_ids), len(ref))
        for i in range(n):
            p.resetJointState(
                self.humanoid,
                self.joint_ids[i],
                targetValue=float(ref[i]),
                targetVelocity=0.0,
                physicsClientId=self.cid
            )

    # --- goal helpers ---------------------------------------------------
    def get_goal_positions(self):
        return self.GOAL_POSITIONS

    def get_current_goal(self):
        return np.array(self.GOAL_POSITIONS[self.current_goal_idx], dtype=np.float64)

    def _update_goal(self, pos):
        goal = self.get_current_goal()
        if (np.linalg.norm(np.array(pos) - goal) < 1.5
                and self.current_goal_idx < len(self.GOAL_POSITIONS) - 1):
            self.current_goal_idx += 1

    # --- distance tracker -----------------------------------------------
    def track_distance(self):
        pos, _ = p.getBasePositionAndOrientation(self.humanoid, physicsClientId=self.cid)
        pos     = np.array(pos)
        delta_y = pos[1] - self.prev_pos[1]
        self.prev_pos = pos
        return -delta_y

    # --- reset ----------------------------------------------------------
    def reset(self):
        pos, _ = p.getBasePositionAndOrientation(self.humanoid, physicsClientId=self.cid)

        # Only lift if completely on the ground — keep X, Y position
        if pos[2] < 0.3:
            p.resetBasePositionAndOrientation(
                self.humanoid,
                [pos[0], pos[1], 1.2],
                self.START_ORN,
                physicsClientId=self.cid
            )

        # Zero all velocity before posing to prevent explosive joint forces
        for bid in self.all_bodies:
            p.resetBaseVelocity(bid, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)
        for j in self.joint_ids:
            p.resetJointState(self.humanoid, j, targetValue=0.0, targetVelocity=0.0, physicsClientId=self.cid)

        # Stage 2: apply reference walk pose at random phase
        if STAGE == 2 and self.motion is not None:
            self._elapsed = np.random.uniform(0.0, self.motion.total_duration)
            self._apply_reference_pose(self._elapsed)

        self.start_time       = time.time()
        self.prev_pos         = np.array(
            p.getBasePositionAndOrientation(self.humanoid, physicsClientId=self.cid)[0],
            dtype=np.float64
        )
        self.current_goal_idx = 0
        self._step_count      = 0

        obs, _ = self.get_obs()
        return obs

    # --- observation ----------------------------------------------------
    def get_obs(self):
        obs = []

        pos, orn = p.getBasePositionAndOrientation(self.humanoid, physicsClientId=self.cid)
        vel, ang = p.getBaseVelocity(self.humanoid, physicsClientId=self.cid)

        pos_np = np.array(pos)
        orn_np = np.array(orn)
        vel_np = np.array(vel)
        ang_np = np.array(ang)

        obs_dict = {
            'position':    {'x': pos[0], 'y': pos[1], 'z': pos[2]},
            'orientation': orn,
            'velocity':    {'x': vel[0], 'y': vel[1], 'z': vel[2]},
        }

        # Goal distances
        for i, goal in enumerate(self.get_goal_positions()):
            dist = abs(pos[1] - goal[1])
            obs_dict[f'goal_{i}_dist'] = dist
            obs.append(dist)

        obs.extend(pos_np)
        obs.extend(vel_np)
        obs.extend(ang_np)
        obs.extend(orn_np)

        # Current joint states
        for j in self.joint_ids:
            state = p.getJointState(self.humanoid, j, physicsClientId=self.cid)
            obs.append(state[0])
            obs.append(state[1])

        # Stage 2: add reference joints and phase to observation
        if STAGE == 2 and self.motion is not None:
            ref   = self.motion.get_frame_at_time(self._elapsed)
            n     = min(len(self.joint_ids), len(ref))
            obs.extend(ref[:n].tolist())
            phase = (self._elapsed % self.motion.total_duration) / self.motion.total_duration
            obs.append(float(phase))

        return np.array(obs, dtype=np.float32), obs_dict

    # --- step -----------------------------------------------------------
    def step(self, action):
        self._step_count += 1

        if STAGE == 2 and self.motion is not None:
            # Advance motion clock
            dt             = self.SUB_STEPS / self.PHYSICS_HZ
            self._elapsed += dt
            ref_joints     = self.motion.get_frame_at_time(self._elapsed)
            n_ref          = min(len(self.joint_ids), len(ref_joints))

        # Apply actions
        for i, joint_idx in enumerate(self.joint_ids):
            current_angle = p.getJointState(self.humanoid, joint_idx, physicsClientId=self.cid)[0]

            if STAGE == 2 and self.motion is not None and i < n_ref:
                # Stage 2: PD tracking toward reference + policy correction
                angle_error = float(ref_joints[i]) - current_angle
                target_vel  = angle_error * 5.0 + float(action[i]) * self.VEL_SCALE
            else:
                # Stage 1: pure velocity control, policy learns from scratch
                target_vel = float(action[i]) * self.VEL_SCALE

            p.setJointMotorControl2(
                bodyUniqueId   = self.humanoid,
                jointIndex     = joint_idx,
                controlMode    = p.VELOCITY_CONTROL,
                targetVelocity = target_vel,
                force          = self.MAX_FORCE,
                physicsClientId= self.cid
            )

        for _ in range(self.SUB_STEPS):
            p.stepSimulation(physicsClientId=self.cid)

        pos, orn     = p.getBasePositionAndOrientation(self.humanoid, physicsClientId=self.cid)
        vel, ang_vel = p.getBaseVelocity(self.humanoid, physicsClientId=self.cid)

        self._update_goal(pos)

        # Upright orientation from quaternion — used in both stages
        rot_matrix  = p.getMatrixFromQuaternion(orn, physicsClientId=self.cid)
        upright_dot = float(rot_matrix[8])   # 1.0 = upright, -1.0 = upside down
        height_fraction  = min(pos[2] / 1.4, 1.0)
        upright_fraction = max(upright_dot, 0.0)

        # ----------------------------------------------------------------
        # STAGE 1 REWARD — just learn to stand upright
        # Run until ep_rew_mean consistently above 5.0, then switch STAGE=2
        # ----------------------------------------------------------------
        if STAGE == 1:
            # Standing reward — tall AND vertical
            standing_reward = height_fraction * upright_fraction * 5.0

            # Getting up reward — reward any upward movement when low
            prev_height  = self.prev_pos[2]
            height_delta = pos[2] - prev_height
            getting_up   = max(height_delta, 0.0) * 20.0 if pos[2] < 1.0 else 0.0

            # Limb activity when on the ground — encourages pushing up
            joint_vels = np.array([
                p.getJointState(self.humanoid, j, physicsClientId=self.cid)[1]
                for j in self.joint_ids
            ])
            limb_activity = float(np.mean(np.abs(joint_vels))) * 1.0 if pos[2] < 0.8 else 0.0

            # Flat on back penalty — scales with how horizontal the torso is
            flat_penalty = -3.0 * (1.0 - upright_fraction)

            # Fall penalty
            fall_penalty = -5.0 if pos[2] < 0.4 else 0.0

            # Spin penalty — discourage spinning in place
            spin_penalty = -0.3 * float(np.linalg.norm(ang_vel))

            reward = (
                standing_reward
                + getting_up
                + limb_activity
                + flat_penalty
                + fall_penalty
                + spin_penalty
            )

            # End episode only on complete collapse
            done = bool(pos[2] < 0.3)

        # ----------------------------------------------------------------
        # STAGE 2 REWARD — DeepMimic motion imitation (Peng et al. 2018)
        # Switch to this once the robot can reliably stand (ep_rew_mean > 5.0)
        # ----------------------------------------------------------------
        else:
            # 1. POSE REWARD — exponential kernel over joint angle error
            current_joints = np.array([
                p.getJointState(self.humanoid, j, physicsClientId=self.cid)[0]
                for j in self.joint_ids
            ])
            pose_error = np.sum(np.square(current_joints[:n_ref] - ref_joints[:n_ref]))
            r_pose     = float(np.exp(-2.0 * pose_error))

            # 2. FORWARD VELOCITY REWARD — normalised to [0, 1]
            forward_vel = float(-vel[1])
            r_vel       = min(max(forward_vel, 0.0), 2.0) / 2.0

            # 3. END EFFECTOR REWARD — feet height matches walk cycle phase
            phase    = (self._elapsed % self.motion.total_duration) / self.motion.total_duration
            right_up = np.sin(2 * np.pi * phase)
            left_up  = np.sin(2 * np.pi * phase + np.pi)

            right_foot_idx = min(5,  len(self.joint_ids) - 1)
            left_foot_idx  = min(10, len(self.joint_ids) - 1)

            right_state = p.getLinkState(self.humanoid, self.joint_ids[right_foot_idx], physicsClientId=self.cid)
            left_state  = p.getLinkState(self.humanoid, self.joint_ids[left_foot_idx],  physicsClientId=self.cid)
            right_h     = float(right_state[0][2])
            left_h      = float(left_state[0][2])

            r_end = 0.5 * (
                np.exp(-10.0 * (right_h - 0.1 * max(right_up, 0.0)) ** 2) +
                np.exp(-10.0 * (left_h  - 0.1 * max(left_up,  0.0)) ** 2)
            )

            # 4. ALIVE REWARD — upright and above fall threshold
            r_alive = 1.0 if (pos[2] > 0.8 and upright_dot > 0.5) else 0.0

            reward = (
                self.W_POSE    * r_pose  +
                self.W_VEL     * r_vel   +
                self.W_END_EFF * r_end   +
                self.W_ALIVE   * r_alive
            )

            done = bool(pos[2] < 0.5 or upright_dot < 0.0)

        obs, _ = self.get_obs()
        return obs, reward, done