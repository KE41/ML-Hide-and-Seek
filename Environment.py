# Environment Setup and Creation - Pybullet

import pybullet as p
import pybullet_data
import numpy as np
import time

"""Environment Creation"""

# Store world body IDs so reset() can remove only the humanoid, not the map
_WORLD_BODY_COUNT = None  # set after map is built, before humanoid is loaded


def create_environment(gui):

    # FIX: removed p.disconnect() — each env gets its own client ID via
    # p.connect(), so disconnecting here would kill other parallel envs.
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
            cameraDistance=11,
            cameraYaw=270,
            cameraPitch=-40,
            cameraTargetPosition=[0, 0, 0],
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

    # Symmetrical fixed shapes (original 8-point circle)

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

    # Record how many bodies exist BEFORE loading the humanoid.
    # reset() will remove every body with ID >= this count and reload fresh.
    world_body_count = p.getNumBodies(physicsClientId=cid)

    env = HumanoidEnv(cid, world_body_count)
    env._load_humanoid()   # load humanoid for the first time
    return env


# ---------------------------------------------------------------------------
class HumanoidEnv:

    START_POS = [0, 3.5, 0.5]  # torso centre height when humanoid stands on plane.urdf
    START_ORN = [0, 0, 0, 1]

    GOAL_POSITIONS = [
        [0,  0.0, 1.0],
        [0, -3.5, 1.0],
        [0, -7.0, 1.0],
    ]

    # Physics / action scaling
    PHYSICS_HZ  = 240
    SUB_STEPS   = 4
    MAX_FORCE   = 80
    VEL_SCALE   = 1.0

    def __init__(self, cid, world_body_count):
        self.cid              = cid
        self.world_body_count = world_body_count
        self.humanoid         = None
        self.all_bodies       = []
        self.joint_ids        = []
        self.start_time       = time.time()
        self.prev_pos         = np.array(self.START_POS, dtype=np.float64)
        self.current_goal_idx = 0
        self._step_count      = 0   # tracks steps for action penalty

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
        print("Loaded MJCF body IDs:", humanoid_bodies)

        self.all_bodies = list(humanoid_bodies)
        self.humanoid   = humanoid_bodies[-1]

        self.joint_ids = []
        for i in range(p.getNumJoints(self.humanoid, physicsClientId=self.cid)):
            info = p.getJointInfo(self.humanoid, i, physicsClientId=self.cid)
            if info[2] != p.JOINT_FIXED:
                self.joint_ids.append(i)

        print(f"Controllable joints: {len(self.joint_ids)}")

        for bid in self.all_bodies:
            p.resetBasePositionAndOrientation(bid, self.START_POS, self.START_ORN, physicsClientId=self.cid)
            p.resetBaseVelocity(bid, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)

        for j in self.joint_ids:
            p.resetJointState(self.humanoid, j, targetValue=0.0, targetVelocity=0.0, physicsClientId=self.cid)

        for j in self.joint_ids:
            p.setJointMotorControl2(
                self.humanoid, j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.0,
                force=500,
                physicsClientId=self.cid
            )
        for _ in range(120):
            p.stepSimulation(physicsClientId=self.cid)

        for bid in self.all_bodies:
            p.resetBaseVelocity(bid, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)

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
        # Just reposition the existing humanoid instead of reloading from disk
        for bid in self.all_bodies:
            p.resetBasePositionAndOrientation(bid, self.START_POS, self.START_ORN, physicsClientId=self.cid)
            p.resetBaseVelocity(bid, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)

        for j in self.joint_ids:
            p.resetJointState(self.humanoid, j, targetValue=0.0, targetVelocity=0.0, physicsClientId=self.cid)

        # Fewer warmup steps — just enough to settle
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.cid)

        for bid in self.all_bodies:
            p.resetBaseVelocity(bid, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)

        self.start_time = time.time()
        self.prev_pos = np.array(self.START_POS, dtype=np.float64)
        self.current_goal_idx = 0
        self._step_count = 0
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

        for i, goal in enumerate(self.get_goal_positions()):
            dist = abs(pos[1] - goal[1])
            obs_dict[f'goal_{i}_dist'] = dist
            obs.append(dist)

        obs.extend(pos_np)
        obs.extend(vel_np)
        obs.extend(ang_np)
        obs.extend(orn_np)

        episode_duration = time.time() - self.start_time
        avg_velocity     = np.linalg.norm(vel_np) / max(episode_duration, 1e-6)
        obs_dict['time'] = {'duration': episode_duration, 'avg_vel': avg_velocity}

        for j in self.joint_ids:
            state = p.getJointState(self.humanoid, j, physicsClientId=self.cid)
            obs.append(state[0])
            obs.append(state[1])

        return np.array(obs, dtype=np.float32), obs_dict

    # --- step -----------------------------------------------------------
    def step(self, action):
        self._step_count += 1

        for i, joint_idx in enumerate(self.joint_ids):
            target_vel = float(action[i]) * self.VEL_SCALE
            p.setJointMotorControl2(
                bodyUniqueId=self.humanoid,
                jointIndex=joint_idx,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=target_vel,
                force=self.MAX_FORCE,
                physicsClientId=self.cid
            )

        for _ in range(self.SUB_STEPS):
            p.stepSimulation(physicsClientId=self.cid)

        pos, orn = p.getBasePositionAndOrientation(self.humanoid, physicsClientId=self.cid)
        vel, ang_vel = p.getBaseVelocity(self.humanoid, physicsClientId=self.cid)

        self._update_goal(pos)

        # ----------------------------------------------------------------
        # REWARD FUNCTION
        # Based on:
        #   - PyBullet's own HumanoidBulletEnv (bullet3 gym_locomotion_envs.py)
        #   - HuMam paper: 6-term reward for stability + energy efficiency
        #   - Benchmarking PBRS paper: robust scaling via potential-based terms
        # ----------------------------------------------------------------

        # 1. ALIVE BONUS
        # Flat bonus every step the humanoid stays upright above fall threshold.
        # PyBullet's own humanoid env uses +1.0/step — we use +2.0 to
        # strongly prioritise survival over everything else early in training.
        # This is the single most important signal: just stay alive.
        alive_bonus = 2.0 if pos[2] > 0.4 else 0.0

        # 2. UPRIGHT REWARD
        # Height-proportional reward so partial uprightness is better than
        # fully flat. Standing humanoid torso ~1.4 m → reward up to +1.5/step.
        # Capped at 1.0 so it can't exceed alive_bonus in magnitude.
        upright_fraction = min(pos[2] / 1.4, 1.0)
        upright_reward   = upright_fraction * 1.5

        # 3. FORWARD VELOCITY REWARD
        # Reward forward velocity (-Y direction) scaled by how upright the
        # humanoid is — crawling/sliding on its face gives almost nothing.
        # Coefficient 1.5 from PyBullet humanoid env tuning: enough to
        # motivate walking without overshadowing the alive bonus.
        forward_vel_reward = -vel[1] * 1.5 * upright_fraction

        # 4. DISTANCE PROGRESS REWARD
        # Incremental -Y displacement per step, also upright-gated.
        # Coefficient 0.5: smaller than velocity reward so it supplements
        # rather than dominates (avoids reward hacking via single big lunge).
        forward_dist_reward = self.track_distance() * 0.5 * upright_fraction

        # 5. ENERGY / ELECTRICITY COST
        # Penalise large actions to encourage efficient, smooth movement.
        # PyBullet uses -2.0 * |torque * velocity|; we approximate with
        # -0.005 * sum(action^2) which is gentler but still discourages
        # thrashing. Too large and the agent learns to do nothing.
        electricity_cost = -0.005 * float(np.sum(np.square(action)))

        # 6. STALL TORQUE COST
        # Small penalty for applying force while joints are near-stationary
        # (wastes energy). PyBullet uses -0.1; we match that value.
        joint_vels = np.array([
            p.getJointState(self.humanoid, j, physicsClientId=self.cid)[1] for j in self.joint_ids
        ])
        stall_cost = -0.1 * float(np.sum(np.square(action) * (np.abs(joint_vels) < 0.1)))

        # 7. JOINTS AT LIMIT COST
        # Discourage joints being pinned at their mechanical limits (causes
        # jerky/frozen-limb behaviour). PyBullet uses -0.1 per stuck joint.
        joint_angles = np.array([
            p.getJointState(self.humanoid, j, physicsClientId=self.cid)[0] for j in self.joint_ids
        ])
        joint_info   = [p.getJointInfo(self.humanoid, j, physicsClientId=self.cid) for j in self.joint_ids]
        at_limit     = sum(
            1 for k, info in enumerate(joint_info)
            if abs(joint_angles[k]) > 0.99 * max(abs(info[8]), abs(info[9]), 1e-3)
        )
        joints_at_limit_cost = -0.1 * at_limit

        # 8. SPIN / ANGULAR VELOCITY PENALTY
        # Prevents the torso spinning wildly to farm forward velocity.
        # 0.05 coefficient: light enough not to block turning, heavy enough
        # to stop uncontrolled rotation.
        spin_penalty = -0.05 * float(np.linalg.norm(ang_vel))

        # 9. LATERAL DRIFT PENALTY
        # Keep the humanoid near X=0 (the centreline of the map).
        # 0.1 coefficient: gentle nudge, not a hard wall.
        lateral_penalty = -0.1 * abs(pos[0])

        # 10. GOAL PROXIMITY BONUS
        # Small bonus for closing in on the current waypoint, upright-gated.
        dist_to_goal = np.linalg.norm(np.array(pos) - self.get_current_goal())
        goal_reward  = max(0.0, 5.0 - dist_to_goal) * 0.05 * upright_fraction

        # ---- Final reward ----
        reward = (
            alive_bonus           # +2.0  stay alive — dominant signal
            + upright_reward      # +1.5  be tall
            + forward_vel_reward  # +var  move forward while upright
            + forward_dist_reward # +var  actual displacement
            + goal_reward         # +var  head toward waypoint
            + electricity_cost    # -var  don't thrash joints
            + stall_cost          # -var  don't stall joints
            + joints_at_limit_cost# -var  don't pin joints
            + spin_penalty        # -var  don't spin
            + lateral_penalty     # -var  stay centred
        )

        # Episode ends when torso drops below 0.3 m (humanoid has fallen)
        done = bool(pos[2] < 0.3)

        obs, _ = self.get_obs()
        return obs, reward, done