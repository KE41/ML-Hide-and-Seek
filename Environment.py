# Environment Setup and Creation - Pybullet

import pybullet as p
import pybullet_data
import numpy as np
import time

"""Environment Creation"""

# Store world body IDs so reset() can remove only the humanoid, not the map
_WORLD_BODY_COUNT = None  # set after map is built, before humanoid is loaded


def create_environment(gui):

    try:
        p.disconnect()
    except Exception:
        pass

    cid = p.connect(p.GUI if gui else p.DIRECT)
    if cid < 0:
        raise RuntimeError("Failed to connect to PyBullet")

    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1. / 240.)

    if gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=11,
            cameraYaw=270,
            cameraPitch=-40,
            cameraTargetPosition=[0, 0, 0]
        )

    # ---- Floor ----
    p.loadURDF("plane.urdf")

    # ---- Walls ----
    wall          = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 10, 2])
    tb_wall       = p.createCollisionShape(p.GEOM_BOX, halfExtents=[10, 0.4, 2])
    wall_vis      = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 10, 2],  rgbaColor=[1,0,0,1])
    tb_vis        = p.createVisualShape(p.GEOM_BOX, halfExtents=[10, 0.4, 2], rgbaColor=[0,0,1,1])

    p.createMultiBody(0, wall,    wall_vis, basePosition=[ 10,  0, 2])
    p.createMultiBody(0, wall,    wall_vis, basePosition=[-10,  0, 2])
    p.createMultiBody(0, tb_wall, tb_vis,   basePosition=[  0, 10, 2])
    p.createMultiBody(0, tb_wall, tb_vis,   basePosition=[  0,-10, 2])

    # Symmetrical fixed shapes (original 8-point circle)

    # 1 RED cube
    cube1_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.6])
    cube1_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.6], rgbaColor=[1, 0, 0, 1])
    p.createMultiBody(2.0, cube1_col, cube1_vis, basePosition=[4, 4, 1.8])

    # 2 BLUE cube
    cube2_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.6])
    cube2_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.6], rgbaColor=[0, 0, 1, 1])
    p.createMultiBody(2.0, cube2_col, cube2_vis, basePosition=[-4, -4, 1.8])

    # 3 GREEN cube
    cube3_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.6])
    cube3_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.6], rgbaColor=[0, 1, 0, 1])
    p.createMultiBody(2.0, cube3_col, cube3_vis, basePosition=[4, -4, 1.8])

    # 4 YELLOW cube
    cube4_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.6])
    cube4_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.6], rgbaColor=[1, 1, 0, 1])
    p.createMultiBody(2.0, cube4_col, cube4_vis, basePosition=[-4, 4, 1.8])

    # 5 CYAN rectangle
    rect5_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 1.2])
    rect5_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 1.2], rgbaColor=[0, 1, 1, 1])
    p.createMultiBody(3.0, rect5_col, rect5_vis, basePosition=[0, 6, 2.4])

    # 6 MAGENTA rectangle
    rect6_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 1.2])
    rect6_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 1.2], rgbaColor=[1, 0, 1, 1])
    p.createMultiBody(3.0, rect6_col, rect6_vis, basePosition=[0, -6, 2.4])

    # 7 ORANGE rectangle
    rect7_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 1.2])
    rect7_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 2.2], rgbaColor=[1, 0.5, 0, 1])
    p.createMultiBody(3.0, rect7_col, rect7_vis, basePosition=[6, 0, 2.4])

    # 8 PURPLE rectangle
    rect8_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 1.2])
    rect8_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 1.2], rgbaColor=[0.5, 0, 1, 1])
    p.createMultiBody(3.0, rect8_col, rect8_vis, basePosition=[-6, 0, 2.4])

    # Sphere corner markers (diagonal inside map)
    sphere_col = p.createCollisionShape(p.GEOM_SPHERE, radius=1.60)
    sphere_vis = p.createVisualShape(p.GEOM_SPHERE, radius=1.60, rgbaColor=[0.3, 0.3, 0.3, 2.8])

    z = 2.8
    p.createMultiBody(1.0, sphere_col, sphere_vis, basePosition=[6.2,  6.2, z])
    p.createMultiBody(1.0, sphere_col, sphere_vis, basePosition=[-6.2, 6.2, z])
    p.createMultiBody(1.0, sphere_col, sphere_vis, basePosition=[6.2, -6.2, z])
    p.createMultiBody(1.0, sphere_col, sphere_vis, basePosition=[-6.2,-6.2, z])
    p.createMultiBody(1.0, sphere_col, sphere_vis, basePosition=[3,    0,   z])

    # Record how many bodies exist BEFORE loading the humanoid.
    # reset() will remove every body with ID >= this count and reload fresh.
    world_body_count = p.getNumBodies()

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
    MAX_FORCE   = 80      # enough torque to stand without launching the body
    # Actions from the policy are in [-1,1]; multiply by this to get target rad/s.
    # Lower = slower, more natural movement.
    VEL_SCALE   = 1.0     # actions [-1,1] -> max 1 rad/s — controlled joint movement

    def __init__(self, cid, world_body_count):
        self.cid              = cid
        self.world_body_count = world_body_count  # body IDs below this are map bodies
        self.humanoid         = None
        self.all_bodies       = []
        self.joint_ids        = []
        self.start_time       = time.time()
        self.prev_pos         = np.array(self.START_POS, dtype=np.float64)
        self.current_goal_idx = 0

    # ------------------------------------------------------------------
    # Load (or reload) the humanoid MJCF from scratch.
    # This is the only reliable way to fully reset a multi-body MJCF in
    # PyBullet — resetBasePositionAndOrientation on MJCF bodies is broken
    # because internal constraints between sub-bodies are not re-anchored,
    # so the root body drags the torso back to its old position on the
    # very first stepSimulation() call after the teleport.
    # ------------------------------------------------------------------
    def _load_humanoid(self):
        # Remove any previously loaded humanoid bodies
        for bid in self.all_bodies:
            try:
                p.removeBody(bid)
            except Exception:
                pass

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        humanoid_bodies = p.loadMJCF("mjcf/humanoid.xml",
                                     flags=p.URDF_USE_SELF_COLLISION)
        print("Loaded MJCF body IDs:", humanoid_bodies)

        self.all_bodies = list(humanoid_bodies)
        self.humanoid   = humanoid_bodies[-1]  # torso

        # Collect controllable joint IDs first (needed before settling)
        self.joint_ids = []
        for i in range(p.getNumJoints(self.humanoid)):
            info = p.getJointInfo(self.humanoid, i)
            if info[2] != p.JOINT_FIXED:
                self.joint_ids.append(i)

        print(f"Controllable joints: {len(self.joint_ids)}")

        # Place every sub-body at the start position
        for bid in self.all_bodies:
            p.resetBasePositionAndOrientation(bid, self.START_POS, self.START_ORN)
            p.resetBaseVelocity(bid, [0, 0, 0], [0, 0, 0])

        # Zero all joints so the humanoid starts in a neutral T-pose
        for j in self.joint_ids:
            p.resetJointState(self.humanoid, j, targetValue=0.0, targetVelocity=0.0)

        # Hold joints at zero and let physics settle for a moment so the
        # internal MJCF constraints resolve before we hand control to the policy.
        # Without this the constraint solver applies a large impulse on frame 1
        # that launches the humanoid into the air.
        for j in self.joint_ids:
            p.setJointMotorControl2(
                self.humanoid, j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.0,
                force=500   # high force just for settling, not used during training
            )
        for _ in range(120):   # 120 steps at 240 Hz = 0.5 seconds to settle
            p.stepSimulation()

        # Now zero velocity again after settling
        for bid in self.all_bodies:
            p.resetBaseVelocity(bid, [0, 0, 0], [0, 0, 0])

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
        pos, _ = p.getBasePositionAndOrientation(self.humanoid)
        pos     = np.array(pos)
        delta_y = pos[1] - self.prev_pos[1]
        self.prev_pos = pos
        return -delta_y  # positive = moved in -Y (forward)

    # --- reset ----------------------------------------------------------
    def reset(self):
        # Reload the MJCF completely — this is the correct PyBullet reset for
        # multi-body MJCF files.  resetBasePositionAndOrientation is unreliable
        # because it does not re-anchor the internal root-to-torso constraint,
        # causing the torso to snap back to the root's fallen position on the
        # first stepSimulation() after the teleport.
        self._load_humanoid()

        self.start_time       = time.time()
        self.prev_pos         = np.array(self.START_POS, dtype=np.float64)
        self.current_goal_idx = 0

        obs, _ = self.get_obs()
        return obs

    # --- observation ----------------------------------------------------
    def get_obs(self):
        obs = []

        pos, orn = p.getBasePositionAndOrientation(self.humanoid)
        vel, ang = p.getBaseVelocity(self.humanoid)

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
            state = p.getJointState(self.humanoid, j)
            obs.append(state[0])  # joint angle
            obs.append(state[1])  # joint velocity

        return np.array(obs, dtype=np.float32), obs_dict

    # --- step -----------------------------------------------------------
    def step(self, action):
        for i, joint_idx in enumerate(self.joint_ids):
            # Scale action [-1,1] → target velocity in rad/s.
            # Scale action to target velocity in rad/s
            target_vel = float(action[i]) * self.VEL_SCALE
            p.setJointMotorControl2(
                bodyUniqueId=self.humanoid,
                jointIndex=joint_idx,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=target_vel,
                force=self.MAX_FORCE
            )

        for _ in range(self.SUB_STEPS):
            p.stepSimulation()

        pos, orn = p.getBasePositionAndOrientation(self.humanoid)
        vel, _   = p.getBaseVelocity(self.humanoid)

        self._update_goal(pos)

        # --- UPRIGHT REWARD (most important — teaches standing before walking) ---
        # The humanoid torso is ~1.4m tall when standing. Reward height directly
        # so the agent learns that being tall is always better than lying flat.
        # Clipped to [0, 1] so a standing robot gets +3.0 every step.
        upright_reward = min(pos[2] / 1.4, 1.0) * 3.0

        # --- FORWARD PROGRESS (secondary — only useful once it can stand) ---
        # Reward moving in -Y direction, but scale by how upright it is so that
        # sliding on its face doesn't give the same reward as walking upright.
        upright_fraction = min(pos[2] / 1.4, 1.0)          # 0 when flat, 1 when standing
        forward_reward   = -vel[1] * 2.0 * upright_fraction
        forward_reward  += self.track_distance() * 0.5 * upright_fraction

        # --- STABILITY (penalise spinning / falling sideways) ---
        # Angular velocity penalty keeps the torso from spinning wildly
        _, ang_vel = p.getBaseVelocity(self.humanoid)
        spin_penalty     = np.linalg.norm(ang_vel) * 0.1
        lateral_penalty  = abs(pos[0]) * 0.2   # stay near X=0

        # --- GOAL PROXIMITY ---
        dist_to_goal = np.linalg.norm(np.array(pos) - self.get_current_goal())
        goal_reward  = max(0.0, 5.0 - dist_to_goal) * 0.1 * upright_fraction

        reward = (
            upright_reward      # stand up — always rewarded
            + forward_reward    # walk forward — only rewarded when upright
            + goal_reward
            - spin_penalty
            - lateral_penalty
        )

        done = bool(pos[2] < 0.3)

        obs, _ = self.get_obs()
        return obs, reward, done