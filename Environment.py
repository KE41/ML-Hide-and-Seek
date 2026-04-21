# Environment Setup and Creation - Pybullet

import pybullet as p
import pybullet_data
import numpy as np
import time

"""Environment Creation"""


def create_environment(gui):  # change to false for DIRECT

    try:
        p.disconnect()
    except Exception:
        pass  # No existing connection, that's fine

    cid = p.connect(p.GUI if gui else p.DIRECT)

    if cid < 0:
        raise RuntimeError("Failed to connect to PyBullet")

    # Disabled mouse movement of objects / robot
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)

    # Disables UI
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    # Pybullet sim settings
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)

    # Fps fix 1
    p.setTimeStep(1. / 400.)  # Remove for DIRECT MODE

    # Base Map - Superflat minecraft
    #p.loadURDF("plane.urdf")

    # Camera
    if gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=11,
            cameraYaw=270,
            cameraPitch=-40,
            cameraTargetPosition=[0, 0, 0]
        )

    # Map -----------------
    # (x,y,z) - max double

    # Collision Shapes (walls + floor only doubled)
    wall = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 10, 2])
    topbottom_wall = p.createCollisionShape(p.GEOM_BOX, halfExtents=[10, 0.4, 2])
    ##floor = p.createCollisionShape(p.GEOM_BOX, halfExtents=[10, 10, 2])

    # Visual shapes
    wall_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 10, 2], rgbaColor=[1, 0, 0, 1])
    topbottom_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[10, 0.4, 2], rgbaColor=[0, 0, 1, 1])

    # Floor
    #floor_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[10, 10, 0.1], rgbaColor=[0.5, 0.5, 0.5, 1])
    #floor_id = p.createMultiBody(0, floor, floor_vis, basePosition=[0, 0, 0.1])
    #p.changeVisualShape(floor_id, -1, rgbaColor=[0.5, 0.5, 0.5, 1]) # change floor to gray

    # Make Bodies

    # Walls (2x map size)
    p.createMultiBody(0, wall, wall_vis, basePosition=[10, 0, 2])
    p.createMultiBody(0, wall, wall_vis, basePosition=[-10, 0, 2])
    p.createMultiBody(0, topbottom_wall, topbottom_vis, basePosition=[0, 10, 2])
    p.createMultiBody(0, topbottom_wall, topbottom_vis, basePosition=[0, -10, 2])

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

    # 6 MAGENTA rectangle (unchanged)
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

    #offset -2.2
    #z = 1.4

    z = 2.8

    p.createMultiBody(1.0, sphere_col, sphere_vis, basePosition=[6.2, 6.2, z])
    p.createMultiBody(1.0, sphere_col, sphere_vis, basePosition=[-6.2, 6.2, z])
    p.createMultiBody(1.0, sphere_col, sphere_vis, basePosition=[6.2, -6.2, z])
    p.createMultiBody(1.0, sphere_col, sphere_vis, basePosition=[-6.2, -6.2, z])
    p.createMultiBody(1.0, sphere_col, sphere_vis, basePosition=[3, 0, z])

    # Humanoid Agent (MJCF - HumanoidBulletEnv style)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    humanoid_bodies = p.loadMJCF("mjcf/humanoid.xml", flags=p.URDF_USE_SELF_COLLISION) #apparently two bodies are present
    print("All MJCF body IDs:", humanoid_bodies)  # print ALL ids before taking [0]
    print("Num bodies loaded:", len(humanoid_bodies))

    root_id = humanoid_bodies[0]
    humanoid_id = humanoid_bodies[-1]
    pos, _ = p.getBasePositionAndOrientation(humanoid_id)
    #print("Body [0] initial position:", pos)  # is this actually the torso?

    p.resetBasePositionAndOrientation(humanoid_id, [0, 3.5, 2.2], [0, 0, 0, 1])
    p.resetBaseVelocity(humanoid_id, [0, 0, 0], [0, 0, 0])

    return HumanoidEnv(humanoid_id, cid, root_id)

# Humanoid Agent Reinforcement Learning

class HumanoidEnv:
    START_POS = [0, 3.5, 1.2] # 0, 3.5 -2.5 working
    START_ORN = [0, 0, 0, 1]

    def __init__(self, humanoid_id, cid, root_id):
        self.humanoid = humanoid_id
        self.root_id = root_id
        self.cid = cid
        self.joint_ids = []

        for i in range(p.getNumJoints(self.humanoid)):
            info = p.getJointInfo(self.humanoid, i)

            if info[2] != p.JOINT_FIXED:
                self.joint_ids.append(i)

        print("joints found:", len(self.joint_ids))

    def reset(self):
        p.resetBasePositionAndOrientation(
            self.root_id, self.START_POS, self.START_ORN
        )
        p.resetBaseVelocity(self.humanoid, [0, 0, 0], [0, 0, 0])

        for j in self.joint_ids:
            p.resetJointState(self.humanoid, j, targetValue=0, targetVelocity=0)

        return self.get_obs()

    def get_obs(self):
        obs = []

        pos, orn = p.getBasePositionAndOrientation(self.humanoid)
        vel, ang = p.getBaseVelocity(self.humanoid)

        obs_dict = {
            'position': {'x': pos[0], 'y': pos[1], 'z': pos[2]},
            'orientation': orn,
            'velocity': {'x': vel[0], 'y': vel[1], 'z': vel[2]}
        }

        # Goal directions to aid movement
        #goal_positions = self.get_goal_positions()  # Returns list of target positions

       # for i, goal in enumerate(goal_positions):
         #   dist_to_goal = np.linalg.norm(
           #     obs_dict['position']['y'] - goal[1],
             #   p=np.inf  # Only consider Y-axis distance
            #)

            obs_dict[f'goal_{i}_dist'] = dist_to_goal

        # Convert to numpy arrays for consistency
        pos = np.array(pos)
        orn = np.array(orn)
        vel = np.array(vel)
        ang = np.array(ang)

        obs.extend(pos)
        obs.extend(vel)
        obs.extend(ang)
        obs.extend(orn)

        # Time calculation
        current_time = time.time()
        episode_duration = current_time - self.start_time
        avg_velocity = np.linalg.norm(vel) / max(episode_duration, 1e-6)

        obs_dict['time'] = {'duration': episode_duration, 'avg_vel': avg_velocity}

        for j in self.joint_ids:
            state = p.getJointState(self.humanoid, j)
            obs.append(state[0])
            obs.append(state[1])

        return np.array(obs, dtype=np.float32), obs_dict

    def step(self, action):
        """
        Executes one step of the environment using the provided action.
        'action' should be a vector corresponding to the joint controls.
        """
        # --- 1. APPLY ACTION (Joint Control) ---
        # We move away from teleporting and instead apply torques/velocities
        # to the humanoid joints based on the neural network output.
        for i, joint_idx in enumerate(self.joint_ids):
            p.setJointMotorControl2(
                bodyUniqueId=self.humanoid,
                jointIndex=joint_idx,
                controlMode=p.VELOCITY_CONTROL,  # Using velocity for more stable training
                targetVelocity=action[i],  # Action value mapped to speed
                force=50  # Strength of the motor
            )

        p.stepSimulation()

        # --- 3. RETRIEVE PHYSICAL STATE ---
        # Get position and velocity directly from the physics engine
        pos, orn = p.getBasePositionAndOrientation(self.humanoid)
        vel, ang_vel = p.getBaseVelocity(self.humanoid)

        #print(f"Pos: {pos}")



        # A. Forward Progress Reward (The primary goal: Move along Y axis)
        forward_velocity = vel[1] * 1.0
        forward_reward = forward_velocity * 2
        distance_travelled = self.track_distance(action)
        forward_reward += distance_travelled * 0.5

        # B. Height/Falling Penalty (Penalize if the torso drops too low)
        height_penalty = 0.0

        if pos[2] < 1.2:
            low_height_val = max(0, 1.2 - pos[2]) * 5.0
            fall_speed_penalty = max(0, -vel[2]) * 2.0
            height_penalty = low_height_val + fall_speed_penalty

        # We penalize the robot for moving away from X = 0
        lateral_penalty = abs(pos[0]) * 0.3

        momentum_bonus = -abs(vel[0]) * 0.1

        goal_rewards = []
        # --- B. GOAL PROXIMITY REWARD (Long-term objective) ---
        goal_rewards = []

        2


        # D. Momentum/Stability Bonus (A negative penalty to reduce wobbling)
        # Penalize excessive side-to-side velocity on the X axis
        stability_penalty = abs(vel[0]) * 0.5

        # FINAL REWARD AGGREGATION
        # Structure: Progress - Height_Penalty - Lateral_Penalty - Stability_Penalty
        reward = forward_reward - height_penalty - lateral_penalty - stability_penalty + momentum_bonus

        # --- 5. TERMINATION LOGIC ---
        # Terminate episode if the robot falls below a certain height threshold
        done = pos[2] < 0.5

        # --- 6. OBSERVATION ---
        obs = self.get_obs()

        return obs, reward, done