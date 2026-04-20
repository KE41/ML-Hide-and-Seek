# Environment Setup and Creation - Pybullet

import pybullet as p
import pybullet_data
import numpy as np

"""Environment Creation"""


def create_environment(gui=True):  # change to false for DIRECT

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

    # Fps fix 1
    p.setTimeStep(1. / 200.)

    # Pybullet sim settings
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)

    # Base Map - Superflat minecraft
    p.loadURDF("plane.urdf")

    # Camera
    if gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=9,
            cameraYaw=50,
            cameraPitch=-35,
            cameraTargetPosition=[0, 0, 0]
        )

    # Map -----------------
    # (x,y,z) - max double

    # Collision Shapes (walls + floor only doubled)
    wall = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 10, 2])
    topbottom_wall = p.createCollisionShape(p.GEOM_BOX, halfExtents=[10, 0.4, 2])
    floor = p.createCollisionShape(p.GEOM_BOX, halfExtents=[10, 10, 0.2])
    box = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])

    # Visual shapes
    wall_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 10, 2], rgbaColor=[1, 0, 0, 1])
    topbottom_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[10, 0.4, 2], rgbaColor=[0, 0, 1, 1])
    floor_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[10, 10, 0.2], rgbaColor=[0.5, 0.5, 0.5, 1])

    # Make Bodies

    # Walls (2x map size)
    p.createMultiBody(0, wall, wall_vis, basePosition=[10, 0, 2])
    p.createMultiBody(0, wall, wall_vis, basePosition=[-10, 0, 2])
    p.createMultiBody(0, topbottom_wall, topbottom_vis, basePosition=[0, 10, 2])
    p.createMultiBody(0, topbottom_wall, topbottom_vis, basePosition=[0, -10, 2])

    # Floor
    p.createMultiBody(0, floor, floor_vis, basePosition=[0, 0, 0])

    # Symmetrical fixed shapes (original 8-point circle)

    # 1 RED cube
    cube1_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.6])
    cube1_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.6], rgbaColor=[1, 0, 0, 1])
    p.createMultiBody(2.0, cube1_col, cube1_vis, basePosition=[4, 4, 0.8])

    # 2 BLUE cube
    cube2_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.6])
    cube2_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.6], rgbaColor=[0, 0, 1, 1])
    p.createMultiBody(2.0, cube2_col, cube2_vis, basePosition=[-4, -4, 0.8])

    # 3 GREEN cube
    cube3_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.6])
    cube3_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.6], rgbaColor=[0, 1, 0, 1])
    p.createMultiBody(2.0, cube3_col, cube3_vis, basePosition=[4, -4, 0.8])

    # 4 YELLOW cube
    cube4_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.6])
    cube4_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.6], rgbaColor=[1, 1, 0, 1])
    p.createMultiBody(2.0, cube4_col, cube4_vis, basePosition=[-4, 4, 0.8])

    # 5 CYAN rectangle
    rect5_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 1.2])
    rect5_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 1.2], rgbaColor=[0, 1, 1, 1])
    p.createMultiBody(3.0, rect5_col, rect5_vis, basePosition=[0, 6, 1.4])

    # 6 MAGENTA rectangle (unchanged)
    rect6_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 1.2])
    rect6_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 1.2], rgbaColor=[1, 0, 1, 1])
    p.createMultiBody(3.0, rect6_col, rect6_vis, basePosition=[0, -6, 1.4])

    # 7 ORANGE rectangle
    rect7_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 1.2])
    rect7_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 1.2], rgbaColor=[1, 0.5, 0, 1])
    p.createMultiBody(3.0, rect7_col, rect7_vis, basePosition=[6, 0, 1.4])

    # 8 PURPLE rectangle
    rect8_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 1.2])
    rect8_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 1.2], rgbaColor=[0.5, 0, 1, 1])
    p.createMultiBody(3.0, rect8_col, rect8_vis, basePosition=[-6, 0, 1.4])

    # Sphere corner markers (diagonal inside map)

    sphere_col = p.createCollisionShape(p.GEOM_SPHERE, radius=1.60)
    sphere_vis = p.createVisualShape(p.GEOM_SPHERE, radius=1.60, rgbaColor=[0.3, 0.3, 0.3, 1])

    offset = -2.2
    z = 1.8

    p.createMultiBody(1.0, sphere_col, sphere_vis, basePosition=[4 - offset, 4 - offset, z])
    p.createMultiBody(1.0, sphere_col, sphere_vis, basePosition=[-4 + offset, 4 - offset, z])
    p.createMultiBody(1.0, sphere_col, sphere_vis, basePosition=[4 - offset, -4 + offset, z])
    p.createMultiBody(1.0, sphere_col, sphere_vis, basePosition=[-4 + offset, -4 + offset, z])
    p.createMultiBody(1.0, sphere_col, sphere_vis, basePosition=[0, 0, z])

    # Humanoid Agent (MJCF - HumanoidBulletEnv style)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    humanoid_id = p.loadMJCF(
        "mjcf/humanoid.xml"
    )

    humanoid_id = humanoid_id[0]

    p.resetBasePositionAndOrientation(
        humanoid_id,
        [0, -3.5, 1.4],
        [0, 0, 0, 1]
    )

    return HumanoidEnv(humanoid_id, cid)


# Humanoid Agent Reinforcement Learning

class HumanoidEnv:
    START_POS = [0, -3.5, 1.2]
    START_ORN = [0, 0, 0, 1]

    def __init__(self, humanoid_id, cid):
        self.humanoid = humanoid_id
        self.cid = cid
        self.joint_ids = []

        for i in range(p.getNumJoints(self.humanoid)):
            info = p.getJointInfo(self.humanoid, i)

            if info[2] != p.JOINT_FIXED:
                self.joint_ids.append(i)

        print("joints found:", len(self.joint_ids))

    def reset(self):
        p.resetBasePositionAndOrientation(
            self.humanoid, self.START_POS, self.START_ORN
        )
        p.resetBaseVelocity(self.humanoid, [0, 0, 0], [0, 0, 0])

        for j in self.joint_ids:
            p.resetJointState(self.humanoid, j, targetValue=0, targetVelocity=0)

        return self.get_obs()

    def get_obs(self):
        obs = []

        pos, orn = p.getBasePositionAndOrientation(self.humanoid)
        vel, ang = p.getBaseVelocity(self.humanoid)

        obs.extend(pos)
        obs.extend(vel)
        obs.extend(ang)

        for j in self.joint_ids:
            state = p.getJointState(self.humanoid, j)
            obs.append(state[0])
            obs.append(state[1])

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).flatten()

        # apply clipped joint targets (reduced strength for stability)
        for i, j in enumerate(self.joint_ids):
            target = float(np.clip(action[i], -1, 1))
            p.setJointMotorControl2(
                self.humanoid,
                j,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=20
            )

        p.stepSimulation()

        # get state
        pos, _ = p.getBasePositionAndOrientation(self.humanoid)
        vel, _ = p.getBaseVelocity(self.humanoid)

        # reward components
        forward_reward = 20.0 * vel[1]
        alive_bonus = 0.2
        height_penalty = 2.0 * max(0, 0.8 - pos[2])
        stationary_penalty = -1.0 if abs(vel[1]) < 0.05 else 0.0

        reward = forward_reward + alive_bonus - height_penalty

        # termination
        done = pos[2] < 0.8

        obs = self.get_obs()

        return obs, reward, done

    # Reward Function

    def reward(self):
        pos, _ = p.getBasePositionAndOrientation(self.humanoid)
        vel, _ = p.getBaseVelocity(self.humanoid)

        forward_reward = vel[1]  # move forward (y direction)
        alive_bonus = 1.0  # reward for staying alive
        height_penalty = 2.0 * max(0, 0.8 - pos[2])  # penalize falling

        return forward_reward + alive_bonus - height_penalty

    def is_done(self):
        pos, _ = p.getBasePositionAndOrientation(self.humanoid)
        return pos[2] < 0.5
