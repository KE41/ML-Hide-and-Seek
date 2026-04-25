"""
Watch a trained humanoid policy run in the GUI.

Usage:
    python RunPolicy.py                          # loads best_model/best_model.zip
    python RunPolicy.py --model humanoid_ppo_final.zip
    python RunPolicy.py --model checkpoints/humanoid_ppo_5000000_steps.zip
"""

import argparse
import time

from stable_baselines3 import PPO
from GymWrapper import HumanoidGymEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="best_model/best_model.zip",
        help="Path to a .zip checkpoint",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--motion",
        default="humanoid3d_walk.txt",
        help="Motion file used during training",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = PPO.load(args.model)

    env = HumanoidGymEnv(gui=True, motion_path=args.motion)

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            time.sleep(1 / 60)

            if terminated or truncated:
                break

        print(f"Episode {ep}: steps={steps}  total_reward={total_reward:.3f}")

    env.close()


if __name__ == "__main__":
    main()