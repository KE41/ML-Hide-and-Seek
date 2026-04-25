"""
Train the humanoid with PPO (Stable-Baselines3) using motion imitation.
The humanoid learns to walk by matching humanoid3d_walk.txt reference poses.

Install deps once:
    pip install stable-baselines3[extra] gymnasium

Run:
    python TrainPPO.py
    python TrainPPO.py --resume --resume-steps 5000000

Watch tensorboard:
    tensorboard --logdir ppo_logs
"""

import argparse
import os
from functools import partial

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.wrappers import TimeLimit

from GymWrapper import HumanoidGymEnv

# ---------------------------------------------------------------------------
CHECKPOINT_DIR    = "checkpoints"
BEST_MODEL_DIR    = "best_model"
LOG_DIR           = "ppo_logs"
TOTAL_STEPS       = 50_000_000
N_ENVS            = 1      # 32 for desktop pc
MAX_EPISODE_STEPS = 2000    # episodes end after this many steps
MOTION_PATH       = "humanoid3d_walk.txt"  # must be in project folder
# ---------------------------------------------------------------------------


def make_env_fn():
    """
    Each parallel env gets its own PyBullet instance and its own MotionClip.
    TimeLimit forces episodes to end so Monitor can log ep_rew_mean.
    """
    env = HumanoidGymEnv(gui=True, motion_path=MOTION_PATH)
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    env = Monitor(env)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/humanoid_ppo_8500000_steps.zip",
        help="Checkpoint to resume from",
    )
    parser.add_argument(
        "--resume-steps",
        type=int,
        default=5_000_000,
        help="Steps already trained — fixes progress bar",
    )
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,        exist_ok=True)

    env = make_vec_env(make_env_fn, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    checkpoint_cb = CheckpointCallback(
        save_freq   = 500_000 // N_ENVS,
        save_path   = CHECKPOINT_DIR,
        name_prefix = "humanoid_ppo",
        verbose     = 1,
    )

    # Larger network for motion imitation — needs to learn complex pose matching
    policy_kwargs = dict(
        net_arch      = dict(pi=[1024, 512], vf=[1024, 512]),
        activation_fn = __import__("torch").nn.Tanh,
    )

    if args.resume:
        print(f"Resuming from {args.checkpoint}")
        model = PPO.load(
            args.checkpoint,
            env             = env,
            tensorboard_log = LOG_DIR,
        )
        steps_to_run = TOTAL_STEPS - args.resume_steps
    else:
        model = PPO(
            policy          = "MlpPolicy",
            env             = env,
            device          = "cpu",
            verbose         = 1,
            tensorboard_log = LOG_DIR,
            n_steps         = 4096,
            batch_size      = 256,
            n_epochs        = 10,
            learning_rate   = 3e-4,
            ent_coef        = 0.01,    # slightly higher for motion imitation exploration
            clip_range      = 0.2,
            gae_lambda      = 0.95,
            gamma           = 0.99,
            vf_coef         = 0.5,
            max_grad_norm   = 0.5,
            policy_kwargs   = policy_kwargs,
        )
        steps_to_run = TOTAL_STEPS

    print(f"Training for {steps_to_run:,} steps across {N_ENVS} parallel envs …")
    print(f"Motion file: {MOTION_PATH}")
    print("Watch live:  tensorboard --logdir ppo_logs\n")

    model.learn(
        total_timesteps     = steps_to_run,
        callback            = [checkpoint_cb],
        reset_num_timesteps = not args.resume,
        progress_bar        = True,
    )

    model.save("humanoid_ppo_final")
    print("Done — saved humanoid_ppo_final.zip")


if __name__ == "__main__":
    main()