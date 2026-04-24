"""
Train the humanoid with PPO (Stable-Baselines3).
Replaces main.py — Environment.py and PyTorchPolicy.py are untouched.

Install deps once:
    pip install stable-baselines3[extra] gymnasium

Run:
    python TrainPPO.py
    python TrainPPO.py --resume       # continue from last checkpoint
"""

#pip install torch-directml
# pip install tensorboard
# pip install stable-baselines3[extra]
# Ensure ur using python 3.10.11

# to see data run tensorboard --logdir ppo_logs on terminal in folder

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from functools import partial

from GymWrapper import HumanoidGymEnv

# ---------------------------------------------------------------------------
CHECKPOINT_DIR    = "checkpoints"
BEST_MODEL_DIR    = "best_model"
LOG_DIR           = "ppo_logs"
TOTAL_STEPS       = 50_000_000
N_ENVS            = 1      # 32 parallel environments - desktop pc
MAX_EPISODE_STEPS = 2000    # FIX: forces episodes to end so Monitor logs rewards
# ---------------------------------------------------------------------------

def make_env_fn():
    return make_env(gui=True)

def make_env(gui):
    """
    Wrap in TimeLimit FIRST so episodes terminate after MAX_EPISODE_STEPS,
    then Monitor so SB3 can log ep_rew_mean to TensorBoard.
    Without TimeLimit, if the robot never falls, episodes never end and
    TensorBoard shows nothing.
    """
    env = HumanoidGymEnv(gui=gui)
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    env = Monitor(env)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/humanoid_ppo_5200000_steps.zip",
        help="Checkpoint path to resume from",
    )
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,        exist_ok=True)

    # FIX: gui=False during training — GUI slows everything down and can
    # interfere with logging flushes. Use RunPolicy.py to watch results.

    env = make_vec_env(
        make_env_fn,
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv  # add this
    )

    policy_kwargs = dict(
        net_arch      = dict(pi=[512, 512], vf=[512, 512]),
        activation_fn = __import__("torch").nn.Tanh,
    )

    # FIX: resume block no longer calls learn() — the single learn() call
    # at the bottom handles both fresh and resumed runs.
    if args.resume:
        print(f"Resuming from {args.checkpoint}")
        model = PPO.load(
            args.checkpoint,
            env             = env,
            tensorboard_log = LOG_DIR,  # FIX: must re-attach log dir on load
        )
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
            ent_coef        = 0.005,
            clip_range      = 0.2,
            gae_lambda      = 0.95,
            gamma           = 0.99,
            vf_coef         = 0.5,
            max_grad_norm   = 0.5,
            policy_kwargs   = policy_kwargs,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq   = 500_000 // N_ENVS, # Checkpoint save freq
        save_path   = CHECKPOINT_DIR,
        name_prefix = "humanoid_ppo",
        verbose     = 1,
    )

    print(f"Training for {TOTAL_STEPS:,} steps across {N_ENVS} parallel envs …")
    print("Watch live:  tensorboard --logdir ppo_logs\n")

    model.learn(
        total_timesteps     = TOTAL_STEPS,
        callback            = [checkpoint_cb],
        reset_num_timesteps = not args.resume,  # FIX: True on fresh run, False on resume
        progress_bar        = True,
    )

    model.save("humanoid_ppo_final")
    print("Done — saved humanoid_ppo_final.zip")


# Run first, then run RunPolicy.py to watch the result.

# Use:
# python TrainPPO.py --resume
# to resume from default checkpoint, or:
# python TrainPPO.py --resume --checkpoint checkpoints/humanoid_ppo_500000_steps.zip

if __name__ == "__main__":
    main()