"""
Train the humanoid with PPO (Stable-Baselines3).
Replaces main.py — Environment.py and PyTorchPolicy.py are untouched.

Install deps once:
    pip install stable-baselines3[extra] gymnasium

Run:
    python TrainPPO.py
    python TrainPPO.py --resume       # continue from last checkpoint
"""

#pip install stablebaseline3 + [extra]
#pip install torch-directml
# pip install tensorboard
# pip install stable-baselines3[extra]
# Ensure ur using python 3.10.11

# to see data run tensorboard --logdir ppo_logs on terminal in folder

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

from GymWrapper import HumanoidGymEnv

# ---------------------------------------------------------------------------
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_DIR = "best_model"
LOG_DIR        = "ppo_logs"
TOTAL_STEPS    = 50_000_000
N_ENVS         = 4   # parallel environments — remove for hide and seek
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,        exist_ok=True)

    # Run N_ENVS environments in parallel for faster step collection
    env = make_vec_env(lambda: HumanoidGymEnv(gui=False), n_envs=N_ENVS)

    policy_kwargs = dict(
        net_arch      = dict(pi=[512, 512], vf=[512, 512]),
        activation_fn = __import__("torch").nn.Tanh,
    )

    if args.resume:
        checkpoints = sorted(
            [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".zip")],
            key=lambda f: int(f.split("_")[-2]) if f.split("_")[-2].isdigit() else 0
        )
        if checkpoints:
            latest = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
            print(f"Resuming from {latest}")
            model = PPO.load(latest, env=env, tensorboard_log=LOG_DIR)
        else:
            print("No checkpoint found — starting fresh.")
            args.resume = False

    if not args.resume:
        model = PPO(
            policy          = "MlpPolicy",
            env             = env,
            device          = "cpu",        # ← string "cpu", not the torch.cpu module
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
        save_freq   = 100_000,
        save_path   = CHECKPOINT_DIR,
        name_prefix = "humanoid_ppo",
        verbose     = 1,
    )

    print(f"Training for {TOTAL_STEPS:,} steps across {N_ENVS} parallel envs …")
    print("Watch live:  tensorboard --logdir ppo_logs\n")

    model.learn(
        total_timesteps     = TOTAL_STEPS,
        callback            = [checkpoint_cb],
        reset_num_timesteps = not args.resume,
        progress_bar        = True,
    )

    model.save("humanoid_ppo_final")
    print("Done — saved humanoid_ppo_final.zip")


# Run first, then run RunPolicy.py to watch the result.
if __name__ == "__main__":
    main()