"""
Train the humanoid with PPO (Stable-Baselines3).
Replaces main.py — Environment.py and PyTorchPolicy.py are untouched.

Install deps once:
    pip install stable-baselines3[extra] gymnasium

Run:
    python train_ppo.py
    python train_ppo.py --gui          # watch it in real-time (slow)
    python train_ppo.py --resume       # continue from last checkpoint
"""

#pip install stablebaseline3 + [extra]
#pip install torch-directml

import argparse
import os
import torch_directml
device = torch_directml.device()

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from GymWrapper import HumanoidGymEnv

# ---------------------------------------------------------------------------
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_DIR = "best_model"
LOG_DIR        = "ppo_logs"
TOTAL_STEPS    = 10_000_000   # increase for better walking (20M+ recommended)
# ---------------------------------------------------------------------------


def make_env(gui: bool = True):
    env = HumanoidGymEnv(gui=gui)
    env = Monitor(env)            # records episode rewards/lengths to CSV
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui",    action="store_true", help="Enable PyBullet GUI")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,        exist_ok=True)

    env = HumanoidGymEnv(gui=True)  # ← create directly, not through make_env
    env = Monitor(env)

    # Sanity-check the wrapped env (prints warnings if something is wrong)
    print("Checking environment …")
    check_env(env, warn=True)
    print("Environment OK.\n")

    # ------------------------------------------------------------------
    # PPO hyper-parameters tuned for humanoid locomotion.
    #
    #   n_steps        – steps collected per update (larger = more stable)
    #   batch_size     – mini-batch size for gradient updates
    #   n_epochs       – passes over each rollout buffer
    #   learning_rate  – conservative for stability
    #   ent_coef       – entropy bonus encourages exploration
    #   clip_range     – PPO clipping (0.2 is standard)
    #   gae_lambda     – GAE smoothing; 0.95 is standard
    #   gamma          – discount; 0.99 good for long-horizon walking
    #   vf_coef        – value-function loss weight
    #   max_grad_norm  – gradient clipping
    #   policy_kwargs  – 2×512 Tanh matches your original network size
    # ------------------------------------------------------------------
    policy_kwargs = dict(
        net_arch   = dict(pi=[512, 512], vf=[512, 512]),
        activation_fn = __import__("torch").nn.Tanh,
    )

    if args.resume:
        # Find the latest checkpoint
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
            policy         = "MlpPolicy",
            env            = env,
            verbose        = 1,
            tensorboard_log= LOG_DIR,
            n_steps        = 4096,
            batch_size     = 256,
            n_epochs       = 10,
            learning_rate  = 3e-4,
            ent_coef       = 0.005,
            clip_range     = 0.2,
            gae_lambda     = 0.95,
            gamma          = 0.99,
            vf_coef        = 0.5,
            max_grad_norm  = 0.5,
            policy_kwargs  = policy_kwargs,
        )

    # Save a checkpoint every 100 k steps
    checkpoint_cb = CheckpointCallback(
        save_freq   = 100_000,
        save_path   = CHECKPOINT_DIR,
        name_prefix = "humanoid_ppo",
        verbose     = 1,
    )

    # Keep a copy of the best model (by mean episode reward)
    # eval_env  = make_env(gui=False)
    # eval_cb   = EvalCallback(
    #     eval_env,
    #     best_model_save_path = BEST_MODEL_DIR,
    #     log_path             = LOG_DIR,
    #     eval_freq            = 50_000,
    #     n_eval_episodes      = 3,
    #     deterministic        = True,
    #     verbose              = 1,
    # )

    print(f"Training for {TOTAL_STEPS:,} steps …")
    print("Watch live:  tensorboard --logdir ppo_logs\n")

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=[checkpoint_cb],  # ← removed eval_cb
        reset_num_timesteps=not args.resume,
        progress_bar=True,
    )

    model.save("humanoid_ppo_final")
    print("Done — saved humanoid_ppo_final.zip")

# Run first then RunPolicy.py second.

if __name__ == "__main__":
    main()