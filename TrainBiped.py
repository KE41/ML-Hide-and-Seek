"""
TrainBiped.py
-------------
Train a PPO locomotion policy for biped2d_pybullet.urdf.

Run this BEFORE Runhideseek.py so the bipeds have a policy to load.

Usage
-----
    python TrainBiped.py                    # default 1M steps
    python TrainBiped.py --steps 2000000    # longer training

GPU acceleration
----------------
Stable Baselines 3 uses PyTorch.  On Windows with an AMD GPU, install
torch-directml and the device will be detected automatically:

    pip install torch-directml

For NVIDIA GPUs PyTorch CUDA is used automatically if available.
CPU fallback is always available.

Output
------
Saves biped_policy/best_model.zip  (used by BipedNNWalker)
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from BipedGymEnv import BipedGymEnv


def _detect_device():
    """Return the best available torch device string."""
    try:
        import torch
        if torch.cuda.is_available():
            print("[TrainBiped] Using CUDA GPU")
            return "cuda"
    except ImportError:
        pass

    try:
        import torch_directml
        dml = torch_directml.device()
        print(f"[TrainBiped] Using DirectML device: {dml}")
        return dml
    except (ImportError, Exception):
        pass

    print("[TrainBiped] Using CPU")
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",   type=int,   default=1_000_000,
                        help="Total training timesteps (default 1M)")
    parser.add_argument("--envs",    type=int,   default=4,
                        help="Parallel training environments")
    parser.add_argument("--resume",  action="store_true",
                        help="Resume from biped_policy/best_model.zip if present")
    args = parser.parse_args()

    os.makedirs("biped_policy", exist_ok=True)

    device = _detect_device()

    # Vectorised training envs
    env = make_vec_env(BipedGymEnv, n_envs=args.envs, vec_env_cls=SubprocVecEnv)

    # Separate eval env (single, serial)
    eval_env = make_vec_env(BipedGymEnv, n_envs=1)

    policy_path = "biped_policy/best_model.zip"

    if args.resume and os.path.exists(policy_path):
        print(f"[TrainBiped] Resuming from {policy_path}")
        model = PPO.load(policy_path, env=env, device=device)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps        = 2048,
            batch_size     = 256,
            n_epochs       = 10,
            learning_rate  = 3e-4,
            gamma          = 0.99,
            gae_lambda     = 0.95,
            clip_range     = 0.2,
            ent_coef       = 0.005,
            vf_coef        = 0.5,
            max_grad_norm  = 0.5,
            policy_kwargs  = dict(net_arch=[256, 256]),
            verbose        = 1,
            tensorboard_log= "biped_logs/",
            device         = device,
        )

    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=200.0, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = "biped_policy/",
        log_path             = "biped_logs/",
        eval_freq            = 20_000,
        n_eval_episodes      = 5,
        callback_on_new_best = stop_cb,
        verbose              = 1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq  = 100_000,
        save_path  = "biped_policy/",
        name_prefix= "biped_ckpt",
    )

    print(f"\n[TrainBiped] Training for {args.steps:,} steps on {args.envs} envs...")
    print( "[TrainBiped] Policy will be saved to biped_policy/best_model.zip\n")

    model.learn(total_timesteps=args.steps, callback=[eval_cb, ckpt_cb])

    model.save("biped_policy/final_model")
    print("\n[TrainBiped] Done.  Run:  python Runhideseek.py")


if __name__ == "__main__":
    main()
