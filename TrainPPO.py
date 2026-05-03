"""
Train the humanoid with PPO (Stable-Baselines3) using motion imitation.
The humanoid learns to walk by matching humanoid3d_walk.txt reference poses.

Install deps once:
    pip install stable-baselines3[extra] gymnasium pybullet

Run:
    python TrainPPO.py
    python TrainPPO.py --resume
    python TrainPPO.py --resume --checkpoint checkpoints/humanoid_ppo_5000000_steps.zip

Watch TensorBoard:
    tensorboard --logdir ppo_logs
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from gymnasium.wrappers import TimeLimit

from GymWrapper import HumanoidGymEnv

# ===========================================================================
# PARAMETERS — edit these before running
# ===========================================================================

MOTION_PATH       = "humanoid3d_walk.txt"   # reference motion file

TOTAL_STEPS       = 50_000_000              # total env steps to train for
N_ENVS            = 1                       # 32 for not use 1 for GUI)
MAX_EPISODE_STEPS = 2_000                   # truncate episodes after this many steps

# PPO hyperparameters
N_STEPS           = 4_096                   # rollout buffer steps per env per update
BATCH_SIZE        = 256                     # minibatch size
N_EPOCHS          = 10                      # passes over rollout buffer per update
LEARNING_RATE     = 3e-4
ENT_COEF          = 0.01                    # entropy bonus (helps exploration for motion imitation)
CLIP_RANGE        = 0.2
GAE_LAMBDA        = 0.95
GAMMA             = 0.99
VF_COEF           = 0.5
MAX_GRAD_NORM     = 0.5

# Network architecture
POLICY_NET_ARCH   = dict(pi=[1024, 512], vf=[1024, 512])

# Paths
CHECKPOINT_DIR    = "checkpoints"
BEST_MODEL_DIR    = "best_model"
LOG_DIR           = "ppo_logs"
FINAL_MODEL_PATH  = "humanoid_ppo_final"
CHECKPOINT_FREQ   = 500_000                 # save a checkpoint every N steps

# ===========================================================================


def make_env_fn(gui: bool = False):
    """Factory that returns a single wrapped environment."""
    def _init():
        env = HumanoidGymEnv(gui=gui, motion_path=MOTION_PATH)
        env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
        env = Monitor(env)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a checkpoint zip",
    )
    parser.add_argument(
        "--checkpoint",
        default=f"{CHECKPOINT_DIR}/humanoid_ppo_1000000_steps.zip",
        help="Checkpoint .zip to resume from (used with --resume)",
    )
    parser.add_argument(
        "--resume-steps",
        type=int,
        default=0,
        help="Steps already completed in the checkpoint (fixes progress bar)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=False,
        help="Open PyBullet GUI (forces N_ENVS=1 and DummyVecEnv)",
    )
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,        exist_ok=True)

    # GUI mode must run single-process (PyBullet GUI can't be forked)
    n_envs     = 1 if args.gui else N_ENVS
    vec_cls    = DummyVecEnv if (args.gui or n_envs == 1) else SubprocVecEnv

    print(f"Creating {n_envs} environment(s) via {vec_cls.__name__} …")
    env = make_vec_env(
        make_env_fn(gui=args.gui),
        n_envs      = n_envs,
        vec_env_cls = vec_cls,
    )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    checkpoint_cb = CheckpointCallback(
        save_freq   = max(CHECKPOINT_FREQ // n_envs, 1),
        save_path   = CHECKPOINT_DIR,
        name_prefix = "humanoid_ppo",
        verbose     = 1,
    )

    # Eval env uses the same vec type as training env to avoid SB3 warning
    eval_env = make_vec_env(
        make_env_fn(gui=False),
        n_envs      = 1,
        vec_env_cls = vec_cls,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = BEST_MODEL_DIR,
        log_path             = LOG_DIR,
        eval_freq            = max(CHECKPOINT_FREQ // n_envs, 1),
        n_eval_episodes      = 5,
        deterministic        = True,
        verbose              = 1,
    )

    callbacks = [checkpoint_cb, eval_cb]

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    policy_kwargs = dict(
        net_arch      = POLICY_NET_ARCH,
        activation_fn = __import__("torch").nn.Tanh,
    )

    if args.resume:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(
                f"Checkpoint not found: {args.checkpoint}\n"
                f"Available checkpoints: {os.listdir(CHECKPOINT_DIR)}"
            )
        print(f"Resuming from: {args.checkpoint}  (steps already done: {args.resume_steps:,})")
        model = PPO.load(
            args.checkpoint,
            env             = env,
            tensorboard_log = LOG_DIR,
        )
        steps_to_run = max(TOTAL_STEPS - args.resume_steps, 0)
    else:
        print("Starting fresh training run.")
        model = PPO(
            policy          = "MlpPolicy",
            env             = env,
            device          = "cpu",
            verbose         = 1,
            tensorboard_log = LOG_DIR,
            n_steps         = N_STEPS,
            batch_size      = BATCH_SIZE,
            n_epochs        = N_EPOCHS,
            learning_rate   = LEARNING_RATE,
            ent_coef        = ENT_COEF,
            clip_range      = CLIP_RANGE,
            gae_lambda      = GAE_LAMBDA,
            gamma           = GAMMA,
            vf_coef         = VF_COEF,
            max_grad_norm   = MAX_GRAD_NORM,
            policy_kwargs   = policy_kwargs,
        )
        steps_to_run = TOTAL_STEPS

    print()
    print(f"  Motion file      : {MOTION_PATH}")
    print(f"  Parallel envs    : {n_envs}")
    print(f"  Max episode steps: {MAX_EPISODE_STEPS:,}")
    print(f"  Steps to run     : {steps_to_run:,}")
    print(f"  Checkpoints      : {CHECKPOINT_DIR}/")
    print(f"  Best model       : {BEST_MODEL_DIR}/")
    print(f"  TensorBoard logs : {LOG_DIR}/")
    print()
    print("  Watch live:  tensorboard --logdir", LOG_DIR)
    print()

    model.learn(
        total_timesteps     = steps_to_run,
        callback            = callbacks,
        reset_num_timesteps = not args.resume,
        progress_bar        = True,
    )

    model.save(FINAL_MODEL_PATH)
    print(f"Training complete — saved {FINAL_MODEL_PATH}.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()