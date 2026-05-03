"""
Continuous standing/walking trainer.
- No entropy coefficient
- Uses motion file directly for imitation reward
- Only resets on fall
- Gradient update every UPDATE_EVERY steps

Run:
    python TrainPPO.py
    python TrainPPO.py --resume
"""

import argparse
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
import torch.optim as optim
import pybullet as p

from Environment import create_environment, STAGE

CHECKPOINT_PATH = "stand_checkpoint.pt"
UPDATE_EVERY    = 512
TOTAL_STEPS     = 50_000_000
GUI             = True
MOTION_PATH     = "humanoid3d_walk.txt"


# ------------------------------------------------------------------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, act_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        mean = self.net(x)
        std  = torch.exp(self.log_std.clamp(-3, 1))
        return mean, std

    def sample(self, x):
        mean, std = self.forward(x)
        dist      = D.Normal(mean, std)
        action    = dist.sample()
        log_prob  = dist.log_prob(action).sum(dim=-1)
        return action, log_prob


# ------------------------------------------------------------------
class MotionImitationReward:
    """
    Reads humanoid3d_walk.txt and computes a pose-matching reward
    each step by comparing actual joint angles to the reference frame.

    The file stores quaternions per joint (4 floats each).
    We convert to a single scalar per joint using the scalar (w) component
    so it can be compared against PyBullet's 1-DOF joint angles.
    """

    def __init__(self, path: str, joint_ids: list, cid: int, humanoid_id: int):
        with open(path, "r") as f:
            data = json.load(f)

        frames = data["Frames"]
        self.frame_duration = frames[0][0]
        # columns 1-7 are root pos+quat, columns 8+ are joint data
        self.raw_frames     = np.array([frame[8:] for frame in frames], dtype=np.float32)
        self.num_frames     = len(self.raw_frames)
        self.total_duration = self.num_frames * self.frame_duration

        self.joint_ids  = joint_ids
        self.cid        = cid
        self.humanoid   = humanoid_id
        self.elapsed    = 0.0

        # Each joint in DeepMimic uses 4 floats (quaternion).
        # We extract just the w component (index 0 of each group of 4)
        # as a proxy scalar angle for comparison.
        n_joints   = len(joint_ids)
        refs       = []
        for frame in self.raw_frames:
            frame_refs = []
            for i in range(n_joints):
                base = i * 4
                if base < len(frame):
                    # w component of quaternion — ranges -1 to 1, same as joint angle range
                    frame_refs.append(float(frame[base]))
                else:
                    frame_refs.append(0.0)
            refs.append(frame_refs)

        self.joint_refs = np.array(refs, dtype=np.float32)  # [num_frames, n_joints]
        print(f"[MotionImitation] {self.num_frames} frames | "
              f"{self.total_duration:.2f}s total | "
              f"{n_joints} joints tracked")

    def reset(self):
        self.elapsed = 0.0

    def step(self, dt: float) -> float:
        """Advance clock and return imitation reward for current joint state."""
        self.elapsed += dt

        # Get reference frame at current time (looping)
        t      = (self.elapsed % self.total_duration) / self.frame_duration
        idx_lo = int(t) % self.num_frames
        idx_hi = (idx_lo + 1) % self.num_frames
        alpha  = t - int(t)
        ref    = (1.0 - alpha) * self.joint_refs[idx_lo] + alpha * self.joint_refs[idx_hi]

        # Get actual joint angles from sim
        actual = np.array([
            p.getJointState(self.humanoid, j, physicsClientId=self.cid)[0]
            for j in self.joint_ids
        ], dtype=np.float32)

        # Exponential kernel — same as DeepMimic paper
        error  = np.sum(np.square(actual - ref))
        reward = float(np.exp(-2.0 * error))
        return reward


# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    env = create_environment(gui=GUI, motion_path=MOTION_PATH)

    obs_raw, _ = env.get_obs()
    obs_dim    = len(obs_raw)
    act_dim    = len(env.joint_ids)
    print(f"obs_dim={obs_dim}  act_dim={act_dim}  stage={STAGE}")

    # Wire up motion imitation reward
    motion_reward = MotionImitationReward(
        path       = MOTION_PATH,
        joint_ids  = env.joint_ids,
        cid        = env.cid,
        humanoid_id= env.humanoid,
    )

    policy    = PolicyNet(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    start_step = 0
    falls      = 0

    if args.resume and os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, weights_only=True)
        policy.load_state_dict(ckpt["policy"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        falls      = ckpt.get("falls", 0)
        print(f"Resumed from step {start_step}, falls so far: {falls}")
    else:
        print("Starting fresh.")

    gamma      = 0.99
    log_probs  = []
    rewards    = []
    total_step = start_step
    dt         = 1.0 / 240.0

    obs = env.reset()
    motion_reward.reset()

    while total_step < TOTAL_STEPS:

        obs_tensor       = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action, log_prob = policy.sample(obs_tensor)
        action           = action.squeeze(0)
        log_prob         = log_prob.squeeze(0)
        action_np        = torch.clamp(action, -1.0, 1.0).detach().numpy()

        if GUI:
            time.sleep(dt)

        obs, env_reward, done = env.step(action_np)

        # Add motion imitation reward on top of env reward
        imitation_reward = motion_reward.step(dt)
        reward = env_reward + imitation_reward

        log_probs.append(log_prob)
        rewards.append(reward)
        total_step += 1

        if done:
            falls += 1
            obs = env.reset()
            motion_reward.reset()
            print(f"  Fall #{falls} at step {total_step}")

        # Gradient update every UPDATE_EVERY steps
        if len(rewards) >= UPDATE_EVERY:
            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)

            returns_t   = torch.tensor(returns, dtype=torch.float32)
            returns_t   = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
            log_probs_t = torch.stack(log_probs)
            advantages  = returns_t - returns_t.mean()

            loss = -(log_probs_t * advantages).mean()  # no entropy term

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

            avg_r = sum(rewards) / len(rewards)
            print(f"Step: {total_step:>8} | Avg reward: {avg_r:>7.3f} | Falls: {falls}")

            log_probs = []
            rewards   = []

        if total_step % 10_000 == 0:
            torch.save({
                "policy":    policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step":      total_step,
                "falls":     falls,
            }, CHECKPOINT_PATH)
            print(f"  → Checkpoint saved at step {total_step}")

    print("Training complete.")


if __name__ == "__main__":
    main()