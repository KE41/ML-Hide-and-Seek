"""
Continuous standing trainer — Stage 1.
Uses the same continuous-loop style as the original main.py:
  - No episode time limit
  - Only resets on fall (done=True from Environment)
  - Gradient update every UPDATE_EVERY steps
  - Checkpoint saves and resumes automatically

Run:
    python TrainPPO.py
    python TrainPPO.py --resume
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.optim as optim

from Environment import create_environment

CHECKPOINT_PATH = "stand_checkpoint.pt"
UPDATE_EVERY    = 512   # steps between gradient updates
TOTAL_STEPS     = 50_000_000
GUI             = True


# ------------------------------------------------------------------
# Minimal inline policy — same architecture as original PyTorchPolicy
# ------------------------------------------------------------------
import torch.nn as nn
import torch.distributions as D

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    env = create_environment(gui=GUI)

    # get_obs returns (array, dict) in this branch
    obs_raw, _ = env.get_obs()
    obs_dim    = len(obs_raw)
    act_dim    = len(env.joint_ids)
    print(f"obs_dim={obs_dim}  act_dim={act_dim}")

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

    # First obs
    obs = env.reset()

    while total_step < TOTAL_STEPS:

        obs_tensor        = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action, log_prob  = policy.sample(obs_tensor)
        action            = action.squeeze(0)
        log_prob          = log_prob.squeeze(0)
        action_np         = torch.clamp(action, -1.0, 1.0).detach().numpy()

        if GUI:
            time.sleep(1.0 / 240.0)

        obs, reward, done = env.step(action_np)

        log_probs.append(log_prob)
        rewards.append(reward)
        total_step += 1

        # Only reset on actual fall — continuous otherwise
        if done:
            falls += 1
            obs = env.reset()
            print(f"  Fall #{falls} at step {total_step}")

        # ----------------------------------------------------------
        # Gradient update every UPDATE_EVERY steps
        # ----------------------------------------------------------
        if len(rewards) >= UPDATE_EVERY:
            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)

            returns_t  = torch.tensor(returns, dtype=torch.float32)
            returns_t  = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

            log_probs_t = torch.stack(log_probs)
            advantages  = returns_t - returns_t.mean()

            # Entropy bonus decays over time: explore early, exploit later
            entropy_coeff = max(0.0001, 0.01 * (0.999 ** (total_step / UPDATE_EVERY)))
            entropy_bonus = entropy_coeff * log_probs_t.detach().mean()
            loss = -(log_probs_t * advantages).mean() + entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

            avg_r = sum(rewards) / len(rewards)
            print(
                f"Step: {total_step:>8} | "
                f"Avg reward: {avg_r:>7.3f} | "
                f"Falls: {falls} | "
                f"Entropy coeff: {entropy_coeff:.5f}"
            )

            log_probs = []
            rewards   = []

        # Checkpoint every 10k steps
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