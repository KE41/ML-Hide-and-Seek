import pybullet as p
import numpy as np
import torch
import torch.optim as optim
import time

from Environment import create_environment
from PyTorchPolicy import PolicyNet

SAVE_PATH  = "policy.pt"
SAVE_EVERY = 500   # save weights every N steps

def main():
    env = create_environment(gui=True)

    # Read initial obs without touching physics
    obs, _ = env.get_obs()
    print("obs shape:", obs.shape)

    obs_dim = len(obs)
    act_dim = len(env.joint_ids)
    print(f"obs_dim={obs_dim}  act_dim={act_dim}")

    policy    = PolicyNet(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    # Load saved weights if they exist
    start_step = 0
    try:
        checkpoint  = torch.load(SAVE_PATH)
        policy.load_state_dict(checkpoint['policy'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step  = checkpoint['step']
        print(f"Resumed from step {start_step}")
    except FileNotFoundError:
        print("No checkpoint found, starting fresh")

    gamma     = 0.99
    PHYSICS_HZ = 240
    SUB_STEPS  = 4
    step_dt    = SUB_STEPS / PHYSICS_HZ

    # Collect experience in rolling windows instead of episodes.
    # The robot NEVER resets — it just keeps walking (or falling) indefinitely.
    WINDOW = 300   # how many steps to collect before doing one gradient update

    log_probs = []
    rewards   = []
    step      = start_step

    while True:

        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action, log_prob = policy.sample(obs_tensor)
        action   = action.squeeze(0)
        log_prob = log_prob.squeeze(0)
        action   = torch.clamp(action, -1.0, 1.0)

        obs, reward, done = env.step(action.detach().numpy())

        time.sleep(step_dt)  # pace GUI to real-time

        log_probs.append(log_prob)
        rewards.append(reward)
        step += 1

        # When the window is full, update the policy and clear the buffer
        if len(rewards) >= WINDOW:
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)

            returns    = torch.tensor(returns, dtype=torch.float32)
            returns    = (returns - returns.mean()) / (returns.std() + 1e-8)
            log_probs_t = torch.stack(log_probs)
            advantages  = returns - returns.mean()

            entropy_bonus = 0.001 * log_probs_t.detach().mean()
            loss = -(log_probs_t * advantages).mean() + entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Step {step:8d} | reward/window {sum(rewards):8.2f} | loss {loss.item():.4f}")

            log_probs = []
            rewards   = []

        # Save checkpoint periodically
        if step % SAVE_EVERY == 0:
            torch.save({
                'step':      step,
                'policy':    policy.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, SAVE_PATH)
            print(f"  -> Saved checkpoint at step {step}")


if __name__ == "__main__":
    main()