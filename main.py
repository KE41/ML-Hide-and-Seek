import pybullet as p
import numpy as np
import torch
import torch.optim as optim
import time

from Environment import create_environment
from PyTorchPolicy import PolicyNet


def main():
    env = create_environment(gui=True)

    obs = env.reset()
    print("Reset obs shape:", obs.shape)
    print("Reset obs values:", obs[:10])  # first 10 values

    obs_dim = len(obs)
    act_dim = len(env.joint_ids)

    policy = PolicyNet(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    gamma = 0.99

    for episode in range(3000000):

        obs = env.reset() #cleared every episode?

        log_probs = []
        rewards = []

        done = False
        steps = 0

        while not done and steps < 300:
            print(f"Step {steps}, done={done}")

            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            action, log_prob = policy.sample(obs_tensor)

            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            print("log_prob shape:", log_prob.shape)  # should be []  or [1]

            action = torch.clamp(action, -1.0, 1.0)

            # Keep the window open
            time.sleep(1. / 200.)

            obs, reward, done = env.step(action.detach().numpy())

            log_probs.append(log_prob)
            rewards.append(reward)

            steps += 1

        returns = []
        G = 0

        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs = torch.stack(log_probs)

        baseline = returns.mean()
        advantages = returns - baseline

        entropy_bonus = 0.001 * log_probs.detach().mean()
        loss = -(log_probs * advantages).mean() + entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Episode:", episode, "Reward:", sum(rewards))


if __name__ == "__main__":
    main()