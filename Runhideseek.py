"""
RunHideSeek.py
--------------
Entry point for the hide-and-seek simulation.

Boots the existing PyBullet world from Environment.create_environment(),
then hands it to HideSeekEnv which adds two humanoid robots and runs the game.

Usage
-----
    python RunHideSeek.py
    python RunHideSeek.py --rounds 20
    python RunHideSeek.py --rounds 5 --duration 30 --seeker-speed 3.5 --hider-speed 2.5

Arguments
---------
  --rounds        Number of rounds to play  (default: 10)
  --duration      Seconds of sim-time per round (default: 20)
  --seeker-speed  Max speed of the seeker humanoid (default: 3.0)
  --hider-speed   Max speed of the hider  humanoid (default: 2.8)

The MJCF humanoid from the original environment is still present but frozen.
Two new URDF-based humanoid robots are added:
  Seeker : humanoid/humanoid.urdf  — full 3-D humanoid, orange, RL Q-table agent
  Hider  : biped/biped2d_pybullet.urdf — biped robot, cyan, rule-based agent

Both robots move kinematically with sine-wave joint walking animation.

Scoring
-------
  • Seeker earns 1 point per round it catches the hider within time limit
  • Hider  earns 1 point per round it evades until time expires
  • Final scores printed and shown in the GUI at the end

Agent strategies
----------------
  Seeker : Q-table reinforcement learning (ε-greedy over discretised state).
           Learns across rounds — epsilon decays so it exploits more later.
  Hider  : Rule-based potential-field controller.
           Repels away from seeker, seeks nearest obstacle for cover.
"""

import argparse
import sys
import os

# Make sure local files are importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Environment import create_environment   # unchanged original
from HideSeekEnv import HideSeekEnv          # new extension module


def main():
    parser = argparse.ArgumentParser(description="PyBullet Hide-and-Seek simulation")
    parser.add_argument("--rounds",       type=int,   default=10,
                        help="Number of rounds")
    parser.add_argument("--duration",     type=float, default=20.0,
                        help="Sim-time per round (seconds)")
    parser.add_argument("--seeker-speed", type=float, default=1.5,
                        help="Seeker game-AI target speed (m/s)")
    parser.add_argument("--hider-speed",  type=float, default=1.2,
                        help="Hider game-AI target speed (m/s)")
    parser.add_argument("--policy",       type=str,
                        default="best_model/best_model.zip",
                        help="Path to trained PPO checkpoint for the seeker humanoid")
    args = parser.parse_args()

    print("=" * 60)
    print("  PyBullet Hide-and-Seek  — Two Humanoids Edition")
    print("  Seeker : MJCF humanoid  (PPO policy + steering)  [orange]")
    print("  Hider  : MJCF humanoid  (CPG oscillator walking) [cyan]")
    print(f"  Policy : {args.policy}")
    print("=" * 60)

    base_env = create_environment(gui=True)

    game = HideSeekEnv(
        base_env       = base_env,
        round_duration = args.duration,
        seeker_speed   = args.seeker_speed,
        hider_speed    = args.hider_speed,
        policy_path    = args.policy,
    )

    seeker_pts, hider_pts = game.run(num_rounds=args.rounds)

    print()
    print("=" * 60)
    print(f"  Final scores after {args.rounds} rounds:")
    print(f"    Seeker (PPO humanoid) : {seeker_pts} point(s)")
    print(f"    Hider  (CPG biped)    : {hider_pts} point(s)")
    if seeker_pts > hider_pts:
        print("  Winner: SEEKER — PPO humanoid caught the CPG humanoid!")
    elif hider_pts > seeker_pts:
        print("  Winner: HIDER  — CPG humanoid evaded the PPO seeker!")
    else:
        print("  Result: TIE")
    print("=" * 60)


if __name__ == "__main__":
    main()