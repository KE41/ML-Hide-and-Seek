"""
HideSeekEnv.py
--------------
Hide-and-seek game using two biped robots controlled by BipedNNWalker.

  Seeker  — orange biped, RL Q-table game AI for direction
  Hider   — cyan biped,   potential-field game AI for direction

Both robots use a trained PPO neural-network locomotion policy; what differs
is the high-level game strategy (SeekerAgent vs HiderAgent) that decides
which direction to walk each step.

Game rules
----------
  • Each round lasts ROUND_DURATION sim-time seconds.
  • Seeker scores if XY distance < CATCH_DIST and has line-of-sight.
  • Hider scores if it survives the full round.
  • Scores persist across rounds.
"""

import math
import os
import random
import subprocess
import sys
import time

import pybullet as p

from HideSeekAgents  import SeekerAgent, HiderAgent
from BipedNNWalker   import BipedNNWalker


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROUND_DURATION = 30.0
CATCH_DIST     = 1.5        # XY metres
STEP_DT        = 1.0 / 30.0
SIM_STEPS_PER_DECISION = 8
MAP_LIMIT      = 9.0        # soft arena boundary (metres from centre)

SEEKER_COLOUR = [1.0, 0.3, 0.0, 1.0]   # orange
HIDER_COLOUR  = [0.0, 0.8, 1.0, 1.0]   # cyan

SPAWN_Z = 0.6   # spawn height above ground (biped torso is ~1.4m off base)

SEEKER_START = [0.0, 0.0, SPAWN_Z]

HIDER_STARTS = [
    [ 7.0,  7.0, SPAWN_Z],
    [-7.0,  7.0, SPAWN_Z],
    [ 7.0, -7.0, SPAWN_Z],
    [-7.0, -7.0, SPAWN_Z],
    [ 0.0,  7.5, SPAWN_Z],
    [ 0.0, -7.5, SPAWN_Z],
    [ 7.5,  0.0, SPAWN_Z],
    [-7.5,  0.0, SPAWN_Z],
]

LOS_HIT_COL  = [1.0, 0.0, 0.0, 1.0]
LOS_MISS_COL = [0.0, 1.0, 0.0, 0.3]


# ---------------------------------------------------------------------------

def _clamp_to_map(pos, vx, vy):
    """Redirect velocity away from the arena boundary."""
    x, y = pos[0], pos[1]
    margin = 1.5   # start pushing back this far before the hard limit
    limit  = MAP_LIMIT
    if x >  (limit - margin): vx = min(vx, 0.0)
    if x < -(limit - margin): vx = max(vx, 0.0)
    if y >  (limit - margin): vy = min(vy, 0.0)
    if y < -(limit - margin): vy = max(vy, 0.0)
    return vx, vy


class HideSeekEnv:
    """
    Spawns two BipedCPGWalker robots into the existing PyBullet world and
    runs the hide-and-seek game loop.  The base_env's own humanoid is left
    untouched in the scene.
    """

    def __init__(
        self,
        base_env,
        round_duration: float = ROUND_DURATION,
        seeker_speed: float   = 3.0,
        hider_speed: float    = 2.8,
        **_kwargs,              # absorb any leftover args (e.g. policy_path)
    ):
        self.base_env       = base_env
        self.cid            = base_env.cid
        self.round_duration = round_duration

        # Remove the MJCF humanoid that create_environment() loaded
        for bid in list(base_env.all_bodies):
            try:
                p.removeBody(bid, physicsClientId=self.cid)
            except Exception:
                pass
        base_env.all_bodies = []
        base_env.humanoid   = None

        # Restore the floor (the MJCF humanoid's embedded geom was the only floor)
        import pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.loadURDF("plane.urdf", physicsClientId=self.cid)

        # ---- Game AI (high-level direction decisions) ----
        self.seeker_agent = SeekerAgent(speed=seeker_speed)
        self.hider_agent  = HiderAgent(speed=hider_speed)

        # ---- Locomotion: both agents are BipedNNWalkers ----
        self.seeker_walker = BipedNNWalker(
            self.cid, start_pos=SEEKER_START, colour=SEEKER_COLOUR,
            policy_path="biped_policy/best_model.zip",
        )
        self.hider_walker = BipedNNWalker(
            self.cid, start_pos=HIDER_STARTS[0], colour=HIDER_COLOUR,
            policy_path="biped_policy/best_model.zip",
        )

        self.seeker_id = self.seeker_walker.biped_id
        self.hider_id  = self.hider_walker.biped_id

        # Scores
        self.seeker_score = 0
        self.hider_score  = 0
        self.round_number = 0

        # Launch background NN training if no policy exists yet
        self._train_proc = None
        policy_file = "biped_policy/best_model.zip"
        if not os.path.exists(policy_file):
            print("[HideSeekEnv] No trained policy found — launching TrainBiped.py in background.")
            print("[HideSeekEnv] Bipeds will use CPG walking until training produces a checkpoint.")
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TrainBiped.py")
            self._train_proc = subprocess.Popen(
                [sys.executable, script, "--steps", "500000", "--envs", "2"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            print(f"[HideSeekEnv] Training PID: {self._train_proc.pid}")
        else:
            print(f"[HideSeekEnv] Policy found at '{policy_file}' — bipeds will use NN locomotion.")

        # Debug overlay IDs
        self._los_line_id  = None
        self._label_seeker = None
        self._label_hider  = None
        self._label_round  = None
        self._label_timer  = None
        self._label_status = None

        self._init_labels()

        print("[HideSeekEnv] Initialised.")
        print(f"  Seeker : BipedCPGWalker  (orange, Q-table game AI)")
        print(f"  Hider  : BipedCPGWalker  (cyan,   potential-field AI)")
        print(f"  Round  : {self.round_duration}s  |  Catch dist: {CATCH_DIST}m")

    # ------------------------------------------------------------------
    # Overlays
    # ------------------------------------------------------------------

    def _init_labels(self):
        c = self.cid
        self._label_seeker = p.addUserDebugText(
            "SEEKER (CPG)  0", [0, 9.5, 4.5],
            textColorRGB=[1.0, 0.4, 0.0], textSize=1.6, physicsClientId=c,
        )
        self._label_hider = p.addUserDebugText(
            "HIDER  (CPG)  0", [0, -9.5, 4.5],
            textColorRGB=[0.0, 0.8, 1.0], textSize=1.6, physicsClientId=c,
        )
        self._label_round = p.addUserDebugText(
            "Round 0", [-9, 0, 5.5],
            textColorRGB=[1, 1, 1], textSize=1.4, physicsClientId=c,
        )
        self._label_timer = p.addUserDebugText(
            "Time: --", [9, 0, 5.5],
            textColorRGB=[1, 1, 0.2], textSize=1.4, physicsClientId=c,
        )
        self._label_status = p.addUserDebugText(
            "", [0, 0, 5.5],
            textColorRGB=[1, 1, 1], textSize=1.6, physicsClientId=c,
        )

    def _update_labels(self, time_left: float, status: str = ""):
        c = self.cid
        p.addUserDebugText(
            f"SEEKER (CPG)  {self.seeker_score}", [0, 9.5, 4.5],
            textColorRGB=[1.0, 0.4, 0.0], textSize=1.6,
            replaceItemUniqueId=self._label_seeker, physicsClientId=c,
        )
        p.addUserDebugText(
            f"HIDER  (CPG)  {self.hider_score}", [0, -9.5, 4.5],
            textColorRGB=[0.0, 0.8, 1.0], textSize=1.6,
            replaceItemUniqueId=self._label_hider, physicsClientId=c,
        )
        p.addUserDebugText(
            f"Round {self.round_number}", [-9, 0, 5.5],
            textColorRGB=[1, 1, 1], textSize=1.4,
            replaceItemUniqueId=self._label_round, physicsClientId=c,
        )
        p.addUserDebugText(
            f"Time: {time_left:.1f}s", [9, 0, 5.5],
            textColorRGB=[1, 1, 0.2], textSize=1.4,
            replaceItemUniqueId=self._label_timer, physicsClientId=c,
        )
        p.addUserDebugText(
            status, [0, 0, 5.5],
            textColorRGB=[1, 1, 1], textSize=1.6,
            replaceItemUniqueId=self._label_status, physicsClientId=c,
        )

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------

    def _has_line_of_sight(self, from_pos, to_pos):
        result = p.rayTest(from_pos, to_pos, physicsClientId=self.cid)
        if not result:
            return False
        hit_body = result[0][0]
        los      = (hit_body == self.hider_id)
        colour   = LOS_HIT_COL if los else LOS_MISS_COL
        if self._los_line_id is None:
            self._los_line_id = p.addUserDebugLine(
                from_pos, to_pos, colour, lineWidth=2, physicsClientId=self.cid,
            )
        else:
            self._los_line_id = p.addUserDebugLine(
                from_pos, to_pos, colour, lineWidth=2,
                replaceItemUniqueId=self._los_line_id,
                physicsClientId=self.cid,
            )
        return los

    # ------------------------------------------------------------------
    # Round management
    # ------------------------------------------------------------------

    def _reset_round(self, hider_start=None):
        if hider_start is None:
            hider_start = random.choice(HIDER_STARTS)
        self.seeker_walker.reset(SEEKER_START)
        self.hider_walker.reset(hider_start)
        self.seeker_agent.reset()
        self.hider_agent.reset()
        for _ in range(20):
            p.stepSimulation(physicsClientId=self.cid)

    # ------------------------------------------------------------------
    # Main game loop
    # ------------------------------------------------------------------

    def run(self, num_rounds: int = 100):
        print(f"\n[HideSeekEnv] Starting {num_rounds} rounds!\n")

        for round_idx in range(1, num_rounds + 1):
            self.round_number = round_idx
            self._reset_round()
            self._update_labels(self.round_duration, f"Round {round_idx} — GO!")
            time.sleep(0.8)

            sim_time    = 0.0
            caught      = False
            last_los    = False    # hider visible on the most recent step?
            step_reward = -0.01

            print(f"  Round {round_idx:>2} | "
                  f"Hider: {self.hider_score}  Seeker: {self.seeker_score}")

            while sim_time < self.round_duration and not caught:

                seeker_pos = self.seeker_walker.get_pos()
                hider_pos  = self.hider_walker.get_pos()

                los = self._has_line_of_sight(seeker_pos, hider_pos)
                last_los = los    # remember for end-of-round scoring
                dist_xy = math.sqrt(
                    (seeker_pos[0] - hider_pos[0]) ** 2 +
                    (seeker_pos[1] - hider_pos[1]) ** 2
                )

                if los and dist_xy < CATCH_DIST:
                    caught = True
                    self.seeker_score += 1
                    msg = (f"SEEKER CATCHES! "
                           f"(Seeker {self.seeker_score} – {self.hider_score} Hider)")
                    self._update_labels(0.0, msg)
                    print(f"    CAUGHT at t={sim_time:.1f}s  dist={dist_xy:.2f}m")
                    time.sleep(1.5)
                    break

                catch_reward = 1.0 if caught else step_reward
                svx, svy = self.seeker_agent.step(
                    seeker_pos,
                    hider_pos if los else None,
                    los,
                    catch_reward,
                )
                svx, svy = _clamp_to_map(seeker_pos, svx, svy)
                self.seeker_walker.step(svx, svy, STEP_DT)

                hvx, hvy = self.hider_agent.step(hider_pos, seeker_pos)
                hvx, hvy = _clamp_to_map(hider_pos, hvx, hvy)
                self.hider_walker.step(hvx, hvy, STEP_DT)

                for _ in range(SIM_STEPS_PER_DECISION):
                    p.stepSimulation(physicsClientId=self.cid)

                sim_time  += STEP_DT
                time_left  = max(0.0, self.round_duration - sim_time)
                self._update_labels(time_left,
                                    "DETECTED!" if los else "searching...")
                time.sleep(STEP_DT)

            if not caught:
                if last_los:
                    # Hider was in the seeker's line-of-sight when the
                    # timer expired -> the seeker spotted it at the buzzer,
                    # so this round goes to the SEEKER, not the hider.
                    self.seeker_score += 1
                    msg = (f"SEEKER SPOTS AT BUZZER! "
                           f"(Seeker {self.seeker_score} – {self.hider_score} Hider)")
                    self._update_labels(0.0, msg)
                    print(f"    Hider was still visible at round end "
                          f"— point to SEEKER")
                    time.sleep(1.5)
                else:
                    self.hider_score += 1
                    msg = (f"HIDER SURVIVES! "
                           f"(Seeker {self.seeker_score} – {self.hider_score} Hider)")
                    self._update_labels(0.0, msg)
                    print(f"    Hider survived the round unseen!")
                    time.sleep(1.5)

        winner = (
            "SEEKER (orange biped)" if self.seeker_score > self.hider_score else
            "HIDER  (cyan biped)"   if self.hider_score  > self.seeker_score else
            "TIE"
        )
        final = (
            f"GAME OVER | {num_rounds} rounds | "
            f"Seeker {self.seeker_score} – {self.hider_score} Hider | "
            f"WINNER: {winner}"
        )
        print(f"\n  {final}")
        self._update_labels(0.0, "GAME OVER")
        p.addUserDebugText(
            final, [0, 0, 7.5],
            textColorRGB=[1, 1, 0], textSize=1.2,
            physicsClientId=self.cid,
        )
        time.sleep(5.0)
        return self.seeker_score, self.hider_score