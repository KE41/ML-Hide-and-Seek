"""
HideSeekAgents.py
-----------------
Two distinct agent strategies for the hide-and-seek game:

  SeekerAgent  — Reinforcement-Learning driven (Q-table with discretised state).
                 Learns across rounds which movement directions lead to finding
                 the hider, exploiting a simple ε-greedy policy.

  HiderAgent   — Rule-based / potential-field controller.
                 Maintains a repulsion field away from the seeker and toward
                 the nearest obstacle that offers line-of-sight cover.

Both agents control a simple sphere body spawned by HideSeekEnv.
Joint-level humanoid control is intentionally not used so the two agents
remain clearly distinct and comparable without interfering with the existing
HumanoidEnv humanoid.
"""

import numpy as np
import pybullet as p
import random
import math


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _angle_toward(source_pos, target_pos):
    """Return yaw angle (radians) from source toward target in XY plane."""
    dx = target_pos[0] - source_pos[0]
    dy = target_pos[1] - source_pos[1]
    return math.atan2(dy, dx)


def _dist_xy(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# ---------------------------------------------------------------------------
# Seeker — ε-greedy Q-table RL agent
# ---------------------------------------------------------------------------

class SeekerAgent:
    """
    Strategy: Reinforcement Learning (Q-table, ε-greedy).

    State  : discretised (dx_bin, dy_bin) relative to hider's last known position.
    Actions: 8 compass directions + stay.

    The seeker does NOT know the hider's exact position — it only gets
    the hider position when line-of-sight returns True (detected).
    Between detections it follows the last known position, then explores.
    """

    # 8 compass directions + stay
    DIRECTIONS = [
        ( 1,  0),   # E
        ( 1,  1),   # NE
        ( 0,  1),   # N
        (-1,  1),   # NW
        (-1,  0),   # W
        (-1, -1),   # SW
        ( 0, -1),   # S
        ( 1, -1),   # SE
        ( 0,  0),   # stay
    ]
    N_ACTIONS = len(DIRECTIONS)

    # Q-table grid bins
    BINS = 8
    HALF = BINS // 2

    def __init__(self, speed: float = 3.0, epsilon: float = 0.25, lr: float = 0.1, gamma: float = 0.9):
        self.speed   = speed
        self.epsilon = epsilon
        self.lr      = lr
        self.gamma   = gamma

        # Q[dx_bin, dy_bin, action] — initialised optimistically
        self.Q = np.zeros((self.BINS, self.BINS, self.N_ACTIONS), dtype=np.float32)

        self._last_known_hider = None   # last detected hider position
        self._prev_state       = None
        self._prev_action      = None
        self._steps_since_seen = 0

    # ------------------------------------------------------------------ helpers

    def _discretise(self, seeker_pos, target_pos):
        dx = target_pos[0] - seeker_pos[0]
        dy = target_pos[1] - seeker_pos[1]
        # bin into [-HALF, HALF) grid
        bx = int(np.clip(dx / 2.0 + self.HALF, 0, self.BINS - 1))
        by = int(np.clip(dy / 2.0 + self.HALF, 0, self.BINS - 1))
        return bx, by

    def _select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.N_ACTIONS)
        return int(np.argmax(self.Q[state[0], state[1]]))

    def _update_q(self, state, action, reward, next_state):
        best_next = float(np.max(self.Q[next_state[0], next_state[1]]))
        td_target = reward + self.gamma * best_next
        td_error  = td_target - self.Q[state[0], state[1], action]
        self.Q[state[0], state[1], action] += self.lr * td_error

    # ------------------------------------------------------------------ public

    def reset(self):
        """Called at the start of each round."""
        self._last_known_hider = None
        self._prev_state       = None
        self._prev_action      = None
        self._steps_since_seen = 0
        # Decay epsilon slightly each round so the agent exploits more over time
        self.epsilon = max(0.05, self.epsilon * 0.97)

    def step(self, seeker_pos, hider_pos_if_visible, detected: bool, reward: float):
        """
        Returns (vx, vy) velocity command in world XY.

        seeker_pos           : (x, y, z) current seeker position
        hider_pos_if_visible : (x, y, z) hider position — only valid when detected=True
        detected             : whether hider is currently in LOS
        reward               : game reward for Q update (+1 catch, -0.01/step)
        """
        if detected:
            self._last_known_hider = hider_pos_if_visible
            self._steps_since_seen = 0
        else:
            self._steps_since_seen += 1

        # When hider is visible, chase directly — no Q-table needed.
        if detected and hider_pos_if_visible is not None:
            ddx = hider_pos_if_visible[0] - seeker_pos[0]
            ddy = hider_pos_if_visible[1] - seeker_pos[1]
            dist = math.sqrt(ddx ** 2 + ddy ** 2) + 1e-6
            return (ddx / dist) * self.speed, (ddy / dist) * self.speed

        # No LOS — use Q-table to decide search direction.
        if self._last_known_hider is None:
            angle = random.uniform(0, 2 * math.pi)
            return math.cos(angle) * self.speed, math.sin(angle) * self.speed

        target = self._last_known_hider
        state  = self._discretise(seeker_pos, target)

        # Q update: reward getting closer to last known position.
        if self._prev_state is not None:
            # Shaped reward: positive if state bin distance shrank
            prev_dist = math.sqrt(
                (self._prev_state[0] - self.HALF) ** 2 +
                (self._prev_state[1] - self.HALF) ** 2
            )
            curr_dist = math.sqrt(
                (state[0] - self.HALF) ** 2 +
                (state[1] - self.HALF) ** 2
            )
            shaped = reward + 0.5 * (prev_dist - curr_dist)
            self._update_q(self._prev_state, self._prev_action, shaped, state)

        action = self._select_action(state)
        self._prev_state  = state
        self._prev_action = action

        dx, dy = self.DIRECTIONS[action]

        # Add jitter after extended search to break loops
        if self._steps_since_seen > 60:
            angle = math.atan2(dy + 1e-9, dx + 1e-9) + random.uniform(-0.8, 0.8)
            dx = math.cos(angle)
            dy = math.sin(angle)

        return dx * self.speed, dy * self.speed


# ---------------------------------------------------------------------------
# Hider — rule-based potential-field controller
# ---------------------------------------------------------------------------

class HiderAgent:
    """
    Strategy: Rule-based potential fields.

    Forces:
      • Repulsion  from seeker (large, distance-squared falloff)
      • Attraction to nearest obstacle centroid (for cover)
      • Wall repulsion (keeps hider away from boundary)
      • Small random noise (breaks symmetry)

    This is a deliberate algorithmic contrast to the Q-learning seeker,
    making the performance comparison meaningful.
    """

    # Static obstacle centroids from Environment.py
    OBSTACLE_POSITIONS = [
        ( 4,  4),   # red cube
        (-4, -4),   # blue cube
        ( 4, -4),   # green cube
        (-4,  4),   # yellow cube
        ( 0,  6),   # cyan rect
        ( 0, -6),   # magenta rect
        ( 6,  0),   # orange rect
        (-6,  0),   # purple rect
        ( 6.2,  6.2), # corner sphere
        (-6.2,  6.2),
        ( 6.2, -6.2),
        (-6.2, -6.2),
        ( 3,    0),
    ]

    WALL_LIMIT = 8.5   # stay inside this radius from centre

    def __init__(self, speed: float = 2.8):
        self.speed = speed
        self._noise_angle = 0.0

    def reset(self):
        self._noise_angle = random.uniform(0, 2 * math.pi)

    def step(self, hider_pos, seeker_pos):
        """
        Returns (vx, vy) velocity command in world XY.
        """
        hx, hy = hider_pos[0], hider_pos[1]
        sx, sy = seeker_pos[0], seeker_pos[1]

        fx, fy = 0.0, 0.0

        # 1. Repulsion from seeker (scales with proximity)
        sdx = hx - sx
        sdy = hy - sy
        d2  = sdx ** 2 + sdy ** 2 + 0.01
        repulsion_strength = 12.0 / d2
        fx += sdx * repulsion_strength
        fy += sdy * repulsion_strength

        # 2. Attraction toward the obstacle that offers best cover
        #    "Best cover" = closest obstacle that is roughly between hider and seeker
        best_score    = -1e9
        best_obs      = None
        seeker_angle  = math.atan2(sy - hy, sx - hx)

        for ox, oy in self.OBSTACLE_POSITIONS:
            obs_angle = math.atan2(oy - hy, ox - hx)
            # angular alignment: how much is the obstacle between hider and seeker?
            angle_diff = abs(math.atan2(
                math.sin(obs_angle - seeker_angle),
                math.cos(obs_angle - seeker_angle)
            ))
            alignment = max(0.0, math.pi - angle_diff)   # pi = perfect cover

            obs_dist = math.sqrt((ox - hx) ** 2 + (oy - hy) ** 2) + 0.1
            score    = alignment / obs_dist              # close + aligned = best

            if score > best_score:
                best_score = score
                best_obs   = (ox, oy)

        if best_obs is not None:
            bx = best_obs[0] - hx
            by = best_obs[1] - hy
            bd = math.sqrt(bx ** 2 + by ** 2) + 0.1
            # attraction decays if already near the obstacle
            attraction = max(0.0, 1.0 - 1.2 / bd)
            fx += (bx / bd) * attraction * 3.0
            fy += (by / bd) * attraction * 3.0

        # 3. Wall repulsion
        wall_margin = self.WALL_LIMIT
        dist_from_centre = math.sqrt(hx ** 2 + hy ** 2)
        if dist_from_centre > wall_margin - 1.5:
            fx -= hx * 2.0
            fy -= hy * 2.0

        # 4. Slow noise rotation to break dead zones
        self._noise_angle += random.uniform(-0.3, 0.3)
        fx += math.cos(self._noise_angle) * 0.3
        fy += math.sin(self._noise_angle) * 0.3

        # Normalise and scale to speed
        magnitude = math.sqrt(fx ** 2 + fy ** 2) + 1e-6
        vx = (fx / magnitude) * self.speed
        vy = (fy / magnitude) * self.speed

        return vx, vy