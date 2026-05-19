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

# ---------------------------------------------------------------------------
# Policy-Gradient learner (REINFORCE) — deliberately a DIFFERENT RL family
# from the seeker's tabular Q-learning, so the two are comparable.
# ---------------------------------------------------------------------------

class PolicyGradientLearner:
    """
    Monte-Carlo REINFORCE with a linear-softmax policy and a running
    baseline.  Contrast with SeekerAgent's Q-table:

      • policy-based, not value-based   (learns π(a|s) directly)
      • on-policy                       (trains on its own samples)
      • Monte-Carlo                     (whole-episode return, no bootstrap)
      • stochastic softmax policy       (exploration via entropy, not ε)
      • episodic update                 (one gradient step per round)

    Policy:  logits = W · φ(s),  π = softmax(logits)
    Update :  ΔW = lr · Σ_t (G_t − b) · (onehot(a_t) − π_t) ⊗ φ_t
              with G_t the discounted return and b an EMA baseline
              (variance reduction), plus a small entropy bonus so the
              policy keeps exploring instead of collapsing early.
    """

    def __init__(self, n_features: int, n_actions: int,
                 lr: float = 0.02, gamma: float = 0.95,
                 entropy_beta: float = 0.01, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W            = rng.normal(0.0, 0.01,
                                       size=(n_actions, n_features))
        self.n_actions    = n_actions
        self.lr           = lr
        self.gamma        = gamma
        self.entropy_beta = entropy_beta
        self.baseline     = 0.0           # EMA of mean episode return
        self.episode_count = 0
        self.last_return   = 0.0
        self._traj         = []           # list of (phi, action, prob, r)

    # ---- policy ------------------------------------------------------
    def _softmax(self, logits):
        z = logits - np.max(logits)
        e = np.exp(np.clip(z, -50.0, 50.0))
        return e / (np.sum(e) + 1e-12)

    def act(self, phi):
        """Sample an action from the current stochastic policy."""
        phi   = np.asarray(phi, dtype=np.float64)
        probs = self._softmax(self.W @ phi)
        a     = int(np.random.choice(self.n_actions, p=probs))
        return a, probs

    def record(self, phi, action, probs, reward):
        self._traj.append(
            (np.asarray(phi, dtype=np.float64), action,
             np.asarray(probs, dtype=np.float64), float(reward)))

    # ---- episodic REINFORCE update ----------------------------------
    def finish_episode(self):
        """Run one policy-gradient update over the finished trajectory."""
        if not self._traj:
            return
        T = len(self._traj)

        # discounted returns G_t
        returns = np.zeros(T)
        g = 0.0
        for t in reversed(range(T)):
            g = self._traj[t][3] + self.gamma * g
            returns[t] = g

        mean_G = float(np.mean(returns))
        self.last_return = float(returns[0])
        # EMA baseline for variance reduction
        self.baseline += 0.1 * (mean_G - self.baseline)

        gradW = np.zeros_like(self.W)
        for t in range(T):
            phi, a, probs, _ = self._traj[t]
            adv      = returns[t] - self.baseline
            onehot   = np.zeros(self.n_actions)
            onehot[a] = 1.0
            # ∇ log π(a|s) for linear-softmax = (onehot − π) ⊗ φ
            gradW += adv * np.outer(onehot - probs, phi)
            # entropy bonus keeps the policy from collapsing too soon
            ent_grad = -(probs * (np.log(probs + 1e-12) + 1.0))
            gradW += self.entropy_beta * np.outer(ent_grad, phi)

        gradW /= T
        # gradient clipping for stability
        norm = np.linalg.norm(gradW)
        if norm > 5.0:
            gradW *= 5.0 / norm
        self.W += self.lr * gradW

        self.episode_count += 1
        self._traj.clear()


class HiderAgent:
    """
    Strategy: Rule-based potential fields  +  a REINFORCE policy that
    learns WHICH field-derived strategy to use.

    Layer 1 (unchanged, rule-based potential fields):
      • Repulsion  from seeker (large, distance-squared falloff)
      • Attraction to nearest obstacle centroid (for cover)
      • Wall repulsion (keeps hider away from boundary)
      • Small random noise (breaks symmetry)

    Layer 2 (learned, Monte-Carlo policy gradient — REINFORCE):
      The field math above yields several candidate headings (follow the
      blended field, flee straight away, commit to cover, or juke
      laterally).  A linear-softmax policy, trained by REINFORCE at the
      end of every round on the hider's own survival reward, learns the
      state-dependent probability of each.  Action 0 reproduces the
      original pure rule-based behaviour, so learning can only add to it.

    This is a deliberate algorithmic contrast to the seeker: the seeker
    learns by VALUE-based tabular Q-learning; the hider learns by
    POLICY-based episodic policy gradient — different RL families on a
    comparable decision, which is the point of the comparison.
    """

    # ---- learned-strategy action set (all derived from the field) ----
    #   0 : follow the blended potential field   (original behaviour)
    #   1 : flee straight away from the seeker    (aggressive evasion)
    #   2 : commit to the best cover obstacle     (break line of sight)
    #   3 : juke laterally across the seeker line (dodge pursuit)
    N_STRATEGIES = 4
    N_FEATURES   = 6   # [bias, d_seek, sinθ, cosθ, cover, wall]

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

    def __init__(self, speed: float = 2.8, learn: bool = True):
        self.speed = speed
        self._noise_angle = 0.0

        # Layer-2 learner (REINFORCE). learn=False ⇒ pure rule-based,
        # which is handy for an A/B baseline against the learning hider.
        self.learn   = learn
        self.learner = PolicyGradientLearner(
            n_features=self.N_FEATURES, n_actions=self.N_STRATEGIES)
        self._pending = None   # (phi, action, probs) awaiting its reward
        self._prev_dist = None # seeker distance last step (reward shaping)

    def reset(self):
        # End-of-round: run the Monte-Carlo policy-gradient update on the
        # round that just finished, then start a fresh episode.
        if self.learn:
            self.learner.finish_episode()
        self._pending   = None
        self._prev_dist = None
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

        # Normalise and scale to speed  (this is the rule-based heading)
        magnitude = math.sqrt(fx ** 2 + fy ** 2) + 1e-6
        vx = (fx / magnitude) * self.speed
        vy = (fy / magnitude) * self.speed

        # ------------------------------------------------------------------
        # Layer 2 — REINFORCE policy decides which strategy heading to use.
        # All candidate headings are derived from quantities the rule-based
        # field already computed, so nothing about Layer 1 is discarded.
        # ------------------------------------------------------------------
        if not self.learn:
            return vx, vy

        sdx, sdy = hx - sx, hy - sy
        d_seek   = math.sqrt(sdx * sdx + sdy * sdy) + 1e-6

        # ---- credit the PREVIOUS step's action with this step's reward ----
        # Hider reward (self-contained, survival-oriented): stay far from
        # the seeker, gain a bonus while good cover is available, small
        # penalty near the wall, tiny step cost.
        if self._pending is not None:
            wall = math.sqrt(hx * hx + hy * hy)
            r = (min(d_seek / 6.0, 1.0)
                 + 0.4 * (1.0 if best_score > 0.6 else 0.0)
                 - 0.3 * (1.0 if wall > self.WALL_LIMIT - 1.0 else 0.0)
                 - 0.01)
            if self._prev_dist is not None and d_seek < self._prev_dist:
                r -= 0.1                       # discourage closing the gap
            phi_p, a_p, pr_p = self._pending
            self.learner.record(phi_p, a_p, pr_p, r)
        self._prev_dist = d_seek

        # ---- build the policy state features φ(s) ----
        seeker_bearing = math.atan2(sdy, sdx)        # away-from-seeker dir
        wall_d = math.sqrt(hx * hx + hy * hy) / self.WALL_LIMIT
        phi = np.array([
            1.0,                                     # bias
            min(d_seek / 8.0, 1.5),                  # distance to seeker
            math.sin(seeker_bearing),
            math.cos(seeker_bearing),
            1.0 if best_score > 0.6 else 0.0,        # is good cover available
            min(wall_d, 1.5),                        # proximity to wall
        ], dtype=np.float64)

        action, probs = self.learner.act(phi)
        self._pending = (phi, action, probs)

        # ---- candidate headings (all from existing field quantities) ----
        field_ang = math.atan2(vy, vx)               # action 0 (rule-based)
        away_ang  = seeker_bearing                    # action 1
        if best_obs is not None:                      # action 2
            cover_ang = math.atan2(best_obs[1] - hy, best_obs[0] - hx)
        else:
            cover_ang = field_ang
        juke_ang  = away_ang + math.pi / 2.0          # action 3

        chosen = (field_ang, away_ang, cover_ang, juke_ang)[action]
        vx = math.cos(chosen) * self.speed
        vy = math.sin(chosen) * self.speed
        return vx, vy