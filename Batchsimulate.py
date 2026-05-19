"""
BatchSimulate.py
----------------
Run MANY hide-and-seek simulations in parallel, fully headless
(PyBullet DIRECT mode), and collect results for analysis — built to
compare the seeker's tabular Q-learning against the hider's REINFORCE
policy gradient over many independent seeded runs.

Why a separate file
-------------------
The existing game logic in HideSeekEnv / HideSeekAgents / BipedNNWalker
is left untouched.  Two batch-only concerns are handled here, per worker,
without editing those files:

  • HideSeekEnv.run() contains time.sleep() calls for GUI pacing.  Each
    worker is its own process, so we neutralise time.sleep there safely
    (a 20 s round would otherwise take 20 real seconds).
  • HideSeekEnv.__init__ launches TrainBiped.py as a subprocess when no
    biped_policy/best_model.zip exists — that would spawn one trainer
    PER simulation.  We drop a placeholder file to suppress it.  The
    rewritten BipedNNWalker is CPG-based and ignores the policy anyway.

Usage
-----
    python BatchSimulate.py                          # 16 sims, all cores
    python BatchSimulate.py --sims 64 --workers 8
    python BatchSimulate.py --sims 200 --rounds 10 --duration 20
    python BatchSimulate.py --sims 8 --rounds 2 --duration 3   # quick test

Arguments
---------
  --sims         Total number of independent simulations   (default 16)
  --workers      Parallel processes (default: min(sims, CPU count))
  --rounds       Rounds per simulation                      (default 10)
  --duration     Sim-seconds per round                      (default 20)
  --seeker-speed Seeker game-AI target speed (m/s)           (default 1.5)
  --hider-speed  Hider  game-AI target speed (m/s)           (default 1.2)
  --seed-base    Base RNG seed; sim i uses seed-base + i     (default 0)
  --out          Output filename prefix              (default batch_results)

Outputs
-------
  <prefix>.csv           one row per simulation (scores, winner, learning
                         metrics, wall time, seed)
  <prefix>_summary.json  aggregate statistics across all simulations
  A summary table is also printed to the console.

Each simulation runs with its own seed (seed-base + index) so the runs
are reproducible and suitable for variance analysis.
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import statistics
import sys
import time


# ===========================================================================
# Worker — one full headless simulation (runs in its own process)
# ===========================================================================

def run_one_simulation(task):
    """
    Execute a single DIRECT-mode hide-and-seek simulation.

    `task` is a tuple (picklable for the 'spawn' start method):
        (sim_id, rounds, duration, seeker_speed, hider_speed, seed)

    Returns a flat result dict (never raises — failures are captured in
    an "error" field so one bad run can't abort the whole batch).
    """
    sim_id, rounds, duration, seeker_speed, hider_speed, seed = task

    # Heavy imports are kept INSIDE the worker so importing this module
    # (e.g. for unit tests) does not require PyBullet.
    import random
    import numpy as np

    wall_start = time.perf_counter()

    # --- neutralise HideSeekEnv's GUI-pacing sleeps (process-local) ----
    _real_sleep = time.sleep
    time.sleep = lambda *a, **k: None

    # --- reproducible but distinct per simulation ---------------------
    random.seed(seed)
    np.random.seed(seed)

    result = {
        "sim_id": sim_id, "seed": seed, "rounds": rounds,
        "duration": duration, "error": "",
    }

    base = None
    try:
        import pybullet as p
        import pybullet_data

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from HideSeekEnv import HideSeekEnv

        # Minimal stand-in for create_environment()'s base_env.
        # HideSeekEnv clears all_bodies / humanoid and loads its own
        # plane, so this is all it actually needs — and it avoids any
        # dependency on Environment.py for batch runs.
        class _DirectBaseEnv:
            def __init__(self):
                self.cid = p.connect(p.DIRECT)
                p.setAdditionalSearchPath(
                    pybullet_data.getDataPath(),
                    physicsClientId=self.cid)
                p.setGravity(0, 0, -9.8, physicsClientId=self.cid)
                p.setTimeStep(1.0 / 240.0, physicsClientId=self.cid)
                self.all_bodies = []
                self.humanoid = None

        base = _DirectBaseEnv()

        game = HideSeekEnv(
            base_env       = base,
            round_duration = duration,
            seeker_speed   = seeker_speed,
            hider_speed    = hider_speed,
        )

        seeker_score, hider_score = game.run(num_rounds=rounds)

        winner = ("seeker" if seeker_score > hider_score else
                  "hider"  if hider_score > seeker_score else "tie")

        result.update({
            "seeker_score": int(seeker_score),
            "hider_score":  int(hider_score),
            "winner":       winner,
            "seeker_win_rate": round(seeker_score / max(rounds, 1), 4),
        })

        # ---- learning telemetry (defensive: agents may evolve) ----
        sa = getattr(game, "seeker_agent", None)
        if sa is not None:
            result["seeker_epsilon"] = round(
                float(getattr(sa, "epsilon", float("nan"))), 5)
            Q = getattr(sa, "Q", None)
            if Q is not None:
                result["seeker_q_mean"] = round(float(np.mean(Q)), 5)
                result["seeker_q_max"]  = round(float(np.max(Q)), 5)
                result["seeker_q_nonzero"] = int(np.count_nonzero(Q))

        ha = getattr(game, "hider_agent", None)
        learner = getattr(ha, "learner", None) if ha is not None else None
        if learner is not None:
            result["hider_episodes"]   = int(getattr(
                learner, "episode_count", 0))
            result["hider_last_return"] = round(float(getattr(
                learner, "last_return", float("nan"))), 5)
            result["hider_baseline"]   = round(float(getattr(
                learner, "baseline", float("nan"))), 5)
            W = getattr(learner, "W", None)
            if W is not None:
                result["hider_policy_wnorm"] = round(
                    float(np.linalg.norm(W)), 5)

    except Exception as exc:                      # capture, don't crash
        import traceback
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-800:]
    finally:
        if base is not None:
            try:
                import pybullet as p
                p.disconnect(base.cid)
            except Exception:
                pass
        time.sleep = _real_sleep              # restore (tidiness)

    result["wall_time_s"] = round(time.perf_counter() - wall_start, 2)
    return result


# ===========================================================================
# Aggregation + output (pure functions — unit-testable without PyBullet)
# ===========================================================================

def aggregate(results):
    """Compute summary statistics over a list of result dicts."""
    ok  = [r for r in results if not r.get("error")]
    bad = [r for r in results if r.get("error")]

    def col(key):
        return [r[key] for r in ok if isinstance(r.get(key), (int, float))]

    def stats(vals):
        if not vals:
            return {"mean": None, "std": None, "min": None, "max": None}
        return {
            "mean": round(statistics.fmean(vals), 4),
            "std":  round(statistics.pstdev(vals), 4)
                    if len(vals) > 1 else 0.0,
            "min":  round(min(vals), 4),
            "max":  round(max(vals), 4),
        }

    winners = [r.get("winner") for r in ok]
    n = len(ok)
    summary = {
        "simulations_total":   len(results),
        "simulations_ok":      n,
        "simulations_failed":  len(bad),
        "seeker_wins":  winners.count("seeker"),
        "hider_wins":   winners.count("hider"),
        "ties":         winners.count("tie"),
        "seeker_win_pct": round(100.0 * winners.count("seeker") / n, 2)
                          if n else None,
        "hider_win_pct":  round(100.0 * winners.count("hider") / n, 2)
                          if n else None,
        "seeker_score":      stats(col("seeker_score")),
        "hider_score":       stats(col("hider_score")),
        "seeker_win_rate":   stats(col("seeker_win_rate")),
        "hider_last_return": stats(col("hider_last_return")),
        "hider_baseline":    stats(col("hider_baseline")),
        "seeker_epsilon":    stats(col("seeker_epsilon")),
        "wall_time_s":       stats(col("wall_time_s")),
    }
    if bad:
        summary["failures"] = [
            {"sim_id": r.get("sim_id"), "error": r.get("error")}
            for r in bad
        ]
    return summary


def write_outputs(results, summary, prefix):
    """Write per-sim CSV and aggregate JSON; return their paths."""
    # union of all keys (sims may carry different optional metrics)
    keys = []
    for r in results:
        for k in r:
            if k not in keys and k != "traceback":
                keys.append(k)

    csv_path = f"{prefix}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in sorted(results, key=lambda x: x.get("sim_id", 0)):
            w.writerow(r)

    json_path = f"{prefix}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return csv_path, json_path


def print_summary(summary):
    s = summary
    print("\n" + "=" * 64)
    print("  BATCH RESULTS")
    print("=" * 64)
    print(f"  Simulations : {s['simulations_ok']} ok / "
          f"{s['simulations_total']} total"
          + (f"  ({s['simulations_failed']} FAILED)"
             if s['simulations_failed'] else ""))
    print(f"  Seeker wins : {s['seeker_wins']:>4}  "
          f"({s['seeker_win_pct']}%)")
    print(f"  Hider  wins : {s['hider_wins']:>4}  "
          f"({s['hider_win_pct']}%)")
    print(f"  Ties        : {s['ties']:>4}")
    print("-" * 64)

    def line(name, st):
        if st["mean"] is None:
            print(f"  {name:<20} (no data)")
        else:
            print(f"  {name:<20} mean={st['mean']:<9} "
                  f"std={st['std']:<9} "
                  f"[{st['min']} .. {st['max']}]")

    line("Seeker score",      s["seeker_score"])
    line("Hider score",       s["hider_score"])
    line("Seeker win rate",   s["seeker_win_rate"])
    line("Hider REINFORCE G", s["hider_last_return"])
    line("Hider baseline",    s["hider_baseline"])
    line("Seeker epsilon",    s["seeker_epsilon"])
    line("Wall time / sim s", s["wall_time_s"])
    print("=" * 64)


# ===========================================================================
# Orchestrator
# ===========================================================================

def _suppress_autotrain():
    """
    Drop a placeholder biped_policy/best_model.zip so HideSeekEnv does
    NOT launch a TrainBiped.py subprocess per simulation.  Only created
    if absent — a real policy is never overwritten.
    """
    pol = os.path.join("biped_policy", "best_model.zip")
    if not os.path.exists(pol):
        os.makedirs("biped_policy", exist_ok=True)
        open(pol, "wb").close()
        print(f"[batch] created placeholder {pol} to suppress per-sim "
              f"auto-training (CPG walker ignores it).")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Parallel headless hide-and-seek batch simulator "
                    "(PyBullet DIRECT mode).")
    parser.add_argument("--sims",         type=int,   default=32)
    parser.add_argument("--workers",      type=int,   default=0,
                        help="Parallel processes (0 = min(sims, CPUs))")
    parser.add_argument("--rounds",       type=int,   default=10)
    parser.add_argument("--duration",     type=float, default=20.0)
    parser.add_argument("--seeker-speed", type=float, default=1.5)
    parser.add_argument("--hider-speed",  type=float, default=1.2)
    parser.add_argument("--seed-base",    type=int,   default=0)
    parser.add_argument("--out",          type=str,
                        default="batch_results")
    args = parser.parse_args(argv)

    cpu = os.cpu_count() or 1
    workers = (args.workers if args.workers > 0
               else min(args.sims, cpu))
    workers = max(1, min(workers, args.sims))

    print("=" * 64)
    print("  Hide-and-Seek BATCH  (headless / DIRECT mode)")
    print(f"  sims={args.sims}  workers={workers}  "
          f"rounds={args.rounds}  duration={args.duration}s")
    print(f"  seeker_speed={args.seeker_speed}  "
          f"hider_speed={args.hider_speed}  seed_base={args.seed_base}")
    print("=" * 64)

    _suppress_autotrain()

    tasks = [
        (i, args.rounds, args.duration,
         args.seeker_speed, args.hider_speed, args.seed_base + i)
        for i in range(args.sims)
    ]

    results = []
    t0 = time.perf_counter()

    if workers == 1:
        # serial path (still DIRECT) — easiest to debug
        for k, t in enumerate(tasks, 1):
            r = run_one_simulation(t)
            results.append(r)
            tag = "OK " if not r.get("error") else "ERR"
            print(f"  [{k:>4}/{args.sims}] {tag} sim {r['sim_id']:>4}  "
                  f"{r.get('winner','-'):>6}  "
                  f"{r.get('wall_time_s','?')}s")
    else:
        # 'spawn' avoids inheriting any PyBullet state into workers and
        # works the same on Windows and Linux.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            done = 0
            for r in pool.imap_unordered(run_one_simulation, tasks):
                results.append(r)
                done += 1
                tag = "OK " if not r.get("error") else "ERR"
                print(f"  [{done:>4}/{args.sims}] {tag} "
                      f"sim {r['sim_id']:>4}  "
                      f"{r.get('winner','-'):>6}  "
                      f"{r.get('wall_time_s','?')}s",
                      flush=True)

    elapsed = time.perf_counter() - t0
    summary = aggregate(results)
    summary["batch_wall_time_s"] = round(elapsed, 2)
    summary["workers"] = workers

    csv_path, json_path = write_outputs(results, summary, args.out)
    print_summary(summary)
    print(f"\n  Total wall time : {elapsed:.1f}s "
          f"({elapsed / max(args.sims,1):.2f}s/sim effective)")
    print(f"  Per-sim CSV     : {os.path.abspath(csv_path)}")
    print(f"  Summary JSON    : {os.path.abspath(json_path)}")
    return summary


if __name__ == "__main__":
    main()