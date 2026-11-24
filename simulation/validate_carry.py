"""Validate transport across colors/directions and exercise failure checks."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics

import numpy as np

from carry_env import ROOT, CarryEnv
from run_carry import run_episode


def failure_checks() -> dict:
    outcomes = {}
    for scenario in ("stationary", "release", "collision"):
        env = CarryEnv(max_seconds=5.0)
        try:
            env.reset()
            if scenario == "release":
                for finger, target in (("left", 0.04875), ("right", -0.04875)):
                    actuator = env.model.actuator(f"right_arm_gripper_{finger}_joint_servo").id
                    env.hold_targets[actuator] = target
            while not env.done:
                _, info = env.step([0.18, 0.0] if scenario == "collision" else [0.0, 0.0])
            outcomes[scenario] = info["failure_reason"]
        finally:
            env.close()
    expected = {"stationary": "timeout", "release": "lost_grasp", "collision": "collision"}
    if outcomes != expected:
        raise AssertionError(f"Failure checks differ: expected {expected}, got {outcomes}")
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "generated/validation" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--teacher", choices=("smooth", "stop-turn"), default="smooth")
    parser.add_argument("--compare", action="store_true", help="Pair the original and smooth teachers.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--colors", nargs="+", choices=("blue", "red", "green", "yellow"),
                        default=("blue", "red", "green", "yellow"))
    parser.add_argument("--start-jitter", type=float, default=0.03)
    parser.add_argument("--heading-jitter-deg", type=float, default=0.0)
    args = parser.parse_args()
    if args.seeds < 1 or not np.isfinite([args.start_jitter, args.heading_jitter_deg]).all() or min(args.start_jitter, args.heading_jitter_deg) < 0:
        parser.error("seeds must be positive; jitter ranges must be finite and nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    env = CarryEnv()
    results = []
    try:
        profiles = ("stop-turn", "smooth") if args.compare else (args.teacher,)
        for profile in profiles:
            for goal in ("A", "B"):
                for color in args.colors:
                    for seed in range(args.seed, args.seed + args.seeds):
                        result = run_episode(env, args.output / profile / f"{goal}_{color}_{seed}",
                                             color=color, goal=goal, seed=seed,
                                             jitter=args.start_jitter, record=False,
                                             teacher_profile=profile,
                                             heading_jitter=float(np.deg2rad(args.heading_jitter_deg)))
                        result["teacher_profile"] = profile
                        results.append(result)
                        print(json.dumps(result), flush=True)
    finally:
        env.close()
    checks = failure_checks()
    summary = {"episodes": results, "successes": sum(item["success"] for item in results),
               "total": len(results), "failure_checks": checks}
    summary["profiles"] = {}
    for profile in profiles:
        rows = [row for row in results if row["teacher_profile"] == profile]
        successful = [row for row in rows if row["success"]]
        summary["profiles"][profile] = {
            "successes": len(successful), "total": len(rows),
            "mean_seconds_on_success": statistics.mean(row["seconds"] for row in successful) if successful else None,
            "mean_position_error_m_on_success": statistics.mean(row["position_error_m"] for row in successful) if successful else None,
            "worst_slip_m": max(row["worst_slip_m"] for row in rows),
        }
    if args.compare:
        pairs = {}
        for row in results:
            pairs.setdefault((row["goal"], row["color"], row["seed"]), {})[row["teacher_profile"]] = row
        matched = [pair for pair in pairs.values() if all(row["success"] for row in pair.values())]
        summary["successful_pairs"] = len(matched)
        if matched:
            before = statistics.mean(pair["stop-turn"]["seconds"] for pair in matched)
            after = statistics.mean(pair["smooth"]["seconds"] for pair in matched)
            summary["paired_mean_time_reduction_percent"] = (before - after) / before * 100
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "episodes"}), flush=True)
    if not all(item["success"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
