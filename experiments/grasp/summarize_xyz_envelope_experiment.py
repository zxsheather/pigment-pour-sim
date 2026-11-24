"""Summarize the completed normalization-only comparison and takeover pilot."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize_reactive_experiment import annotate_counts, orientation_error, rollout_diagnostics


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("experiments/grasp/runs"))
    args = parser.parse_args()
    runs = args.runs
    output = runs / "xyz_envelope_eval_v5"
    experiment = runs / "train_xyz_envelope_3000_v5"
    final = read(output / "final_summary.json")
    assert final["completed"] and final["all_trials_valid"]
    trace = [json.loads(line) for line in (experiment / "learning_curve.jsonl").read_text().splitlines()]
    assert [r["completed_steps"] for r in trace] == list(range(1, 3001))
    losses = np.asarray([r["loss"] for r in trace])
    assert np.isfinite(losses).all() and all(np.isfinite(r["grad_norm"]) for r in trace)
    trained = read(experiment / "report.json")
    assert trained["completed"] and trained["fp32_master_weights"]
    bounds = read(runs / "xyz_envelope_v5/envelope.json")
    assert hashlib.sha256((experiment / "norm_stats.json").read_bytes()).hexdigest() == bounds["updated_statistics_sha256"]
    fixed = read(runs / "xyz_envelope_v5/fixed_ranges.json")
    full = read(runs / "xyz_envelope_v5/full_horizon_ranges.json")
    assert fixed["overall"]["max_position_clip_error_mm"] == full["overall"]["max_position_clip_error_mm"] == 0
    compared = read(output / "comparison.json")
    for row in compared:
        path = (runs / "reactive_eval_v4/openloop_reactive_003000/report.json"
                if row["label"] == "original_003000" else output / ("openloop_" + row["label"]) / "report.json")
        opened = read(path)
        row["validation_execute8_orientation_deg"] = orientation_error(
            [r for r in opened["records"] if r["split"] == "unused_internal_episode"])

    labels = ["original_003000", "xyz_003000"]
    reports = {label: read(output / ("final_" + label) / "report.json") for label in labels}
    keys = [{(c["batch"], c["episode"]) for c in report["cases"]} for report in reports.values()]
    assert keys[0] == keys[1] and len(keys[0]) == 14
    assert all(r["completed"] and all(c["trial_valid"] for c in r["cases"]) for r in reports.values())
    pilots = {}
    for name in ["reactive", "axis", "endon", "endon_pilot", "endon_clearance_pilot", "precontact"]:
        path = runs / ("takeover_" + name + "_v5") / "manifest.json"
        if not path.exists():
            continue
        manifest = read(path)
        assert manifest["source_split"] == "train_episode only"
        assert not manifest["used_for_xyz_normalization_training"]
        entries = manifest["episodes"]
        pilots[name] = {"attempts": len(entries), "accepted": sum(e["accepted"] for e in entries),
            "invalid": sum(not e["numerically_valid"] for e in entries),
            "upright": {"attempts": sum(e.get("initial_bottle_tilt_deg", 0) <= 35 for e in entries),
                        "accepted": sum(e["accepted"] and e.get("initial_bottle_tilt_deg", 0) <= 35 for e in entries)},
            "fallen": {"attempts": sum(e.get("initial_bottle_tilt_deg", 0) > 35 for e in entries),
                       "accepted": sum(e["accepted"] and e.get("initial_bottle_tilt_deg", 0) > 35 for e in entries)},
            "source_cases_reused_for_teacher_development": True}
    data = {}
    for name in ["reactive_train_v4", "xyz_holdout_v5", "xyz_recovery_holdout_v5"]:
        m = read(runs / name / "manifest.json")
        data[name] = {"attempts": len(m["episodes"]), "accepted": m["accepted_episodes"]}
    summary = {"completed": True, "training_updates": 3000,
        "loss_first50_mean": float(losses[:50].mean()), "loss_last50_mean": float(losses[-50:].mean()),
        "loss_not_directly_comparable_across_changed_normalizers": True,
        "training_minutes_excluding_initialization": (trace[-1]["wall_time"] - trace[0]["wall_time"]) / 60,
        "peak_allocated_gb": trained["peak_allocated_gb"], "peak_reserved_gb": trained["peak_reserved_gb"],
        "envelope": bounds, "full_horizon_clipping": full["overall"], "data": data,
        "comparison": compared, "final_evaluation": final, "teacher_pilot": pilots,
        "teacher_pilot_not_model_success_or_used_in_this_training": True,
        "rollout_diagnostics": {label: rollout_diagnostics(output / ("final_" + label)) for label in labels},
        "limits": ["one inference seed and a reconstructed simulator",
                   "recovery tasks are separate from complete tabletop pickup",
                   "takeover pilot uses two training positions; no independent teacher generalization claim",
                   "no real robot, pouring, liquid control or complete task tested"]}
    takeover_audit = read(runs / "xyz_envelope_v5/takeover_audit.json")
    assert takeover_audit["passed"]
    summary["corrective_dataset_audit"] = takeover_audit
    names = ["Original XYZ\n3,000 updates", "Expanded XYZ\n1,000 updates", "Expanded XYZ\n3,000 updates"]
    colors = ["#a0acb9", "#4b89ae", "#23816b"]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#f7f8f1", "axes.facecolor": "#fffef8"}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        fig.suptitle("Wall-OSS-0.5 | Matched XYZ-normalization comparison", fontsize=17, fontweight="bold")
        ax = axes[0, 0]
        ax.plot(np.arange(1, 3001), losses, color="#8fb7ab", alpha=.25, linewidth=.6)
        ax.plot(np.arange(50, 3001), np.convolve(losses, np.ones(50) / 50, mode="valid"), color=colors[2])
        ax.set(title="New training loss (normalizer-specific scale)", xlabel="Optimizer updates",
               ylabel="Flow-matching loss", yscale="log")
        ax = axes[0, 1]
        values = [r["openloop"]["unused_internal_episode"]["execute8_xyz_cm"] for r in compared]
        bars = ax.bar(names, values, color=colors)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + .05, f"{value:.2f}", ha="center")
        ax.set(title="Expert observations: same 8 validation episodes", ylabel="First-8-action XYZ error (cm)")
        ax = axes[1, 0]
        metrics = [reports[label]["aggregates"]["external_holdout"] for label in labels]
        bars = ax.bar(["Original XYZ\n3,000 updates", "Expanded XYZ\n3,000 updates"],
                      [m["successes"] / m["trials"] for m in metrics], color=[colors[0], colors[2]])
        annotate_counts(ax, bars, metrics)
        ax.set(title="Primary: 8 new tabletop pickup positions", ylabel="Stable pickup fraction", ylim=(0, 1.17))
        ax = axes[1, 1]
        groups = ["hand_offset", "grasp_miss", "slip"]
        for index, label in enumerate(labels):
            metrics = [reports[label]["aggregates"]["external_recovery_" + group] for group in groups]
            bars = ax.bar(np.arange(3) + (index - .5) * .36,
                [m["successes"] / m["trials"] for m in metrics], width=.34,
                color=[colors[0], colors[2]][index], label=["Original XYZ", "Expanded XYZ"][index])
            annotate_counts(ax, bars, metrics)
        ax.set_xticks(np.arange(3), ["Hand offset", "Off-center grip", "Slip recovery"])
        ax.set(title="Secondary: 2 new recovery cases per condition", ylabel="Stable recovery fraction", ylim=(0, 1.17))
        ax.legend(fontsize=8)
        for ax in axes.flat:
            ax.grid(axis="y", alpha=.15)
        fig.savefig(output / "xyz_envelope_comparison.png", dpi=180)
        plt.close(fig)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ["completed", "training_updates", "training_minutes_excluding_initialization",
        "peak_allocated_gb", "full_horizon_clipping", "teacher_pilot"]}))


if __name__ == "__main__":
    main()
