"""Summarize corrective-data training from completed, paired evaluations.

The external tabletop pickup set is the primary result. Recovery tasks are
reported separately, including slip setups that already used a scripted lift.
"""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


def read(path):
    return json.loads(path.read_text())


def orientation_error(records):
    errors = []
    for record in records:
        prediction = np.asarray(record["prediction"])
        target = np.asarray(record["target"])
        count = min(8, len(target))
        delta = (Rotation.from_euler("xyz", prediction[:count, 3:6])
                 * Rotation.from_euler("xyz", target[:count, 3:6]).inv())
        errors.append(float(np.degrees(delta.magnitude()).mean()))
    return float(np.mean(errors))


def annotate_counts(ax, bars, metrics):
    for bar, metric in zip(bars, metrics):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + .025,
                f"{metric['successes']}/{metric['trials']}",
                ha="center", fontsize=10)


def rollout_diagnostics(folder):
    report = read(folder / "report.json")
    records = []
    for case in report["cases"]:
        telemetry = read(folder / case["directory"] / "telemetry.json")
        poses = np.asarray([row["right_ee"] for row in telemetry])
        commands = np.asarray([row["requested_pose"] for row in telemetry])
        records.append({
            "split": case["split"], "episode": case["episode"],
            "directory": case["directory"], "success": case["success"],
            "min_grasp_distance_cm": case["min_grasp_distance_m"] * 100,
            "max_bottle_bottom_lift_cm": case["max_bottle_bottom_lift_m"] * 100,
            "max_held_seconds": case["max_held_seconds"],
            "bilateral_contact_steps": case["bilateral_contact_steps"],
            "max_tool_angle_from_teacher_deg": float(np.degrees(
                Rotation.from_euler("xyz", poses[:, 3:6]).magnitude()).max()),
            "max_command_angle_from_teacher_deg": float(np.degrees(
                Rotation.from_euler("xyz", commands[:, 3:6]).magnitude()).max())})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=Path("experiments/grasp/runs"))
    parser.add_argument("--orientation-diagnostic", type=Path,
                        help="Optional assisted validation run, never used for checkpoint selection")
    args = parser.parse_args()
    final = read(args.evaluation / "final_summary.json")
    assert final["completed"] and final["all_trials_valid"]
    trained = read(args.experiment / "report.json")
    assert trained["completed"] and trained["fp32_master_weights"]
    comparison = read(args.evaluation / "comparison.json")
    trace = [json.loads(line) for line in
             (args.experiment / "learning_curve.jsonl").read_text().splitlines()]
    steps = np.asarray([r["completed_steps"] for r in trace])
    losses = np.asarray([r["loss"] for r in trace])
    assert np.array_equal(steps, np.arange(1, 3001))
    assert np.isfinite(losses).all()
    assert all(np.isfinite(r["grad_norm"]) for r in trace)

    for item in comparison:
        opened = read(args.evaluation / f"openloop_{item['label']}" / "report.json")
        item["validation_execute8_orientation_deg"] = orientation_error(
            [r for r in opened["records"] if r["split"] == "unused_internal_episode"])

    paired = {}
    for label in final["paired_external"]:
        report = read(args.evaluation / f"final_{label}" / "report.json")
        assert report["completed"] and all(c["trial_valid"] for c in report["cases"])
        paired[label] = report
    selected = final["selection"]["candidate"]["label"]
    labels = ["reference_v3", selected]
    primary_keys = [{(c["batch"], c["episode"]) for c in paired[label]["cases"]
                     if c["split"] == "external_holdout"} for label in labels]
    assert len(primary_keys[0]) == 8 and primary_keys[0] == primary_keys[1]

    manifests = {}
    training_pose_angles = []
    for group in ["train", "validation", "holdout", "recovery_holdout"]:
        manifest = read(args.runs / f"reactive_{group}_v4" / "manifest.json")
        accepted = [e for e in manifest["episodes"] if e["accepted"]]
        if group == "train":
            for entry in accepted:
                folder = args.runs / "reactive_train_v4" / f"episode_{entry['episode']:04d}"
                with np.load(folder / "trajectory.npz") as trajectory:
                    training_pose_angles.extend(np.degrees(Rotation.from_euler(
                        "xyz", trajectory["state"][:, 10:13]).magnitude()).tolist())
        manifests[group] = {
            "attempted": len(manifest["episodes"]), "accepted": len(accepted),
            "frames": sum(e["frames"] for e in accepted),
            "teacher_source_sha256": manifest["teacher_source_sha256"],
            "by_kind": {kind: sum(e["kind"] == kind for e in accepted)
                        for kind in sorted({e["kind"] for e in accepted})}}
    summary = {
        "completed": True, "training_steps": int(steps[-1]),
        "trainable_parameters": 399754240,
        "peak_allocated_gb": trained["peak_allocated_gb"],
        "peak_reserved_gb": trained["peak_reserved_gb"],
        "loss_first50_mean": float(losses[:50].mean()),
        "loss_last50_mean": float(losses[-50:].mean()),
        "wall_training_seconds_excluding_initialization":
            float(trace[-1]["wall_time"] - trace[0]["wall_time"]),
        "normalization_sha256": hashlib.sha256(
            (args.experiment / "norm_stats.json").read_bytes()).hexdigest(),
        "data": manifests, "checkpoints": comparison,
        "final_evaluation": final,
        "training_observed_tool_angle_max_deg": max(training_pose_angles),
        "rollout_diagnostics": {
            "closedloop_" + row["label"]: rollout_diagnostics(
                args.evaluation / ("closedloop_" + row["label"])) for row in comparison},
        "primary_task": "8 unseen tabletop pickups; no scripted grasp after initialization",
        "recovery_tasks_are_separate": True,
        "claim_limits": ["single inference seed and one reconstructed simulator",
                         "teacher, coverage, normalization and update count changed together",
                         "no real robot or complete pouring task evaluated"]}
    summary["rollout_diagnostics"].update({
        "final_" + label: rollout_diagnostics(args.evaluation / ("final_" + label))
        for label in labels})
    clipping_file = args.runs / "reactive_audit_v4/action_ranges.json"
    if clipping_file.exists():
        clipping = read(clipping_file)
        assert clipping["official_collator_roundtrip_verified"]
        assert clipping["norm_stats_sha256"] == summary["normalization_sha256"]
        summary["action_target_clipping"] = {k: clipping[k] for k in
            ["overall", "by_kind", "by_observed_phase", "scope"]}
    if args.orientation_diagnostic:
        assisted = read(args.orientation_diagnostic / "report.json")
        unassisted_folder = args.evaluation / "closedloop_reactive_001000"
        unassisted = read(unassisted_folder / "report.json")
        assert assisted["completed"] and all(c["trial_valid"] for c in assisted["cases"])
        assert assisted["control_mode"] == "model_position_gripper_fixed_orientation"
        assert assisted["fixed_tool_orientation_rpy"] == [0., 0., 0.]
        assert not assisted["external_holdout_used"]
        expected = {(c["batch"], c["episode"]) for c in unassisted["cases"]
                    if c["split"] == "unused_internal_episode"}
        assert len(expected) == 8
        assert expected == {(c["batch"], c["episode"]) for c in assisted["cases"]}
        summary["assisted_orientation_diagnostic"] = {
            "checkpoint_steps": 1000, "selection_impact": False,
            "external_used": False, "aggregates": assisted["aggregates"],
            "rollouts": rollout_diagnostics(args.orientation_diagnostic)}

    names = ["Previous data\n1,000 updates", "Reactive data\n1,000 updates",
             "Reactive data\n3,000 updates"]
    colors = ["#a0acb9", "#4b89ae", "#23816b"]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#f7f8f1", "axes.facecolor": "#fffef8"}):
        fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
        fig.suptitle("Wall-OSS-0.5 | Reactive demonstrations and model-only execution",
                     fontsize=16, fontweight="bold")
        ax = axes[0, 0]
        window = 50
        ax.plot(steps, losses, color="#8fb7ab", alpha=.3, linewidth=.6)
        ax.plot(steps[window - 1:], np.convolve(losses, np.ones(window) / window, mode="valid"),
                color="#23816b", linewidth=1.5)
        ax.set(title="New training loss (50-update moving mean)",
               xlabel="Completed optimizer updates", ylabel="Flow-matching loss", yscale="log")
        ax = axes[0, 1]
        ax.bar(names, [r["openloop"]["unused_internal_episode"]["execute8_xyz_cm"]
                       for r in comparison], color=colors)
        ax.set(title="Action position error on 8 validation episodes",
               ylabel="Mean first-8-action XYZ distance (cm)")
        ax = axes[1, 0]
        ax.bar(names, [r["validation_execute8_orientation_deg"] for r in comparison], color=colors)
        ax.set(title="Action orientation error on validation observations",
               ylabel="Mean first-8-action angle error (degrees)")
        ax = axes[1, 1]
        metrics = [r["closedloop"]["unused_internal_episode"] for r in comparison]
        bars = ax.bar(names, [m["successes"] / m["trials"] for m in metrics], color=colors)
        annotate_counts(ax, bars, metrics)
        ax.set(title="Validation: stable tabletop pickups", ylabel="Success fraction", ylim=(0, 1.17))
        ax = axes[2, 0]
        metrics = [paired[label]["aggregates"]["external_holdout"] for label in labels]
        bars = ax.bar(["Previous model", "Selected new model"],
                      [m["successes"] / m["trials"] for m in metrics],
                      color=[colors[0], "#23816b"])
        annotate_counts(ax, bars, metrics)
        ax.set(title="Primary: 8 fresh tabletop pickup positions",
               ylabel="Stable pickup fraction", ylim=(0, 1.17))
        ax = axes[2, 1]
        groups = ["hand_offset", "grasp_miss", "slip"]
        for index, label in enumerate(labels):
            metrics = [paired[label]["aggregates"]["external_recovery_" + group]
                       for group in groups]
            bars = ax.bar(np.arange(3) + (index - .5) * .36,
                          [m["successes"] / m["trials"] for m in metrics], width=.34,
                          color=[colors[0], "#23816b"][index],
                          label=["Previous model", "Selected new model"][index])
            annotate_counts(ax, bars, metrics)
        ax.set_xticks(np.arange(3), ["Hand offset", "Off-center grip", "Slip recovery"])
        ax.set(title="Secondary: recovery cases (2 per condition)",
               ylabel="Stable recovery fraction", ylim=(0, 1.17))
        ax.legend(fontsize=8)
        for ax in axes.flat:
            ax.grid(axis="y", alpha=.15)
        fig.savefig(args.evaluation / "reactive_comparison.png", dpi=180)
        plt.close(fig)
        if args.orientation_diagnostic:
            raw_diagnostics = [r for r in rollout_diagnostics(unassisted_folder)
                               if r["split"] == "unused_internal_episode"]
            fixed_diagnostics = summary["assisted_orientation_diagnostic"]["rollouts"]
            metrics = [unassisted["aggregates"]["unused_internal_episode"],
                       assisted["aggregates"]["unused_internal_episode"]]
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
            fig.suptitle("Validation-only diagnostic | Same 1,000-step model and 8 initial states",
                         fontsize=14, fontweight="bold")
            conditions = ["Model only", "Fixed orientation"]
            bars = axes[0].bar(conditions, [m["successes"] / m["trials"] for m in metrics],
                               color=["#4b89ae", "#c18a3d"])
            annotate_counts(axes[0], bars, metrics)
            axes[0].set(title="Stable tabletop pickups", ylabel="Success fraction", ylim=(0, 1.17))
            for axis, field, title, unit in [
                    (axes[1], "max_tool_angle_from_teacher_deg", "Executed tool orientation", "Maximum angle from teacher (degrees)"),
                    (axes[2], "min_grasp_distance_cm", "Closest approach to bottle", "Minimum tool-to-grasp distance (cm)")]:
                for index, records in enumerate([raw_diagnostics, fixed_diagnostics]):
                    values = [r[field] for r in records]
                    axis.scatter(np.full(len(values), index), values,
                                 color=["#4b89ae", "#c18a3d"][index], alpha=.7)
                axis.set_xticks([0, 1], conditions)
                axis.set(title=title, ylabel=unit, xlim=(-.5, 1.5), ylim=(0, None))
            for ax in axes:
                ax.grid(axis="y", alpha=.15)
            fig.savefig(args.evaluation / "orientation_diagnostic.png", dpi=180)
            plt.close(fig)
    (args.evaluation / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in
                      ["data", "checkpoints", "final_evaluation", "rollout_diagnostics"]}))


if __name__ == "__main__":
    main()
