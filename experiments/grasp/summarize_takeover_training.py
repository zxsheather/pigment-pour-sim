"""Verify and summarize the matched additional-training correction experiment."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from summarize_reactive_experiment import annotate_counts, rollout_diagnostics


def read(path):
    return json.loads(path.read_text())


def verify_load(report):
    audit = report["load_audit"]
    assert not audit["missing_non_tied_keys"] and not audit["unexpected_keys"]
    assert not audit["unloaded_meta_parameters"]
    assert audit["action_expert_overlay_tensors"] == 257
    assert audit["normalization_from_current_run_files"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("experiments/grasp/runs"))
    args = parser.parse_args()
    runs = args.runs
    output = runs / "takeover_training_eval_v6"
    protocol = read(output / "protocol.json")
    group = read(runs / "takeover_training_v6/group_split.json")
    assert group["no_source_family_overlap"] and group["corrective_episodes"] == 9
    config_comparison = read(runs / "takeover_training_v6/config_comparison.json")
    assert config_comparison["passed"] and config_comparison["shared_normalization_file"]
    bounds = read(runs / "takeover_training_v6/xyz_envelope.json")
    assert bounds["all_other_statistics_identical"] and not bounds["external_data_used"]
    clipping = read(runs / "takeover_training_v6/action_range_audit.json")
    assert clipping["official_collator_roundtrip_verified"]
    assert clipping["overall"]["max_position_clip_error_mm"] == 0
    inputs = read(runs / "takeover_training_v6/input_contract.json")
    assert inputs["proprioception_matches"] and inputs["prefix_tokens_match"]
    assert all(r["pixel_values_max_abs_error"] == 0 and r["image_grid_equal"]
               and r["agent_mask_equal"] and r["train_action_tokens"] == r["serve_action_tokens"] == 32
               for r in inputs["frames"])
    normalizer = runs / "takeover_training_v6/common_norm_stats.json"
    assert hashlib.sha256(normalizer.read_bytes()).hexdigest() == bounds["updated_statistics_sha256"]
    summary = {"completed": True, "protocol": protocol, "group_split": group,
               "bounds": bounds, "clipping": clipping["overall"],
               "input_contract_frames_checked": len(inputs["frames"]), "models": {}}
    traces, closed, opened, normalizers = {}, {}, {}, {}
    for cohort in ["control", "augmented"]:
        trained = runs / f"train_takeover_{cohort}_3000_v6"
        report = read(trained / "report.json")
        assert report["completed"] and report["fp32_master_weights"]
        init = report["action_expert_initialization"]
        assert init["all_parameters_match_before_optimizer"] and not init["optimizer_state_restored"]
        assert init["source_completed_updates"] == 3000 and init["parameter_tensors"] == 257
        normalizers[cohort] = {name: torch.load(trained / "evaluation/adapted" / name,
            map_location="cpu", weights_only=True) for name in ["normalizer_action.pth", "normalizer_propri.pth"]}
        trace = [json.loads(line) for line in (trained / "learning_curve.jsonl").read_text().splitlines()]
        assert [r["completed_steps"] for r in trace] == list(range(1, 3001))
        assert all(np.isfinite(r["loss"]) and np.isfinite(r["grad_norm"]) for r in trace)
        traces[cohort] = trace
        membership = read(trained / "dataset_membership.json")
        assert membership["passed"] and membership["all_corrective_episodes_in_gradient_training"]
        # The sampler is world-size one, batch one, without replacement per epoch.
        # More updates than its length proves every selected frame was visited.
        assert len(trace) >= membership["training_frames"]
        if cohort == "augmented":
            assert membership["corrective_episode_ids"] == list(range(9))
            assert all(membership["gradient_frames_by_episode"][str(e)] > 0 for e in range(9))
        opened[cohort] = read(output / f"{cohort}_openloop_internal/report.json")
        verify_load(opened[cohort])
        closed[cohort] = {}
        for split in ["internal", "external"]:
            closed[cohort][split] = read(output / f"{cohort}_closedloop_{split}/report.json")
            tested = closed[cohort][split]
            verify_load(tested)
            assert tested["completed"] and all(c["trial_valid"] for c in tested["cases"])
            for case in tested["cases"]:
                if case["split"] == "seen_prefix_recovery":
                    init_case = case["recorded_prefix_initialization"]
                    assert max(init_case["robot_state_error"], init_case["bottle_position_error"]) < 1e-6
                    assert case["task"] == "in_sample_prefix_recovery"
        summary["models"][cohort] = {"initialization": init, "membership": membership,
            "additional_updates": len(trace), "logical_total_weight_updates": 6000,
            "loss_first50_mean": float(np.mean([r["loss"] for r in trace[:50]])),
            "loss_last50_mean": float(np.mean([r["loss"] for r in trace[-50:]])),
            "training_minutes": (trace[-1]["wall_time"] - trace[0]["wall_time"]) / 60,
            "peak_allocated_gb": report["peak_allocated_gb"],
            "openloop": opened[cohort]["aggregates"],
            "closedloop": {s: tested["aggregates"] for s, tested in closed[cohort].items()},
            "external_rollout_diagnostics": rollout_diagnostics(output / f"{cohort}_closedloop_external")}
    assert summary["models"]["control"]["initialization"] == summary["models"]["augmented"]["initialization"]
    for name in normalizers["control"]:
        control, augmented = normalizers["control"][name], normalizers["augmented"][name]
        assert set(control) == set(augmented)
        assert all(torch.equal(control[key], augmented[key]) for key in control)
    summary["exported_normalizers_identical_between_cohorts"] = True
    for split in ["internal", "external"]:
        keys = [{(c["split"], c["batch"], c["episode"]) for c in closed[cohort][split]["cases"]}
                for cohort in ["control", "augmented"]]
        assert keys[0] == keys[1] and len(keys[0]) == (12 if split == "internal" else 14)
    summary["all_52_closedloop_trials_valid"] = True
    summary["all_9_corrections_contribute_gradient_frames"] = True
    summary["limits"] = ["Nine corrections cover two previously seen training scenes and pre-fall states only",
        "One inference seed, eight new normal positions and two cases per recovery type",
        "The original internal validation remains available for development",
        "No physical robot, navigation, liquid control or complete task tested"]
    colors = {"control": "#8995a3", "augmented": "#27866f"}
    labels = {"control": "Original data +3k", "augmented": "Add 9 corrections +3k"}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#f7f8f1", "axes.facecolor": "#fffef8"}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        fig.suptitle("Wall-OSS-0.5 | Adding 9 model-state corrections", fontsize=17, fontweight="bold")
        ax = axes[0, 0]
        for cohort in colors:
            loss = np.asarray([r["loss"] for r in traces[cohort]])
            ax.plot(np.arange(50, 3001), np.convolve(loss, np.ones(50) / 50, mode="valid"),
                    color=colors[cohort], label=labels[cohort])
        ax.set(title="Training loss, 50-update average", xlabel="Additional updates", ylabel="Flow loss", yscale="log")
        ax.legend(fontsize=8)
        ax = axes[0, 1]
        values = [opened[c]["aggregates"]["unused_internal_episode"]["execute8_xyz_cm"] for c in colors]
        bars = ax.bar(list(labels.values()), values, color=list(colors.values()))
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + .03, f"{value:.2f}", ha="center")
        ax.set(title="Same eight validation episodes", ylabel="First-8-action XYZ error (cm)")
        ax = axes[1, 0]
        metrics = [closed[c]["external"]["aggregates"]["external_holdout"] for c in colors]
        bars = ax.bar(list(labels.values()), [m["successes"] / m["trials"] for m in metrics], color=list(colors.values()))
        annotate_counts(ax, bars, metrics)
        ax.set(title="Primary: eight new full pickup positions", ylabel="Stable pickup fraction", ylim=(0, 1.17))
        ax = axes[1, 1]
        groups = ["hand_offset", "grasp_miss", "slip"]
        for index, cohort in enumerate(colors):
            metrics = [closed[cohort]["external"]["aggregates"]["external_recovery_" + g] for g in groups]
            bars = ax.bar(np.arange(3) + (index - .5) * .36,
                [m["successes"] / m["trials"] for m in metrics], width=.34,
                color=colors[cohort], label=labels[cohort])
            annotate_counts(ax, bars, metrics)
        ax.set_xticks(np.arange(3), ["Hand offset", "Off-center grip", "Slip recovery"])
        ax.set(title="Separate recovery tasks, two cases each", ylabel="Stable recovery fraction", ylim=(0, 1.17))
        ax.legend(fontsize=8)
        for ax in axes.flat:
            ax.grid(axis="y", alpha=.15)
        fig.savefig(output / "takeover_training_comparison.png", dpi=180)
        plt.close(fig)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"completed": True, "all_52_closedloop_trials_valid": True,
        "models": {c: {k: m[k] for k in ["training_minutes", "peak_allocated_gb", "openloop", "closedloop"]}
                   for c, m in summary["models"].items()}}))


if __name__ == "__main__":
    main()
