"""Verify the matched diagnostic and separate contact loss from low clearance."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from feedback_trace import read_trace


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/grasp/runs/feedback_diagnostic_v7"))
    parser.add_argument("--runs", type=Path, default=Path("experiments/grasp/runs"))
    args = parser.parse_args()
    root = args.output
    protocol = read(root / "protocol.json")
    replay = read(root / "replay/report.json")
    assert replay["completed"] and replay["valid_trials"] == 4
    assert all(max(c["replay_max_abs_errors"].values()) == 0 for c in replay["cases"])
    checkpoint = args.runs / protocol["checkpoint"] / "action_expert.safetensors"
    assert hashlib.file_digest(checkpoint.open("rb"), "sha256").hexdigest() == protocol["action_expert_sha256"]
    groups = ["direct8", "direct2", "hold8", "hold2"]
    summary = {"completed": True, "protocol": protocol, "replay_matches_original_exactly": True,
               "all_trials_valid": True, "groups": {}}
    cases, traces, latency = {}, {}, {}
    for name in [*groups, "repeat8"]:
        report = read(root / name / "report.json")
        assert report["completed"] and report["valid_trials"] == 4
        audit = report["load_audit"]
        assert not audit["missing_non_tied_keys"] and not audit["unexpected_keys"]
        assert not audit["unloaded_meta_parameters"] and audit["action_expert_overlay_tensors"] == 257
        assert report["physics_paused_during_inference"]
        records, calls = [], []
        for index, case in enumerate(report["cases"]):
            assert case["trial_valid"]
            assert (case["batch"], case["episode"], case["split"]) == tuple(
                protocol["cases"][index][key] for key in ["batch", "episode", "split"])
            assert abs(case["simulated_seconds"] - case["control_steps"] * .04) < 1e-7
            assert case["success"] or case["control_steps"] == 384
            folder = root / name / case["directory"]
            diagnostic = case["diagnostics"]
            rows = read_trace(folder / "physics_trace.jsonl.gz")
            assert len(rows) == round(case["simulated_seconds"] / diagnostic["physics_timestep_s"])
            assert bool(case["success"]) == (diagnostic["max_bilateral_lifted_s"] >= 1 - 1e-9)
            macro = read(folder / "telemetry.json")
            chunks = np.asarray(read(folder / "actions.json"))
            raw = chunks[:, :report["execute_steps"]].reshape(-1, 7)[:case["control_steps"]]
            applied = np.asarray([r["requested_pose"] for r in macro])
            assert len(applied) == len(raw) == case["control_steps"]
            assert np.array_equal(raw[:, :6], applied[:, :6]), "An arm command was modified"
            latch = diagnostic["gripper_latch"]
            if latch is None:
                assert np.array_equal(raw, applied)
            else:
                assert report["assistance_uses_simulator_contact_identity"]
                after = np.asarray([r["time_s"] - .04 >= latch["time_s"] - 1e-8 for r in macro])
                assert not after.any() or np.allclose(applied[after, 6], latch["gripper_command"], atol=1e-8, rtol=0)
            nearest = min(macro, key=lambda r: r["grasp_distance_m"])
            distance = lambda a, b: float(np.linalg.norm(np.asarray(a) - np.asarray(b))) * 100
            record = {"batch": case["batch"], "episode": case["episode"], "split": case["split"],
                "success": case["success"], "latched": latch is not None,
                "max_bilateral_contact_s": diagnostic["max_bilateral_contact_s"],
                "max_bilateral_lifted_s": diagnostic["max_bilateral_lifted_s"],
                "max_lift_cm": diagnostic["max_bottle_bottom_lift_m"] * 100,
                "first_sustained_hold_break": diagnostic["first_sustained_hold_break"],
                "nearest_grasp_sample": {"time_s": nearest["time_s"],
                    "actual_distance_cm": nearest["grasp_distance_m"] * 100,
                    "requested_distance_cm": distance(nearest["requested_tool_xyz"], nearest["bottle_xyz"]),
                    "target_tracking_cm": distance(nearest["requested_tool_xyz"], nearest["actual_tool_xyz"])}}
            original = args.runs / "takeover_training_eval_v6/augmented_closedloop_internal" / case["directory"]
            old = np.asarray(read(original / "actions.json"))
            record["initial_chunk_xyz_difference_from_v6_m"] = float(np.max(np.abs(chunks[0, :, :3] - old[0, :, :3])))
            record["initial_chunk_full_difference_from_v6"] = float(np.max(np.abs(chunks[0] - old[0])))
            differences = [np.abs(np.asarray(Image.open(folder / f"initial_{camera}.png")).astype(int)
                - np.asarray(Image.open(original / f"initial_{camera}.png")).astype(int))
                for camera in ["face_view", "right_wrist_view"]]
            record["initial_image_difference"] = {"max_0_255": max(int(d.max()) for d in differences),
                "changed_channels": sum(int((d > 0).sum()) for d in differences),
                "total_channels": sum(d.size for d in differences)}
            record["initial_images_match_v6"] = record["initial_image_difference"]["changed_channels"] == 0
            # A separate exact audit found one image with 12 channels differing
            # by one grayscale level. Preserve this deviation; tolerate only
            # sparse 8-bit rounding-scale differences, never a scene change.
            assert record["initial_image_difference"]["max_0_255"] <= 1
            assert record["initial_image_difference"]["changed_channels"] / record["initial_image_difference"]["total_channels"] <= 1e-4
            if name in ["direct8", "repeat8"]:
                common = min(len(chunks), len(old))
                record["max_action_difference_from_v6"] = float(np.max(np.abs(chunks[:common] - old[:common])))
                expected = read(original / "telemetry.json")
                record["max_bottle_difference_from_v6_m"] = float(np.max(np.abs(
                    np.asarray([r["bottle_xyz"] for r in macro[:len(expected)]])
                    - np.asarray([r["bottle_xyz"] for r in expected[:len(macro)]]))))
            records.append(record)
            calls.extend(read(folder / "inferences.json"))
            traces[name, index] = rows
        cases[name] = report["cases"]
        latency[name] = [c["inference_wall_seconds"] for c in calls]
        summary["groups"][name] = {"trials": 4, "valid": 4,
            "successes": sum(r["success"] for r in records),
            "contact_at_least_80ms": sum(r["max_bilateral_contact_s"] >= .08 - 1e-9 for r in records),
            "lifted_contact_at_least_80ms": sum(r["max_bilateral_lifted_s"] >= .08 - 1e-9 for r in records),
            "latch_triggered": sum(r["latched"] for r in records),
            "median_inference_s": statistics.median(latency[name]),
            "inference_calls": len(calls),
            "inference_calls_slower_than_action_window": sum(c["inference_wall_seconds"] > c["nominal_action_window_s"] for c in calls),
            "cases": records}
    summary["all_16_model_trials_valid"] = True
    summary["all_20_model_trials_including_repeat_valid"] = True
    probes = read(root / "replay_probes/report.json")
    assert probes["completed"] and len(probes["cases"]) == 4 and probes["not_model_successes"]
    assert not probes["model_inference_executed"] and not probes["model_adapts_to_intervention"]
    interventions = [p["intervention"] for p in probes["cases"]]
    assert all(abs(p["time_s"] - interventions[0]["time_s"]) < 1e-9 for p in interventions)
    assert all(np.allclose(p["actual_tool_pose"], interventions[0]["actual_tool_pose"], rtol=0, atol=1e-9)
               for p in interventions)
    summary["counterfactual_replay_probes"] = {"same_recorded_case_and_trigger_state": True,
        "not_model_or_hardware_successes": True, "cases": [{"mode": p["mode"],
            "trial_valid": p["trial_valid"], "success_criterion_reached": p["success_criterion_reached"],
            "intervention": p["intervention"],
            "max_bilateral_contact_s": p["diagnostics"]["max_bilateral_contact_s"],
            "max_bilateral_lifted_s": p["diagnostics"]["max_bilateral_lifted_s"],
            "first_sustained_hold_break": p["diagnostics"]["first_sustained_hold_break"]}
            for p in probes["cases"]]}
    assert all(p["trial_valid"] for p in probes["cases"])
    summary["repeatability_control"] = {"same_four_cases_checkpoint_and_seed": True,
        "initial_images_identical": all(r["initial_images_match_v6"]
            for group in summary["groups"].values() for r in group["cases"]),
        "initial_image_max_abs_difference_0_255": max(r["initial_image_difference"]["max_0_255"]
            for group in summary["groups"].values() for r in group["cases"]),
        "initial_image_total_changed_channels": sum(r["initial_image_difference"]["changed_channels"]
            for group in summary["groups"].values() for r in group["cases"]),
        "repeat8_is_additional_not_a_replacement": True,
        "first_chunk_xyz_max_difference_from_v6_m": max(
            r["initial_chunk_xyz_difference_from_v6_m"] for group in summary["groups"].values() for r in group["cases"]),
        "millimeter_scale_prediction_variation_cause_not_established": True}
    summary["limits"] = ["Four already-used development starts and one flow-noise seed",
        "Shorter execution changes the number of inferences and fresh flow-noise samples",
        "Physics remains paused during inference; this is not a real-time control result",
        "Contact-latched gripper uses privileged simulator contact identity and is separately scored",
        "Substep contact/height interruption identifies temporal events, not a unique physical cause",
        "Repeated process starts can produce small initial prediction differences; small contact-duration differences are not reliable evidence alone"]
    colors = ["#8995a3", "#27866f", "#b68d54", "#4a81ab"]
    labels = ["Direct / 8", "Direct / 2", "Grip latch / 8", "Grip latch / 2"]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#f7f8f1", "axes.facecolor": "#fffef8"}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        fig.suptitle("Wall-OSS-0.5 | Feedback and gripper diagnostics", fontsize=17, fontweight="bold")
        ax = axes[0, 0]
        values = [summary["groups"][g]["successes"] for g in groups]
        bars = ax.bar(labels, np.asarray(values) / 4, color=colors)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value / 4 + .025, f"{value}/4", ha="center")
        ax.set(title="Stable pickup: same 4 development starts", ylim=(0, 1.16), ylabel="Success fraction")
        ax = axes[0, 1]
        metrics = ["contact_at_least_80ms", "lifted_contact_at_least_80ms", "successes"]
        for i, g in enumerate(groups):
            ax.plot(range(3), [summary["groups"][g][k] for k in metrics], marker="o", color=colors[i], label=labels[i])
        ax.set_xticks(range(3), ["Bilateral >=80ms", "Lifted >=80ms", "Stable >=1s"])
        ax.set(title="Which stages were reached?", ylim=(-.15, 4.4), ylabel="Starts out of four")
        ax.legend(fontsize=8)
        ax = axes[1, 0]
        for i, g in enumerate(groups):
            values = [c["diagnostics"]["max_bilateral_lifted_s"] for c in cases[g]]
            ax.plot(range(1, 5), values, marker="o", label=labels[i], color=colors[i])
        ax.axhline(1, color="#b35353", linestyle="--", label="Stable criterion")
        ax.set(title="Longest uninterrupted lifted bilateral grasp", xlabel="Development start", ylabel="Seconds", xticks=range(1, 5))
        ax.legend(fontsize=8)
        ax = axes[1, 1]
        ax.bar(np.arange(4) - .17, [statistics.median(latency[g]) for g in groups], width=.34,
               color=colors, label="Measured median inference")
        ax.bar(np.arange(4) + .17, [.32, .08, .32, .08], width=.34, color="#d9d9cb", label="Nominal action window")
        ax.set_xticks(range(4), labels)
        ax.set(title="Physics paused during inference: no realtime claim", ylabel="Wall / simulated seconds")
        ax.legend(fontsize=8)
        for ax in axes.flat:
            ax.grid(axis="y", alpha=.15)
        fig.savefig(root / "feedback_comparison.png", dpi=180)
        plt.close(fig)
        # Same predeclared first development start, not a best-case selection.
        fig, axes = plt.subplots(5, 2, figsize=(14, 12), constrained_layout=True)
        for col, name in enumerate(["direct8", "direct2"]):
            rows = traces[name, 0]
            t = [r["time_s"] for r in rows]
            axes[0, col].set_title(labels[groups.index(name)] + " | first development start")
            for field, label in [("gripper_raw", "Model request"), ("gripper_left", "Actual left"), ("gripper_right", "Actual right")]:
                axes[0, col].plot(t, [r[field] for r in rows], label=label)
            axes[0, col].set_ylabel("Gripper motor units")
            axes[0, col].legend(fontsize=8)
            axes[1, col].plot(t, [r["lift_m"] * 100 for r in rows])
            axes[1, col].axhline(5, linestyle="--", color="#b35353")
            axes[1, col].set_ylabel("Bottle bottom lift (cm)")
            axes[2, col].step(t, [r["contacts"] for r in rows], where="post")
            axes[2, col].set_ylabel("Finger contacts")
            axes[3, col].plot(t, [r["tracking_rotation_deg"] for r in rows], label="Tool rotation tracking error")
            axes[3, col].plot(t, [r["bottle_tilt_deg"] for r in rows], label="Bottle tilt")
            axes[3, col].set_ylabel("Degrees")
            axes[3, col].legend(fontsize=8)
            axes[4, col].plot(t, [r["held_s"] for r in rows])
            axes[4, col].set(ylabel="Stable-condition timer (s)", xlabel="Simulated seconds")
            for ax in axes[:, col]:
                ax.set_xlim(0, 15.36)
                ax.grid(alpha=.15)
        fig.suptitle("Physics-step trace | Model commands and actual contact / clearance", fontsize=16)
        fig.savefig(root / "first_start_trace.png", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        values = [p["diagnostics"]["max_bilateral_lifted_s"] for p in probes["cases"]]
        bars = ax.bar(["Freeze issued grip", "Tighter close 1.7", "Hold actual tool pose", "Pose + tighter close"],
                     values, color=["#8995a3", "#b68d54", "#27866f", "#4a81ab"])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + .04, f"{value:.3f}s", ha="center")
        ax.axhline(1, color="#b35353", linestyle="--", label="Stable criterion")
        ax.set(title="One recorded-policy counterfactual | Privileged trigger, no new model inference",
               ylabel="Longest lifted bilateral hold (s)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=.15)
        fig.savefig(root / "counterfactual_probes.png", dpi=180)
        plt.close(fig)
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"completed": True, "all_20_model_trials_including_repeat_valid": True,
        "groups": {g: {k: v for k, v in row.items() if k != "cases"} for g, row in summary["groups"].items()}}), flush=True)


if __name__ == "__main__":
    main()
