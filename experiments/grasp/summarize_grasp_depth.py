"""Verify insertion mechanics and keep recorded replays separate from inference."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from feedback_trace import read_trace
from grasp_depth_probe import APPROACHES, CASE, read


def check_case(root, case, raw_commands=None, lift=True):
    folder = root / case["directory"]
    rows = read_trace(folder / "physics_trace.jsonl.gz")
    macro = read(folder / "telemetry.json")
    assert case["trial_valid"] and len(macro) == 384
    assert abs(macro[-1]["time_s"] - 15.36) < 1e-7
    assert abs(len(rows) * case["diagnostics"]["physics_timestep_s"] - 15.36) < 1e-7
    assert abs(rows[-1]["held_s"] - case["end_bilateral_lifted_s"]) < 1e-8
    assert case["end_bilateral_at_least_3s"] == (rows[-1]["held_s"] >= 3.)
    handoff = case["handoff"]
    anchor_world = None
    for step, row in enumerate(macro):
        applied = np.asarray(row["requested_pose"])
        after = handoff is not None and step * .04 >= handoff["time_s"] - 1e-9
        if not after:
            raw = raw_commands[step] if raw_commands is not None else np.asarray(row["raw_model_pose"])
            expected = np.asarray(raw).copy()
            fraction = min(1., step * .04 / .4)
            expected[:3] += Rotation.from_euler("xyz", expected[3:6]).apply([case["depth_mm"] / 1000 * fraction, 0., 0.])
            np.testing.assert_allclose(applied, expected, atol=1e-9, rtol=0)
        else:
            np.testing.assert_allclose(applied[3:6], handoff["tool_pose"][3:6], atol=1e-9, rtol=0)
            assert applied[6] == 2.2
            if anchor_world is None:
                anchor_world = np.asarray(row["requested_tool_xyz"])
            fraction = np.clip((step * .04 - handoff["time_s"] - .2) / 2., 0., 1.) if lift else 0.
            smooth = 10 * fraction ** 3 - 15 * fraction ** 4 + 6 * fraction ** 5
            np.testing.assert_allclose(row["requested_tool_xyz"], anchor_world + [0., 0., .12 * smooth], atol=1e-6, rtol=0)
    return {"id": case.get("approach", case.get("seed")), "depth_mm": case["depth_mm"],
        "original_criterion_reached": case["original_criterion_reached"],
        "end_bilateral_s": case["end_bilateral_lifted_s"], "end_retained_s": case["end_retained_s"],
        "final_lift_cm": case["final_lift_m"] * 100,
        "end_3s": case["end_bilateral_at_least_3s"],
        "contact_tip_margin_mm": handoff["contact_behind_tip_m"] * 1000 if handoff and handoff["contact_behind_tip_m"] is not None else None,
        "bottle_x_at_handoff_mm": handoff["bottle_x_in_hand_m"] * 1000 if handoff else None,
        "final_bottle_x_mm": rows[-1]["bottle_x_in_hand_m"] * 1000,
        "final_palm_contacts": rows[-1]["palm_contact_bodies"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("experiments/grasp/runs/grasp_depth_v8"))
    parser.add_argument("--runs", type=Path, default=Path("experiments/grasp/runs"))
    args = parser.parse_args()
    root = args.root
    hold, lifted = read(root / "report.json"), read(root / "lift_report.json")
    live_reports = [read(root / (name + "_report.json")) for name in ["live", "live_extended"]]
    assert hold["completed"] and lifted["completed"] and all(r["completed"] for r in live_reports)
    assert len(hold["cases"]) == len(lifted["cases"]) == 12
    live = [c for report in live_reports for c in report["cases"]]
    assert len(live) == 12 and {(c["seed"], c["depth_mm"]) for c in live} == {(s, d) for s in [0, 1, 2] for d in [0, 10, 20, 30]}
    for report in live_reports:
        audit = report["load_audit"]
        assert not audit["missing_non_tied_keys"] and not audit["unexpected_keys"]
        assert not audit["unloaded_meta_parameters"] and audit["action_expert_overlay_tensors"] == 257
    records = {"hold_replay": [], "lift_replay": [], "fresh_inference": []}
    for label, report, dirname, lifting in [("hold_replay", hold, "replays", False), ("lift_replay", lifted, "lift_replays", True)]:
        for case in report["cases"]:
            case_root = root / dirname
            if case["depth_mm"] == 0:
                assert case["baseline_prefix_max_abs_error"] == 0
            raw_path = args.runs / APPROACHES[case["approach"]] / CASE / "actions.json"
            raw_chunks = read(raw_path)
            raw = np.asarray(raw_chunks)[:, :8].reshape(-1, 7)
            records[label].append(check_case(case_root, case, raw, lifting))
    for case in live:
        records["fresh_inference"].append(check_case(root, case))
    summary = {"completed": True, "all_36_trials_valid": True, "no_new_training": True,
        "same_fixed_scene_only": True, "all_successes_are_hybrid_or_privileged_replays": True,
        "original_geometry": read(root / "geometry_inspection.json"), "records": records,
        "scores": {label: {str(depth): {"trials": sum(r["depth_mm"] == depth for r in rows),
            "original_1s": sum(r["depth_mm"] == depth and r["original_criterion_reached"] for r in rows),
            "end_3s": sum(r["depth_mm"] == depth and r["end_3s"] for r in rows)}
            for depth in [0, 10, 20, 30]} for label, rows in records.items()},
        "live_protocols": [r["protocol"] for r in live_reports],
        "limits": ["One fixed scene; recorded approach variants and repeats are not independent positions",
            "10mm was selected from replay data; 20/30mm inference was an additional development screen after partial 10mm failures",
            "Contact identity triggers the common program controller; no deployable grasp detector tested",
            "Physics paused during policy inference; no real-time or hardware claim",
            "Nominal target offset and actual seating depth differ because contact timing and object motion change"]}
    colors = ["#8995a3", "#27866f", "#4a81ab", "#b68d54"]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#f7f8f1", "axes.facecolor": "#fffef8"}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        fig.suptitle("Grasp seating depth | Fixed-scene hybrid control", fontsize=17, fontweight="bold")
        for ax, label, title in [(axes[0, 0], "lift_replay", "Recorded approaches + identical controlled lift"),
                                  (axes[0, 1], "fresh_inference", "Fresh model approaches + same program, seeds 0/1/2")]:
            counts = [summary["scores"][label][str(d)]["end_3s"] for d in [0, 10, 20, 30]]
            bars = ax.bar(["0 mm", "+10 mm", "+20 mm", "+30 mm"], np.asarray(counts) / 3, color=colors)
            for bar, value in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width() / 2, value / 3 + .025, f"{value}/3", ha="center")
            ax.set(title=title, ylabel="Bilateral elevated grasp >=3s at end", ylim=(0, 1.17))
        ax = axes[1, 0]
        for i, depth in enumerate([0, 10, 20, 30]):
            values = [r["contact_tip_margin_mm"] for r in records["lift_replay"] if r["depth_mm"] == depth and r["contact_tip_margin_mm"] is not None]
            ax.scatter([i] * len(values), values, color=colors[i], s=45)
        ax.set(xticks=range(4), xticklabels=["0 mm", "+10 mm", "+20 mm", "+30 mm"],
               title="Measured contact location at grasp confirmation", ylabel="Contact behind fingertip (mm)")
        ax = axes[1, 1]
        for i, depth in enumerate([0, 10, 20, 30]):
            rows = sorted([r for r in records["fresh_inference"] if r["depth_mm"] == depth], key=lambda r: r["id"])
            ax.plot([r["id"] for r in rows], [r["end_bilateral_s"] for r in rows], marker="o", label=f"{depth} mm", color=colors[i])
        ax.axhline(3, color="#b35353", linestyle="--", label="End-hold criterion")
        ax.set(title="Fresh inference: final continuous elevated bilateral hold", xlabel="Inference seed", ylabel="Seconds", xticks=[0, 1, 2])
        ax.legend(fontsize=8)
        for ax in axes.flat:
            ax.grid(axis="y", alpha=.15)
        fig.savefig(root / "grasp_depth_comparison.png", dpi=180)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        for ax, depth in zip(axes, [0, 10]):
            case = next(c for c in lifted["cases"] if c["approach"] == "original" and c["depth_mm"] == depth)
            ax.imshow(plt.imread(root / "lift_replays" / case["directory"] / "grasp_closeup.png"))
            margin = case["handoff"]["contact_behind_tip_m"] * 1000
            ax.set_title(f"Target offset {depth} mm | contact ~{margin:.1f} mm behind tip")
            ax.axis("off")
        fig.suptitle("Evaluation closeups | Same recorded approach, different insertion targets", fontsize=13)
        fig.savefig(root / "grasp_depth_closeups.png", dpi=180)
        plt.close(fig)
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"completed": True, "all_36_trials_valid": True, "scores": summary["scores"]}), flush=True)


if __name__ == "__main__":
    main()
