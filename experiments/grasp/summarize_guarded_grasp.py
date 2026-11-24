"""Audit paired controls, report all outcomes and preserve privileged labels."""
import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from feedback_trace import read_trace
from grasp_depth_probe import read
from guarded_grasp import GuardConfig, closing_geometry_valid, geometry_ready


def check_case(root, case):
    folder = root / case["directory"]
    rows = read_trace(folder / "physics_trace.jsonl.gz")
    macro = read(folder / "telemetry.json")
    geometries = read(folder / "geometry.json")
    assert case["trial_valid"] and len(macro) == len(geometries) == 384
    assert abs(macro[-1]["time_s"] - 15.36) < 1e-7
    assert abs(len(rows) * case["diagnostics"]["physics_timestep_s"] - 15.36) < 1e-7
    assert abs(rows[-1]["held_s"] - case["end_bilateral_lifted_s"]) < 1e-8
    assert case["end_bilateral_at_least_3s"] == (rows[-1]["held_s"] >= 3.)
    handoff = case["handoff"]
    anchor_world = None
    closing_events = []
    if case["mode"] == "guarded":
        cfg = GuardConfig(**case["guard"]["config"])
        commands = read(folder / "guard_commands.json")
        assert len(commands) == 384
        closing_events = [e for e in case["guard"]["events"] if e["to"] == "close"]
        for event in closing_events:
            assert event["from"] == "ready" and geometry_ready(event["geometry"], cfg)
        for row, control in zip(macro, commands):
            np.testing.assert_array_equal(row["requested_pose"], control["applied"])
            if row["requested_pose"][6] < cfg.close_request_below:
                assert control["phase"] in ["close", "lift_hold"]
        if handoff:
            assert closing_events
            event_index = min(range(len(rows)), key=lambda i: abs(rows[i]["time_s"] - handoff["time_s"]))
            row = rows[event_index]
            at_handoff = {"center_m": [row[f"bottle_{axis}_in_hand_m"] for axis in "xyz"],
                "axis_error_deg": geometries[row["control_step"]]["after"]["axis_error_deg"]}
            assert closing_geometry_valid(at_handoff)
    for step, row in enumerate(macro):
        applied = np.asarray(row["requested_pose"])
        after = handoff is not None and step * .04 >= handoff["time_s"] - 1e-9
        if after:
            np.testing.assert_allclose(applied[3:6], handoff["tool_pose"][3:6], atol=1e-9, rtol=0)
            assert applied[6] == 2.2
            if anchor_world is None:
                anchor_world = np.asarray(row["requested_tool_xyz"])
            fraction = np.clip((step * .04 - handoff["time_s"] - .2) / 2., 0., 1.)
            smooth = 10 * fraction ** 3 - 15 * fraction ** 4 + 6 * fraction ** 5
            np.testing.assert_allclose(row["requested_tool_xyz"], anchor_world + [0., 0., .12 * smooth], atol=1e-6, rtol=0)
        elif case["mode"] == "baseline":
            np.testing.assert_array_equal(applied, row["raw_policy_reference"])
    return {"name": case["name"], "mode": case["mode"], "kind": case["kind"],
        "seed": case["seed"], "source_case": case["source_case"],
        "original_1s": case["original_criterion_reached"], "end_3s": case["end_bilateral_at_least_3s"],
        "end_bilateral_s": case["end_bilateral_lifted_s"], "final_lift_cm": case["final_lift_m"] * 100,
        "final_bottle_tilt_deg": rows[-1]["bottle_tilt_deg"],
        "max_posthandoff_tilt_deg": max((r["bottle_tilt_deg"] for r in rows if handoff and r["time_s"] >= handoff["time_s"]), default=None),
        "handoff_tip_margin_mm": handoff["contact_behind_tip_m"] * 1000 if handoff else None,
        "handoff_bottle_x_mm": handoff["bottle_x_in_hand_m"] * 1000 if handoff else None,
        "inference_calls": case["inference_calls"], "closure_attempts": len(closing_events),
        "phases": dict(Counter(r["phase"] for r in geometries)),
        "abort_reason": case["guard"]["abort_reason"] if case["guard"] else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("experiments/grasp/runs/guarded_grasp_v9"))
    parser.add_argument("--suites", nargs="+", default=["replay", "normal", "external"])
    parser.add_argument("--repeatability", type=Path, help="Reuse the matching frozen policy repeatability audit")
    parser.add_argument("--depth-reference", type=Path, default=Path("experiments/grasp/runs/grasp_depth_v8"))
    args = parser.parse_args()
    records, protocols, pairing = {}, {}, {}
    for suite in args.suites:
        root = args.root / suite
        report = read(root / "report.json")
        assert report["completed"]
        protocols[suite] = report["protocol"]
        if suite != "replay":
            audit = report["load_audit"]
            assert not audit["missing_non_tied_keys"] and not audit["unexpected_keys"]
            assert not audit["unloaded_meta_parameters"] and audit["action_expert_overlay_tensors"] == 257
        groups = {}
        for case in report["cases"]:
            groups.setdefault(case["name"], {})[case["mode"]] = case
        pairing[suite] = []
        for name, pair in groups.items():
            assert set(pair) == {"baseline", "guarded"}
            assert pair["baseline"]["initial_observation"] == pair["guarded"]["initial_observation"]
            if suite != "replay":
                initial_left = np.load(root / pair["baseline"]["directory"] / "initial_physics.npz")
                initial_right = np.load(root / pair["guarded"]["directory"] / "initial_physics.npz")
                for key in ["integration_state", "control"]:
                    np.testing.assert_array_equal(initial_left[key], initial_right[key])
                left = read(root / pair["baseline"]["directory"] / "actions.json")
                right = read(root / pair["guarded"]["directory"] / "actions.json")
                # Before the first intervention, both freshly sampled first chunks match.
                np.testing.assert_allclose(left[0], right[0], atol=1e-6, rtol=0)
                left_macro = read(root / pair["baseline"]["directory"] / "telemetry.json")
                right_macro = read(root / pair["guarded"]["directory"] / "telemetry.json")
                shared = 0
                for a, b in zip(left_macro, right_macro):
                    intervention = (not np.array_equal(a["requested_pose"], a["raw_policy_reference"])
                        or not np.array_equal(b["requested_pose"], b["raw_policy_reference"]))
                    if intervention:
                        break
                    for key in ["requested_pose", "right_ee", "bottle_xyz", "held_seconds"]:
                        np.testing.assert_allclose(a[key], b[key], atol=1e-9, rtol=0)
                    shared += 1
                common_chunks = (shared + 7) // 8
                left_inference = read(root / pair["baseline"]["directory"] / "inferences.json")
                right_inference = read(root / pair["guarded"]["directory"] / "inferences.json")
                for a, b in zip(left_inference[:common_chunks], right_inference[:common_chunks]):
                    if "rng_seed" in a or "rng_seed" in b:
                        assert a["rng_seed"] == b["rng_seed"]
                        assert a["input_sha256"] == b["input_sha256"]
                        assert a["physics_state_sha256"] == b["physics_state_sha256"]
                differences = [float(np.max(np.abs(np.asarray(a) - b)))
                    for a, b in zip(left[:common_chunks], right[:common_chunks])]
                assert not differences or max(differences) < 1e-6
                pairing[suite].append({"name": name, "shared_policy_prefix_steps": shared,
                    "shared_inference_chunks": common_chunks,
                    "prediction_cache_hits": sum(v.get("shared_prediction", False) for v in [*left_inference, *right_inference]),
                    "rendered_pixel_differences": sum(sum(v.get("rendered_pixel_differences", {}).values())
                        for v in [*left_inference, *right_inference]),
                    "max_common_chunk_abs_difference": max(differences, default=0.)})
        records[suite] = [check_case(root, c) for c in report["cases"]]
    if "replay" in records:
        old = read(args.depth_reference / "lift_report.json")
        for case in records["replay"]:
            if case["mode"] == "baseline":
                previous = next(c for c in old["cases"] if c["approach"] == case["name"] and c["depth_mm"] == 0)
                assert abs(case["end_bilateral_s"] - previous["end_bilateral_lifted_s"]) < 1e-9
                assert abs(case["final_lift_cm"] / 100 - previous["final_lift_m"]) < 1e-9
                before = read(args.depth_reference / "lift_replays" / previous["directory"] / "telemetry.json")
                after = read(args.root / "replay" / f'{case["name"]}_baseline' / "telemetry.json")
                errors = [float(np.max(np.abs(np.asarray(a[key]) - b[key])))
                    for a, b in zip(before, after) for key in ["right_ee", "bottle_xyz", "held_seconds", "requested_pose"]]
                assert max(errors) == 0.
                case["baseline_replay_max_abs_error"] = max(errors)
    repeatability = read(args.repeatability or args.root / "repeatability/report.json")
    assert repeatability["completed"] and repeatability["model_inputs_identical"] and repeatability["exactly_repeatable"]
    scores = {suite: {mode: {"trials": sum(r["mode"] == mode for r in rows),
        "original_1s": sum(r["mode"] == mode and r["original_1s"] for r in rows),
        "end_3s": sum(r["mode"] == mode and r["end_3s"] for r in rows)}
        for mode in ["baseline", "guarded"]} for suite, rows in records.items()}
    summary = {"completed": True, "all_trials_valid": True, "protocols": protocols, "records": records,
        "pairing_audits": pairing,
        "repeatability": {k: repeatability[k] for k in ["repeats", "model_inputs_identical", "max_abs_differences", "exactly_repeatable"]},
        "scores": scores, "limits": ["Privileged simulation object geometry drives local alignment",
        "Paired approaches share RGB and policy samples only at identical full integration states and inference seeds; not independent per-branch inference",
        "Both groups use simulator contact identity and identical relative program lift",
        "No new training; model RGB/state inputs contain no object truth",
        "Previously used development scenes; no untouched evaluation or hardware/real-time claim"]}
    with plt.rc_context({"font.family": "DejaVu Sans", "figure.facecolor": "#f7f8f1", "axes.facecolor": "#fffef8"}):
        fig, axes = plt.subplots(1, len(records), figsize=(5 * len(records), 4.5), squeeze=False, constrained_layout=True)
        fig.suptitle("Guarded preclosure | Privileged hybrid simulation", fontsize=17, fontweight="bold")
        titles = {"replay": "Recorded approaches (development)", "normal": "Shared fresh policy approach | fixed start", "external": "Shared fresh policy approach | 8 starts"}
        for ax, (suite, score) in zip(axes.flat, scores.items()):
            values = [score[m]["end_3s"] / score[m]["trials"] for m in ["baseline", "guarded"]]
            bars = ax.bar(["Contact + lift baseline", "Geometry guard + lift"], values, color=["#8995a3", "#27866f"])
            for bar, mode in zip(bars, ["baseline", "guarded"]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .025,
                    f'{score[mode]["end_3s"]}/{score[mode]["trials"]}', ha="center", fontsize=14)
            ax.set(title=titles[suite], ylabel="Elevated bilateral hold >=3s at end", ylim=(0, 1.17))
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.15)
        fig.savefig(args.root / "guarded_grasp_comparison.png", dpi=180)
        plt.close(fig)
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"completed": True, "scores": scores}), flush=True)


if __name__ == "__main__":
    main()
