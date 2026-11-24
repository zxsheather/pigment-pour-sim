"""Test a right-XYZ normalization change with matched 3000-update models.

The 1000-update checkpoint is diagnostic only. Final comparison is fixed in
advance to 3000 vs 3000 updates on new, paired external initial conditions.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=Path("experiments/grasp/runs"))
    parser.add_argument("--reference-final-device", default="1")
    args = parser.parse_args()
    runs = args.runs
    experiment = runs / "train_xyz_envelope_3000_v5"
    output = runs / "xyz_envelope_eval_v5"
    output.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    norm = "local/pigment_grasp_reactive_v4"
    protocol = {"changed_variable": "right-XYZ action normalization bounds only",
        "same_training_pool": True, "same_architecture_and_training_config_except_output_paths": True,
        "final_updates_both_models": 3000, "checkpoint_selection": "fixed 3000 updates before evaluation",
        "step_1000_is_diagnostic_only": True, "seed": 0,
        "control_period_s": .04, "execute_steps": 8, "max_chunks": 48,
        "new_external_seeds": {"clean": 2051, "recovery": 2052},
        "external_data_used_for_training_or_statistics": False,
        "previous_v4_external_results_are_development_evidence_only": True,
        "teacher_takeover_pilot_not_added_to_this_training": True,
        "slip_recovery_is_not_full_tabletop_pickup": True,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))

    def wait_for(path, seconds=7200):
        deadline = time.monotonic() + seconds
        while not path.exists():
            error = experiment / "report.json"
            if error.exists() and read(error).get("error"):
                raise RuntimeError("Training failed; inspect its report")
            if time.monotonic() > deadline:
                raise TimeoutError(str(path))
            time.sleep(10)

    internal = read(runs / "reactive_eval_v4/internal_cases.json")
    assert len(internal) == 10
    internal_file = output / "internal_cases.json"
    internal_file.write_text(json.dumps(internal, indent=2))
    wait_for(runs / "xyz_envelope_v5/fresh_cases_ready.json")
    accepted = {}
    positions = []
    roots = ["reactive_train_v4", "reactive_validation_v4", "reactive_holdout_v4",
             "reactive_recovery_holdout_v4", "xyz_holdout_v5", "xyz_recovery_holdout_v5"]
    for name in roots:
        manifest = read(runs / name / "manifest.json")
        accepted[name] = [e for e in manifest["episodes"] if e["accepted"]]
        positions.append({tuple(e["offset_xy_m"]) for e in accepted[name]})
    assert all(not positions[i] & positions[j] for i in range(6) for j in range(i + 1, 6))
    assert len(accepted["xyz_holdout_v5"]) == 8
    assert len(accepted["xyz_recovery_holdout_v5"]) == 6
    external = [{"split": "external_holdout", "batch": "xyz_holdout_v5",
                 "episode": e["episode"], "kind": e["kind"]} for e in accepted["xyz_holdout_v5"]]
    external += [{"split": "external_recovery_" + e["kind"],
                  "batch": "xyz_recovery_holdout_v5", "episode": e["episode"], "kind": e["kind"]}
                 for e in accepted["xyz_recovery_holdout_v5"]]
    external_file = output / "external_cases.json"
    external_file.write_text(json.dumps(external, indent=2))

    def command(script, checkpoint, destination, cases, extra=()):
        return [sys.executable, str(scripts / script), "--source", str(args.source),
                "--checkpoint", str(checkpoint), "--norm-key", norm,
                "--output", str(destination), "--cases", str(cases), *extra]

    def evaluate(script, checkpoint, destination, cases, extra=()):
        if (destination / "report.json").exists():
            result = read(destination / "report.json")
            assert result.get("completed", bool(result.get("records")))
            return result
        assert not destination.exists(), "Do not overwrite a partial evaluation"
        with destination.with_suffix(".log").open("w") as log:
            subprocess.run(command(script, checkpoint, destination, cases, extra),
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        return read(destination / "report.json")

    old = runs / "reactive_eval_v4"
    reference_open = read(old / "openloop_reactive_003000/report.json")
    reference_closed = read(old / "closedloop_reactive_003000/report.json")
    assert reference_closed["completed"] and all(c["trial_valid"] for c in reference_closed["cases"])
    compared = [{"label": "original_003000", "steps": 3000,
        "cached_internal_evaluation": True,
        "openloop": reference_open["aggregates"], "closedloop": reference_closed["aggregates"]}]
    for steps, checkpoint in [(1000, experiment / "step_001000/evaluation/adapted"),
                              (3000, experiment / "evaluation/adapted")]:
        wait_for(checkpoint / "adaptation.json")
        if steps == 1000:
            with (output / "input_contract.log").open("w") as log:
                subprocess.run([sys.executable, str(scripts / "audit_observation_contract.py"),
                    "--source", str(args.source), "--checkpoint", str(checkpoint.parent / "baseline"),
                    "--config", str(experiment / "train.yml"), "--norm-key", norm,
                    "--phase-frames", "--episode-folder", str(runs / "reactive_train_v4/episode_0000"),
                    "--output", str(output / "input_contract.json")],
                    check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                    stdout=log, stderr=subprocess.STDOUT)
            audit = read(output / "input_contract.json")
            assert audit["proprioception_matches"] and audit["prefix_tokens_match"]
            assert all(f["agent_mask_equal"] and f["image_grid_equal"] and
                       f["pixel_values_max_abs_error"] == 0 and
                       f["train_action_tokens"] == f["serve_action_tokens"] == 32
                       for f in audit["frames"])
        label = f"xyz_{steps:06d}"
        opened = evaluate("evaluate_expert_openloop.py", checkpoint, output / ("openloop_" + label),
                          internal_file, ("--phase-anchors",))
        closed = evaluate("evaluate_expert_closedloop.py", checkpoint, output / ("closedloop_" + label), internal_file)
        assert closed["completed"] and all(c["trial_valid"] for c in closed["cases"])
        row = {"label": label, "steps": steps, "openloop": opened["aggregates"],
               "closedloop": closed["aggregates"]}
        compared.append(row)
        (output / "comparison.json").write_text(json.dumps(compared, indent=2))
        print("VALIDATED " + json.dumps(row), flush=True)

    jobs = []
    for label, checkpoint in [("original_003000", runs / "train_reactive_3000_v4/evaluation/adapted"),
                              ("xyz_003000", experiment / "evaluation/adapted")]:
        destination = output / ("final_" + label)
        if (destination / "report.json").exists():
            assert read(destination / "report.json")["completed"]
            continue
        env = dict(os.environ)
        if label == "original_003000":
            env.update(CUDA_VISIBLE_DEVICES=args.reference_final_device,
                       MUJOCO_EGL_DEVICE_ID=args.reference_final_device)
        log = destination.with_suffix(".log").open("w")
        proc = subprocess.Popen(command("evaluate_expert_closedloop.py", checkpoint, destination,
                                       external_file, ("--external-holdout",)),
                                env=env, stdout=log, stderr=subprocess.STDOUT)
        jobs.append((proc, log))
    for proc, log in jobs:
        code = proc.wait()
        log.close()
        assert code == 0
    reports = {label: read(output / ("final_" + label) / "report.json")
               for label in ["original_003000", "xyz_003000"]}
    assert all(r["completed"] and len(r["cases"]) == 14 and
               all(c["trial_valid"] for c in r["cases"]) for r in reports.values())
    summary = {"completed": True, "all_trials_valid": True,
               "fixed_comparison_updates": 3000, "external_used_for_selection": False,
               "paired_external": {label: r["aggregates"] for label, r in reports.items()}}
    (output / "final_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
