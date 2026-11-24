"""Evaluate a fixed final correction-data checkpoint; never select on external cases."""
from pathlib import Path
import argparse
import json
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=Path("experiments/grasp/runs"))
    parser.add_argument("--cohort", choices=["control", "augmented"], required=True)
    args = parser.parse_args()
    output = args.runs / "takeover_training_eval_v6"
    assert json.loads((output / "cases_ready.json").read_text())["ready"]
    training = args.runs / f"train_takeover_{args.cohort}_3000_v6"
    trained = json.loads((training / "report.json").read_text())
    assert trained["completed"]
    checkpoint = training / "evaluation/adapted"
    description = json.loads((checkpoint / "adaptation.json").read_text())
    assert description["global_step"] == 3000 and description["logical_total_weight_updates"] == 6000
    assert description["action_expert_initialization"]["source_completed_updates"] == 3000
    membership = json.loads((training / "dataset_membership.json").read_text())
    assert membership["passed"] and membership["all_corrective_episodes_in_gradient_training"]
    results = {}
    for mode, case_file in [("openloop", "internal"), ("closedloop", "internal"), ("closedloop", "external")]:
        destination = output / f"{args.cohort}_{mode}_{case_file}"
        assert not destination.exists(), "Do not overwrite an existing evaluation"
        command = [sys.executable, str(Path(__file__).with_name(f"evaluate_expert_{mode}.py")),
            "--source", str(args.source), "--checkpoint", str(checkpoint), "--runs", str(args.runs),
            "--norm-key", "local/pigment_grasp_reactive_v4", "--output", str(destination),
            "--cases", str(output / (case_file + "_cases.json"))]
        if mode == "openloop":
            command.append("--phase-anchors")
        with destination.with_suffix(".log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        report = json.loads((destination / "report.json").read_text())
        if mode == "closedloop":
            assert report["completed"] and all(c["trial_valid"] for c in report["cases"])
        results[mode + "_" + case_file] = report["aggregates"]
    (output / (args.cohort + "_completed.json")).write_text(json.dumps(
        {"completed": True, "cohort": args.cohort, "results": results}, indent=2))
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
