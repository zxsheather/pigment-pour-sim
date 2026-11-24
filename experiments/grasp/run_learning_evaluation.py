"""Evaluate completed learning-curve snapshots and select before final holdout.

Launch on a GPU separate from the single training GPU. Paths are supplied by
the caller; this runner does not connect to machines or upload any artifacts.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--additional-checkpoint", type=Path)
    p.add_argument("--additional-steps", type=int, default=1000)
    p.add_argument("--additional-label", default="fp32_master")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    candidates = [
        (0, args.previous/"evaluation/baseline"),
        (100, args.previous/"evaluation/adapted"),
        (1000, args.experiment/"step_001000/evaluation/adapted"),
        (3000, args.experiment/"step_003000/evaluation/adapted"),
        (6000, args.experiment/"evaluation/adapted"),
    ]
    if args.additional_checkpoint:
        candidates.append((args.additional_steps,args.additional_checkpoint))
    summaries = []

    def execute(script, checkpoint, output, *extra):
        report = output/"report.json"
        if report.exists():
            existing = read(report)
            if existing.get("completed", len(existing.get("records", [])) == 60):
                return existing
            raise RuntimeError(f"Incomplete existing evaluation requires inspection: {output}")
        if output.exists():
            raise RuntimeError(f"Existing output without a report: {output}")
        command = [sys.executable, str(scripts/script), "--source", str(args.source),
                   "--checkpoint", str(checkpoint), "--output", str(output), *extra]
        with output.with_suffix(".log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        return read(report)

    for steps, checkpoint in candidates:
        extra = checkpoint == args.additional_checkpoint
        tag = f"fp32_{steps:06d}" if extra else f"{steps:06d}"
        variant = args.additional_label if extra else "original"
        deadline = time.monotonic() + 7200
        while not (checkpoint/"adaptation.json").exists():
            training_report = args.experiment/"report.json"
            if training_report.exists() and read(training_report).get("error"):
                raise RuntimeError("Training failed; inspect its report before continuing")
            if time.monotonic() > deadline:
                raise TimeoutError(f"Waiting for checkpoint {steps}")
            time.sleep(10)
        opened = execute("evaluate_expert_openloop.py", checkpoint, args.output/f"openloop_{tag}")
        closed = execute("evaluate_expert_closedloop.py", checkpoint, args.output/f"closedloop_{tag}")
        if not all(case["trial_valid"] for case in closed["cases"]):
            raise RuntimeError("An invalid simulation case requires inspection")
        summaries.append({"steps": steps, "tag":tag, "variant":variant,"openloop": opened["aggregates"],
                          "closedloop": closed["aggregates"]})
        (args.output/"comparison.json").write_text(json.dumps(summaries, indent=2))
        print("CHECKPOINT_EVALUATED " + json.dumps(summaries[-1]), flush=True)

    def rank(item):
        return (item["closedloop"]["unused_internal_episode"]["successes"],
                -item["openloop"]["unused_internal_episode"]["execute8_xyz_cm"],
                -item["steps"])

    # Keep the untuned baseline in the comparison, but select among updated
    # models for the task-specific adaptation experiment.
    chosen = max([s for s in summaries if s["steps"] > 0], key=rank)
    checkpoint = next(path for steps,path in candidates if steps==chosen["steps"]
                      and (path==args.additional_checkpoint)==(chosen["variant"]!="original"))
    selection = {"steps": chosen["steps"], "variant":chosen["variant"], "rule":
                 "Most successes on 3 unused internal positions; tie: lower internal execute8 XYZ error; tie: earlier step",
                 "external_holdout_used_for_selection": False, "metrics": chosen}
    (args.output/"selection.json").write_text(json.dumps(selection, indent=2))
    final = execute("evaluate_expert_closedloop.py", checkpoint, args.output/"final_external_holdout", "--external-holdout")
    (args.output/"final_summary.json").write_text(json.dumps({
        "completed": True, "selection": selection,
        "external_holdout": final["aggregates"],
        "external_cases_valid": all(c["trial_valid"] for c in final["cases"]),
    }, indent=2))
    print("FINAL_EVALUATION " + json.dumps(final["aggregates"]), flush=True)


if __name__ == "__main__":
    main()
