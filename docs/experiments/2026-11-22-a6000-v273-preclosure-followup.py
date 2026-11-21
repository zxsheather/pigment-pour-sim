"""Extend the matched V247 comparison to the next audited raw checkpoint."""
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
import time


stage = Path(__file__).resolve().parent
runs = stage.parent
target = runs / "cylinder-joint-preclosure-realignment-bc-v247-a6000"
probe = runs / "cylinder-preclosure-official-probe-v257"
script = probe / "probe-v257-official-fixed-state.py"
document = "v257-official-fixed-state-aperture-probe.json"
parent = probe / "v247-parent" / document
first = probe / "v247-step-128" / document
second = probe / "v247-step-256" / document
status_path = stage / "v273-preclosure-followup-status.json"


def read(path):
    return json.loads(path.read_text())


def ref(path):
    return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}


def write(path, data):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def status(phase, **details):
    write(status_path, dict(phase=phase, independent_cases_read=False, full_task_measured=False,
                            timestamp=time.time(), **details))


def candidate():
    return next(runs.glob("*/candidate-ready.json"), None)


def compare(reference, current, step):
    baseline, evaluated = read(reference), read(current)
    audit_path = target / f"checkpoint-{step}-weight-audit.json"
    audit = read(audit_path)
    assert evaluated["model"] == audit["checkpoint"] and evaluated["model_sha256"] == audit["checkpoint_sha256"]
    assert audit["passed"] and baseline["config_sha256"] == evaluated["config_sha256"]
    cases = []
    for before, after in zip(baseline["snapshots"], evaluated["snapshots"], strict=True):
        assert before["label"] == after["label"] and before["evidence"] == after["evidence"]
        trials = []
        for initial, trained in zip(before["probe"], after["probe"], strict=True):
            assert initial["seed"] == trained["seed"] and initial["start"] == trained["start"]
            assert initial["geometry_unchanged"] and trained["geometry_unchanged"]
            measurements = {
                key: [initial[key], trained[key]]
                for key in ("actual_aperture_change_mm", "maximum_tilt_rad", "forward_mm",
                            "corrected_lateral_mm", "executed_ticks", "terminal_reason")
            }
            a, b = measurements["actual_aperture_change_mm"]
            trials.append(dict(seed=initial["seed"], measurements=measurements,
                               aperture_difference_mm=b-a,
                               both_executed_full_block=measurements["executed_ticks"] == [8, 8]))
        assert len(trials) == 3
        equal_horizon = [r["aperture_difference_mm"] for r in trials if r["both_executed_full_block"]]
        cases.append(dict(label=before["label"], tick=before["evidence"]["tick"], trials=trials,
                          same_eight_tick_pairs=len(equal_horizon),
                          median_aperture_difference_mm=median(equal_horizon) if equal_horizon else None))
    assert len(cases) == 3
    return dict(reference=ref(reference), evaluated=ref(current), weight_audit=ref(audit_path),
                cases=cases, full_task_measured=False, independent_cases_read=False)


def wait_for_audit(audit):
    while not audit.is_file():
        if candidate():
            return False
        progress = read(target / "execution-status.json")
        proc = subprocess.run(["ps", "-p", str(progress["driver_pid"]), "-o", "stat=,args="],
                              capture_output=True, text=True).stdout.strip()
        if (not proc or proc.startswith("Z") or "train_and_evaluate_cylinder_joints.py" not in proc):
            return audit.is_file()
        time.sleep(10)
    return not candidate()


def main():
    first_result = stage / "v273-v247-step128-alignment.json"
    write(first_result, compare(parent, first, 128))
    status("waiting_for_v247_step256_audit", step128=ref(first_result))
    audit = target / "checkpoint-256-weight-audit.json"
    if not wait_for_audit(audit):
        status("yielded_to_candidate" if candidate() else "v247_ended_before_step256_audit",
               step128=ref(first_result))
        return
    assert read(audit)["passed"]
    status("running_v247_256_matched_probe", step128=ref(first_result), step256_audit=ref(audit))
    with (stage / "v273-v247-step256-probe.log").open("x") as log:
        result = subprocess.run([sys.executable, str(script), "--v247-step", "256"],
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        if candidate():
            status("yielded_to_candidate", step128=ref(first_result))
            return
        status("v247_256_probe_failed", exit_code=result.returncode, step128=ref(first_result))
        result.check_returncode()
    comparison = stage / "v273-v247-step256-alignment.json"
    write(comparison, {"versus_parent": compare(parent, second, 256),
                       "versus_step128": compare(first, second, 256)})
    status("complete", step128=ref(first_result), step256=ref(comparison))


if __name__ == "__main__":
    main()
