"""Interpret physics-step contact and height measurements independently."""
import gzip
import json


def intervals(rows, predicate, dt):
    result, start = [], None
    for i, row in enumerate(rows):
        if predicate(row) and start is None:
            start = i
        if start is not None and (not predicate(row) or i == len(rows) - 1):
            end = i if predicate(row) else i - 1
            result.append({"start_s": rows[start]["time_s"] - dt,
                "end_s": rows[end]["time_s"], "duration_s": (end - start + 1) * dt,
                "start_index": start, "end_index": end})
            start = None
    return result


def summarize_trace(rows, dt):
    assert rows and dt > 0
    contact = intervals(rows, lambda r: r["contacts"] == 2, dt)
    held = intervals(rows, lambda r: r["contacts"] == 2 and r["lift_m"] >= .05, dt)
    lifted = intervals(rows, lambda r: r["lift_m"] >= .05, dt)
    breaks = []
    for run in held:
        if run["end_index"] == len(rows) - 1:
            continue
        after = rows[run["end_index"] + 1]
        reasons = []
        if after["contacts"] != 2:
            reasons.append("finger_contact_lost")
        if after["lift_m"] < .05:
            reasons.append("height_below_5cm")
        assert reasons
        breaks.append({"time_s": after["time_s"], "preceding_held_s": run["duration_s"],
                       "reasons": reasons, "before": rows[run["end_index"]], "after": after})
    sustained = [b for b in breaks if b["preceding_held_s"] >= .08 - 1e-9]
    return {"physics_steps": len(rows), "physics_timestep_s": dt,
        "max_bilateral_contact_s": max((r["duration_s"] for r in contact), default=0.),
        "max_bilateral_lifted_s": max((r["duration_s"] for r in held), default=0.),
        "max_above_5cm_s": max((r["duration_s"] for r in lifted), default=0.),
        "max_bottle_bottom_lift_m": max(r["lift_m"] for r in rows),
        "first_contact_s": next((r["time_s"] for r in rows if r["contacts"] > 0), None),
        "first_bilateral_contact_s": next((r["time_s"] for r in rows if r["contacts"] == 2), None),
        "first_bilateral_lift_s": next((r["time_s"] for r in rows if r["contacts"] == 2 and r["lift_m"] >= .05), None),
        "first_sustained_hold_break": sustained[0] if sustained else None,
        "held_break_counts": {reason: sum(reason in b["reasons"] for b in breaks)
                              for reason in ["finger_contact_lost", "height_below_5cm"]},
        "held_intervals": held, "bilateral_intervals": contact,
        "held_breaks": [{k: v for k, v in b.items() if k not in ["before", "after"]} for b in breaks]}


def write_trace(path, rows):
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def read_trace(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]
