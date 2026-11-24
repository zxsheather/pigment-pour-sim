"""Reproduce an accepted correction's initial state using recorded commands.

This is an explicitly assisted, in-sample recovery probe. It is never scored
as autonomous pickup from the ready pose. No physics state is teleported.
"""
import json

import numpy as np

from collect_expert import state_vector
from reactive_expert import apply_setup, reset_recording


def replay_takeover_start(env, metadata, runs):
    assert metadata["accepted"] and metadata["source_split"] == "train_episode"
    parent = runs / metadata["source_batch"] / f"episode_{metadata['source_episode']:04d}"
    parent_metadata = json.loads((parent / "episode.json").read_text())
    evaluation = runs / metadata["source_evaluation"]
    report = json.loads((evaluation / "report.json").read_text())
    matches = [case for case in report["cases"] if case["split"] == "train_episode"
               and case["batch"] == metadata["source_batch"]
               and case["episode"] == metadata["source_episode"]]
    assert len(matches) == 1 and matches[0]["trial_valid"]
    recorded = evaluation / matches[0]["directory"]
    chunks = np.asarray(json.loads((recorded / "actions.json").read_text()))
    commands = chunks[:, :8].reshape(-1, 7)
    prefix = metadata["prefix_steps"]
    assert 1 <= prefix <= len(commands)
    apply_setup(env, parent_metadata["setup"])
    for command in commands[:prefix]:
        env.execute(command)
    expected = json.loads((recorded / "telemetry.json").read_text())[prefix - 1]
    robot_error = float(np.max(np.abs(state_vector(env.observation()[0]) - metadata["initial_robot_state"])))
    bottle_error = float(np.max(np.abs(env.d.xpos[env.bottle] - expected["bottle_xyz"])))
    assert max(robot_error, bottle_error) < 1e-6, "Correction start did not replay exactly"
    reset_recording(env)
    return {"kind": "recorded_model_prefix", "prefix_steps": prefix,
            "robot_state_error": robot_error, "bottle_position_error": bottle_error,
            "in_sample_training_scene": True, "full_ready_pose_pickup": False}
