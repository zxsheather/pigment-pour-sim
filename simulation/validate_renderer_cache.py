"""Compare cached renderers with the old per-resolution recreation behavior.

Uses identical preserved physics for both render paths, without a model or
physics step. Checks raster differences, reading results and integration-state
parity. EGL repeats can differ sparsely by two 8-bit intensity levels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from carry_env import CAMERAS
from cylinder_reader import read_volume
from inspect_cylinder import CylinderEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--controller-config", type=Path, required=True)
    parser.add_argument("--snapshots", type=int, nargs="+", default=[0, 50])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((args.episode / "summary.json").read_text())
    case = source["case"]
    controller = json.loads(args.controller_config.read_text())
    env = CylinderEnv(case["volume"], image_size=768, **controller["environment"],
                      friction_scale=case.get("friction_scale", 1.),
                      mass_scale=case.get("mass_scale", 1.))
    saved = np.load(args.episode / "recovery_physics.npz")
    spec = int(saved["state_spec"])
    rows = []
    legacy = None
    creations = 0
    start = time.monotonic()
    try:
        for index in args.snapshots:
            mujoco.mj_setState(env.model, env.data, saved["full_state"][index], spec)
            mujoco.mj_forward(env.model, env.data)
            before = np.empty(mujoco.mj_stateSize(env.model, spec))
            mujoco.mj_getState(env.model, env.data, before, spec)
            for size in [448, 768]:
                for camera in CAMERAS:
                    if legacy is None or legacy.width != size:
                        if legacy is not None:
                            legacy.close()
                        legacy = mujoco.Renderer(env.model, height=size, width=size)
                        creations += 1
                    legacy.update_scene(env.data, camera=camera)
                    reference = legacy.render().copy()
                    legacy_repeat = legacy.render().copy()
                    cached = env.render(camera, width=size)
                    cached_repeat = env.render(camera, width=size)
                    difference = abs(reference.astype(int)-cached.astype(int))
                    changed_fraction = np.any(difference, axis=2).mean()
                    reader_equal = (camera != "face_view" or read_volume(reference) == read_volume(cached))
                    legacy_difference = abs(reference.astype(int)-legacy_repeat.astype(int))
                    cached_difference = abs(cached.astype(int)-cached_repeat.astype(int))
                    rows.append(dict(snapshot=index, width=size, camera=camera,
                                     pixels_exactly_equal=bool(np.array_equal(reference, cached)),
                                     max_channel_difference=int(difference.max()),
                                     changed_pixel_fraction=float(changed_fraction),
                                     mean_channel_difference=float(difference.mean()),
                                     reader_equal=reader_equal,
                                     within_raster_tolerance=bool(difference.max() <= 2 and changed_fraction <= .001),
                                     legacy_repeat_max_difference=int(legacy_difference.max()),
                                     legacy_repeat_changed_fraction=float(np.any(legacy_difference,axis=2).mean()),
                                     cached_repeat_max_difference=int(cached_difference.max()),
                                     cached_repeat_changed_fraction=float(np.any(cached_difference,axis=2).mean()),
                                     image_sha256=hashlib.sha256(cached.tobytes()).hexdigest()))
            after = np.empty_like(before)
            mujoco.mj_getState(env.model, env.data, after, spec)
            assert np.array_equal(before, after)
        identities = {size: id(renderer) for size, renderer in env._renderers.items()}
        for _ in range(20):
            for size in [448, 768]:
                env.render("face_view", width=size)
                assert id(env._renderers[size]) == identities[size]
        assert set(env._renderers) == {448, 768}
    finally:
        if legacy is not None:
            legacy.close()
        env.close()
    assert not env._renderers and env.renderer is None
    passed=all(row["within_raster_tolerance"] and row["reader_equal"] for row in rows)
    report = dict(passed=passed, comparisons=rows, integration_states_unchanged=True,
                  readers_equal=all(row["reader_equal"] for row in rows), cached_renderer_count=2,
                  additional_alternating_renders_without_recreation=40,
                  legacy_renderer_creations_for_reference_images=creations,
                  cached_renderers_closed=True, elapsed_wall_seconds=time.monotonic()-start,
                  source_sha256=hashlib.sha256((Path(__file__).parent/"carry_env.py").read_bytes()).hexdigest(),
                  pixel_tolerance="At most 2/255 channel difference and 0.1% changed pixels; RGB reader must be exactly equal",
                  tolerance_basis="Unmodified same-context EGL repeats in validation-v3 also showed sparse 2-level differences; initial exact/1-level failures retained. Physical task gates unchanged.",
                  note="Raster/physics and renderer-lifecycle check only; no policy task score. Not bitwise image identity.")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not passed:raise SystemExit(1)


if __name__ == "__main__":
    main()
