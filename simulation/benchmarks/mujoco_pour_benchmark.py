"""Run and score a fixed-start pour using a caller-supplied controller."""

import json
from dataclasses import dataclass
from math import floor, isfinite
from pathlib import Path

import mujoco
import numpy as np

from simulation.motion.motion import BottlePose, MoveResult
from simulation.pour_system import PourController, PourSystem
from simulation.world_model.collision import workspace_contacts
from simulation.world_model.paths import ROOT, SCENE_CONFIG
from simulation.world_model.water import PourWater

POSITION_TOLERANCE_M = .005
ANGLE_TOLERANCE_RAD = np.deg2rad(1.)
MAX_VOLUME_ERROR_ML = 1.
MAX_SPILL_ML = .5


@dataclass(frozen=True)
class PourScore:
    signed_error_ml: float
    absolute_error_ml: float
    success: bool


def score_pour(
    requested_ml: float, collected_ml: float, spilled_ml: float, *, completed: bool
) -> PourScore:
    """Evaluate final volumes once the discharged water has landed."""
    error = collected_ml - requested_ml
    absolute_error = abs(error)
    return PourScore(
        error,
        absolute_error,
        completed and absolute_error <= MAX_VOLUME_ERROR_ML and spilled_ml <= MAX_SPILL_ML,
    )


@dataclass(frozen=True)
class PourRun:
    status: str
    elapsed_s: float
    settling_s: float
    requested_ml: float
    initial_ml: float
    collected_ml: float
    spilled_ml: float
    remaining_ml: float
    retract_pose: BottlePose
    final_pose: BottlePose
    grasp_retained: bool
    contacts: tuple[tuple[int, int], ...]
    motion_status: MoveResult
    score: PourScore


def run_episode(
    controller: PourController,
    requested_ml: float = 20.,
    *,
    scene_path: Path = ROOT / "generated" / "pour_start.xml",
    timeout_s: float = 60.,
) -> PourRun:
    """Run one headless attached-bottle episode at a 50 ms control cadence."""
    if not isfinite(requested_ml) or requested_ml <= 0:
        raise ValueError("Expected a finite positive requested volume")
    if not isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Expected a finite positive episode timeout")

    config = json.loads(SCENE_CONFIG.read_text())
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("pour_start").id)
    mujoco.mj_forward(model, data)
    dt = model.opt.timestep
    bottle = model.body("held_water_bottle").id

    if model.body_weldid[bottle] == bottle:
        raise ValueError("The benchmark requires a rigidly attached bottle")
    contacts = set(workspace_contacts(model, data))
    # Use the same servo settling period as the water demo, before episode time.
    for _ in range(round(1. / dt)):
        if contacts:
            break
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        contacts.update(workspace_contacts(model, data))
    data.time = 0.
    system = PourSystem(model, data, controller)
    retract_pose = system.bottle_pose()
    water = PourWater(model, data, config)
    system.start(requested_ml)
    status = "contact" if contacts else "timeout"

    # Step only up to the deadline. No controller tick can occur at or after it.
    for _ in range(floor(timeout_s / dt + 1e-9)):
        if contacts:
            break
        system.tick(water.collected_ml)
        if system.status != "running":
            status = system.status
            break
        water.apply_load(data)
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        water.step(data, dt)
        contacts.update(workspace_contacts(model, data))
        if contacts:
            status = "contact"
            break

    system.stop(status)
    final_pose = system.bottle_pose()
    if status == "finished":
        angular_error = np.empty(3)
        mujoco.mju_subQuat(angular_error, np.asarray(final_pose.bottle_quat_wxyz),
                           np.asarray(retract_pose.bottle_quat_wxyz))
        retracted = (np.linalg.norm(np.subtract(final_pose.mouth_xyz_m, retract_pose.mouth_xyz_m))
                     <= POSITION_TOLERANCE_M
                     and np.linalg.norm(angular_error) <= ANGLE_TOLERANCE_RAD)
        status = "completed" if (retracted and system.motion_status.status == "reached"
                                 and water.remaining_ml < water.initial_ml) else "incomplete"

    # At termination, freeze the robot and score only water released
    # during the episode. The workspace and cylinder are static.
    settling_steps = 0
    while water.parcels:
        water.advance_parcels(dt)
        settling_steps += 1

    return PourRun(
        status=status,
        elapsed_s=data.time,
        settling_s=settling_steps * dt,
        requested_ml=requested_ml,
        initial_ml=water.initial_ml,
        collected_ml=water.collected_ml,
        spilled_ml=water.spilled_ml,
        remaining_ml=water.remaining_ml,
        retract_pose=retract_pose,
        final_pose=final_pose,
        grasp_retained=True,  # The pour_start scene attaches the bottle rigidly.
        contacts=tuple(sorted(contacts)),
        motion_status=system.motion_status,
        score=score_pour(requested_ml, water.collected_ml, water.spilled_ml,
                         completed=status == "completed"),
    )
