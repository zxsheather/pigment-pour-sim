"""Run the controller and motion layer together at their simulation cadences."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np

from simulation.motion.motion import BottlePose, MoveResult, PourMotion

@dataclass(frozen=True)
class PourObservation:
    time_s: float
    bottle_pose: BottlePose
    cylinder_mouth_xyz_m: tuple[float, float, float]
    cylinder_quat_wxyz: tuple[float, float, float, float]
    collected_ml: float
    requested_ml: float
    motion_status: MoveResult
    last_request: MoveResult | None = None


@dataclass(frozen=True)
class Move:
    target: BottlePose
    duration_s: float = 1.6


# Hold keeps the current motion request in effect.
PourController = Callable[[PourObservation], Move | Literal["hold", "finish", "abort"]]



class PourSystem:
    """Apply controller decisions directly to the robot's motion layer."""

    def __init__(self, model, data, controller: PourController):
        self.controller = controller
        self.motion = PourMotion(model)
        self.data = data
        self.dt = model.opt.timestep
        self.control_stride = max(1, round(.05 / self.dt))
        self.motion_status = MoveResult("idle")
        self.last_request = None
        self.status = "idle"
        self._step = 0
        self._mouth = model.site("held_bottle_mouth").id
        self._bottle = model.body("held_water_bottle").id
        self._cylinder = model.site("cylinder_mouth").id

    def start(self, requested_ml: float):
        if self.status != "idle":
            raise RuntimeError("Create a fresh pouring system for each task")
        self.requested_ml = requested_ml
        self.status = "running"

    def bottle_pose(self):
        return BottlePose(tuple(self.data.site(self._mouth).xpos),
                          tuple(self.data.body(self._bottle).xquat))

    def tick(self, collected_ml: float):
        """Run one control tick. The simulation runner advances physics."""
        if self.status != "running":
            return
        if self._step % self.control_stride == 0:
            cylinder = self.data.site(self._cylinder)
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, cylinder.xmat)
            action = self.controller(PourObservation(
                self.data.time, self.bottle_pose(), tuple(cylinder.xpos), tuple(quat),
                collected_ml, self.requested_ml, self.motion_status, self.last_request,
            ))
            if isinstance(action, Move):
                self.last_request = self.motion.request_move(action.target, action.duration_s, self.data)
            elif action in ("finish", "abort"):
                self.stop("finished" if action == "finish" else "aborted")
                return
            elif action != "hold":
                raise ValueError("Expected a Move, hold, finish, or abort from the controller")

        self.motion_status = self.motion.update(self.dt, self.data)
        self._step += 1
        if self.motion_status.status == "failed":
            self.stop("motion_failed")

    def stop(self, status: str):
        """Stop advancing requests and trajectories for this task."""
        if self.status == "running":
            self.status = status
