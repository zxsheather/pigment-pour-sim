"""Feedback controller for the fixed-start attached-bottle benchmark."""

from typing import Literal

import mujoco
import numpy as np

from simulation.motion.motion import BottlePose
from simulation.pour_system import Move, PourObservation


class ScriptedPourController:
    """Create a fresh controller for each episode."""

    tail_ml = 9.0

    def __init__(self):
        self._moves: tuple[Move, ...] = ()
        self._next_move = 0
        self._last_collected_ml = 0.0
        self._last_progress_s: float | None = None
        self._abort_after_retract = False

    def __call__(self, observation: PourObservation) -> Move | Literal["hold", "finish", "abort"]:
        if observation.last_request and observation.last_request.status == "rejected":
            return "abort"
        if not self._moves:
            start = observation.bottle_pose
            cx, cy, cz = observation.cylinder_mouth_xyz_m
            height = cz + 0.195
            upright = start.bottle_quat_wxyz
            raised = BottlePose((*start.mouth_xyz_m[:2], height), upright)
            aimed = BottlePose((cx - 0.011, cy, height), upright)
            turn, tipped = np.empty(4), np.empty(4)
            mujoco.mju_axisAngle2Quat(turn, np.array([0., 1., 0.]), np.deg2rad(105))
            mujoco.mju_mulQuat(tipped, turn, np.asarray(upright))
            self._moves = (
                Move(raised), Move(aimed),
                Move(BottlePose(aimed.mouth_xyz_m, tuple(tipped)), 2.4),
                Move(aimed, 2.4), Move(raised), Move(start),
            )

        status = observation.motion_status.status
        if self._next_move == 3:
            if observation.collected_ml > self._last_collected_ml:
                self._last_progress_s = observation.time_s
                self._last_collected_ml = observation.collected_ml
            if self._last_progress_s is None and status == "reached":
                self._last_progress_s = observation.time_s
            stalled = (
                self._last_progress_s is not None
                and observation.time_s - self._last_progress_s >= 2.0
            )
            # Require some measured discharge before interrupting the tilt.
            cutoff = (
                observation.collected_ml > 0.
                and observation.collected_ml + self.tail_ml >= observation.requested_ml
            )
            if not (cutoff or stalled):
                return "hold"
            self._abort_after_retract = stalled
        elif status not in ("idle", "reached"):
            return "hold"

        if self._next_move == len(self._moves):
            return "abort" if self._abort_after_retract else "finish"
        move = self._moves[self._next_move]
        self._next_move += 1
        return move
