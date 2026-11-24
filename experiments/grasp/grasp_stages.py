"""Optional grasp/lift handoff based only on gripper proprioception.

This controller never sees object poses or simulator contact identities.
It holds a model-issued close command after sustained opposing resistance.
The arm trajectory continues to come from the model. Thresholds are for this
uncalibrated simulator, not for deployment on hardware.
"""
from dataclasses import dataclass


@dataclass
class GraspStages:
    blocked_seconds: float = 0.
    held_gripper: float | None = None
    required_seconds: float = .12
    min_opening: float = 1.8
    min_effort_n: float = 1.

    @property
    def phase(self):
        return "lift" if self.held_gripper is not None else "grasp"

    def gripper_command(self, requested):
        return float(requested) if self.held_gripper is None else self.held_gripper

    def observe(self, requested, opening, effort, dt):
        if self.held_gripper is not None:
            return False
        closing = 0. <= max(0., float(requested)) < 1.5
        blocked = (closing and min(opening) > self.min_opening
                   and effort[0] < -self.min_effort_n
                   and effort[1] > self.min_effort_n)
        self.blocked_seconds = self.blocked_seconds + dt if blocked else 0.
        if self.blocked_seconds >= self.required_seconds:
            self.held_gripper = max(0., min(4.5, float(requested)))
            return True
        return False
