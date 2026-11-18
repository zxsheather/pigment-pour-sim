"""Reference feedback controller for the idealized pouring benchmark."""

from simulation.benchmarks.pour_benchmark import Action, Observation


TIME_STEP_S = 0.05
POSITION_LIMIT_MPS = 0.12
START_FLOW_RAD = 0.9
POUR_TILT_RAD = 1.6
MAX_FLOW_MLPS = 12.0
ALIGNMENT_TOLERANCE_M = 0.002


def _position_command(error_m: float) -> float:
    """Correct as much of the remaining error as this tick permits."""
    return max(-POSITION_LIMIT_MPS, min(POSITION_LIMIT_MPS, error_m / TIME_STEP_S))


class ScriptedPolicy:
    """Align the mouth, tip, dose, and return upright for one episode."""

    def __call__(self, observation: Observation) -> Action:
        height_error = 0.06 - observation.dz
        vx = _position_command(-observation.dx)
        vy = _position_command(-observation.dy)
        vz = _position_command(height_error)
        alignment_error = max(abs(observation.dx), abs(observation.dy), abs(height_error))
        aligned = alignment_error <= ALIGNMENT_TOLERANCE_M

        # At 1 rad/s the nominal linear flow ramp integrates to this amount
        # while the bottle returns from its operating angle to the flow cutoff.
        return_volume_ml = 0.5 * MAX_FLOW_MLPS * (POUR_TILT_RAD - START_FLOW_RAD)
        # Collected volume only increases, so this condition stays true while righting.
        should_add_volume = observation.goal_ml - observation.collected_ml > return_volume_ml
        target_tilt = POUR_TILT_RAD if aligned and should_add_volume else 0.0

        tilt_rate = max(-1.0, min(1.0, (target_tilt - observation.tilt) / TIME_STEP_S))
        return Action(vx, vy, vz, tilt_rate)
