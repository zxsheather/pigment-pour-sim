from simulation.benchmarks.mujoco_pour_benchmark import run_episode
from simulation.controllers.pour_scripted import ScriptedPourController
from simulation.pour_system import Move


def test_scripted_pour_progress(scenes):
    controller = ScriptedPourController()
    phase_names = iter(("raise", "aim", "tilt", "return upright", "retract", "lower"))

    def report_progress(observation):
        action = controller(observation)
        if isinstance(action, Move):
            label = next(phase_names, "move")
        elif action in ("finish", "abort"):
            label = action
        else:
            return action
        print(f"{observation.time_s:5.2f} s | {label} | "
              f"collected {observation.collected_ml:.3f} mL")
        return action

    result = run_episode(report_progress, requested_ml=20., scene_path=scenes["pour_start"])
    print(f"Final: {result.status} | collected {result.collected_ml:.3f} mL | "
          f"spill {result.spilled_ml:.3f} mL | error {result.score.signed_error_ml:+.3f} mL")
    assert result.status == "completed"
    assert result.score.success
