"""Record the scripted controller running the MuJoCo benchmark."""

from contextlib import ExitStack

import mujoco
from PIL import Image, ImageDraw

from simulation.benchmarks.mujoco_pour_benchmark import run_episode
from simulation.controllers.pour_scripted import ScriptedPourController
from simulation.pour_system import Move
from simulation.world_model.paths import ROOT


def main():
    controller = ScriptedPourController()
    phases = iter(("Raise", "Aim", "Tilt", "Return upright", "Retract", "Lower"))
    phase = "Start"
    requested_ml = 20.
    interval = .125
    next_frame = 0.
    frames = []

    def control(observation):
        nonlocal phase
        action = controller(observation)
        if isinstance(action, Move):
            phase = next(phases, "Move")
        return action

    with ExitStack() as stack:
        renderer = None

        def render(model, data, water):
            nonlocal renderer, next_frame
            if data.time + 1e-9 < next_frame:
                return
            next_frame += interval
            if renderer is None:
                renderer = stack.enter_context(mujoco.Renderer(model, width=560, height=380))
            renderer.update_scene(data, camera="pour_start_detail")
            water.draw(renderer.scene, data)
            frame = Image.fromarray(renderer.render())
            overlay = ImageDraw.Draw(frame)
            overlay.rectangle((0, 0, frame.width, 43), fill="#17242b")
            overlay.text((8, 5), f"{data.time:5.2f} s  |  {phase}  |  Target {requested_ml:.0f} mL",
                         fill="white")
            overlay.text((8, 23), f"Collected {water.collected_ml:.3f} mL  |  "
                         f"Spill {water.spilled_ml:.3f} mL", fill="white")
            frames.append(frame)

        run = run_episode(control, requested_ml=requested_ml, observe_frame=render)

    final = frames[-1].copy()
    draw = ImageDraw.Draw(final)
    draw.rectangle((0, 0, final.width, 43), fill="#17242b")
    draw.text((8, 5), f"{run.elapsed_s:.2f} s  |  {run.status}  |  "
              f"Error {run.score.signed_error_ml:+.3f} mL", fill="white")
    draw.text((8, 23), f"Collected {run.collected_ml:.3f} mL  |  Spill {run.spilled_ml:.3f} mL",
              fill="white")
    frames.append(final)
    path = ROOT / "generated" / "scripted_pour.gif"
    durations = [round(interval * 1000)] * (len(frames) - 1) + [1800]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print(f"{run.status}: {run.collected_ml:.3f} mL collected, {run.spilled_ml:.3f} mL spilled. Saved {path}")
    if not run.score.success:
        raise RuntimeError("Scripted pour did not pass the benchmark")


if __name__ == "__main__":
    main()
