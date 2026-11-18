import mujoco
import numpy as np
import pytest

from simulation.benchmarks.mujoco_pour_benchmark import run_episode, score_pour
from simulation.motion.motion import BottlePose
from simulation.pour_system import Move


@pytest.mark.parametrize(
    "collected, spilled, completed, signed_error, success",
    [
        (20., 0., True, 0., True),
        (15., 0., True, -5., False),
        (0., 20., True, -20., False),
        (20., 1., True, 0., False),
        (20., 0., False, 0., False),
    ],
)
def test_score_pour(collected, spilled, completed, signed_error, success):
    score = score_pour(20., collected, spilled, completed=completed)
    assert score.signed_error_ml == signed_error
    assert score.absolute_error_ml == abs(signed_error)
    assert score.success is success


def test_no_op_times_out_from_repeatable_reset(scenes):
    observations = []

    def hold(observation):
        observations.append(observation)
        return "hold"

    first = run_episode(hold, scene_path=scenes["pour_start"], timeout_s=.1)
    first_observations = observations.copy()
    observations.clear()
    second = run_episode(hold, scene_path=scenes["pour_start"], timeout_s=.1)
    assert first == second
    assert observations == first_observations
    assert observations[0].time_s == 0.
    assert observations[0].collected_ml == 0.
    assert observations[0].motion_status.status == "idle"
    assert first.status == "timeout"
    assert first.elapsed_s == pytest.approx(.1)
    assert first.retract_pose == observations[0].bottle_pose
    assert first.collected_ml == first.spilled_ml == 0.
    assert first.initial_ml == first.remaining_ml == 100.
    assert first.score.signed_error_ml == -20.
    assert not first.score.success
    assert not first.contacts


def test_routes_move_and_abort(scenes):
    statuses = []

    def controller(observation):
        status = observation.motion_status.status
        statuses.append(status)
        if status == "idle":
            return Move(observation.bottle_pose, duration_s=.1)
        if status == "reached":
            return "abort"
        return "hold"

    run = run_episode(controller, scene_path=scenes["pour_start"], timeout_s=1.)
    assert "executing" in statuses
    assert run.status == "aborted"
    assert run.motion_status.status == "reached"


def test_rejection_does_not_hide_active_move_status(scenes):

    def controller(observation):
        if observation.motion_status.status == "idle":
            x, y, z = observation.bottle_pose.mouth_xyz_m
            return Move(BottlePose((x, y, z + .04),
                                   observation.bottle_pose.bottle_quat_wxyz), 1.2)
        if observation.last_request.status == "accepted":
            return Move(observation.bottle_pose, duration_s=0.)
        assert observation.last_request.status == "rejected"
        return "abort" if observation.motion_status.status == "reached" else "hold"

    run = run_episode(controller, scene_path=scenes["pour_start"], timeout_s=3.)
    assert run.status == "aborted"
    assert run.motion_status.status == "reached"


def test_deadline_stops_active_move(scenes):
    times = []

    def controller(observation):
        times.append(observation.time_s)
        if len(times) == 1:
            return Move(observation.bottle_pose, duration_s=1.)
        return "hold"

    run = run_episode(controller, scene_path=scenes["pour_start"], timeout_s=.103)
    assert run.status == "timeout"
    assert run.elapsed_s == pytest.approx(.1)
    assert max(times) < .103
    assert not run.score.success


def test_finish_without_pouring_is_incomplete(scenes):
    run = run_episode(lambda _: "finish", scene_path=scenes["pour_start"])
    assert run.status == "incomplete"
    assert run.elapsed_s == 0.
    assert not run.score.success


def test_completed_pour_runs_through_system_and_benchmark(scenes):
    moves = None

    def controller(observation):
        nonlocal moves
        if moves is None:
            start = observation.bottle_pose
            cx, cy, cz = observation.cylinder_mouth_xyz_m
            height = cz + .195
            upright = start.bottle_quat_wxyz
            raised = BottlePose((*start.mouth_xyz_m[:2], height), upright)
            above = BottlePose((cx - .011, cy, height), upright)
            turn, tipped = np.empty(4), np.empty(4)
            mujoco.mju_axisAngle2Quat(turn, np.array([0., 1., 0.]), np.deg2rad(105))
            mujoco.mju_mulQuat(tipped, turn, np.asarray(upright))
            moves = iter((Move(raised), Move(above), Move(BottlePose(
                above.mouth_xyz_m, tuple(tipped)), 2.4), Move(above, 2.4),
                Move(raised), Move(start)))
        if observation.motion_status.status in ("idle", "reached"):
            return next(moves, "finish")
        return "hold"

    run = run_episode(controller, scene_path=scenes["pour_start"], timeout_s=20.)
    assert run.status == "completed", run
    assert run.collected_ml > 0.
    assert run.spilled_ml <= .5
    assert run.initial_ml == pytest.approx(run.remaining_ml + run.collected_ml + run.spilled_ml)
