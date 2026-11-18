"""Compare pour controllers in a task-space environment with simplified flow."""

import argparse
from dataclasses import dataclass
from math import hypot, isfinite
from random import Random

DT = 0.05
MAX_STEPS = 160
CAPTURE_RADIUS = 0.012
MIN_HEIGHT = 0.035
MAX_HEIGHT = 0.09


@dataclass(frozen=True)
class Observation:
    dx: float
    dy: float
    dz: float
    tilt: float
    collected_ml: float
    remaining_ml: float
    goal_ml: float


@dataclass(frozen=True)
class Action:
    vx: float
    vy: float
    vz: float
    tilt_rate: float


@dataclass
class Episode:
    observations: list[Observation]
    actions: list[Action]
    spill_ml: float
    success: bool

    @property
    def elapsed_s(self):
        return len(self.actions) * DT

    @property
    def error_ml(self):
        final = self.observations[-1]
        return abs(final.goal_ml - final.collected_ml)


def cases(seed, count):
    rng = Random(seed)
    return [
        Observation(rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05),
                    rng.uniform(0.05, 0.10), 0., 0., 100., (15., 20., 25.)[i % 3])
        for i in range(count)
    ]


def aligned(obs):
    return hypot(obs.dx, obs.dy) <= CAPTURE_RADIUS and MIN_HEIGHT <= obs.dz <= MAX_HEIGHT


def clip(value, low, high):
    return max(low, min(high, value))


def run_episode(policy, initial):
    obs = initial
    observations, actions = [obs], []
    spill = 0.
    for _ in range(MAX_STEPS):
        command = policy(obs)
        values = (command.vx, command.vy, command.vz, command.tilt_rate)
        if not all(isfinite(v) for v in values):
            raise ValueError("Controller returned a non-finite command")
        action = Action(*(clip(v, -limit, limit)
                          for v, limit in zip(values, (0.12, 0.12, 0.12, 1.))))
        tilt = clip(obs.tilt + action.tilt_rate * DT, 0., 1.8)
        transferred = min(obs.remaining_ml, 12. * DT * clip((tilt - 0.9) / 0.7, 0., 1.))
        next_obs = Observation(
            obs.dx + action.vx * DT, obs.dy + action.vy * DT, obs.dz + action.vz * DT,
            tilt, obs.collected_ml, obs.remaining_ml - transferred, obs.goal_ml,
        )
        captured = aligned(next_obs)
        if captured:
            next_obs = Observation(
                next_obs.dx, next_obs.dy, next_obs.dz, tilt,
                obs.collected_ml + transferred, next_obs.remaining_ml, obs.goal_ml,
            )
        else:
            spill += transferred
        actions.append(action)
        observations.append(next_obs)
        obs = next_obs
        if spill > 0.5 or obs.collected_ml > obs.goal_ml + 1.:
            break
        if captured and obs.tilt <= 0.1 and abs(obs.collected_ml - obs.goal_ml) <= 1.:
            return Episode(observations, actions, spill, True)
    return Episode(observations, actions, spill, False)


def evaluate(policy_factory, initial_conditions):
    return [run_episode(policy_factory(), initial) for initial in initial_conditions]


def report(episodes):
    count = len(episodes)
    print(f"Success: {sum(ep.success for ep in episodes)}/{count}")
    print(f"Mean absolute volume error: {sum(ep.error_ml for ep in episodes) / count:.3f} mL")
    print(f"Maximum absolute volume error: {max(ep.error_ml for ep in episodes):.3f} mL")
    print(f"Maximum spill: {max(ep.spill_ml for ep in episodes):.3f} mL")
    print(f"Mean time: {sum(ep.elapsed_s for ep in episodes) / count:.2f} s")


def main():
    from simulation.controllers.pour_scripted import ScriptedPolicy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cases", type=int, default=60)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")
    report(evaluate(ScriptedPolicy, cases(args.seed, args.cases)))


if __name__ == "__main__":
    main()
