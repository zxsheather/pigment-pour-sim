"""Fit and run a compact imitation policy for the idealized pour benchmark."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from simulation.world_model.paths import ROOT

from simulation.benchmarks.pour_benchmark import Action, cases, report, run_episode


LINEAR_LIMIT = 0.12


def observation_arrays(observations):
    positions = np.asarray([(o.dx, o.dy, o.dz) for o in observations])
    angles = np.asarray([o.tilt for o in observations])
    deficits = np.asarray([o.goal_ml - o.collected_ml for o in observations])
    return positions, angles, deficits


def fit_tree(features, targets, max_depth=7, min_leaf=8):
    """Fit a small squared-error decision tree with axis-aligned splits."""
    nodes = []

    def grow(inputs, outputs, depth):
        node = len(nodes)
        prediction = float(outputs.mean())
        nodes.append((-1, 0., -1, -1, prediction))
        if depth == max_depth or len(outputs) < 2 * min_leaf:
            return node
        best_cost = float(np.sum((outputs - prediction) ** 2))
        best = None
        counts = np.arange(1, len(outputs))
        for column in range(inputs.shape[1]):
            order = np.argsort(inputs[:, column])
            values = inputs[order, column]
            ordered_targets = outputs[order]
            split_sums = np.cumsum(ordered_targets)[:-1]
            other_sums = ordered_targets.sum() - split_sums
            costs = np.dot(ordered_targets, ordered_targets)
            costs = costs - split_sums ** 2 / counts - other_sums ** 2 / (len(outputs) - counts)
            valid = (values[:-1] < values[1:]) & (counts >= min_leaf)
            valid &= len(outputs) - counts >= min_leaf
            cut = int(np.argmin(np.where(valid, costs, np.inf)))
            if valid[cut] and costs[cut] < best_cost - 1e-9:
                best_cost = costs[cut]
                threshold = float((values[cut] + values[cut + 1]) / 2)
                best = column, threshold
        if best is not None:
            column, threshold = best
            left = inputs[:, column] <= threshold
            lower = grow(inputs[left], outputs[left], depth + 1)
            upper = grow(inputs[~left], outputs[~left], depth + 1)
            nodes[node] = (column, threshold, lower, upper, prediction)
        return node

    grow(features, targets, 0)
    return np.asarray(nodes)


class LearnedPolicy:
    """A fitted translation model and tilt tree with observation-only inference."""

    def __init__(self, slopes, centers, nodes):
        self.slopes = slopes
        self.centers = centers
        self.nodes = nodes

    @classmethod
    def fit(cls, observations, actions):
        positions, angles, deficits = observation_arrays(observations)
        translation = np.asarray([(a.vx, a.vy, a.vz) for a in actions])
        slopes, centers = [], []
        for axis in range(3):
            unsaturated = np.abs(translation[:, axis]) < LINEAR_LIMIT - 1e-6
            samples = positions[unsaturated, axis]
            design = np.column_stack((samples, np.ones(len(samples))))
            slope, intercept = np.linalg.lstsq(design, translation[unsaturated, axis], rcond=None)[0]
            slopes.append(slope)
            centers.append(-intercept / slope)
        centers = np.asarray(centers)
        offset = np.max(np.abs(positions - centers), axis=1)
        features = np.column_stack((offset, angles, deficits))
        nodes = fit_tree(features, np.asarray([a.tilt_rate for a in actions]))
        return cls(np.asarray(slopes), centers, nodes)

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, slopes=self.slopes, centers=self.centers, nodes=self.nodes)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as model:
            return cls(model["slopes"], model["centers"], model["nodes"])

    def __call__(self, observation):
        position = np.asarray([observation.dx, observation.dy, observation.dz])
        error = position - self.centers
        velocity = np.clip(self.slopes * error, -LINEAR_LIMIT, LINEAR_LIMIT)
        features = (np.max(np.abs(error)), observation.tilt,
                    observation.goal_ml - observation.collected_ml)
        node = 0
        while True:
            column, threshold, lower, upper, prediction = self.nodes[node]
            if column < 0:
                return Action(*map(float, velocity), float(prediction))
            node = int(lower if features[int(column)] <= threshold else upper)


def train():
    from simulation.controllers.pour_scripted import ScriptedPolicy

    rng = np.random.default_rng(41)
    teacher = ScriptedPolicy()
    observations, actions = [], []
    for initial in cases(seed=41, count=120):
        episode = run_episode(ScriptedPolicy(), initial)
        if not episode.success:
            continue
        for observation, action in zip(episode.observations, episode.actions):
            observations.append(observation)
            actions.append(action)

            # Query the teacher near demonstrations to cover small deviations
            # that a learned controller can encounter during its own rollouts.
            dx, dy, dz = rng.uniform(-0.003, 0.003, size=3)
            collected = max(0., observation.collected_ml + rng.uniform(-0.4, 0.4))
            nearby = replace(
                observation,
                dx=observation.dx + dx, dy=observation.dy + dy, dz=observation.dz + dz,
                tilt=max(0., observation.tilt + rng.uniform(-0.04, 0.04)),
                collected_ml=collected,
                remaining_ml=observation.remaining_ml - (collected - observation.collected_ml),
            )
            observations.append(nearby)
            actions.append(teacher(nearby))
    if not observations:
        raise RuntimeError("No successful scripted demonstrations were recorded")
    print(f"Training on {len(actions)} teacher-labeled observations")
    return LearnedPolicy.fit(observations, actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT /
                        "generated" / "pour_learned.npz")
    parser.add_argument("--load", action="store_true", help="Evaluate existing fitted weights")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cases", type=int, default=60)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")
    if args.model.suffix != ".npz":
        parser.error("--model must have a .npz extension")
    if not args.load:
        train().save(args.model)
        print(f"Saved {args.model}")
    policy = LearnedPolicy.load(args.model)
    episodes = [run_episode(policy, initial) for initial in cases(args.seed, args.cases)]
    print(f"Held-out seed {args.seed}")
    report(episodes)


if __name__ == "__main__":
    main()
