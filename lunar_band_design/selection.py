"""D-optimal greedy selection and a small genetic-search audit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SelectionResult:
    indices: tuple[int, ...]
    objective: float


def contrast_vectors(response: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Whiten component contrasts relative to the final component."""
    if response.shape[1] < 2:
        raise ValueError("At least two components are required")
    return (response[:, :-1] - response[:, [-1]]) / sigma[:, None]


def objective(vectors: np.ndarray, indices: tuple[int, ...] | list[int]) -> float:
    dimension = vectors.shape[1]
    fisher = np.eye(dimension)
    for index in indices:
        vector = vectors[index]
        fisher += np.outer(vector, vector)
    sign, value = np.linalg.slogdet(fisher)
    return float(value) if sign > 0 else -np.inf


def compatible(
    candidate: int,
    selected: list[int] | tuple[int, ...],
    centers: np.ndarray,
    widths: np.ndarray,
    spacing_factor: float,
) -> bool:
    for existing in selected:
        required = spacing_factor * 0.5 * (widths[candidate] + widths[existing])
        if abs(centers[candidate] - centers[existing]) < required:
            return False
    return True


def greedy_select(
    vectors: np.ndarray,
    centers: np.ndarray,
    widths: np.ndarray,
    count: int,
    spacing_factor: float,
) -> SelectionResult:
    """Sequentially maximize regularized log-determinant gain."""
    dimension = vectors.shape[1]
    fisher_inverse = np.eye(dimension)
    selected: list[int] = []
    for _ in range(count):
        gains = np.full(vectors.shape[0], -np.inf)
        for index, vector in enumerate(vectors):
            if index in selected or not compatible(
                index, selected, centers, widths, spacing_factor
            ):
                continue
            gains[index] = np.log1p(vector @ fisher_inverse @ vector)
        best = int(np.argmax(gains))
        if not np.isfinite(gains[best]):
            raise ValueError("Spacing constraint leaves too few feasible bands")
        vector = vectors[best]
        denominator = 1.0 + vector @ fisher_inverse @ vector
        fisher_inverse -= (
            np.outer(fisher_inverse @ vector, vector @ fisher_inverse) / denominator
        )
        selected.append(best)
    ordered = tuple(sorted(selected, key=lambda i: centers[i]))
    return SelectionResult(ordered, objective(vectors, ordered))


def _random_feasible(
    rng: np.random.Generator,
    candidate_count: int,
    count: int,
    centers: np.ndarray,
    widths: np.ndarray,
    spacing_factor: float,
) -> tuple[int, ...] | None:
    order = rng.permutation(candidate_count)
    selected: list[int] = []
    for index in order:
        if compatible(int(index), selected, centers, widths, spacing_factor):
            selected.append(int(index))
            if len(selected) == count:
                return tuple(sorted(selected))
    return None


def genetic_select(
    vectors: np.ndarray,
    centers: np.ndarray,
    widths: np.ndarray,
    count: int,
    spacing_factor: float,
    seed: int,
    population_size: int = 40,
    generations: int = 50,
) -> SelectionResult:
    """Small, deterministic evolutionary cross-check seeded by the greedy solution."""
    rng = np.random.default_rng(seed)
    greedy = greedy_select(vectors, centers, widths, count, spacing_factor)
    population: set[tuple[int, ...]] = {greedy.indices}
    while len(population) < population_size:
        member = _random_feasible(
            rng, vectors.shape[0], count, centers, widths, spacing_factor
        )
        if member is None:
            raise ValueError("Could not initialize a feasible genetic population")
        population.add(member)
    for _ in range(generations):
        ranked = sorted(population, key=lambda x: objective(vectors, x), reverse=True)
        elite = ranked[: max(4, population_size // 5)]
        next_population: set[tuple[int, ...]] = set(elite)
        attempts = 0
        while len(next_population) < population_size and attempts < population_size * 100:
            attempts += 1
            parent_a, parent_b = rng.choice(len(elite), size=2, replace=True)
            pool = list(set(elite[int(parent_a)]) | set(elite[int(parent_b)]))
            rng.shuffle(pool)
            child: list[int] = []
            for index in pool:
                if compatible(index, child, centers, widths, spacing_factor):
                    child.append(index)
                if len(child) == count:
                    break
            if rng.random() < 0.75 and child:
                child.pop(int(rng.integers(len(child))))
            random_order = rng.permutation(vectors.shape[0])
            for index in random_order:
                index = int(index)
                if index not in child and compatible(
                    index, child, centers, widths, spacing_factor
                ):
                    child.append(index)
                if len(child) == count:
                    break
            if len(child) == count:
                next_population.add(tuple(sorted(child)))
        population = next_population
        while len(population) < population_size:
            member = _random_feasible(
                rng, vectors.shape[0], count, centers, widths, spacing_factor
            )
            if member is not None:
                population.add(member)
    best = max(population, key=lambda x: objective(vectors, x))
    return SelectionResult(best, objective(vectors, best))
