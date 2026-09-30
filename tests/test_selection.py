import numpy as np

from lunar_band_design.selection import genetic_select, greedy_select


def test_greedy_is_deterministic_and_respects_spacing():
    vectors = np.column_stack([np.linspace(0, 2, 30), np.cos(np.linspace(0, 3, 30))])
    centers = np.linspace(1, 10, 30)
    widths = np.full(30, 0.5)
    first = greedy_select(vectors, centers, widths, 5, 1.0)
    second = greedy_select(vectors, centers, widths, 5, 1.0)
    assert first == second
    chosen = centers[list(first.indices)]
    assert np.min(np.diff(chosen)) >= 0.5


def test_genetic_audit_preserves_known_greedy_baseline():
    vectors = np.column_stack([np.linspace(0, 2, 30), np.cos(np.linspace(0, 3, 30))])
    centers = np.linspace(1, 10, 30)
    widths = np.full(30, 0.5)
    greedy = greedy_select(vectors, centers, widths, 5, 1.0)
    genetic = genetic_select(
        vectors, centers, widths, 5, 1.0, seed=42, population_size=12, generations=5
    )
    assert genetic.objective >= greedy.objective
