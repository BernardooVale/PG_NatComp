import numpy as np
from gp.tree import Individual, Node
from gp.selection import tournament_selection, roulette_selection, epsilon_lexicase_selection


def test_tournament_selection():
    pop = [
        Individual(Node("x0"), fitness=10.0),
        Individual(Node("x1"), fitness=2.0),
        Individual(Node(1.0), fitness=5.0),
    ]
    rng = np.random.default_rng(42)

    # Tournament with k = pop_size must always pick the absolute best (fitness=2.0)
    best = tournament_selection(pop, tournament_size=3, rng=rng)
    assert best.fitness == 2.0


def test_roulette_selection_uniform_when_equal():
    # If all fitness values are identical, selection probabilities should be uniform
    pop = [Individual(Node("x0"), fitness=1.0) for _ in range(4)]
    rng = np.random.default_rng(42)

    counts = [0] * 4
    for _ in range(4000):
        selected = roulette_selection(pop, rng)
        idx = pop.index(selected)
        counts[idx] += 1

    # Check that each index is chosen roughly 1000 times (between 850 and 1150)
    for c in counts:
        assert 800 <= c <= 1200


def test_lexicase_never_returns_empty_and_handles_mad():
    pop = [
        Individual(Node("x0"), fitness=1.0),
        Individual(Node("x1"), fitness=2.0),
        Individual(Node(1.0), fitness=3.0),
    ]
    # Error matrix: 3 individuals, 5 cases
    error_matrix = np.array([
        [0.1, 0.5, 0.2, 0.4, 0.3],
        [0.2, 0.1, 0.3, 0.5, 0.2],
        [0.4, 0.6, 0.5, 0.7, 0.6],
    ])
    rng = np.random.default_rng(42)

    for _ in range(50):
        selected = epsilon_lexicase_selection(pop, error_matrix, "mad", rng)
        assert selected in pop

    # Test classic lexicase ('none')
    for _ in range(50):
        selected_classic = epsilon_lexicase_selection(pop, error_matrix, "none", rng)
        assert selected_classic in pop
