"""Tests for abmem.services.hpo.abc_optimizer."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import pytest

from abmem.services.hpo.abc_optimizer import (
    ABCSettings,
    ArtificialBeeColony,
    CandidateRequest,
    CandidateScore,
    fitness_from_score,
    onlooker_probabilities,
    weighted_score,
)
from abmem.services.hpo.pareto import pareto_front
from abmem.services.hpo.param_space import ParamSpace


def _space(dim: int, low: float = -5.0, high: float = 5.0) -> ParamSpace:
    return ParamSpace.from_bounds(
        {f"x{d}": {"low": low, "high": high} for d in range(dim)}
    )


def _evaluator(
    space: ParamSpace,
    f: Callable[[np.ndarray], tuple[float, float]],
    log: list[CandidateRequest] | None = None,
) -> Callable[[Sequence[CandidateRequest]], list[CandidateScore]]:
    def evaluate(batch: Sequence[CandidateRequest]) -> list[CandidateScore]:
        out = []
        for req in batch:
            if log is not None:
                log.append(req)
            params = space.decode(req.x)
            out.append(CandidateScore(req, f(np.array(list(params.values())))))
        return out

    return evaluate


def _neg_sphere(z: np.ndarray) -> tuple[float, float]:
    return (-float(np.sum(z**2)), 0.0)


# ----------------------------------------------------------------- helpers


def test_fitness_branches() -> None:
    assert fitness_from_score(0.0) == 1.0
    assert fitness_from_score(-3.0) == pytest.approx(0.25)  # c = 3 >= 0
    assert fitness_from_score(2.0) == pytest.approx(3.0)  # c = -2 < 0 -> 1 + |c|
    assert fitness_from_score(-math.inf) == 0.0
    with pytest.raises(ValueError):
        fitness_from_score(math.nan)


def test_weighted_score() -> None:
    assert weighted_score((2.0, 4.0), (0.5, 0.25)) == pytest.approx(2.0)
    assert weighted_score((-math.inf, -math.inf), (1.0, 0.0)) == -math.inf
    assert weighted_score((1.0, math.nan), (1.0, 0.0)) == -math.inf


def test_onlooker_probabilities() -> None:
    assert np.allclose(onlooker_probabilities([1.0, 3.0]), [0.25, 0.75])
    assert np.allclose(onlooker_probabilities([2.0, 2.0, 2.0, 2.0]), 0.25)
    assert np.allclose(onlooker_probabilities([0.0, 0.0]), 0.5)
    with pytest.raises(ValueError):
        onlooker_probabilities([])
    with pytest.raises(ValueError):
        onlooker_probabilities([-1.0, 1.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"colony_size": 1, "max_iter": 1, "limit": 1},
        {"colony_size": 2, "max_iter": 0, "limit": 1},
        {"colony_size": 2, "max_iter": 1, "limit": 0},
        {"colony_size": 2.5, "max_iter": 1, "limit": 1},
        {"colony_size": 2, "max_iter": 1, "limit": 1, "mode": "pareto"},
        {"colony_size": 2, "max_iter": 1, "limit": 1, "weights": (0.0, 0.0)},
        {"colony_size": 2, "max_iter": 1, "limit": 1, "weights": (math.inf, 0.0)},
        {"colony_size": 2, "max_iter": 1, "limit": 1, "weights": (1.0,)},
    ],
)
def test_settings_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ABCSettings(**kwargs)


# ------------------------------------------------------------- convergence


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_converges_on_negated_sphere_4d(seed: int) -> None:
    space = _space(4)
    settings = ABCSettings(colony_size=10, max_iter=50, limit=20)
    abc = ArtificialBeeColony(
        space, settings, _evaluator(space, _neg_sphere), np.random.default_rng(seed)
    )
    result = abc.run()
    assert not result.stopped
    assert result.best_objectives[0] >= -1e-2
    best_params = np.array(list(space.decode(result.best_x).values()))
    assert result.best_objectives[0] == pytest.approx(-np.sum(best_params**2))
    # History is non-decreasing and complete.
    assert [it for it, _ in result.history] == list(range(51))
    scores = [s for _, s in result.history]
    assert all(b >= a for a, b in zip(scores, scores[1:]))
    assert scores[-1] == pytest.approx(
        weighted_score(result.best_objectives, settings.weights)
    )


def test_evaluation_count_and_batches() -> None:
    space = _space(2)
    batches: list[list[CandidateRequest]] = []
    inner = _evaluator(space, _neg_sphere)

    def evaluate(batch: Sequence[CandidateRequest]) -> list[CandidateScore]:
        batches.append(list(batch))
        return inner(batch)

    settings = ABCSettings(colony_size=6, max_iter=5, limit=3)
    abc = ArtificialBeeColony(space, settings, evaluate, np.random.default_rng(7))
    abc.run()
    n_evals = sum(len(b) for b in batches)
    assert n_evals == 6 + 5 * 2 * 6 + abc.n_scouts
    for batch in batches:
        assert len({(r.iteration, r.phase) for r in batch}) == 1
    assert batches[0][0].phase == "INIT" and len(batches[0]) == 6


# ------------------------------------------------------------ determinism


def test_deterministic_under_fixed_seed() -> None:
    space = _space(3)
    settings = ABCSettings(colony_size=5, max_iter=10, limit=4)

    def run(seed: int) -> tuple:
        log: list[CandidateRequest] = []
        r = ArtificialBeeColony(
            space,
            settings,
            _evaluator(space, _neg_sphere, log),
            np.random.default_rng(seed),
        ).run()
        return r, np.array([req.x for req in log])

    (r1, x1), (r2, x2), (r3, x3) = run(42), run(42), run(43)
    assert np.array_equal(r1.best_x, r2.best_x)
    assert r1.history == r2.history
    assert np.array_equal(x1, x2)
    assert x1.shape != x3.shape or not np.array_equal(x1, x3)


# -------------------------------------------------------------- moves/bounds


def test_bounds_respected_with_optimum_at_corner() -> None:
    space = ParamSpace.from_bounds(
        {
            "a": {"low": 0.0, "high": 1.0},
            "lr": {"low": 1e-4, "high": 1e-1, "scale": "log"},
            "n": {"low": 1, "high": 5, "scale": "int"},
        }
    )
    log: list[CandidateRequest] = []

    def f(z: np.ndarray) -> tuple[float, float]:
        return (float(z[0] + z[1] + z[2]), 0.0)

    settings = ABCSettings(colony_size=6, max_iter=30, limit=5)
    result = ArtificialBeeColony(
        space, settings, _evaluator(space, f, log), np.random.default_rng(3)
    ).run()
    xs = np.array([r.x for r in log])
    assert np.all((xs >= 0.0) & (xs <= 1.0))
    best = space.decode(result.best_x)
    assert best["n"] == 5
    assert isinstance(best["n"], int)
    assert best["a"] > 0.9


def test_only_one_dimension_changes_per_move() -> None:
    space = _space(5)
    holder: dict[str, ArtificialBeeColony] = {}
    inner = _evaluator(space, _neg_sphere)
    checked = 0

    def evaluate(batch: Sequence[CandidateRequest]) -> list[CandidateScore]:
        nonlocal checked
        abc = holder["abc"]
        for req in batch:
            if req.phase in ("EMPLOYED", "ONLOOKER"):
                diff = np.count_nonzero(req.x != abc.sources[req.source_index])
                assert diff <= 1
                checked += 1
        return inner(batch)

    abc = ArtificialBeeColony(
        space, ABCSettings(8, 10, 5), evaluate, np.random.default_rng(11)
    )
    holder["abc"] = abc
    abc.run()
    assert checked == 10 * 2 * 8


def test_colony_size_two_runs() -> None:
    space = _space(2)
    result = ArtificialBeeColony(
        space,
        ABCSettings(2, 5, 2),
        _evaluator(space, _neg_sphere),
        np.random.default_rng(0),
    ).run()
    assert result.best_x.shape == (2,)


# ----------------------------------------------------------- onlooker phase


def test_onlooker_selection_follows_fitness_chi_square() -> None:
    """Onlooker picks of a fixed population follow p_i = fit_i / sum fit."""
    space = _space(2)
    init_scores = [-4.0, -1.0, 0.0, 1.0, 3.0]  # fitness 0.2, 0.5, 1, 2, 4
    counts = np.zeros(5)

    def evaluate(batch: Sequence[CandidateRequest]) -> list[CandidateScore]:
        out = []
        for req in batch:
            if req.phase == "INIT":
                out.append(CandidateScore(req, (init_scores[req.source_index], 0.0)))
            else:
                if req.phase == "ONLOOKER":
                    counts[req.source_index] += 1
                out.append(
                    CandidateScore(req, (-math.inf, -math.inf))
                )  # never accepted
        return out

    iters = 2000
    abc = ArtificialBeeColony(
        space, ABCSettings(5, iters, 10**9), evaluate, np.random.default_rng(5)
    )
    abc.run()
    fit = np.array([fitness_from_score(s) for s in init_scores])
    expected = fit / fit.sum() * counts.sum()
    assert counts.sum() == 5 * iters
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    assert chi2 < 18.47  # chi-square critical value, df = 4, alpha = 0.001
    assert abc.n_scouts == 0


def test_failed_evaluations_never_replace_sources() -> None:
    space = _space(2)

    def evaluate(batch: Sequence[CandidateRequest]) -> list[CandidateScore]:
        return [
            CandidateScore(
                req, (1.0, 0.0) if req.phase == "INIT" else (-math.inf, -math.inf)
            )
            for req in batch
        ]

    abc = ArtificialBeeColony(
        space, ABCSettings(3, 4, 100), evaluate, np.random.default_rng(0)
    )
    result = abc.run()
    assert np.all(abc.scores == 1.0)
    assert result.best_objectives == (1.0, 0.0)


# -------------------------------------------------------------- scout phase


def test_scout_replaces_source_after_limit_failures() -> None:
    space = _space(2)
    log: list[CandidateRequest] = []
    trials_before_scout: list[int] = []
    holder: dict[str, ArtificialBeeColony] = {}

    def evaluate(batch: Sequence[CandidateRequest]) -> list[CandidateScore]:
        out = []
        for req in batch:
            log.append(req)
            if req.phase == "SCOUT":
                trials_before_scout.append(int(holder["abc"].trials[req.source_index]))
                out.append(CandidateScore(req, (-50.0, 0.0)))  # worse, still accepted
            elif req.phase == "INIT":
                out.append(CandidateScore(req, (-1.0, 0.0)))
            else:
                out.append(CandidateScore(req, (-10.0, 0.0)))  # never improves
        return out

    limit = 1
    abc = ArtificialBeeColony(
        space, ABCSettings(4, 1, limit), evaluate, np.random.default_rng(1)
    )
    holder["abc"] = abc
    abc.run()
    scouts = [r for r in log if r.phase == "SCOUT"]
    assert len(scouts) == 1  # at most one scout per cycle
    assert trials_before_scout[0] > limit
    i = scouts[0].source_index
    assert abc.trials[i] == 0
    assert np.array_equal(abc.sources[i], scouts[0].x)
    assert abc.scores[i] == -50.0


def test_scout_never_triggers_with_large_limit() -> None:
    space = _space(2)
    log: list[CandidateRequest] = []
    abc = ArtificialBeeColony(
        space,
        ABCSettings(4, 20, 10**6),
        _evaluator(space, _neg_sphere, log),
        np.random.default_rng(2),
    )
    abc.run()
    assert abc.n_scouts == 0
    assert all(r.phase != "SCOUT" for r in log)


def test_scout_does_not_trigger_at_exactly_limit() -> None:
    space = _space(2)

    def evaluate(batch: Sequence[CandidateRequest]) -> list[CandidateScore]:
        return [
            CandidateScore(r, (0.0 if r.phase == "INIT" else -1.0, 0.0)) for r in batch
        ]

    abc = ArtificialBeeColony(
        space, ABCSettings(2, 1, 1), evaluate, np.random.default_rng(0)
    )
    # Force a deterministic counter: patch onlooker phase out by inspecting after employed only.
    abc._init_phase()
    abc._employed_phase(1)
    assert np.all(abc.trials == 1)
    assert abc._scout_phase(1)
    assert abc.n_scouts == 0


# ------------------------------------------------------------ stop/contract


def test_should_stop_before_second_batch() -> None:
    space = _space(2)
    calls = {"n": 0}
    inner = _evaluator(space, _neg_sphere)

    def evaluate(batch: Sequence[CandidateRequest]) -> list[CandidateScore]:
        calls["n"] += 1
        return inner(batch)

    abc = ArtificialBeeColony(
        space,
        ABCSettings(3, 10, 5),
        evaluate,
        np.random.default_rng(0),
        should_stop=lambda: calls["n"] >= 1,
    )
    result = abc.run()
    assert result.stopped
    assert calls["n"] == 1
    assert result.best_x.shape == (2,)
    assert result.history == ((0, result.history[0][1]), (1, result.history[0][1]))


def test_should_stop_before_any_batch() -> None:
    space = _space(2)
    result = ArtificialBeeColony(
        space,
        ABCSettings(3, 10, 5),
        _evaluator(space, _neg_sphere),
        np.random.default_rng(0),
        lambda: True,
    ).run()
    assert result.stopped
    assert result.best_x.size == 0
    assert result.best_objectives == (-math.inf, -math.inf)
    assert result.history == ()


def test_malformed_evaluate_result_raises() -> None:
    space = _space(2)
    with pytest.raises(ValueError):
        ArtificialBeeColony(
            space, ABCSettings(3, 1, 5), lambda b: [], np.random.default_rng(0)
        ).run()
    with pytest.raises(ValueError):
        ArtificialBeeColony(
            space,
            ABCSettings(3, 1, 5),
            lambda b: [CandidateScore(r, (0.0, 0.0)) for r in reversed(b)],
            np.random.default_rng(0),
        ).run()


def test_request_arrays_are_read_only() -> None:
    space = _space(2)

    def evaluate(batch: Sequence[CandidateRequest]) -> list[CandidateScore]:
        with pytest.raises(ValueError):
            batch[0].x[0] = 0.5
        return [CandidateScore(r, (0.0, 0.0)) for r in batch]

    ArtificialBeeColony(
        space, ABCSettings(2, 1, 5), evaluate, np.random.default_rng(0)
    ).run()


# ---------------------------------------------------------------- archive


def test_archive_is_pareto_front_of_evaluations() -> None:
    space = ParamSpace.from_bounds(
        {"x": {"low": -1.0, "high": 2.0}, "y": {"low": 0.0, "high": 1.0}}
    )
    log: list[CandidateScore] = []

    def evaluate(batch: Sequence[CandidateRequest]) -> list[CandidateScore]:
        out = []
        for req in batch:
            p = space.decode(req.x)
            x, y = p["x"], p["y"]
            obj = (-(x**2) - y, -((x - 1.0) ** 2) - y)
            out.append(CandidateScore(req, obj))
        log.extend(out)
        return out

    result = ArtificialBeeColony(
        space,
        ABCSettings(6, 15, 5, weights=(0.5, 0.5)),
        evaluate,
        np.random.default_rng(9),
    ).run()
    expected = {log[i].objectives for i in pareto_front([s.objectives for s in log])}
    got = {s.objectives for s in result.archive}
    assert got == expected
    # True front: y = 0 and x in [0, 1].
    for s in result.archive:
        p = space.decode(s.request.x)
        assert -1e-9 <= p["x"] <= 1.0 + 1e-9
    # Selection is driven by the weighted objective (optimum x = 0.5, y = 0).
    assert space.decode(result.best_x)["x"] == pytest.approx(0.5, abs=0.1)


def test_archive_excludes_non_finite_objectives() -> None:
    space = _space(2)
    result = ArtificialBeeColony(
        space,
        ABCSettings(2, 1, 5),
        lambda b: [CandidateScore(r, (-math.inf, -math.inf)) for r in b],
        np.random.default_rng(0),
    ).run()
    assert result.archive == ()
