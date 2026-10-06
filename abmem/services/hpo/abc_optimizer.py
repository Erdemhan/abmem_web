"""Artificial Bee Colony optimizer, batch-synchronous variant.

Implements the ABC algorithm of Karaboga (2005) and Karaboga and Basturk
(2007) for maximisation of a weighted single objective (OQ-11) over the unit
cube of a :class:`~abmem.services.hpo.param_space.ParamSpace`.

Algorithm (SN = ``colony_size`` food sources, D dimensions):

1. Initialisation: SN sources drawn uniformly from ``[0, 1]^D`` and evaluated.
2. Employed phase: for every source ``i`` a neighbour ``k != i`` and a
   dimension ``j`` are drawn uniformly, ``v_ij = x_ij + phi * (x_ij - x_kj)``
   with ``phi ~ U[-1, 1]``, clipped to ``[0, 1]``. All other coordinates are
   copied from ``x_i``. Greedy selection keeps the better of ``x_i`` and
   ``v_i``. The trial counter of ``i`` is reset on improvement and
   incremented otherwise.
3. Onlooker phase: SN onlookers pick sources by roulette wheel with
   ``p_i = fit_i / sum_n fit_n`` and apply the same move and greedy selection.
4. Scout phase: the single source with the largest trial counter is replaced
   by a uniform random source if its counter exceeds ``limit``.

Fitness of a source with objective score ``s`` uses the cost ``c = -s``:
``fit = 1 / (1 + c)`` if ``c >= 0`` and ``fit = 1 + |c|`` otherwise
(Karaboga and Basturk, 2007).

Batch-synchronous variant (OQ-19): all candidates of one phase are generated
from the same snapshot of the food sources and evaluated as one batch through
the injected ``evaluate`` callable. Greedy selection is then applied in
submission order (source order in the employed phase, draw order in the
onlooker phase). In the classic algorithm onlookers act sequentially and see
each other's updates. Here several onlookers drawing the same source all move
from its snapshot position, and the selections are applied one after another.

Evaluations whose objectives are not finite (for example failed simulations
reported as ``(-inf, -inf)``) receive the score ``-inf`` and fitness 0, so they
never replace a source with a finite score.

The optimizer has no Django or simulation dependency. With a fixed
``numpy.random.Generator`` and a deterministic ``evaluate`` callable a run is
fully reproducible.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from abmem.services.hpo.pareto import dominates
from abmem.services.hpo.param_space import ParamSpace

HPOPhaseName = Literal["INIT", "EMPLOYED", "ONLOOKER", "SCOUT"]
"""Phase names, equal to the member names of ``abmem.models.enums.HPOPhase``."""


@dataclass(frozen=True)
class ABCSettings:
    """Settings of one ABC run.

    Attributes:
        colony_size: Number of food sources SN (equal to the number of
            employed bees and of onlooker bees), at least 2.
        max_iter: Number of cycles (employed, onlooker, scout), at least 1.
        limit: Trial-counter threshold for the scout phase, at least 1.
        mode: Objective mode. Only ``'weighted'`` is accepted (OQ-11: the
            Pareto front is information only and never used for selection).
        weights: ``(w_agent, w_market)``. Score ``= w1 * agent + w2 * market``.
    """

    colony_size: int
    max_iter: int
    limit: int
    mode: Literal["weighted"] = "weighted"
    weights: tuple[float, float] = (1.0, 0.0)

    def __post_init__(self) -> None:
        """Validates the settings.

        Raises:
            ValueError: If a value is out of range, the mode is not
                ``'weighted'`` or the weights are malformed.
        """
        for name, minimum in (("colony_size", 2), ("max_iter", 1), ("limit", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer, got {value!r}")
            if value < minimum:
                raise ValueError(f"{name} must be >= {minimum}, got {value}")
        if self.mode != "weighted":
            raise ValueError(
                f"mode must be 'weighted', got {self.mode!r} (OQ-11: Pareto front is information only)"
            )
        weights = tuple(float(w) for w in self.weights)
        if len(weights) != 2:
            raise ValueError(f"weights must have two entries, got {self.weights!r}")
        if not all(math.isfinite(w) for w in weights):
            raise ValueError(f"weights must be finite, got {self.weights!r}")
        if all(w == 0.0 for w in weights):
            raise ValueError("at least one weight must be non-zero")
        object.__setattr__(self, "weights", weights)


@dataclass(frozen=True, eq=False)
class CandidateRequest:
    """A unit-cube point submitted for evaluation.

    Attributes:
        iteration: 0 for the initialisation, ``1..max_iter`` for the cycles.
        phase: Phase that generated the candidate.
        source_index: Index of the food source the candidate belongs to.
        x: Read-only array of shape ``(D,)`` in ``[0, 1]^D``.
    """

    iteration: int
    phase: HPOPhaseName
    source_index: int
    x: np.ndarray


@dataclass(frozen=True, eq=False)
class CandidateScore:
    """Evaluation result of a :class:`CandidateRequest`.

    Attributes:
        request: The evaluated request.
        objectives: ``(agent_score, market_score)``, larger is better.
    """

    request: CandidateRequest
    objectives: tuple[float, float]


EvaluateBatch = Callable[[Sequence[CandidateRequest]], Sequence[CandidateScore]]


@dataclass(frozen=True, eq=False)
class ABCResult:
    """Outcome of :meth:`ArtificialBeeColony.run`.

    Attributes:
        best_x: Best evaluated unit-cube point (shape ``(D,)``), or an empty
            array if the run was stopped before any evaluation.
        best_objectives: Objectives of ``best_x``, ``(-inf, -inf)`` if nothing
            was evaluated.
        archive: All evaluated candidates with finite objectives that no other
            evaluated candidate Pareto-dominates (information only, OQ-11).
        history: ``(iteration, best scalar score so far)`` after the
            initialisation (iteration 0) and after each cycle.
        stopped: True if ``should_stop`` ended the run early.
    """

    best_x: np.ndarray
    best_objectives: tuple[float, float]
    archive: tuple[CandidateScore, ...]
    history: tuple[tuple[int, float], ...]
    stopped: bool


def weighted_score(objectives: Sequence[float], weights: Sequence[float]) -> float:
    """Computes the weighted scalar objective.

    Args:
        objectives: ``(agent_score, market_score)``.
        weights: ``(w1, w2)``.

    Returns:
        ``w1 * agent_score + w2 * market_score``, or ``-inf`` if any objective
        is NaN or infinite (failed evaluation).
    """
    values = [float(o) for o in objectives]
    if not all(math.isfinite(v) for v in values):
        return -math.inf
    return float(sum(w * v for w, v in zip(weights, values)))


def fitness_from_score(score: float) -> float:
    """Maps a score (maximised) to the ABC fitness (Karaboga and Basturk, 2007).

    With cost ``c = -score``: ``1 / (1 + c)`` if ``c >= 0``, else ``1 + |c|``.

    Args:
        score: Scalar objective, ``-inf`` for failed evaluations.

    Returns:
        Non-negative fitness, 0 for ``score = -inf``.

    Raises:
        ValueError: If ``score`` is NaN or ``+inf``.
    """
    if math.isnan(score) or score == math.inf:
        raise ValueError(f"score must be finite or -inf, got {score!r}")
    if score == -math.inf:
        return 0.0
    cost = -score
    if cost >= 0.0:
        return 1.0 / (1.0 + cost)
    return 1.0 + abs(cost)


def onlooker_probabilities(fitness: Sequence[float]) -> np.ndarray:
    """Roulette-wheel selection probabilities ``p_i = fit_i / sum fit``.

    Args:
        fitness: Non-negative fitness per food source.

    Returns:
        Probability vector. Uniform if all fitness values are zero.

    Raises:
        ValueError: If ``fitness`` is empty or contains negative or
            non-finite values.
    """
    fit = np.asarray(fitness, dtype=float)
    if fit.ndim != 1 or fit.size == 0:
        raise ValueError("fitness must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(fit)) or np.any(fit < 0.0):
        raise ValueError(f"fitness must be finite and non-negative, got {fit}")
    total = float(fit.sum())
    if total <= 0.0:
        return np.full(fit.size, 1.0 / fit.size)
    return fit / total


class ArtificialBeeColony:
    """Batch-synchronous ABC maximising a weighted objective on a ParamSpace."""

    def __init__(
        self,
        space: ParamSpace,
        settings: ABCSettings,
        evaluate: EvaluateBatch,
        rng: np.random.Generator,
        should_stop: Callable[[], bool] = lambda: False,
    ) -> None:
        """Creates the optimizer.

        Args:
            space: Search space. The optimizer works in its unit cube.
            settings: Colony size, cycles, limit and objective weights.
            evaluate: Called once per phase with the batch of requests.
                Must return one :class:`CandidateScore` per request in the
                same order.
            rng: The only source of randomness.
            should_stop: Checked before every batch. Returning True ends the
                run with ``stopped=True``.
        """
        self.space = space
        self.settings = settings
        self._evaluate = evaluate
        self._rng = rng
        self._should_stop = should_stop
        sn, dim = settings.colony_size, space.dim
        self.sources = np.zeros((sn, dim), dtype=float)
        self.objectives: list[tuple[float, float]] = [(-math.inf, -math.inf)] * sn
        self.scores = np.full(sn, -math.inf)
        self.trials = np.zeros(sn, dtype=int)
        self.n_scouts = 0
        self._best: CandidateScore | None = None
        self._best_score = -math.inf
        self._archive: list[CandidateScore] = []
        self._history: list[tuple[int, float]] = []

    # ------------------------------------------------------------------ moves
    def _neighbour_move(self, snapshot: np.ndarray, i: int) -> np.ndarray:
        """Generates ``v_i`` from ``x_i`` by changing one random dimension.

        Args:
            snapshot: Food sources of shape ``(SN, D)``.
            i: Index of the source to move.

        Returns:
            New point of shape ``(D,)`` in ``[0, 1]^D``.
        """
        sn, dim = snapshot.shape
        k = int(self._rng.integers(sn - 1))
        if k >= i:
            k += 1
        j = int(self._rng.integers(dim))
        phi = float(self._rng.uniform(-1.0, 1.0))
        v = snapshot[i].copy()
        v[j] = min(1.0, max(0.0, v[j] + phi * (v[j] - snapshot[k, j])))
        return v

    @staticmethod
    def _request(
        iteration: int, phase: HPOPhaseName, index: int, x: np.ndarray
    ) -> CandidateRequest:
        """Builds a request with a read-only copy of ``x``."""
        arr = np.array(x, dtype=float, copy=True)
        arr.setflags(write=False)
        return CandidateRequest(
            iteration=iteration, phase=phase, source_index=index, x=arr
        )

    # ------------------------------------------------------------- evaluation
    def _submit(self, requests: list[CandidateRequest]) -> list[CandidateScore] | None:
        """Evaluates one batch, or returns None if a stop was requested.

        Args:
            requests: Candidates of one phase.

        Returns:
            Scores in request order, or None if ``should_stop`` is True.

        Raises:
            ValueError: If the callable returns a malformed result.
        """
        if self._should_stop():
            return None
        results = list(self._evaluate(requests))
        if len(results) != len(requests):
            raise ValueError(
                f"evaluate returned {len(results)} scores for {len(requests)} requests"
            )
        checked: list[CandidateScore] = []
        for req, res in zip(requests, results):
            if not isinstance(res, CandidateScore):
                raise ValueError(
                    f"evaluate must return CandidateScore, got {type(res).__name__}"
                )
            got = res.request
            if (got.iteration, got.phase, got.source_index) != (
                req.iteration,
                req.phase,
                req.source_index,
            ):
                raise ValueError("evaluate must return scores in request order")
            objectives = tuple(float(o) for o in res.objectives)
            if len(objectives) != 2:
                raise ValueError(
                    f"objectives must have two entries, got {res.objectives!r}"
                )
            score = CandidateScore(
                request=req, objectives=(objectives[0], objectives[1])
            )
            self._record(score)
            checked.append(score)
        return checked

    def _record(self, score: CandidateScore) -> None:
        """Updates the best-so-far candidate and the non-dominated archive."""
        scalar = weighted_score(score.objectives, self.settings.weights)
        if self._best is None or scalar > self._best_score:
            self._best, self._best_score = score, scalar
        if not all(math.isfinite(o) for o in score.objectives):
            return
        if any(dominates(a.objectives, score.objectives) for a in self._archive):
            return
        self._archive = [
            a for a in self._archive if not dominates(score.objectives, a.objectives)
        ]
        self._archive.append(score)

    def _greedy(self, score: CandidateScore) -> None:
        """Greedy selection between a source and its candidate."""
        i = score.request.source_index
        scalar = weighted_score(score.objectives, self.settings.weights)
        if scalar > self.scores[i]:
            self._set_source(i, score, scalar)
        else:
            self.trials[i] += 1

    def _set_source(self, i: int, score: CandidateScore, scalar: float) -> None:
        """Replaces food source ``i`` and resets its trial counter."""
        self.sources[i] = score.request.x
        self.objectives[i] = score.objectives
        self.scores[i] = scalar
        self.trials[i] = 0

    def fitness(self) -> np.ndarray:
        """Fitness of the current food sources."""
        return np.array([fitness_from_score(float(s)) for s in self.scores])

    # ----------------------------------------------------------------- phases
    def _init_phase(self) -> bool:
        """Initialises and evaluates SN uniform sources. Returns False on stop."""
        reqs = [
            self._request(0, "INIT", i, self.space.sample(self._rng))
            for i in range(self.settings.colony_size)
        ]
        results = self._submit(reqs)
        if results is None:
            return False
        for res in results:
            i = res.request.source_index
            self._set_source(
                i, res, weighted_score(res.objectives, self.settings.weights)
            )
        return True

    def _employed_phase(self, t: int) -> bool:
        """Employed bee phase of cycle ``t``. Returns False on stop."""
        snapshot = self.sources.copy()
        reqs = [
            self._request(t, "EMPLOYED", i, self._neighbour_move(snapshot, i))
            for i in range(self.settings.colony_size)
        ]
        results = self._submit(reqs)
        if results is None:
            return False
        for res in results:
            self._greedy(res)
        return True

    def _onlooker_phase(self, t: int) -> bool:
        """Onlooker bee phase of cycle ``t``. Returns False on stop."""
        snapshot = self.sources.copy()
        sn = self.settings.colony_size
        probs = onlooker_probabilities(self.fitness())
        chosen = self._rng.choice(sn, size=sn, replace=True, p=probs)
        reqs = [
            self._request(t, "ONLOOKER", int(i), self._neighbour_move(snapshot, int(i)))
            for i in chosen
        ]
        results = self._submit(reqs)
        if results is None:
            return False
        for res in results:
            self._greedy(res)
        return True

    def _scout_phase(self, t: int) -> bool:
        """Scout phase of cycle ``t``. Returns False on stop."""
        i = int(np.argmax(self.trials))
        if self.trials[i] <= self.settings.limit:
            return True
        reqs = [self._request(t, "SCOUT", i, self.space.sample(self._rng))]
        results = self._submit(reqs)
        if results is None:
            return False
        res = results[0]
        self._set_source(i, res, weighted_score(res.objectives, self.settings.weights))
        self.n_scouts += 1
        return True

    # -------------------------------------------------------------------- run
    def _result(self, stopped: bool) -> ABCResult:
        """Assembles the result."""
        if self._best is None:
            best_x = np.empty(0, dtype=float)
            best_obj: tuple[float, float] = (-math.inf, -math.inf)
        else:
            best_x = np.array(self._best.request.x, dtype=float, copy=True)
            best_obj = self._best.objectives
        return ABCResult(
            best_x=best_x,
            best_objectives=best_obj,
            archive=tuple(self._archive),
            history=tuple(self._history),
            stopped=stopped,
        )

    def run(self) -> ABCResult:
        """Runs initialisation and ``max_iter`` cycles.

        Returns:
            The best candidate, the non-dominated archive, the history of the
            best score and whether the run was stopped early.

        Raises:
            ValueError: If ``evaluate`` returns a malformed result.
        """
        if not self._init_phase():
            return self._result(stopped=True)
        self._history.append((0, self._best_score))
        for t in range(1, self.settings.max_iter + 1):
            for phase in (
                self._employed_phase,
                self._onlooker_phase,
                self._scout_phase,
            ):
                if not phase(t):
                    self._history.append((t, self._best_score))
                    return self._result(stopped=True)
            self._history.append((t, self._best_score))
        return self._result(stopped=False)
