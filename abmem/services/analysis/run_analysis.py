"""Read-only analysis of a finished run, computed from the database (PLAN 2.5).

:func:`build_run_analysis` loads the run settings, periods, offers and step
logs of a run with a fixed number of queries and aggregates them into the
series and tables of the analysis page. Nothing is written to the database.

Budget series. ``market_service.run_period`` updates the budget of an agent
once per period, ``B_t = round_{0.01}(B_{t-1} + G_t)`` with half-even
rounding, where ``G_t`` is the sum of the offer profits
(:func:`abmem.services.market.accounting.offer_profit`) of the agent in
period ``t``. The run does not store ``B_0``. The series is therefore
reconstructed backwards from the stored final ``Agent.budget = B_N`` by
inverting each rounding step on the 0.01 grid, which recovers the initial
budget exactly. When a half-even tie makes two grid values map to the same
``B_t`` (possible only if ``B_{t-1} + G_t`` lies exactly halfway between two
cents), the lower one is taken, so ``B_0`` is then exact up to 0.01 per such
period. By construction the series ends at the stored final budget.

Scores. The corrected scores are the values of
:func:`abmem.services.hpo.objective.score_simulation` over the whole run and,
if ``evaluation_periods > 0``, over the evaluation window. The legacy scores
are :func:`abmem.services.analysis.legacy_scores.legacy_composite_scores` on
the rows of the legacy JSON export (budget after each period).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Decimal
from typing import TYPE_CHECKING, Any, Final

from abmem.models.enums import (
    MarketStrategy,
    SimulationMode,
    SimulationState,
)
from abmem.services.analysis.legacy_scores import (
    LEGACY_PRICE_CAP,
    legacy_composite_scores,
)

if TYPE_CHECKING:
    from abmem.models import Simulation

BUDGET_QUANTUM: Final[Decimal] = Decimal("0.01")
"""Grid of ``Agent.budget`` (``market_service.BUDGET_QUANTUM``)."""


class RunAnalysisUnavailable(ValueError):
    """Raised when a simulation has no analysis (not a finished run, no data)."""


@dataclass(frozen=True)
class PlantSetting:
    """One plant of an agent.

    Attributes:
        resource: Resource name.
        capacity: Capacity in MWh.
    """

    resource: str
    capacity: int


@dataclass(frozen=True)
class AgentSetting:
    """Configuration of one run agent.

    Attributes:
        name: Agent name.
        algorithm: Algorithm key.
        baseline: True for a non-learning baseline.
        failstack_enabled: Fail-stack switch (always False for baselines).
        failstack_limit: Fail-stack limit L.
        failstack_delta: Fail-stack price step delta (USD/MWh).
        failstack_min_margin: Fail-stack minimum margin.
        algorithm_params: Per-agent algorithm parameter overrides.
        plants: Plants of the agent.
        algorithm_label: Display label of the algorithm from the registry, the
            key when the algorithm is no longer registered.
        effective_params: Parameters the policy actually used, stored when
            the run started (PLAN 2.12), None for runs started before the
            snapshot existed.
    """

    name: str
    algorithm: str
    baseline: bool
    failstack_enabled: bool
    failstack_limit: int
    failstack_delta: float
    failstack_min_margin: float
    algorithm_params: dict[str, Any]
    plants: list[PlantSetting]
    algorithm_label: str = ""
    effective_params: dict[str, Any] | None = None


@dataclass(frozen=True)
class CheckpointSetting:
    """Saved policy of one agent of the run (finding A9).

    Attributes:
        agent: Agent name (empty if the agent row is gone).
        algorithm: Algorithm key.
        file_size: Size of the checkpoint file in bytes.
        periods_trained: Learning periods including warm-start ancestors.
        parent_run: Run of the checkpoint the policy was loaded from, if any.
    """

    agent: str
    algorithm: str
    file_size: int
    periods_trained: int
    parent_run: int | None


@dataclass(frozen=True)
class RunSettings:
    """Settings of the run.

    Attributes:
        simulation_id: Primary key of the run.
        name: Run name.
        seed: Run seed.
        observation_lags: Observation lag set.
        evaluation_periods: Size of the evaluation window (0 = none).
        period_number: Number of periods N.
        mode: Simulation mode name.
        strategy: Market strategy name.
        lower_bound: Lower bid bound (USD/MWh).
        upper_bound: Upper bid bound (USD/MWh).
        agents: Agent settings in id order.
        warm_start_from: Run whose saved policies initialised this run, None
            for fresh policies.
        freeze_policies: The loaded policies ran without learning.
        checkpoints: Policies saved at the end of this run, in agent order.
        clearing_algorithm: Key of the clearing algorithm of the market.
        clearing_algorithm_label: Display label of the clearing algorithm.
        effective_clearing_params: Parameters the clearing algorithm actually
            used (PLAN 2.12), None for runs started before the snapshot
            existed.
    """

    simulation_id: int
    name: str
    seed: int
    observation_lags: list[int]
    evaluation_periods: int
    period_number: int
    mode: str
    strategy: str
    lower_bound: int
    upper_bound: int
    agents: list[AgentSetting]
    warm_start_from: int | None = None
    freeze_policies: bool = False
    checkpoints: list[CheckpointSetting] = field(default_factory=list)
    clearing_algorithm: str = ""
    clearing_algorithm_label: str = ""
    effective_clearing_params: dict[str, Any] | None = None


@dataclass(frozen=True)
class PeriodRow:
    """Market outcome of one period.

    Attributes:
        period: Period number.
        demand: Demand (MWh).
        met_demand: Met demand (MWh).
        ptf: Market clearing price (USD/MWh), None if missing.
        market_volume: Offered volume (MWh).
    """

    period: int
    demand: float
    met_demand: float
    ptf: float | None
    market_volume: int


@dataclass(frozen=True)
class OfferRow:
    """One cleared offer.

    Attributes:
        period: Period number.
        agent: Agent name.
        resource: Resource (plant) name.
        amount: Offered quantity (MWh).
        offer_price: Offered price (USD/MWh).
        accepted: Whether any quantity was accepted.
        acceptance_amount: Accepted quantity (MWh).
        acceptance_price: Acceptance price (USD/MWh), None if not set.
        profit: Offer profit ``g`` (USD).
    """

    period: int
    agent: str
    resource: str
    amount: int
    offer_price: float
    accepted: bool
    acceptance_amount: float
    acceptance_price: float | None
    profit: float


@dataclass(frozen=True)
class StepLogRow:
    """One ``AgentStepLog`` row.

    Attributes:
        period: Decision period t.
        reward: Reward of period t-1.
        critic_loss: Critic loss of the update at t.
        actor_loss: Actor loss of the update at t.
        mean_q: Mean Q value of the update at t.
        exploration_scale: Exploration scale used at t.
        intervened: True if the fail-stack rule changed any plant price.
        learning_active: True if the policy learned at t.
        evaluation_phase: True if t lies in the evaluation window.
    """

    period: int
    reward: float | None
    critic_loss: float | None
    actor_loss: float | None
    mean_q: float | None
    exploration_scale: float | None
    intervened: bool
    learning_active: bool
    evaluation_phase: bool


@dataclass(frozen=True)
class OfferStats:
    """Offer statistics of an agent or of one of its resources.

    Attributes:
        offers: Number of offers.
        accepted_offers: Number of accepted offers.
        acceptance_rate: Accepted offers in percent of all offers.
        mean_offer_price: Mean offered price.
        mean_accepted_offer_price: Mean offered price of accepted offers,
            None without accepted offers.
        accepted_energy: Sum of accepted quantities (MWh).
        revenue: Sum of ``acceptance_amount * acceptance_price`` (USD).
        profit: Sum of offer profits (USD).
    """

    offers: int
    accepted_offers: int
    acceptance_rate: float
    mean_offer_price: float | None
    mean_accepted_offer_price: float | None
    accepted_energy: float
    revenue: float
    profit: float


@dataclass(frozen=True)
class AgentSummary:
    """Summary row of one agent.

    Attributes:
        name: Agent name.
        algorithm: Algorithm name.
        initial_budget: Reconstructed initial budget ``B_0``.
        final_budget: Stored final budget ``B_N``.
        budget_change: ``B_N - B_0``.
        mean_budget_change: ``(B_N - B_0) / N``.
        stats: Offer statistics over all resources.
        resources: Offer statistics per resource name.
    """

    name: str
    algorithm: str
    initial_budget: float
    final_budget: float
    budget_change: float
    mean_budget_change: float
    stats: OfferStats
    resources: dict[str, OfferStats]


@dataclass(frozen=True)
class ScorePair:
    """Corrected agent score and market score over a set of periods.

    Attributes:
        first_period: First scored period.
        last_period: Last scored period.
        agent_score: Mean corrected agent score, None if not computable.
        market_score: Corrected market score, None if not computable.
        error: Reason when the scores could not be computed.
    """

    first_period: int
    last_period: int
    agent_score: float | None
    market_score: float | None
    error: str = ""


@dataclass(frozen=True)
class CorrectedScores:
    """Corrected scores (B9, ``hpo.objective``).

    Attributes:
        price_cap: ``Market.upperBidBound``.
        whole_run: Scores over periods ``1..N``.
        evaluation_window: Scores over the evaluation window, None if
            ``evaluation_periods = 0``.
    """

    price_cap: float
    whole_run: ScorePair
    evaluation_window: ScorePair | None


@dataclass(frozen=True)
class LegacyScores:
    """Legacy composite scores (pre-B9, ``compute_advanced_scores.py``).

    Attributes:
        price_cap: The constant 200 of the legacy script.
        agent_scores: Composite score per agent name.
        mean_agent_score: Mean of ``agent_scores``.
        market_score: Legacy market score.
    """

    price_cap: float
    agent_scores: dict[str, float]
    mean_agent_score: float
    market_score: float


@dataclass(frozen=True)
class RunAnalysis:
    """Complete analysis of one finished run.

    Attributes:
        settings: Run settings.
        periods: One row per period, by period number.
        offers: One row per offer, by period, agent id and offer id.
        budgets: Budget series per agent name, index t = budget after period t
            (index 0 = initial budget), length N + 1.
        step_logs: Step-log rows per agent name, by period. Agents without
            rows (baselines) map to an empty list.
        agents: Summary per agent in id order.
        corrected: Corrected scores.
        legacy: Legacy scores.
    """

    settings: RunSettings
    periods: list[PeriodRow]
    offers: list[OfferRow]
    budgets: dict[str, list[float]]
    step_logs: dict[str, list[StepLogRow]]
    agents: list[AgentSummary]
    corrected: CorrectedScores
    legacy: LegacyScores

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation.

        Returns:
            Nested dicts and lists of the dataclass fields.
        """
        return asdict(self)


def _enum_name(enum_cls: Any, value: Any) -> str:
    """Return the member name of an enum field value.

    Args:
        enum_cls: ``django_enumfield`` enum class.
        value: Stored value or member.

    Returns:
        The member name, or ``str(value)`` for an unknown value.
    """
    member = enum_cls.get(getattr(value, "value", value))
    return member.name if member is not None else str(value)


def _optional_float(value: Any) -> float | None:
    """Convert a nullable numeric value to float."""
    return None if value is None else float(value)


def _quantize(value: Decimal) -> Decimal:
    """Round to the budget grid as ``market_service.run_period`` does."""
    return value.quantize(BUDGET_QUANTUM, rounding=ROUND_HALF_EVEN)


def previous_budget(after: Decimal, profit: Decimal) -> Decimal:
    """Invert one budget update ``after = round_{0.01}(before + profit)``.

    Args:
        after: Budget after the period (on the 0.01 grid).
        profit: Summed offer profit of the period.

    Returns:
        The grid value ``before`` with ``round(before + profit) == after``.
        If two grid values satisfy this (half-even tie), the lower one. If
        none does (``after`` not produced by such an update), the rounded
        difference ``round(after - profit)``.
    """
    exact = after - profit
    low = exact.quantize(BUDGET_QUANTUM, rounding=ROUND_FLOOR)
    high = exact.quantize(BUDGET_QUANTUM, rounding=ROUND_CEILING)
    for candidate in (low, high):
        if _quantize(candidate + profit) == after:
            return candidate
    return _quantize(exact)


def reconstruct_budget_series(
    final_budget: Decimal, period_profits: list[Decimal]
) -> list[Decimal]:
    """Reconstruct ``B_0..B_N`` from the final budget and per-period profits.

    Args:
        final_budget: Stored budget after the last period ``B_N``.
        period_profits: ``G_1..G_N``, the summed offer profits per period.

    Returns:
        ``[B_0, ..., B_N]`` with ``B_N == final_budget``.
    """
    series = [Decimal(final_budget)]
    for profit in reversed(period_profits):
        series.append(previous_budget(series[-1], profit))
    series.reverse()
    return series


def _offer_stats(rows: list[OfferRow]) -> OfferStats:
    """Aggregate offer statistics.

    Args:
        rows: Offer rows (may be empty).

    Returns:
        The statistics. Means are None for an empty selection.
    """
    accepted = [row for row in rows if row.accepted]
    count = len(rows)
    return OfferStats(
        offers=count,
        accepted_offers=len(accepted),
        acceptance_rate=100.0 * len(accepted) / count if count else 0.0,
        mean_offer_price=(
            math.fsum(row.offer_price for row in rows) / count if count else None
        ),
        mean_accepted_offer_price=(
            math.fsum(row.offer_price for row in accepted) / len(accepted)
            if accepted
            else None
        ),
        accepted_energy=math.fsum(row.acceptance_amount for row in rows),
        revenue=math.fsum(
            row.acceptance_amount * row.acceptance_price
            for row in accepted
            if row.acceptance_price is not None
        ),
        profit=math.fsum(row.profit for row in rows),
    )


def _score_pair(simulation: Simulation, periods: range) -> ScorePair:
    """Compute the corrected scores over ``periods``.

    Args:
        simulation: The run.
        periods: Period numbers to score.

    Returns:
        The scores, or None values with the error message when
        ``score_simulation`` rejects the run.
    """
    from abmem.services.hpo.objective import score_simulation

    try:
        scores = score_simulation(simulation, periods)
    except ValueError as exc:
        return ScorePair(periods.start, periods.stop - 1, None, None, str(exc))
    return ScorePair(
        periods.start, periods.stop - 1, scores.agent_score, scores.market_score
    )


def unavailable_reason(simulation: Simulation) -> str | None:
    """Return why a simulation has no analysis, or None if it has one.

    Checks the cheap conditions of :func:`build_run_analysis` (scenario
    template, state, stored periods) with at most one query.

    Args:
        simulation: The simulation.

    Returns:
        The message, or None.
    """
    from abmem.models.models import Period

    if simulation.proxy:
        return f"Simulation {simulation.pk} is a scenario template and has no analysis."
    if simulation.state != SimulationState.FINISHED:
        return (
            f"Simulation {simulation.pk} is "
            f"{_enum_name(SimulationState, simulation.state)}. "
            "The analysis is available when the run has FINISHED."
        )
    if not Period.objects.filter(market__simulation=simulation).exists():
        return (
            f"Simulation {simulation.pk} has no stored periods or offers "
            "(for example a purged hyperparameter search trial)."
        )
    return None


def build_run_analysis(simulation: Simulation) -> RunAnalysis:
    """Aggregate the analysis of a finished run from the database (read-only).

    Args:
        simulation: A finished run (``proxy=False``, state FINISHED).

    Returns:
        The analysis.

    Raises:
        RunAnalysisUnavailable: If ``simulation`` is a scenario template, not
            FINISHED, or has no periods or offers (for example a purged HPO
            evaluation run).
    """
    from abmem.models import AgentStepLog, PolicyCheckpoint
    from abmem.models.models import Offer, Period, Plant
    from abmem.services.algorithms import registry as algorithm_registry
    from abmem.services.market.clearing_algorithms.registry import clearing_label_for
    from abmem.services.hpo.objective import scored_periods
    from abmem.services.market.accounting import offer_profit, outcome_from_offer

    if simulation.proxy:
        raise RunAnalysisUnavailable(
            f"Simulation {simulation.pk} is a scenario template and has no analysis."
        )
    if simulation.state != SimulationState.FINISHED:
        raise RunAnalysisUnavailable(
            f"Simulation {simulation.pk} is {_enum_name(SimulationState, simulation.state)}. "
            "The analysis is available when the run has FINISHED."
        )
    market = simulation.market
    agents = list(market.agent_set.order_by("id"))
    plants: dict[int, list[PlantSetting]] = defaultdict(list)
    for agent_id, resource_name, capacity in (
        Plant.objects.filter(portfolio__agent__market=market)
        .order_by("portfolio__agent_id", "id")
        .values_list("portfolio__agent_id", "resource__name", "capacity")
    ):
        plants[agent_id].append(PlantSetting(resource_name, int(capacity)))

    period_objects = list(
        Period.objects.filter(market=market).order_by("periodNumber", "id")
    )
    offer_objects = list(
        Offer.objects.filter(period__market=market)
        .select_related("resource")
        .order_by("period__periodNumber", "agent_id", "id")
    )
    if not period_objects or not offer_objects:
        raise RunAnalysisUnavailable(
            f"Simulation {simulation.pk} has no stored periods or offers "
            "(for example a purged hyperparameter search trial)."
        )
    if any(offer.acceptanceAmount is None for offer in offer_objects):
        raise RunAnalysisUnavailable(
            f"Simulation {simulation.pk} has offers that were not cleared."
        )

    period_number_of = {period.id: period.periodNumber for period in period_objects}
    ptf_of = {period.periodNumber: period.ptf for period in period_objects}
    names = {agent.id: agent.name for agent in agents}
    numbers = [period.periodNumber for period in period_objects]
    # Index of the budget after period number t in the series B_0..B_N.
    index_of = {number: index + 1 for index, number in enumerate(numbers)}
    if any(ptf is None for ptf in ptf_of.values()):
        raise RunAnalysisUnavailable(
            f"Simulation {simulation.pk} has periods without a clearing price."
        )

    periods = [
        PeriodRow(
            period=period.periodNumber,
            demand=float(period.demand),
            met_demand=float(period.metDemand),
            ptf=_optional_float(period.ptf),
            market_volume=int(period.marketVolume),
        )
        for period in period_objects
    ]

    offers: list[OfferRow] = []
    period_profit: dict[int, dict[int, Decimal]] = defaultdict(
        lambda: defaultdict(Decimal)
    )
    offers_by_agent: dict[int, list[OfferRow]] = defaultdict(list)
    for offer in offer_objects:
        number = period_number_of[offer.period_id]
        profit = offer_profit(outcome_from_offer(offer))
        period_profit[offer.agent_id][number] += profit
        row = OfferRow(
            period=number,
            agent=names[offer.agent_id],
            resource=offer.resource.name,
            amount=int(offer.amount),
            offer_price=float(offer.offerPrice),
            accepted=bool(offer.acceptance),
            acceptance_amount=float(offer.acceptanceAmount),
            acceptance_price=_optional_float(offer.acceptancePrice),
            profit=float(profit),
        )
        offers.append(row)
        offers_by_agent[offer.agent_id].append(row)

    budgets_decimal: dict[int, list[Decimal]] = {
        agent.id: reconstruct_budget_series(
            Decimal(agent.budget),
            [period_profit[agent.id].get(number, Decimal(0)) for number in numbers],
        )
        for agent in agents
    }
    budgets = {
        names[agent_id]: [float(value) for value in series]
        for agent_id, series in budgets_decimal.items()
    }

    step_logs: dict[str, list[StepLogRow]] = {agent.name: [] for agent in agents}
    for values in (
        AgentStepLog.objects.filter(agent__market=market)
        .order_by("agent_id", "period__periodNumber")
        .values(
            "agent_id",
            "period__periodNumber",
            "reward",
            "critic_loss",
            "actor_loss",
            "mean_q",
            "exploration_scale",
            "intervened_plants",
            "learning_active",
            "evaluation_phase",
        )
    ):
        step_logs[names[values["agent_id"]]].append(
            StepLogRow(
                period=values["period__periodNumber"],
                reward=values["reward"],
                critic_loss=values["critic_loss"],
                actor_loss=values["actor_loss"],
                mean_q=values["mean_q"],
                exploration_scale=values["exploration_scale"],
                intervened=any(
                    bool(flag) for flag in values["intervened_plants"] or []
                ),
                learning_active=bool(values["learning_active"]),
                evaluation_phase=bool(values["evaluation_phase"]),
            )
        )

    agent_settings: list[AgentSetting] = []
    summaries: list[AgentSummary] = []
    n_periods = len(period_objects)
    for agent in agents:
        algorithm_name = str(agent.algorithm_type)
        baseline = algorithm_registry.is_baseline(algorithm_name)
        agent_settings.append(
            AgentSetting(
                name=agent.name,
                algorithm=algorithm_name,
                algorithm_label=algorithm_registry.label_for(algorithm_name),
                baseline=baseline,
                failstack_enabled=bool(agent.failstack_enabled) and not baseline,
                failstack_limit=int(agent.failstack_limit),
                failstack_delta=float(agent.failstack_delta),
                failstack_min_margin=float(agent.failstack_min_margin),
                algorithm_params=dict(agent.algorithm_params or {}),
                plants=plants.get(agent.id, []),
                effective_params=(
                    dict(agent.effective_params)
                    if agent.effective_params is not None
                    else None
                ),
            )
        )
        rows = offers_by_agent.get(agent.id, [])
        by_resource: dict[str, list[OfferRow]] = {}
        for row in rows:
            by_resource.setdefault(row.resource, []).append(row)
        series = budgets_decimal[agent.id]
        change = series[-1] - series[0]
        summaries.append(
            AgentSummary(
                name=agent.name,
                algorithm=algorithm_name,
                initial_budget=float(series[0]),
                final_budget=float(series[-1]),
                budget_change=float(change),
                mean_budget_change=float(change) / n_periods,
                stats=_offer_stats(rows),
                resources={
                    resource: _offer_stats(resource_rows)
                    for resource, resource_rows in by_resource.items()
                },
            )
        )

    settings = RunSettings(
        simulation_id=int(simulation.pk),
        name=simulation.name,
        seed=int(simulation.seed),
        observation_lags=list(simulation.observation_lags),
        evaluation_periods=int(simulation.evaluation_periods),
        period_number=int(simulation.periodNumber),
        mode=_enum_name(SimulationMode, simulation.mode),
        strategy=_enum_name(MarketStrategy, market.strategy),
        clearing_algorithm=str(market.clearing_algorithm),
        clearing_algorithm_label=clearing_label_for(market.clearing_algorithm),
        effective_clearing_params=(
            dict(market.effective_clearing_params)
            if market.effective_clearing_params is not None
            else None
        ),
        lower_bound=int(market.lowerBidBound),
        upper_bound=int(market.upperBidBound),
        agents=agent_settings,
        warm_start_from=simulation.warm_start_from_id,
        freeze_policies=bool(simulation.freeze_policies),
        checkpoints=[
            CheckpointSetting(
                agent=checkpoint.agent.name if checkpoint.agent is not None else "",
                algorithm=str(checkpoint.algorithm_type),
                file_size=int(checkpoint.file_size),
                periods_trained=int(checkpoint.periods_trained),
                parent_run=(
                    checkpoint.parent.simulation_id
                    if checkpoint.parent is not None
                    else None
                ),
            )
            for checkpoint in PolicyCheckpoint.objects.filter(simulation=simulation)
            .select_related("agent", "parent")
            .order_by("agent_id", "id")
        ],
    )

    total = int(simulation.periodNumber)
    window = (
        _score_pair(simulation, scored_periods(simulation))
        if int(simulation.evaluation_periods) > 0
        else None
    )
    corrected = CorrectedScores(
        price_cap=float(market.upperBidBound),
        whole_run=_score_pair(simulation, range(1, total + 1)),
        evaluation_window=window,
    )

    # Legacy rows: offers in export order, budget after the period of the row.
    legacy_rows = []
    for row, offer in zip(offers, offer_objects):
        ptf = ptf_of[row.period]
        legacy_rows.append(
            {
                "period": row.period,
                "agent": row.agent,
                "amount": row.amount,
                "offer_price": row.offer_price,
                "market_price": float(ptf),
                "accepted": row.accepted,
                "budget": float(budgets_decimal[offer.agent_id][index_of[row.period]]),
            }
        )
    agent_scores, market_score = legacy_composite_scores(legacy_rows)
    legacy = LegacyScores(
        price_cap=LEGACY_PRICE_CAP,
        agent_scores=agent_scores,
        mean_agent_score=math.fsum(agent_scores.values()) / len(agent_scores),
        market_score=market_score,
    )

    return RunAnalysis(
        settings=settings,
        periods=periods,
        offers=offers,
        budgets=budgets,
        step_logs=step_logs,
        agents=summaries,
        corrected=corrected,
        legacy=legacy,
    )
