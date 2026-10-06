"""Per-run state of a simulation (ARCHITECTURE.md 5.4, ADR-002, B10).

A :class:`SimulationContext` owns everything a run needs between periods:
the validated demand series, the observation normaliser, the PTF history and
one :class:`AgentRuntimeState` (policy, generator, plants, fail-stack
counters, pending decision) per agent. It replaces the module-level lists of
the former runtime, so two runs in one process cannot share agents or
policies.
"""

from __future__ import annotations

import enum
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from abmem.services.agent.failstack import FailStackParams, FailStackState
from abmem.services.agent.observation import ALLOWED_LAG_SETS, ObservationNormalizer
from abmem.services.diagnostics.step_logger import StepLogger
from abmem.services.market.accounting import OfferOutcome
from abmem.services.market.demand import DemandDataError, DemandSeries

from .seeding import seed_everything, spawn_agent_rngs

if TYPE_CHECKING:
    from abmem.models import Agent, Market, Simulation
    from abmem.models.enums import AlgorithmType
    from abmem.models.models import Period, Plant
    from abmem.services.algorithms.base import ActionResult, BaseAgentAlgorithm

MARKET_DATA_FILE: Final[str] = "marketData.xlsx"
DEMAND_COLUMN: Final[str] = "DEMAND"


class SimulationValidationError(ValueError):
    """Raised by :meth:`SimulationContext.build` when a run cannot start."""


class RunPhase(enum.Enum):
    """Phase of a period: training with exploration, or evaluation (OQ-5)."""

    TRAIN = "train"
    EVALUATE = "evaluate"


@dataclass
class PendingDecision:
    """Decision of period t-1 that becomes a transition at period t.

    Attributes:
        obs_norm: Normalised observation the decision was based on.
        action: Output of ``select_action``.
        u_applied: Normalised action of the prices actually submitted.
        intervened: True if the fail-stack rule replaced any plant price.
    """

    obs_norm: np.ndarray
    action: ActionResult
    u_applied: np.ndarray
    intervened: bool


@dataclass
class AgentRuntimeState:
    """Runtime state of one agent during a run.

    Attributes:
        agent: The run agent.
        policy: Policy built from the agent's algorithm type.
        rng: Generator of the agent (policy noise, pre-learning actions,
            fail-stack draws).
        plants: Plants ordered by ``Plant.id``, defines the action index order.
        total_capacity: Sum of plant capacities (B2).
        failstack: Consecutive-failure counters per plant.
        failstack_params: None when the rule is disabled or the policy is a
            baseline.
        prev: Pending decision of t-1, None if it must not be stored.
        last_outcomes: Cleared outcomes of the agent's offers of t-1.
    """

    agent: Agent
    policy: BaseAgentAlgorithm
    rng: np.random.Generator
    plants: tuple[Plant, ...]
    total_capacity: Decimal
    failstack: FailStackState
    failstack_params: FailStackParams | None
    prev: PendingDecision | None = None
    last_outcomes: tuple[OfferOutcome, ...] = ()


@dataclass
class SimulationContext:
    """Everything a run carries from period to period.

    Attributes:
        simulation: The run.
        market: Market of the run.
        seed: Run seed.
        lags: Observation lag set (``Simulation.observation_lags``).
        evaluation_start: First period of the evaluation window,
            ``periodNumber - evaluation_periods + 1``.
        price_low: ``Market.lowerBidBound``.
        price_high: ``Market.upperBidBound``.
        demand: Validated demand series.
        normalizer: Observation normaliser shared by all agents (C4).
        agents: Runtime states ordered by ``Agent.id``.
        ptf_history: PTF by period number, filled after each clearing.
        step_logger: Buffer of ``AgentStepLog`` rows.
        export_json: Write the JSON offer export at the end of the run.
        current_period: Period being run, set by ``market_service.run_period``.
    """

    simulation: Simulation
    market: Market
    seed: int
    lags: tuple[int, ...]
    evaluation_start: int
    price_low: Decimal
    price_high: Decimal
    demand: DemandSeries
    normalizer: ObservationNormalizer
    agents: list[AgentRuntimeState]
    ptf_history: dict[int, Decimal]
    step_logger: StepLogger
    export_json: bool = True
    current_period: Period | None = field(default=None)

    def phase_for(self, t: int) -> RunPhase:
        """Return the phase of period ``t``.

        Args:
            t: Period number.

        Returns:
            ``RunPhase.EVALUATE`` if ``t >= evaluation_start``, else
            ``RunPhase.TRAIN``.
        """
        return RunPhase.EVALUATE if t >= self.evaluation_start else RunPhase.TRAIN

    @classmethod
    def build(
        cls,
        simulation: Simulation,
        algorithm_overrides: Mapping[str, Any] | None = None,
        *,
        forced_algorithm: AlgorithmType | None = None,
        export_json: bool = True,
        demand: DemandSeries | None = None,
    ) -> SimulationContext:
        """Validate a run, seed all generators and build every policy.

        Parameters per agent (OQ-9): ``AlgorithmDefault.params`` of the
        effective algorithm, overridden by ``Agent.algorithm_params`` (only
        when the effective algorithm is the agent's own), overridden by
        ``algorithm_overrides``.

        Args:
            simulation: Saved run with a market, agents, portfolios and plants.
            algorithm_overrides: HPO parameter overrides applied to every agent.
            forced_algorithm: Algorithm used for every agent instead of
                ``Agent.algorithm_type`` (HPO, K7).
            export_json: Stored on the context for the run loop.
            demand: Demand series to use. Read from ``marketData.xlsx`` when
                None.

        Returns:
            The context. No ``Period`` row has been created.

        Raises:
            SimulationValidationError: If any precondition of the run fails.
        """
        from abmem.models import AlgorithmDefault
        from abmem.models.enums import AlgorithmType as AlgorithmTypeEnum
        from abmem.models.models import Plant
        from abmem.services.algorithms import registry
        from abmem.services.algorithms.base import PolicySpec

        market = simulation.market
        period_number = int(simulation.periodNumber)
        if period_number < 1:
            raise SimulationValidationError(
                f"periodNumber must be >= 1, got {period_number}"
            )
        low = Decimal(market.lowerBidBound)
        high = Decimal(market.upperBidBound)
        if not high > low:
            raise SimulationValidationError(
                f"upperBidBound ({high}) must be greater than lowerBidBound ({low})"
            )
        lags = tuple(int(k) for k in (simulation.observation_lags or ()))
        if lags not in ALLOWED_LAG_SETS:
            raise SimulationValidationError(
                f"observation_lags {list(lags)} not allowed, expected one of "
                f"{[list(s) for s in sorted(ALLOWED_LAG_SETS)]}"
            )
        evaluation_periods = int(simulation.evaluation_periods)
        if evaluation_periods < 0 or (
            evaluation_periods > 0 and evaluation_periods >= period_number
        ):
            raise SimulationValidationError(
                f"evaluation_periods must satisfy 0 <= evaluation_periods < "
                f"periodNumber ({period_number}), got {evaluation_periods}"
            )

        try:
            if demand is None:
                from abmem.constants.paths import DATA_DIR

                demand = DemandSeries.from_excel(
                    DATA_DIR / MARKET_DATA_FILE, column=DEMAND_COLUMN
                )
            demand.validate_horizon(period_number)
        except DemandDataError as error:
            raise SimulationValidationError(str(error)) from error

        agents = list(market.agent_set.order_by("id"))
        if not agents:
            raise SimulationValidationError(f"market {market.pk} has no agents")
        plants_by_agent: dict[int, list[Plant]] = defaultdict(list)
        plant_rows = (
            Plant.objects.filter(portfolio__agent__market=market)
            .select_related("resource", "portfolio")
            .order_by("id")
        )
        for plant in plant_rows:
            plants_by_agent[plant.portfolio.agent_id].append(plant)

        seed = int(simulation.seed)
        try:
            seed_everything(seed)
            rngs = spawn_agent_rngs(seed, len(agents))
        except ValueError as error:
            raise SimulationValidationError(str(error)) from error

        defaults = {
            AlgorithmTypeEnum(int(row.algorithm_type)): dict(row.params or {})
            for row in AlgorithmDefault.objects.all()
        }
        normalizer = ObservationNormalizer.for_market(lags, float(low), float(high))
        states: list[AgentRuntimeState] = []
        for agent, rng in zip(agents, rngs):
            plants = tuple(plants_by_agent.get(agent.pk, ()))
            if not plants:
                raise SimulationValidationError(f"agent {agent.name!r} has no plants")
            if any(p.capacity < 0 for p in plants):
                raise SimulationValidationError(
                    f"agent {agent.name!r} has a plant with negative capacity"
                )
            total_capacity = sum((Decimal(p.capacity) for p in plants), Decimal(0))
            if total_capacity <= 0:
                raise SimulationValidationError(
                    f"agent {agent.name!r} has no plant with positive capacity"
                )
            own_type = AlgorithmTypeEnum(int(agent.algorithm_type))
            algorithm_type = (
                AlgorithmTypeEnum(int(forced_algorithm))
                if forced_algorithm is not None
                else own_type
            )
            params: dict[str, Any] = dict(defaults.get(algorithm_type, {}))
            if algorithm_type == own_type:
                params.update(agent.algorithm_params or {})
            if algorithm_overrides is not None:
                params.update(algorithm_overrides)
            spec = PolicySpec(
                obs_dim=1 + len(lags),
                action_dim=len(plants),
                price_low=float(low),
                price_high=float(high),
                plant_variable_costs=tuple(
                    float(p.resource.variableCost()) for p in plants
                ),
                plant_static_costs=tuple(
                    float(p.resource.staticCost()) for p in plants
                ),
            )
            try:
                policy = registry.build_policy(algorithm_type, spec, params, rng)
                failstack_params = (
                    FailStackParams.from_agent(agent)
                    if policy.is_learning
                    and algorithm_type not in registry.BASELINE_TYPES
                    else None
                )
            except (KeyError, TypeError, ValueError) as error:
                raise SimulationValidationError(
                    f"agent {agent.name!r} ({algorithm_type.name}): {error}"
                ) from error
            agent.runtime_policy = policy
            states.append(
                AgentRuntimeState(
                    agent=agent,
                    policy=policy,
                    rng=rng,
                    plants=plants,
                    total_capacity=total_capacity,
                    failstack=FailStackState.zeros(len(plants)),
                    failstack_params=failstack_params,
                )
            )

        return cls(
            simulation=simulation,
            market=market,
            seed=seed,
            lags=lags,
            evaluation_start=period_number - evaluation_periods + 1,
            price_low=low,
            price_high=high,
            demand=demand,
            normalizer=normalizer,
            agents=states,
            ptf_history={},
            step_logger=StepLogger(),
            export_json=export_json,
        )
