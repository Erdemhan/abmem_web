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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from abmem.services.agent.failstack import FailStackParams, FailStackState
from abmem.services.agent.observation import (
    ALLOWED_LAG_SETS,
    DEMAND_SCALING_MEAN,
    DEMAND_SCALING_STD,
    ObservationNormalizer,
)
from abmem.services.diagnostics.step_logger import StepLogger
from abmem.services.market.accounting import OfferOutcome
from abmem.services.market.demand import DemandDataError, DemandSeries

from .seeding import seed_everything, spawn_agent_rngs

if TYPE_CHECKING:
    from abmem.services.market.demand_generator import DemandProfile
    from abmem.models import Agent, Market, Simulation
    from abmem.models import PolicyCheckpoint
    from abmem.models.models import Period, Plant
    from abmem.services.algorithms.algorithm_config import AlgorithmConfig
    from abmem.services.algorithms.base import PolicySpec
    from abmem.services.algorithms.base import ActionResult, BaseAgentAlgorithm
    from abmem.services.market.clearing_algorithms.base import BaseClearingAlgorithm

MARKET_DATA_FILE: Final[str] = "marketData.xlsx"
DEMAND_COLUMN: Final[str] = "DEMAND"


def algorithm_defaults() -> dict[str, dict[str, Any]]:
    """Return ``AlgorithmDefault.params`` by algorithm key (OQ-9).

    Returns:
        Mapping of algorithm key to a copy of its default parameters.
    """
    from abmem.models import AlgorithmDefault

    return {
        str(row.algorithm_type): dict(row.params or {})
        for row in AlgorithmDefault.objects.all()
    }


def clearing_defaults() -> dict[str, dict[str, Any]]:
    """Return ``ClearingDefault.params`` by clearing algorithm key (PLAN 2.12).

    Returns:
        Mapping of clearing algorithm key to a copy of its default parameters.
    """
    from abmem.models import ClearingDefault

    return {
        str(row.algorithm_key): dict(row.params or {})
        for row in ClearingDefault.objects.all()
    }


def market_clearing_params(
    market: Market, defaults: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Return the effective parameter overrides of a market's clearing algorithm.

    Args:
        market: Scenario or run market.
        defaults: Output of :func:`clearing_defaults`.

    Returns:
        ``defaults[market.clearing_algorithm]``, overridden by
        ``Market.clearing_params``. The class defaults of the config class
        apply to every missing field.
    """
    params: dict[str, Any] = dict(defaults.get(str(market.clearing_algorithm), {}))
    params.update(market.clearing_params or {})
    return params


def write_parameter_snapshot(
    market: Market,
    clearing: BaseClearingAlgorithm,
    states: list[AgentRuntimeState],
) -> None:
    """Store the parameters a run actually uses on its market and agents (PLAN 2.12).

    ``Agent.effective_params`` receives ``policy.config.to_dict()`` and
    ``Market.effective_clearing_params`` receives ``clearing.config.to_dict()``.
    Two UPDATE statements in total.

    Args:
        market: Market of the run.
        clearing: The clearing algorithm built for the run.
        states: Runtime states of the run agents.
    """
    from abmem.models import Agent

    for state in states:
        state.agent.effective_params = state.policy.config.to_dict()
    Agent.objects.bulk_update([state.agent for state in states], ["effective_params"])
    market.effective_clearing_params = clearing.config.to_dict()
    type(market).objects.filter(pk=market.pk).update(
        effective_clearing_params=market.effective_clearing_params
    )


def overrides_apply(algorithm_type: object, override_algorithm: object | None) -> bool:
    """Return True if HPO overrides apply to an agent with this algorithm (PLAN 2.15).

    Args:
        algorithm_type: Key of the effective algorithm of the agent.
        override_algorithm: Algorithm key the overrides are restricted to, or
            None when they apply to every agent.

    Returns:
        True when ``override_algorithm`` is None or equals ``algorithm_type``.
    """
    return override_algorithm is None or str(algorithm_type) == str(override_algorithm)


def agent_params(
    agent: Agent,
    algorithm_type: str,
    defaults: Mapping[str, Mapping[str, Any]],
    algorithm_overrides: Mapping[str, Any] | None = None,
    *,
    override_algorithm: str | None = None,
) -> dict[str, Any]:
    """Return the effective parameters of an agent's policy (OQ-9).

    Args:
        agent: The agent.
        algorithm_type: Key of the effective algorithm of the agent in this run.
        defaults: Output of :func:`algorithm_defaults`.
        algorithm_overrides: HPO overrides applied last.
        override_algorithm: When given, ``algorithm_overrides`` apply only if
            ``algorithm_type`` equals this key (:func:`overrides_apply`).

    Returns:
        ``defaults[algorithm_type]``, overridden by ``Agent.algorithm_params``
        (only when ``algorithm_type`` is the agent's own), overridden by
        ``algorithm_overrides`` (only when they apply to ``algorithm_type``).
    """
    params: dict[str, Any] = dict(defaults.get(algorithm_type, {}))
    if str(algorithm_type) == str(agent.algorithm_type):
        params.update(agent.algorithm_params or {})
    if algorithm_overrides is not None and overrides_apply(
        algorithm_type, override_algorithm
    ):
        params.update(algorithm_overrides)
    return params


_UNBOUNDED_LEARNING_STARTS: Final[int] = 2**31 - 1


def learner_config(algorithm_key: str, params: Mapping[str, Any]) -> AlgorithmConfig:
    """Return the config of a learning agent with ``learning_starts`` made reachable.

    ``learning_starts`` is raised to ``min_learning_starts()`` when the
    agent's own value is smaller (for example after ``batch_size`` was
    increased), so the automatic common learning start adapts instead of
    failing (PLAN 2.14).

    Args:
        algorithm_key: Key of a learning algorithm.
        params: Effective parameters of the agent (``agent_params``).

    Returns:
        The validated config.

    Raises:
        KeyError: Unknown algorithm key.
        TypeError, ValueError: A parameter is invalid.
    """
    from abmem.services.algorithms import registry

    config_class = registry.get_config_class(algorithm_key)
    probe = config_class.from_dict(
        {**params, "learning_starts": _UNBOUNDED_LEARNING_STARTS}
    )
    own = params.get("learning_starts", config_class().learning_starts)
    if isinstance(own, bool):
        raise TypeError(f"learning_starts must be an integer, got {own!r}")
    own = int(own)
    return config_class.from_dict(
        {**params, "learning_starts": max(own, probe.min_learning_starts())}
    )


def common_learning_starts(
    explicit: int | None, learners: Sequence[tuple[str, AlgorithmConfig]]
) -> int | None:
    """Return the common learning start of a run's learning agents (PLAN 2.14).

    Args:
        explicit: ``Simulation.learning_starts``, None for automatic.
        learners: ``(label, config)`` of every learning agent from
            :func:`learner_config`, the label being
            ``"<agent name> (<algorithm key>)"``.

    Returns:
        ``explicit``, or the largest ``first_update_transitions()`` of the
        learners. None when the run has no learning agent.

    Raises:
        SimulationValidationError: The value is below the technical minimum
            (``min_learning_starts()``) or above the replay buffer of a
            learning agent.
    """
    if not learners:
        return None
    if explicit is None:
        value = max(config.first_update_transitions() for _, config in learners)
    else:
        value = int(explicit)
    for name, config in learners:
        minimum = config.min_learning_starts()
        if value < minimum:
            raise SimulationValidationError(
                f"learning start {value} is below the minimum of agent {name}: "
                f"{minimum} transitions (its batch size, sequence length or "
                "rollout length)"
            )
        limit = config.learning_starts_limit()
        if limit is not None and value > limit:
            raise SimulationValidationError(
                f"learning start {value} exceeds the buffer size of agent "
                f"{name}: {limit} transitions"
            )
    return value


@dataclass(frozen=True)
class LearningStart:
    """Common learning start of a set of learning agents (PLAN 2.14).

    Attributes:
        value: The common learning start ``L`` in stored transitions.
        automatic: True when ``value`` is the automatic value (no scenario
            value is set).
        minimum: Largest technical minimum (``min_learning_starts()``) of the
            learning agents, the smallest value a scenario may set.
        limit: Smallest replay buffer size of the learning agents, the
            largest value a scenario may set. None when no agent has a limit.
    """

    value: int
    automatic: bool
    minimum: int
    limit: int | None


def resolve_learning_start(
    explicit: int | None,
    agents: Sequence[tuple[str, str, Mapping[str, Any]]],
    defaults: Mapping[str, Mapping[str, Any]] | None = None,
) -> LearningStart | None:
    """Return the common learning start of agents that need not be saved (PLAN 2.14).

    Each agent's parameters are resolved as :func:`agent_params` does for an
    agent whose effective algorithm is its own: ``defaults[algorithm_key]``
    overridden by the agent's own parameters. Baselines and unknown
    algorithm keys are ignored. Agents with invalid parameters are skipped,
    because :meth:`SimulationContext.build` reports those with their own
    message.

    Args:
        explicit: ``Simulation.learning_starts``, None for automatic.
        agents: ``(label, algorithm_key, params)`` per agent, ``params``
            being the agent's own ``algorithm_params``.
        defaults: Output of :func:`algorithm_defaults`, read from the
            database when None.

    Returns:
        The resolved learning start, None without learning agents.

    Raises:
        SimulationValidationError: ``explicit`` is out of range for an agent
            (message of :func:`common_learning_starts`).
    """
    from abmem.services.algorithms import registry

    if defaults is None:
        defaults = algorithm_defaults()
    learners: list[tuple[str, AlgorithmConfig]] = []
    for label, key, own_params in agents:
        key = str(key)
        if not registry.is_known(key) or registry.is_baseline(key):
            continue
        params: dict[str, Any] = dict(defaults.get(key, {}))
        params.update(own_params or {})
        try:
            config = learner_config(key, params)
        except (KeyError, TypeError, ValueError):
            continue
        learners.append((label, config))
    value = common_learning_starts(explicit, learners)
    if value is None:
        return None
    limits = [
        limit
        for limit in (config.learning_starts_limit() for _, config in learners)
        if limit is not None
    ]
    return LearningStart(
        value=value,
        automatic=explicit is None,
        minimum=max(config.min_learning_starts() for _, config in learners),
        limit=min(limits) if limits else None,
    )


def first_learning_period(observation_lags: Sequence[int], learning_start: int) -> int:
    """Return the period of the first gradient update of a run (EQ-09a).

    Args:
        observation_lags: Observation lag set of the run.
        learning_start: Common learning start ``L`` in stored transitions.

    Returns:
        ``t0 + L`` with ``t0 = max(observation_lags) + 1``, the first period
        with a complete observation (periods are numbered from 1).
    """
    return max(observation_lags) + 1 + int(learning_start)


def scenario_learning_start(simulation: Simulation) -> int | None:
    """Check the learning start of a scenario or run before it starts (PLAN 2.14).

    Resolves every agent of ``simulation.market`` with the stored defaults
    and agent parameters, as :meth:`SimulationContext.build` does without
    HPO overrides (:func:`resolve_learning_start`).

    Args:
        simulation: Scenario or run with a market.

    Returns:
        The common learning start, None without learning agents.

    Raises:
        SimulationValidationError: The value is out of range for an agent.
    """
    agents = [
        (
            f"{agent.name!r} ({agent.algorithm_type})",
            str(agent.algorithm_type),
            agent.algorithm_params or {},
        )
        for agent in simulation.market.agent_set.order_by("id")
    ]
    resolved = resolve_learning_start(simulation.learning_starts, agents)
    return None if resolved is None else resolved.value


def policy_spec_for(
    plants: tuple[Plant, ...], lags: tuple[int, ...], low: Decimal, high: Decimal
) -> PolicySpec:
    """Return the policy spec of an agent with ``plants`` (ordered by id).

    Args:
        plants: Plants of the agent in action order, resources loaded.
        lags: Observation lag set.
        low: Lower bid bound.
        high: Upper bid bound.

    Returns:
        The spec with ``obs_dim = 1 + len(lags)`` and one action per plant.
    """
    from abmem.services.algorithms.base import PolicySpec

    return PolicySpec(
        obs_dim=1 + len(lags),
        action_dim=len(plants),
        price_low=float(low),
        price_high=float(high),
        plant_variable_costs=tuple(float(p.resource.variableCost()) for p in plants),
        plant_static_costs=tuple(float(p.resource.staticCost()) for p in plants),
    )


def demand_for(simulation: Simulation, periods: int) -> DemandSeries:
    """Return the demand series of a simulation (PLAN 2.8).

    Args:
        simulation: Run or scenario.
        periods: Number of periods to generate for a profile (the Excel
            series is returned whole).

    Returns:
        ``DemandSeries.from_profile`` of ``simulation.demand_profile``, or the
        ``DEMAND`` column of ``marketData.xlsx`` when the profile is NULL.

    Raises:
        DemandDataError: The demand file cannot be read or the stored
            profile is invalid.
    """
    profile = _stored_profile(simulation)
    if profile is not None:
        return DemandSeries.from_profile(profile, periods)
    from abmem.constants.paths import DATA_DIR

    return DemandSeries.from_excel(DATA_DIR / MARKET_DATA_FILE, column=DEMAND_COLUMN)


def demand_scaling_for(simulation: Simulation, periods: int) -> tuple[float, float]:
    """Return the demand observation scaling of a simulation.

    Args:
        simulation: Run or scenario.
        periods: Number of periods of the run (used by seasonal profiles).

    Returns:
        ``DemandProfile.scaling(periods)`` of the stored profile (EQ-02b), or
        ``(DEMAND_SCALING_MEAN, DEMAND_SCALING_STD)`` of the demand file
        (EQ-02) when the profile is NULL.

    Raises:
        DemandDataError: The stored profile is invalid.
    """
    profile = _stored_profile(simulation)
    if profile is None:
        return DEMAND_SCALING_MEAN, DEMAND_SCALING_STD
    return profile.scaling(periods)


def _stored_profile(simulation: Simulation) -> DemandProfile | None:
    """Parse ``simulation.demand_profile``, or return None when it is NULL.

    Raises:
        DemandDataError: The stored profile is invalid.
    """
    if simulation.demand_profile is None:
        return None
    from django.core.exceptions import ValidationError

    from abmem.services.market.demand_generator import DemandProfile

    try:
        return DemandProfile.from_dict(simulation.demand_profile)
    except ValidationError as error:
        raise DemandDataError(
            f"invalid demand profile: {'; '.join(error.messages)}"
        ) from error


def _default_clearing() -> BaseClearingAlgorithm:
    """Return an instance of the default clearing algorithm (``MERIT_ORDER``)."""
    from abmem.services.market.clearing_algorithms.registry import (
        DEFAULT_CLEARING_ALGORITHM,
        get_clearing_algorithm,
    )

    return get_clearing_algorithm(DEFAULT_CLEARING_ALGORITHM)


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
        evaluate_only: The agent runs every period in the evaluation phase
            (a warm-started policy with ``Simulation.freeze_policies``).
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
    evaluate_only: bool = False


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
        warm_start_parents: Checkpoint each warm-started agent was loaded
            from, by ``Agent.id`` (set by ``checkpoints.apply_warm_start``).
        warm_start_warnings: Agents that start fresh in a warm-started run.
        clearing: Clearing algorithm of ``Market.clearing_algorithm``, resolved
            once by :meth:`build` (PLAN 2.10).
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
    warm_start_parents: dict[int, PolicyCheckpoint] = field(default_factory=dict)
    warm_start_warnings: list[str] = field(default_factory=list)
    clearing: BaseClearingAlgorithm = field(default_factory=lambda: _default_clearing())

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
        forced_algorithm: str | None = None,
        override_algorithm: str | None = None,
        export_json: bool = True,
        demand: DemandSeries | None = None,
    ) -> SimulationContext:
        """Validate a run, seed all generators and build every policy.

        Parameters per agent (OQ-9): ``AlgorithmDefault.params`` of the
        effective algorithm, overridden by ``Agent.algorithm_params`` (only
        when the effective algorithm is the agent's own), overridden by
        ``algorithm_overrides``. Parameters of the clearing algorithm:
        ``ClearingDefault.params``, overridden by ``Market.clearing_params``
        (PLAN 2.12). The resolved parameters are written once to
        ``Agent.effective_params`` and ``Market.effective_clearing_params``
        (:func:`write_parameter_snapshot`). Defaults are never read again
        during the run.

        Args:
            simulation: Saved run with a market, agents, portfolios and plants.
            algorithm_overrides: HPO parameter overrides applied to every agent.
            forced_algorithm: Algorithm key used for every agent instead of
                ``Agent.algorithm_type`` (HPO, K7).
            override_algorithm: When given, ``algorithm_overrides`` apply only
                to the agents whose effective algorithm is this key (HPO
                target MATCHING, PLAN 2.15).
            export_json: Stored on the context for the run loop.
            demand: Demand series to use. When None, generated from
                ``simulation.demand_profile`` or read from ``marketData.xlsx``
                when the profile is NULL (:func:`demand_for`).

        Returns:
            The context. No ``Period`` row has been created.

        Raises:
            SimulationValidationError: If any precondition of the run fails,
                including an agent or clearing algorithm key that is not
                registered or invalid parameters.
        """
        from abmem.models.models import Plant
        from abmem.services.algorithms import registry
        from abmem.services.market.clearing_algorithms import (
            registry as clearing_registry,
        )

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
        clearing_key = str(market.clearing_algorithm)
        if not clearing_registry.is_known_clearing(clearing_key):
            raise SimulationValidationError(
                f"market {market.pk}: clearing algorithm {clearing_key!r} is not "
                f"available, known keys: "
                f"{[info.key for info in clearing_registry.available_clearing_algorithms()]}"
            )
        try:
            clearing = clearing_registry.get_clearing_algorithm(
                clearing_key, market_clearing_params(market, clearing_defaults())
            )
        except (TypeError, ValueError) as error:
            raise SimulationValidationError(
                f"market {market.pk}: parameters of the clearing algorithm "
                f"{clearing_key}: {error}"
            ) from error
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
                demand = demand_for(simulation, period_number)
            demand.validate_horizon(period_number)
            demand_mean, demand_std = demand_scaling_for(simulation, period_number)
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

        defaults = algorithm_defaults()
        normalizer = ObservationNormalizer.for_market(
            lags,
            float(low),
            float(high),
            demand_mean=demand_mean,
            demand_std=demand_std,
        )
        # Pass 1: resolve every agent's algorithm and parameters. Pass 2 sets the
        # common learning start (PLAN 2.14) and builds the policies.
        resolved: list[tuple[Agent, np.random.Generator, tuple[Plant, ...], Decimal, str, dict[str, Any]]] = []
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
            algorithm_type = str(
                forced_algorithm
                if forced_algorithm is not None
                else agent.algorithm_type
            )
            if not registry.is_known(algorithm_type):
                raise SimulationValidationError(
                    f"agent {agent.name!r}: algorithm {algorithm_type!r} is not "
                    f"available, known keys: "
                    f"{[info.key for info in registry.available_algorithms()]}"
                )
            params = agent_params(
                agent,
                algorithm_type,
                defaults,
                algorithm_overrides,
                override_algorithm=override_algorithm,
            )
            resolved.append((agent, rng, plants, total_capacity, algorithm_type, params))

        learners: list[tuple[str, AlgorithmConfig]] = []
        for agent, _, _, _, algorithm_type, params in resolved:
            if registry.is_baseline(algorithm_type):
                continue
            try:
                config = learner_config(algorithm_type, params)
            except (KeyError, TypeError, ValueError) as error:
                raise SimulationValidationError(
                    f"agent {agent.name!r} ({algorithm_type}): {error}"
                ) from error
            learners.append((f"{agent.name!r} ({algorithm_type})", config))
        learning_starts = common_learning_starts(simulation.learning_starts, learners)

        states: list[AgentRuntimeState] = []
        for agent, rng, plants, total_capacity, algorithm_type, params in resolved:
            if learning_starts is not None and not registry.is_baseline(algorithm_type):
                params = {**params, "learning_starts": learning_starts}
            spec = policy_spec_for(plants, lags, low, high)
            try:
                policy = registry.build_policy(algorithm_type, spec, params, rng)
                failstack_params = (
                    FailStackParams.from_agent(agent) if policy.is_learning else None
                )
            except (KeyError, TypeError, ValueError) as error:
                raise SimulationValidationError(
                    f"agent {agent.name!r} ({algorithm_type}): {error}"
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

        write_parameter_snapshot(market, clearing, states)
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
            clearing=clearing,
        )
