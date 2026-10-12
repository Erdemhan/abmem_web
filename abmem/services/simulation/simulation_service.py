"""Run loop of a simulation (ARCHITECTURE.md 6.3).

``run`` builds a :class:`SimulationContext`, runs periods ``1..periodNumber``
through ``market_service.run_period`` and records the final state. Any
exception sets ``SimulationState.FAILED`` and ``Simulation.error`` before it
is re-raised (OQ-8). The module holds no mutable module-level state (B10).

Policy persistence (finding A9): a run with ``warm_start_from`` loads the saved
policies right after the context is built, and the policies of every learning
agent are saved once the run is FINISHED (``checkpoints``).
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Final

from django.conf import settings

from abmem.models import Simulation
from abmem.models.enums import (
    MarketState,
    SimulationMode,
    SimulationState,
)
from abmem.models.models import Offer
from abmem.services.market import market_service
from abmem.services.simulation import checkpoints
from abmem.services.simulation.context import (
    SimulationContext,
    SimulationValidationError,
)

logger = logging.getLogger(__name__)

POLL_INTERVAL_S: Final[float] = 0.2
PERIODS_PER_DAY: Final[int] = 24


def _sim_data_dir() -> pathlib.Path:
    """Return the directory of the JSON offer exports.

    Returns:
        ``settings.ABMEM_SIM_DATA_DIR`` if set, else
        ``BASE_DIR / 'abm_ddpg' / 'sim_data'``.
    """
    configured = getattr(settings, "ABMEM_SIM_DATA_DIR", None)
    if configured:
        return pathlib.Path(configured)
    return pathlib.Path(settings.BASE_DIR) / "abm_ddpg" / "sim_data"


def export_offers_json(
    simulation: Simulation, budgets_by_period: Mapping[int, Mapping[str, str]]
) -> pathlib.Path:
    """Write the offers of a run grouped by period and agent to a JSON file.

    Args:
        simulation: Finished run.
        budgets_by_period: Agent budgets after each period, by period number
            and agent name.

    Returns:
        Path of the written file.
    """
    offers = (
        Offer.objects.filter(period__market=simulation.market)
        .select_related("agent", "resource", "period")
        .order_by("period__periodNumber", "agent_id", "id")
    )
    by_period: dict[int, dict[str, Any]] = defaultdict(dict)
    for offer in offers:
        number = offer.period.periodNumber
        entry = by_period[number]
        entry.setdefault("market_price", str(offer.period.ptf))
        budget = budgets_by_period.get(number, {}).get(offer.agent.name)
        entry.setdefault(offer.agent.name, []).append(
            {
                "id": offer.id,
                "agent": offer.agent.name,
                "resource": offer.resource.name,
                "amount": offer.amount,
                "offerPrice": str(offer.offerPrice),
                "acceptance": offer.acceptance,
                "acceptancePrice": str(offer.acceptancePrice),
                "acceptanceAmount": (
                    str(offer.acceptanceAmount)
                    if offer.acceptanceAmount is not None
                    else None
                ),
                "budget": budget,
            }
        )
    output = {
        "simulation_id": simulation.id,
        "total_periods": simulation.periodNumber,
        "offers_by_period": dict(by_period),
    }
    directory = _sim_data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"simulation_{simulation.id}_offers.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=4, ensure_ascii=False)
    return path


def _wait_for_next_period(ctx: SimulationContext) -> None:
    """Block while the market waits for the user in PERIODBYPERIOD mode.

    Args:
        ctx: Run context.
    """
    ctx.market.refresh_from_db(fields=["state"])
    while ctx.market.state == MarketState.PERIODEND:
        time.sleep(POLL_INTERVAL_S)
        ctx.market.refresh_from_db(fields=["state"])


def _mark_failed(simulation: Simulation, error: BaseException) -> None:
    """Store the failure state and message of a run (OQ-8).

    Args:
        simulation: The run.
        error: The exception that ended the run.
    """
    simulation.state = SimulationState.FAILED
    simulation.error = f"{type(error).__name__}: {error}"
    simulation.save(update_fields=["state", "error", "updated_at"])


def _save_checkpoints(ctx: SimulationContext) -> None:
    """Save the policies of a FINISHED run (finding A9).

    The market results of the run are complete at this point, so a failure
    to write a checkpoint does not fail the run. It is logged and recorded in
    ``Simulation.error`` while the state stays FINISHED.

    Args:
        ctx: Context of the finished run.
    """
    try:
        checkpoints.save_run_checkpoints(ctx)
    except Exception as error:
        logger.exception(
            "Policies of simulation %s could not be saved", ctx.simulation.pk
        )
        ctx.simulation.error = (
            f"Policies could not be saved: {type(error).__name__}: {error}"
        )
        ctx.simulation.save(update_fields=["error", "updated_at"])


def run(
    simulation: Simulation | int,
    *,
    algorithm_overrides: Mapping[str, Any] | None = None,
    forced_algorithm: str | None = None,
    override_algorithm: str | None = None,
    export_json: bool = True,
) -> Simulation:
    """Run a simulation from period 1 to ``periodNumber``.

    Args:
        simulation: The run or its primary key.
        algorithm_overrides: HPO parameter overrides for every agent.
        forced_algorithm: Algorithm key used for every agent (HPO, K7).
        override_algorithm: When given, ``algorithm_overrides`` apply only to
            the agents whose effective algorithm is this key (PLAN 2.15).
        export_json: Write the JSON offer export at the end.

    Returns:
        The simulation in state FINISHED.

    Raises:
        ValueError: If the simulation or its market is not in state CREATED.
            The simulation is left unchanged.
        Exception: Any error of the run is re-raised after the simulation has
            been set to FAILED with ``error`` holding the message.
    """
    if not isinstance(simulation, Simulation):
        simulation = Simulation.objects.get(pk=simulation)
    market = simulation.market
    if (
        simulation.state != SimulationState.CREATED
        or market.state != MarketState.CREATED
    ):
        raise ValueError(
            f"simulation {simulation.pk} cannot run: state {simulation.state.name}, "
            f"market state {market.state.name}, both must be CREATED"
        )

    try:
        ctx = SimulationContext.build(
            simulation,
            algorithm_overrides,
            forced_algorithm=forced_algorithm,
            override_algorithm=override_algorithm,
            export_json=export_json,
        )
        try:
            checkpoints.apply_warm_start(ctx)
        except checkpoints.WarmStartError as error:
            raise SimulationValidationError(str(error)) from error
        simulation.state = SimulationState.INITIALIZED
        simulation.currentPeriod = 0
        simulation.save(update_fields=["state", "currentPeriod", "updated_at"])
        ctx.market.state = MarketState.INITIALIZED
        ctx.market.save(update_fields=["state", "updated_at"])

        first_day = int(simulation.day)
        period_by_period = simulation.mode == SimulationMode.PERIODBYPERIOD
        budgets_by_period: dict[int, dict[str, str]] = {}
        for t in range(1, simulation.periodNumber + 1):
            if period_by_period and t > 1:
                _wait_for_next_period(ctx)
            if t == ctx.evaluation_start:
                for rt in ctx.agents:
                    rt.policy.set_frozen(True)
            simulation.currentPeriod = t
            simulation.day = first_day + (t - 1) // PERIODS_PER_DAY
            simulation.save(update_fields=["currentPeriod", "day", "updated_at"])
            market_service.run_period(ctx, t)
            if t == 1:
                simulation.state = SimulationState.STARTED
                simulation.save(update_fields=["state", "updated_at"])
            if ctx.export_json:
                budgets_by_period[t] = {
                    rt.agent.name: str(rt.agent.budget) for rt in ctx.agents
                }

        if ctx.export_json:
            export_offers_json(simulation, budgets_by_period)
        simulation.state = SimulationState.FINISHED
        simulation.save(update_fields=["state", "updated_at"])
    except Exception as error:
        logger.exception("Simulation %s failed", simulation.pk)
        _mark_failed(simulation, error)
        raise
    _save_checkpoints(ctx)
    return simulation
