"""End-to-end web run path (PLAN 1.9, A2).

``views.run_simulation`` -> ``starter_service.start`` ->
``bootstrap.run_simulation_process`` -> ``simulation_service.run``. The spawned
process is replaced by a synchronous call of the same target, so the test
exercises the real entry point on the test database. ``transaction=True``
because ``starter_service.start`` closes all database connections.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from django.test import Client
from django.urls import reverse

from abmem.models import AgentStepLog, Simulation
from abmem.models.enums import EnergyType, MarketState, SimulationState
from abmem.models.models import Offer, Period
from abmem.services.starter import starter_service
from tests.factories import make_resource

pytestmark = pytest.mark.django_db(transaction=True)


class _SyncProcess:
    """Stand-in for ``multiprocessing.Process`` running the target in-process."""

    started: list[tuple[Any, tuple]] = []

    def __init__(self, target: Any, args: tuple = ()) -> None:
        self.target = target
        self.args = args

    def start(self) -> None:
        _SyncProcess.started.append((self.target, self.args))
        self.target(*self.args)


class _SyncContext:
    Process = _SyncProcess


def test_web_run_reaches_finished(monkeypatch, settings, tmp_path) -> None:
    settings.ABMEM_SIM_DATA_DIR = str(tmp_path)
    _SyncProcess.started = []
    monkeypatch.setattr(
        starter_service.multiprocessing, "get_context", lambda method: _SyncContext()
    )
    gas = make_resource(name="web-gas", energy_type=EnergyType.FOSSIL, fuel_cost=20)
    coal = make_resource(name="web-coal", energy_type=EnergyType.FOSSIL, fuel_cost=12)
    client = Client()
    payload = {
        "simulation_name": "web-run",
        "simulationMode": "1",
        "periodNumber": "5",
        "periodType": "0",
        "strategy": "0",
        "lowerBound": "0",
        "upperBound": "200",
        "agents": [
            {
                "name": "g1",
                "type": 1,
                "budget": 0,
                "plants": [{"resource": gas.id, "capacity": 20000}],
            },
            {
                "name": "c1",
                "type": 1,
                "budget": 0,
                "plants": [
                    {"resource": coal.id, "capacity": 15000},
                    {"resource": gas.id, "capacity": 10000},
                ],
            },
        ],
    }
    response = client.post(
        reverse("create_simulation"),
        json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200, response.content
    proxy = Simulation.objects.get(proxy=True, name="web-run")

    response = client.get(f"/dev/simulation/run/{proxy.id}/")

    assert response.status_code == 302
    run = Simulation.objects.get(proxy=False)
    assert response.url == reverse("watch_simulation", kwargs={"simulation_id": run.id})
    assert [args for _, args in _SyncProcess.started] == [(run.id,)]
    from abmem import bootstrap

    assert _SyncProcess.started[0][0] is bootstrap.run_simulation_process
    assert run.state == SimulationState.FINISHED, run.error
    assert run.error == ""
    assert run.currentPeriod == 5
    assert run.market.state == MarketState.PERIODEND
    assert Period.objects.filter(market=run.market).count() == 5
    assert Offer.objects.filter(period__market=run.market).count() == 5 * 3
    assert AgentStepLog.objects.filter(agent__market=run.market).count() == 2 * 4
    # the proxy scenario is untouched
    proxy.refresh_from_db()
    assert proxy.state == SimulationState.CREATED
    assert not Period.objects.filter(market=proxy.market).exists()
    assert (tmp_path / f"simulation_{run.id}_offers.json").is_file()

    status = client.get(reverse("get_simulation_status"), {"simulation_id": run.id})
    assert status.status_code == 200
    body = status.json()
    assert body["status"] == "FINISHED"
    assert "periods" not in body
    assert body["period"] == 5 and body["total"] == 5
    assert body["progress"]["completed"] == 5
    assert body["progress"]["percent"] == 100.0
    assert body["progress"]["finished"] is True
