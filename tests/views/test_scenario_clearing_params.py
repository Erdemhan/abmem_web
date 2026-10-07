"""Per-scenario clearing parameters in the wizard, the YAML and the analysis page (PLAN 2.12)."""

from __future__ import annotations

import json
from typing import Any

import pytest
import yaml
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client
from django.urls import reverse

from abmem import scenario_yaml
from abmem.models import ClearingDefault, Simulation
from abmem.models.enums import AgentType, EnergyType
from abmem.models.models import Resource
from abmem.services.agent import library_service
from abmem.services.simulation import simulation_service
from tests.factories import make_resource
from tests.plugin_helpers import temporary_param_clearing_algorithm
from tests.runtime.conftest import fast_demand  # noqa: F401 (fixture)
from tests.views.wizard_helpers import wizard_config

pytestmark = pytest.mark.django_db

PARAM = "TEST_PARAM_CLEARING"


@pytest.fixture
def fossil() -> Resource:
    """One saved fossil resource."""
    return make_resource(name="cp-gas", energy_type=EnergyType.FOSSIL, fuel_cost=20)


def _payload(resource: Resource, **fields: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "simulation_name": "cp-sim",
        "simulationMode": "1",
        "periodType": "0",
        "periodNumber": "2",
        "strategy": "1",
        "lowerBound": "0",
        "upperBound": "200",
        "seed": "42",
        "evaluation_periods": "0",
        "observation_lags": [1, 24],
        "agents": [
            {
                "mode": "new",
                "name": "CpAgent",
                "type": AgentType.FOSSIL.value,
                "budget": "0",
                "algorithm_type": "MARGINAL_COST",
                "failstack": {
                    "enabled": False,
                    "limit": "6",
                    "delta": "15",
                    "min_margin": "0",
                },
                "plants": [{"resource": str(resource.id), "capacity": "120"}],
            }
        ],
    }
    payload.update(fields)
    return payload


def _post(client: Client, url: str, payload: Any) -> Any:
    return client.post(url, json.dumps(payload), content_type="application/json")


def test_wizard_config_carries_the_clearing_param_fields(client: Client) -> None:
    with temporary_param_clearing_algorithm(PARAM):
        ClearingDefault.objects.create(algorithm_key=PARAM, params={"markup": 2.0})
        config = wizard_config(
            client.get(reverse("create_simulation")).content.decode()
        )
    assert config["initial"]["clearing_params"] == {}
    by_key = {c["key"]: c for c in config["clearing_algorithms"]}
    assert by_key["MERIT_ORDER"]["params"] == []
    fields = {f["name"]: f for f in by_key[PARAM]["params"]}
    # The placeholder value is the global default, the class default is kept too.
    assert fields["markup"]["value"] == 2.0 and fields["markup"]["default"] == 0.5
    page = client.get(reverse("create_simulation")).content.decode()
    assert 'id="clearing_params"' in page and 'id="clearing_params-feedback"' in page


def test_wizard_saves_and_updates_clearing_params(
    client: Client, fossil: Resource
) -> None:
    with temporary_param_clearing_algorithm(PARAM):
        response = _post(
            client,
            reverse("create_simulation"),
            _payload(
                fossil,
                clearing_algorithm=PARAM,
                clearing_params={"rounds": "4", "strict": "true", "weights": "3, 4"},
            ),
        )
        assert response.status_code == 200, response.content
        scenario = Simulation.objects.get(id=response.json()["simulation_id"])
        assert scenario.market.clearing_params == {
            "rounds": 4,
            "strict": True,
            "weights": [3, 4],
        }
        page = client.get(reverse("update_simulation", args=[scenario.id]))
        config = wizard_config(page.content.decode())
        assert config["initial"]["clearing_params"] == scenario.market.clearing_params

        payload = _payload(fossil, clearing_algorithm=PARAM, clearing_params={})
        payload["agents"] = []
        response = _post(
            client, reverse("update_simulation", args=[scenario.id]), payload
        )
        assert response.status_code == 200, response.content
        scenario.market.refresh_from_db()
        assert scenario.market.clearing_params == {}

        run = library_service.clone_scenario_for_run(scenario)
        assert run.market.clearing_params == {}


def test_wizard_rejects_invalid_clearing_params(
    client: Client, fossil: Resource
) -> None:
    with temporary_param_clearing_algorithm(PARAM):
        ClearingDefault.objects.create(algorithm_key=PARAM, params={"markup": 0.0})
        for params, fragment in (
            ({"rounds": "x"}, "rounds: rounds must be a number"),
            ({"rounds": "9"}, "rounds: rounds (9) must be <= 10 * markup + 5"),
            ({"nope": 1}, "Unknown parameter(s) nope"),
            ([1], "must be an object"),
        ):
            response = _post(
                client,
                reverse("create_simulation"),
                _payload(fossil, clearing_algorithm=PARAM, clearing_params=params),
            )
            assert response.status_code == 400, params
            messages = response.json()["errors"]["clearing_params"]
            assert any(fragment in message for message in messages), messages
    # Merit order accepts no parameters at all.
    response = _post(
        client, reverse("create_simulation"), _payload(fossil, clearing_params={"x": 1})
    )
    assert response.status_code == 400
    assert "clearing_params" in response.json()["errors"]
    assert not Simulation.objects.exists()


def _upload(text: str) -> SimpleUploadedFile:
    return SimpleUploadedFile(
        "scenario.yaml", text.encode("utf-8"), content_type="application/x-yaml"
    )


def test_yaml_round_trip_of_clearing_params(client: Client, fossil: Resource) -> None:
    with temporary_param_clearing_algorithm(PARAM):
        response = _post(
            client,
            reverse("create_simulation"),
            _payload(
                fossil, clearing_algorithm=PARAM, clearing_params={"markup": "1.5"}
            ),
        )
        scenario = Simulation.objects.get(id=response.json()["simulation_id"])
        document = scenario_yaml.scenario_document(scenario)
        assert document["market"]["clearing_params"] == {"markup": 1.5}
        assert list(document["market"]) == [
            "strategy",
            "clearing_algorithm",
            "clearing_params",
            "bid_bounds",
        ]
        text = client.get(reverse("export_scenario_yaml", args=[scenario.id])).content
        document = yaml.safe_load(text)
        document["scenario"]["name"] = "cp-imported"
        response = client.post(
            reverse("import_scenario_confirm"),
            {"file": _upload(yaml.safe_dump(document, sort_keys=False))},
        )
        assert response.status_code == 200, response.content
        imported = Simulation.objects.get(name="cp-imported")
        assert imported.market.clearing_params == {"markup": 1.5}

        document["scenario"]["name"] = "cp-bad"
        document["market"]["clearing_params"] = {"markup": -1}
        response = client.post(
            reverse("import_scenario_preview"),
            {"file": _upload(yaml.safe_dump(document, sort_keys=False))},
        )
        assert response.status_code == 400
        assert any(
            "markup must be >= 0" in message
            for message in response.json()["errors"]["market.clearing_params"]
        )


def test_yaml_export_without_overrides_has_no_clearing_params(
    client: Client, fossil: Resource
) -> None:
    response = _post(client, reverse("create_simulation"), _payload(fossil))
    scenario = Simulation.objects.get(id=response.json()["simulation_id"])
    assert "clearing_params" not in scenario_yaml.scenario_document(scenario)["market"]


def test_analysis_page_has_the_parameters_card_and_export(
    client: Client, fossil: Resource, fast_demand: object
) -> None:
    response = _post(client, reverse("create_simulation"), _payload(fossil))
    scenario = Simulation.objects.get(id=response.json()["simulation_id"])
    run = library_service.clone_scenario_for_run(scenario)
    simulation_service.run(run, export_json=False)

    page = client.get(reverse("simulation_analysis", args=[run.id])).content.decode()
    export_url = reverse("simulation_export_parameters", args=[run.id])
    assert 'id="an-params-used"' in page and "Parameters used" in page
    assert f'href="{export_url}"' in page

    response = client.get(export_url)
    assert response.status_code == 200
    assert response["Content-Type"].startswith("application/json")
    assert f"simulation_{run.id}_parameters.json" in response["Content-Disposition"]
    document = json.loads(response.content)
    assert document["clearing_algorithm"] == "MERIT_ORDER"
    assert document["clearing_params"] == {}
    assert document["agents"][0]["algorithm"] == "MARGINAL_COST"
    assert isinstance(document["agents"][0]["params"], dict)

    data = client.get(reverse("simulation_analysis_data", args=[run.id])).json()
    assert data["settings"]["effective_clearing_params"] == {}
    assert data["settings"]["agents"][0]["effective_params"] == (
        document["agents"][0]["params"]
    )

    assert (
        client.get(
            reverse("simulation_export_parameters", args=[scenario.id])
        ).status_code
        == 404
    )
