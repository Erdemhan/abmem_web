"""The Algorithms page: listing and editing default parameters (PLAN 2.12)."""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess
from typing import Any

import pytest
from django.test import Client
from django.urls import reverse

from abmem.models import (
    AlgorithmDefault,
    AlgorithmDefaultOrigin,
    ClearingDefault,
    HPOStudy,
)
from abmem.services import algorithm_catalog
from abmem.services.algorithms import registry as algorithm_registry
from abmem.services.algorithms.algorithm_config import TD3Config
from abmem.services.market.clearing_algorithms import registry as clearing_registry
from tests.plugin_helpers import (
    ParamClearingConfig,
    temporary_agent_algorithm,
    temporary_param_clearing_algorithm,
)
from tests.factories import make_simulation
from tests.runtime.conftest import make_run, resources

pytestmark = pytest.mark.django_db

PARAM = "TEST_PARAM_CLEARING"


def _params_url(kind: str, key: str) -> str:
    return reverse("algorithm_params", args=[kind, key])


def _reset_url(kind: str, key: str) -> str:
    return reverse("algorithm_params_reset", args=[kind, key])


def _post_params(client: Client, kind: str, key: str, params: Any) -> Any:
    return client.post(
        _params_url(kind, key),
        json.dumps({"params": params}),
        content_type="application/json",
    )


def _fields(form: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {field["name"]: field for field in form["fields"]}


# Page ----------------------------------------------------------------------------


def test_page_lists_both_kinds_in_two_columns(client: Client) -> None:
    response = client.get(reverse("algorithm_list"))
    assert response.status_code == 200
    page = response.content.decode()
    assert 'id="agent-algorithms"' in page and 'id="clearing-algorithms"' in page
    assert page.index('id="agent-algorithms"') < page.index('id="clearing-algorithms"')
    for info in algorithm_registry.available_algorithms():
        assert f'data-kind="agent" data-key="{info.key}"' in page
    assert 'data-kind="clearing" data-key="MERIT_ORDER"' in page
    assert "Uniform-price merit order" in page
    # Sidebar entry after the library entries.
    assert 'id="sidebar-algorithms-link"' in page
    assert page.index("sidebar-demand-profiles-link") < page.index(
        "sidebar-algorithms-link"
    )
    assert 'aria-current="page">' in page.split('id="sidebar-algorithms-link"')[1][:40]
    assert "confirm(" not in page and "alert(" not in page


def test_cards_show_kind_badges_usage_counts_and_custom_defaults(
    client: Client,
) -> None:
    r = resources(1)
    scenario = make_run([{"plants": [(r[0], 100)], "algorithm": "TD3"}])
    scenario.proxy = True
    scenario.save()
    AlgorithmDefault.objects.create(algorithm_type="TD3", params={"gamma": 0.9})
    cards, errors = algorithm_catalog.agent_cards()
    assert errors == []
    by_key = {card.key: card for card in cards}
    td3 = by_key["TD3"]
    assert td3.is_learning and not td3.is_on_policy
    assert td3.scenarios == 1 and td3.library_agents == 0
    assert td3.has_custom_defaults
    assert td3.param_count == len(TD3Config.field_names())
    assert by_key["PPO"].is_on_policy
    assert not by_key["RANDOM"].is_learning
    # Learning algorithms come first.
    learning = [card.is_learning for card in cards]
    assert learning == sorted(learning, reverse=True)

    clearing, clearing_errors = algorithm_catalog.clearing_cards()
    assert clearing_errors == []
    assert clearing[0].key == "MERIT_ORDER" and clearing[0].param_count == 0
    assert clearing[0].scenarios == 1

    page = client.get(reverse("algorithm_list")).content.decode()
    td3_card = page.split('data-key="TD3"')[1].split('data-kind="agent"')[0]
    assert "Learning" in td3_card and "Off-policy" in td3_card
    assert 'data-origin="CUSTOM">Custom defaults</span>' in td3_card
    assert re.search(r"0 library agents, 1 scenario\b", td3_card)
    assert "On-policy" in page.split('data-key="PPO"')[1].split('data-kind="agent"')[0]


def test_page_shows_a_module_that_failed_to_import(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    classes = algorithm_registry.AGENT_ALGORITHMS.classes()
    trace = "Traceback (most recent call last):\nImportError: no module named fancy\n"
    monkeypatch.setattr(
        algorithm_registry.AGENT_ALGORITHMS,
        "scan",
        lambda: (classes, {"abmem.services.algorithms.fancy": trace}),
    )
    page = client.get(reverse("algorithm_list")).content.decode()
    assert "Module abmem.services.algorithms.fancy failed to import." in page
    assert "ImportError: no module named fancy" in page
    # The other algorithms are still listed.
    assert 'data-key="DDPG"' in page


def test_a_config_class_that_cannot_be_described_is_not_editable(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(config_class: type) -> tuple[()]:
        raise TypeError("unsupported parameter type <class 'str'>")

    monkeypatch.setattr(algorithm_catalog, "describe_config", broken)
    cards, _ = algorithm_catalog.clearing_cards()
    assert "unsupported parameter type" in cards[0].config_error
    response = client.get(_params_url("clearing", "MERIT_ORDER"))
    assert response.status_code == 409
    assert "unsupported parameter type" in response.json()["errors"]["__all__"][0]


# Parameter form --------------------------------------------------------------------


def test_get_form_of_an_agent_algorithm(client: Client) -> None:
    response = client.get(_params_url("agent", "TD3"))
    assert response.status_code == 200
    form = response.json()
    assert form["kind"] == "agent" and form["key"] == "TD3"
    assert form["label"] == algorithm_registry.label_for("TD3")
    assert form["has_custom_defaults"] is False and form["stored"] == {}
    fields = _fields(form)
    assert list(fields) == list(TD3Config.field_names())
    gamma = fields["gamma"]
    assert gamma["kind"] == "float" and gamma["default"] == 0.99
    assert gamma["value"] == 0.99 and gamma["differs"] is False
    assert gamma["search_bounds"] == [0.9, 0.999, "linear"]
    assert gamma["description"] == "Discount factor, 0 < gamma < 1."
    assert fields["hidden_sizes"]["kind"] == "int_list"


def test_save_stores_only_changed_fields_and_reset_deletes(client: Client) -> None:
    values = {
        name: field["value"]
        for name, field in _fields(
            client.get(_params_url("agent", "TD3")).json()
        ).items()
    }
    values.update({"gamma": "0.95", "hidden_sizes": "64, 64", "max_grad_norm": ""})
    response = _post_params(client, "agent", "TD3", values)
    assert response.status_code == 200, response.content
    row = AlgorithmDefault.objects.get(algorithm_type="TD3")
    assert row.params == {"gamma": 0.95, "hidden_sizes": [64, 64]}
    form = response.json()
    assert form["has_custom_defaults"] is True
    fields = _fields(form)
    assert fields["gamma"]["value"] == 0.95 and fields["gamma"]["differs"] is True
    assert fields["tau"]["differs"] is False

    # Saving the class defaults removes the row.
    values = {name: field["default"] for name, field in fields.items()}
    response = _post_params(client, "agent", "TD3", values)
    assert response.status_code == 200
    assert not AlgorithmDefault.objects.filter(algorithm_type="TD3").exists()

    AlgorithmDefault.objects.create(algorithm_type="TD3", params={"gamma": 0.9})
    response = client.post(_reset_url("agent", "TD3"))
    assert response.status_code == 200
    assert response.json()["has_custom_defaults"] is False
    assert not AlgorithmDefault.objects.filter(algorithm_type="TD3").exists()
    # Reset without a row is harmless.
    assert client.post(_reset_url("agent", "TD3")).status_code == 200


def test_invalid_values_return_errors_per_field(client: Client) -> None:
    AlgorithmDefault.objects.create(algorithm_type="TD3", params={"gamma": 0.9})
    response = _post_params(
        client, "agent", "TD3", {"gamma": "1.5", "policy_delay": "x"}
    )
    assert response.status_code == 400
    errors = response.json()["errors"]
    assert set(errors) == {"policy_delay"}
    response = _post_params(client, "agent", "TD3", {"gamma": "1.5"})
    assert response.json()["errors"] == {"gamma": ["gamma must be in (0, 1), got 1.5"]}
    response = _post_params(client, "agent", "TD3", {"batch_size": "128"})
    assert "learning_starts" in response.json()["errors"]
    response = _post_params(client, "agent", "TD3", {"NOISE": 1})
    assert "__all__" in response.json()["errors"]
    # Nothing was written.
    assert AlgorithmDefault.objects.get(algorithm_type="TD3").params == {"gamma": 0.9}


@pytest.mark.parametrize("body", [b"not json", b"[]", b'{"params": [1]}'])
def test_malformed_bodies_are_rejected(client: Client, body: bytes) -> None:
    response = client.post(
        _params_url("agent", "TD3"), body, content_type="application/json"
    )
    assert response.status_code == 400
    assert "__all__" in response.json()["errors"]


def test_unknown_kind_key_and_method(client: Client) -> None:
    assert client.get(_params_url("robot", "TD3")).status_code == 404
    assert client.get(_params_url("agent", "NOPE")).status_code == 404
    assert client.get(_params_url("clearing", "TD3")).status_code == 404
    assert client.post(_reset_url("agent", "NOPE")).status_code == 404
    assert client.get(_reset_url("agent", "TD3")).status_code == 405
    assert client.put(_params_url("agent", "TD3")).status_code == 405


def test_stale_stored_default_is_reported(client: Client) -> None:
    AlgorithmDefault.objects.create(algorithm_type="TD3", params={"OLD_KEY": 1})
    form = client.get(_params_url("agent", "TD3")).json()
    assert "OLD_KEY" in form["stored_error"]
    assert _fields(form)["gamma"]["value"] == 0.99
    assert form["has_custom_defaults"] is True


# Clearing algorithms -------------------------------------------------------------


def test_merit_order_form_has_no_fields(client: Client) -> None:
    form = client.get(_params_url("clearing", "MERIT_ORDER")).json()
    assert form["fields"] == [] and form["label"] == "Uniform-price merit order"
    response = _post_params(client, "clearing", "MERIT_ORDER", {})
    assert response.status_code == 200
    assert not ClearingDefault.objects.exists()


def test_test_clearing_algorithm_is_listed_and_editable(client: Client) -> None:
    with temporary_param_clearing_algorithm(PARAM):
        page = client.get(reverse("algorithm_list")).content.decode()
        assert f'data-kind="clearing" data-key="{PARAM}"' in page
        assert "5 parameters" in page

        form = client.get(_params_url("clearing", PARAM)).json()
        fields = _fields(form)
        assert {name: f["kind"] for name, f in fields.items()} == {
            "markup": "float",
            "rounds": "int",
            "strict": "bool",
            "weights": "int_list",
            "cap": "optional_float",
        }
        assert fields["markup"]["description"] == "Test value, >= 0."

        response = _post_params(
            client,
            "clearing",
            PARAM,
            {
                "markup": "2",
                "rounds": "3",
                "strict": True,
                "weights": "1, 2",
                "cap": "",
            },
        )
        assert response.status_code == 200, response.content
        assert ClearingDefault.objects.get(algorithm_key=PARAM).params == {
            "markup": 2.0,
            "strict": True,
        }
        assert ParamClearingConfig.from_dict(
            ClearingDefault.objects.get(algorithm_key=PARAM).params
        ) == ParamClearingConfig(markup=2.0, strict=True)

        response = _post_params(client, "clearing", PARAM, {"markup": "-1"})
        assert response.status_code == 400
        assert list(response.json()["errors"]) == ["markup"]

        assert client.post(_reset_url("clearing", PARAM)).status_code == 200
        assert not ClearingDefault.objects.exists()
    assert not clearing_registry.is_known_clearing(PARAM)


def test_discovered_agent_algorithm_is_listed(client: Client) -> None:
    with temporary_agent_algorithm("TEST_AGENT_ALGO", "Test algorithm"):
        page = client.get(reverse("algorithm_list")).content.decode()
        assert 'data-key="TEST_AGENT_ALGO"' in page and "Test algorithm" in page
        assert client.get(_params_url("agent", "TEST_AGENT_ALGO")).status_code == 200


# Origin of the stored default -------------------------------------------------------


def _study(name: str = "td3 search") -> HPOStudy:
    """Saved TD3 study on a proxy scenario."""
    scenario = make_simulation(name="proxy")
    scenario.proxy = True
    scenario.save()
    return HPOStudy.objects.create(
        name=name,
        algorithm_type="TD3",
        scenario=scenario,
        param_bounds={"gamma": {"low": 0.9, "high": 0.99, "scale": "linear"}},
        colony_size=2,
        max_iter=1,
        limit=5,
        search_seeds=[1],
        evaluation_seeds=[9],
        objective={"mode": "weighted", "weights": [1.0, 0.0]},
    )


def _optimized_row(study: HPOStudy) -> AlgorithmDefault:
    return AlgorithmDefault.objects.create(
        algorithm_type="TD3",
        params={"gamma": 0.95},
        origin=AlgorithmDefaultOrigin.OPTIMIZED,
        source_study=study,
    )


def _form_values(client: Client) -> dict[str, Any]:
    return {
        name: field["value"]
        for name, field in _fields(
            client.get(_params_url("agent", "TD3")).json()
        ).items()
    }


def test_new_default_row_is_custom_without_study() -> None:
    row = AlgorithmDefault.objects.create(algorithm_type="TD3", params={})
    row.refresh_from_db()
    assert row.origin == AlgorithmDefaultOrigin.CUSTOM
    assert row.source_study is None


def test_page_shows_the_optimized_badge_with_the_study_link(client: Client) -> None:
    study = _study()
    _optimized_row(study)
    page = client.get(reverse("algorithm_list")).content.decode()
    td3_card = page.split('data-key="TD3"')[1].split('data-kind="agent"')[0]
    watch = reverse("hpo_watch", args=[study.pk])
    assert f'href="{watch}"' in td3_card
    assert (
        f'title="From hyperparameter search &#x27;td3 search&#x27; (#{study.pk})"'
        in (td3_card)
    )
    assert 'data-origin="OPTIMIZED">Optimized</a>' in td3_card
    assert "Custom defaults" not in td3_card

    form = client.get(_params_url("agent", "TD3")).json()
    assert form["origin"]["origin"] == "OPTIMIZED"
    assert form["origin"]["study_id"] == study.pk
    assert form["origin"]["study_name"] == "td3 search"
    assert form["origin"]["watch_url"] == watch


def test_deleted_study_shows_the_fallback_without_link(client: Client) -> None:
    study = _study()
    row = _optimized_row(study)
    study.delete()
    row.refresh_from_db()
    assert row.origin == AlgorithmDefaultOrigin.OPTIMIZED
    assert row.source_study is None
    page = client.get(reverse("algorithm_list")).content.decode()
    td3_card = page.split('data-key="TD3"')[1].split('data-kind="agent"')[0]
    assert 'data-origin="OPTIMIZED_DELETED">Optimized (search deleted)</span>' in (
        td3_card
    )
    assert "js-origin-link" not in td3_card
    form = client.get(_params_url("agent", "TD3")).json()
    assert form["origin"]["origin"] == "OPTIMIZED_DELETED"
    assert form["origin"]["watch_url"] == ""


def test_manual_save_turns_optimized_into_custom(client: Client) -> None:
    study = _study()
    _optimized_row(study)
    values = _form_values(client)
    values["tau"] = "0.01"
    response = _post_params(client, "agent", "TD3", values)
    assert response.status_code == 200, response.content
    row = AlgorithmDefault.objects.get(algorithm_type="TD3")
    assert row.origin == AlgorithmDefaultOrigin.CUSTOM
    assert row.source_study is None
    assert row.params["gamma"] == 0.95 and row.params["tau"] == 0.01
    assert response.json()["origin"]["origin"] == "CUSTOM"
    assert HPOStudy.objects.filter(pk=study.pk).exists()


def test_saving_the_stored_values_keeps_the_origin(client: Client) -> None:
    study = _study()
    _optimized_row(study)
    response = _post_params(client, "agent", "TD3", _form_values(client))
    assert response.status_code == 200, response.content
    row = AlgorithmDefault.objects.get(algorithm_type="TD3")
    assert row.origin == AlgorithmDefaultOrigin.OPTIMIZED
    assert row.source_study_id == study.pk
    assert row.params == {"gamma": 0.95}


def test_reset_deletes_an_optimized_default(client: Client) -> None:
    study = _study()
    _optimized_row(study)
    response = client.post(_reset_url("agent", "TD3"))
    assert response.status_code == 200
    assert response.json()["origin"] is None
    assert not AlgorithmDefault.objects.filter(algorithm_type="TD3").exists()
    page = client.get(reverse("algorithm_list")).content.decode()
    td3_card = page.split('data-key="TD3"')[1].split('data-kind="agent"')[0]
    assert "data-origin=" not in td3_card


def test_clearing_defaults_are_always_custom(client: Client) -> None:
    with temporary_param_clearing_algorithm(PARAM):
        response = _post_params(client, "clearing", PARAM, {"markup": "2.0"})
        assert response.status_code == 200, response.content
        assert response.json()["origin"]["origin"] == "CUSTOM"


NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_algorithms_inline_script_passes_node_check(
    client: Client, tmp_path: pathlib.Path
) -> None:
    _optimized_row(_study())
    html = client.get(reverse("algorithm_list")).content.decode()
    scripts = [
        body
        for attrs, body in re.findall(
            r"<script([^>]*)>(.*?)</script>", html, flags=re.S
        )
        if "src=" not in attrs and "application/json" not in attrs and body.strip()
    ]
    assert scripts
    for index, body in enumerate(scripts):
        path = tmp_path / f"algorithms_{index}.js"
        path.write_text(body, encoding="utf-8")
        result = subprocess.run(
            [NODE, "--check", str(path)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, f"script {index}: {result.stderr}"
