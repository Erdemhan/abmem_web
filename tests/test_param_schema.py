"""Unit tests of the config form description and validation (PLAN 2.12)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pytest

from abmem.services.algorithms.algorithm_config import (
    AlgorithmConfig,
    DDPGConfig,
    FixedMarkupConfig,
    RandomConfig,
    SACConfig,
    TD3Config,
)
from abmem.services.market.clearing_algorithms.config import ClearingConfig
from abmem.services.param_schema import (
    NON_FIELD_ERRORS,
    ParamField,
    attribute_descriptions,
    coerce_field_value,
    describe_config,
    field_kind,
    non_default_params,
    parse_form_value,
    validate_params,
)
from tests.plugin_helpers import ParamClearingConfig

# field_kind / coerce_field_value ---------------------------------------------


@pytest.mark.parametrize(
    ("hint", "kind"),
    [
        (int, "int"),
        (float, "float"),
        (bool, "bool"),
        (float | None, "optional_float"),
        (tuple[int, ...], "int_list"),
    ],
)
def test_field_kind_of_supported_hints(hint: Any, kind: str) -> None:
    assert field_kind(hint) == kind


@pytest.mark.parametrize("hint", [str, list[int], tuple[int, int], int | None, dict])
def test_field_kind_rejects_unsupported_hints(hint: Any) -> None:
    with pytest.raises(TypeError, match="unsupported parameter type"):
        field_kind(hint)


def test_coerce_field_value_converts_and_rejects() -> None:
    assert coerce_field_value("n", int, 3.0) == 3
    assert coerce_field_value("x", float, 2) == 2.0
    assert coerce_field_value("o", float | None, None) is None
    assert coerce_field_value("b", bool, True) is True
    assert coerce_field_value("w", tuple[int, ...], [1, 2.0]) == (1, 2)
    for hint, value in (
        (int, 2.5),
        (int, True),
        (float, math.nan),
        (float, math.inf),
        (float, "1"),
        (bool, 1),
        (tuple[int, ...], "1,2"),
    ):
        with pytest.raises(ValueError):
            coerce_field_value("f", hint, value)


# attribute_descriptions --------------------------------------------------------


def test_attribute_descriptions_merge_the_mro_and_join_continuation_lines() -> None:
    descriptions = attribute_descriptions(TD3Config)
    # Inherited from AlgorithmConfig, continuation line joined.
    assert descriptions["learning_starts"] == (
        "Stored transitions required before the first gradient step (B7, "
        "PLAN 1.5), >= batch_size."
    )
    # Declared by TD3Config itself.
    assert descriptions["policy_delay"] == "Critic updates per actor update, >= 1."
    assert set(AlgorithmConfig.field_names()) <= set(descriptions)


def test_attribute_descriptions_subclass_entry_wins() -> None:
    @dataclass(frozen=True)
    class Base:
        """Base.

        Attributes:
            a: Base text.
            b: Only in the base.
        """

    @dataclass(frozen=True)
    class Child(Base):
        """Child.

        Args:
            ignored: Not an attribute section.

        Attributes:
            a: Child text
                continued.
        """

    descriptions = attribute_descriptions(Child)
    assert descriptions == {"a": "Child text continued.", "b": "Only in the base."}


def test_attribute_descriptions_without_docstring_section() -> None:
    assert attribute_descriptions(ClearingConfig) == {}


# describe_config ---------------------------------------------------------------


def test_describe_config_of_td3() -> None:
    fields = {f.name: f for f in describe_config(TD3Config)}
    assert list(fields) == list(TD3Config.field_names())
    assert fields["gamma"] == ParamField(
        name="gamma",
        kind="float",
        default=0.99,
        description="Discount factor, 0 < gamma < 1.",
        search_bounds=(0.9, 0.999, "linear"),
    )
    assert fields["policy_delay"].kind == "int"
    assert fields["policy_delay"].search_bounds == (1.0, 4.0, "int")
    assert fields["hidden_sizes"].kind == "int_list"
    assert fields["hidden_sizes"].default == [256, 256]
    assert fields["max_grad_norm"].kind == "optional_float"
    assert fields["max_grad_norm"].default is None
    assert fields["batch_size"].search_bounds is None
    assert fields["gamma"].to_dict()["search_bounds"] == [0.9, 0.999, "linear"]


def test_describe_config_of_sac_and_baselines() -> None:
    sac = {f.name: f for f in describe_config(SACConfig)}
    assert sac["target_entropy"].kind == "optional_float"
    assert sac["exploration_sigma"].search_bounds is None
    random = describe_config(RandomConfig)
    assert all(f.search_bounds is None for f in random)
    markup = {f.name: f for f in describe_config(FixedMarkupConfig)}
    assert markup["markup"].default == 0.10
    assert markup["markup"].description.startswith("Relative markup")


def test_describe_config_of_clearing_configs() -> None:
    assert describe_config(ClearingConfig) == ()
    fields = {f.name: f for f in describe_config(ParamClearingConfig)}
    assert {name: f.kind for name, f in fields.items()} == {
        "markup": "float",
        "rounds": "int",
        "strict": "bool",
        "weights": "int_list",
        "cap": "optional_float",
    }
    assert fields["weights"].default == [1, 2]
    assert fields["markup"].description == "Test value, >= 0."
    assert all(f.search_bounds is None for f in fields.values())


# parse_form_value ----------------------------------------------------------------


def _field(kind: str, name: str = "f") -> ParamField:
    return ParamField(name=name, kind=kind, default=None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kind", "raw", "expected"),
    [
        ("int", "12", 12),
        ("int", " 3.0 ", 3),
        ("int", 7, 7),
        ("float", "3e-4", 3e-4),
        ("float", 0.5, 0.5),
        ("float", "-2", -2.0),
        ("optional_float", "", None),
        ("optional_float", None, None),
        ("optional_float", "1.5", 1.5),
        ("bool", True, True),
        ("bool", "false", False),
        ("bool", " TRUE ", True),
        ("int_list", "256, 128", [256, 128]),
        ("int_list", [64, "32"], [64, 32]),
    ],
)
def test_parse_form_value_accepts(kind: str, raw: Any, expected: Any) -> None:
    assert parse_form_value(_field(kind), raw) == expected


@pytest.mark.parametrize(
    ("kind", "raw", "fragment"),
    [
        ("int", "", "is required"),
        ("int", None, "is required"),
        ("int", "2.5", "must be an integer"),
        ("int", "abc", "must be a number"),
        ("float", "nan", "must be finite"),
        ("float", "inf", "must be finite"),
        ("float", True, "must be a real number"),
        ("bool", "yes", "true or false"),
        ("bool", 1, "true or false"),
        ("int_list", "", "non-empty list"),
        ("int_list", [], "non-empty list"),
        ("int_list", "1, x", "must be a number"),
        ("int_list", {"a": 1}, "non-empty list"),
    ],
)
def test_parse_form_value_rejects(kind: str, raw: Any, fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment) as info:
        parse_form_value(_field(kind, "name_x"), raw)
    assert "name_x" in str(info.value)


# validate_params -----------------------------------------------------------------


def test_validate_params_accepts_text_values() -> None:
    result = validate_params(TD3Config, {"gamma": "0.95", "hidden_sizes": "64, 64"})
    assert result.is_valid and result.errors == {}
    assert result.config == TD3Config(gamma=0.95, hidden_sizes=(64, 64))
    assert result.parsed == {"gamma": 0.95, "hidden_sizes": [64, 64]}


def test_validate_params_reports_errors_per_field() -> None:
    result = validate_params(
        TD3Config, {"gamma": "x", "policy_delay": "1.5", "tau": "0.01"}
    )
    assert not result.is_valid and result.config is None
    assert set(result.errors) == {"gamma", "policy_delay"}
    assert result.parsed == {"tau": 0.01}


def test_validate_params_assigns_range_errors_to_the_named_field() -> None:
    result = validate_params(TD3Config, {"gamma": 1.5})
    assert result.errors == {"gamma": ["gamma must be in (0, 1), got 1.5"]}
    cross = validate_params(DDPGConfig, {"batch_size": 128})
    # learning_starts (64) < batch_size: the message names learning_starts first.
    assert list(cross.errors) == ["learning_starts"]


def test_validate_params_unknown_keys_and_wrong_types() -> None:
    result = validate_params(DDPGConfig, {"NOISE_STD": 0.1})
    assert list(result.errors) == [NON_FIELD_ERRORS]
    assert "NOISE_STD" in result.errors[NON_FIELD_ERRORS][0]
    assert validate_params(DDPGConfig, ["gamma"]).errors  # type: ignore[arg-type]
    assert validate_params(DDPGConfig, None).config == DDPGConfig()


def test_validate_params_with_base_values() -> None:
    result = validate_params(
        ParamClearingConfig, {"rounds": "9"}, base={"markup": 0.5, "gone": 1}
    )
    assert result.is_valid
    assert result.parsed == {"rounds": 9}
    assert result.config == ParamClearingConfig(markup=0.5, rounds=9)
    failing = validate_params(
        ParamClearingConfig, {"rounds": "9"}, base={"markup": 0.0}
    )
    assert list(failing.errors) == ["rounds"]


def test_non_default_params() -> None:
    assert non_default_params(TD3Config()) == {}
    config = TD3Config(gamma=0.95, hidden_sizes=(64,), policy_delay=2)
    assert non_default_params(config) == {"gamma": 0.95, "hidden_sizes": [64]}
    assert non_default_params(ParamClearingConfig(strict=True)) == {"strict": True}
