"""Parameters of clearing algorithms: ClearingConfig and the registry (PLAN 2.12)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import FrozenInstanceError, dataclass
from decimal import Decimal

import pytest
from django.core.exceptions import ImproperlyConfigured

from abmem.services.market.clearing import Bid, ClearingResult
from abmem.services.market.clearing_algorithms import registry as clearing_registry
from abmem.services.market.clearing_algorithms.base import BaseClearingAlgorithm
from abmem.services.market.clearing_algorithms.config import ClearingConfig
from abmem.services.market.clearing_algorithms.merit_order import (
    MeritOrderClearing,
    MeritOrderConfig,
)
from tests.plugin_helpers import (
    ParamClearingConfig,
    temporary_param_clearing_algorithm,
)


def test_base_config_has_no_fields() -> None:
    config = ClearingConfig()
    assert config.field_names() == ()
    assert config.to_dict() == {}
    assert ClearingConfig.from_dict(None) == config
    assert ClearingConfig.from_dict({}) == config


def test_from_dict_to_dict_round_trip_and_coercion() -> None:
    config = ParamClearingConfig.from_dict(
        {"markup": 1, "rounds": 4.0, "weights": [3, 4], "cap": 2}
    )
    assert config.markup == 1.0 and isinstance(config.markup, float)
    assert config.rounds == 4 and isinstance(config.rounds, int)
    assert config.weights == (3, 4)
    assert config.cap == 2.0
    data = config.to_dict()
    assert data == {
        "markup": 1.0,
        "rounds": 4,
        "strict": False,
        "weights": [3, 4],
        "cap": 2.0,
    }
    assert ParamClearingConfig.from_dict(data) == config
    with pytest.raises(FrozenInstanceError):
        config.markup = 2.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("params", "fragment"),
    [
        ({"markup": -1}, "markup must be >= 0"),
        ({"rounds": 0}, "rounds must be >= 1"),
        ({"rounds": 2.5}, "rounds must be an integer"),
        ({"strict": "yes"}, "strict must be true or false"),
        ({"weights": "1,2"}, "weights must be a sequence"),
        ({"cap": float("nan")}, "cap must be finite"),
        ({"other": 1}, "unknown parameter"),
    ],
)
def test_from_dict_rejects(params: dict[str, object], fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        ParamClearingConfig.from_dict(params)


def test_from_dict_rejects_non_mappings() -> None:
    with pytest.raises(TypeError, match="mapping"):
        ClearingConfig.from_dict([1])  # type: ignore[arg-type]


def test_merit_order_has_an_empty_config() -> None:
    assert MeritOrderClearing.config_class is MeritOrderConfig
    assert MeritOrderConfig.field_names() == ()
    algorithm = clearing_registry.get_clearing_algorithm("MERIT_ORDER")
    assert algorithm.config == MeritOrderConfig()
    assert (
        clearing_registry.get_clearing_config_class("MERIT_ORDER") is MeritOrderConfig
    )
    with pytest.raises(ValueError, match="unknown parameter"):
        clearing_registry.get_clearing_algorithm("MERIT_ORDER", {"x": 1})


def test_instance_requires_its_config_class() -> None:
    assert MeritOrderClearing().config == MeritOrderConfig()
    with pytest.raises(TypeError, match="requires MeritOrderConfig"):
        MeritOrderClearing(ParamClearingConfig())  # type: ignore[arg-type]


def test_registry_builds_the_algorithm_with_its_parameters() -> None:
    with temporary_param_clearing_algorithm("TEST_PARAM_CLEARING") as cls:
        assert clearing_registry.get_clearing_config_class("TEST_PARAM_CLEARING") is (
            ParamClearingConfig
        )
        algorithm = clearing_registry.get_clearing_algorithm(
            "TEST_PARAM_CLEARING", {"markup": 2.0, "strict": True}
        )
        assert isinstance(algorithm, cls)
        assert algorithm.config == ParamClearingConfig(markup=2.0, strict=True)
        default = clearing_registry.get_clearing_algorithm("TEST_PARAM_CLEARING")
        assert default.config == ParamClearingConfig()
    with pytest.raises(KeyError):
        clearing_registry.get_clearing_config_class("TEST_PARAM_CLEARING")


def test_a_registered_class_with_an_invalid_config_class_is_rejected() -> None:
    @dataclass(frozen=True)
    class NotAClearingConfig:
        """Not derived from ClearingConfig."""

    def clear(
        self: BaseClearingAlgorithm,
        bids: Sequence[Bid],
        demand: Decimal,
        quantum: Decimal,
    ) -> ClearingResult:
        return MeritOrderClearing().clear(bids, demand, quantum)

    with pytest.raises(ImproperlyConfigured, match="config_class"):
        type(
            "BadConfigClearing",
            (BaseClearingAlgorithm,),
            {
                "key": "TEST_BAD_CONFIG",
                "clear": clear,
                "config_class": NotAClearingConfig,
                "__module__": __name__,
            },
        )
    assert not clearing_registry.is_known_clearing("TEST_BAD_CONFIG")
