"""Base class of market clearing algorithms (PLAN 2.10).

A clearing algorithm receives the bids of one period, the demand and the
quantity resolution and returns a :class:`~abmem.services.market.clearing.ClearingResult`.
It decides which quantities are accepted and the clearing price (PTF). It does
not decide what accepted bids are paid: the settlement rule
(``Market.strategy``, pay-as-bid or pay-as-PTF) is applied afterwards by
:func:`~abmem.services.market.clearing.acceptance_prices`.

The engine checks every result with
:func:`~abmem.services.market.clearing.validate_clearing_result` before it
writes anything, so a faulty algorithm stops the run with a
:class:`~abmem.services.market.clearing.ClearingContractError`.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from decimal import Decimal
from typing import Any, ClassVar

from django.core.exceptions import ImproperlyConfigured

from abmem.services.market.clearing import Bid, ClearingResult

from .config import ClearingConfig
from .registry import CLEARING_ALGORITHMS

__all__ = ["BaseClearingAlgorithm"]


class BaseClearingAlgorithm(abc.ABC):
    """Abstract clearing algorithm.

    Subclasses declare ``key`` (stored in ``Market.clearing_algorithm``,
    format ``^[A-Z][A-Z0-9_]{0,63}$``), ``label`` and ``description``, and
    implement :meth:`clear`. A concrete subclass that declares its own ``key``
    registers itself when its module is imported. ``config_class`` (optional,
    a frozen :class:`~.config.ClearingConfig` subclass) declares the
    parameters (PLAN 2.12). Instances are created per run with their
    validated config and must not keep state between periods that changes
    the result of a period.

    Attributes:
        config: Validated parameters, an instance of ``config_class``.
    """

    key: ClassVar[str] = ""
    label: ClassVar[str] = ""
    description: ClassVar[str] = ""
    config_class: ClassVar[type[ClearingConfig]] = ClearingConfig

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register a concrete subclass that declares its own ``key``.

        Raises:
            django.core.exceptions.ImproperlyConfigured: If the key is invalid
                or already used by another class, or ``config_class`` is not
                a ClearingConfig subclass.
        """
        super().__init_subclass__(**kwargs)
        if CLEARING_ALGORITHMS.register(cls):
            config = getattr(cls, "config_class", None)
            if not (isinstance(config, type) and issubclass(config, ClearingConfig)):
                CLEARING_ALGORITHMS.unregister(cls.key)
                raise ImproperlyConfigured(
                    f"clearing algorithm {cls.__module__}.{cls.__qualname__} "
                    "must declare config_class, a subclass of ClearingConfig"
                )

    def __init__(self, config: ClearingConfig | None = None) -> None:
        """Store the validated parameters.

        Args:
            config: Instance of ``config_class``, None for its defaults.

        Raises:
            TypeError: If ``config`` is not an instance of ``config_class``.
        """
        if config is None:
            config = self.config_class()
        if not isinstance(config, self.config_class):
            raise TypeError(
                f"{type(self).__name__} requires {self.config_class.__name__}, "
                f"got {type(config).__name__}"
            )
        self.config = config

    @abc.abstractmethod
    def clear(
        self, bids: Sequence[Bid], demand: Decimal, quantum: Decimal
    ) -> ClearingResult:
        """Clear one period.

        Args:
            bids: Bids with unique keys, non-negative finite prices and
                quantities that are multiples of ``quantum``.
            demand: Demand in MWh, a non-negative multiple of ``quantum``.
            quantum: Resolution of accepted quantities.

        Returns:
            A result that satisfies the contract of
            :func:`~abmem.services.market.clearing.validate_clearing_result`.
        """
