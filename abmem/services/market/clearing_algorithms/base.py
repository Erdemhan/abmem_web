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

from abmem.services.market.clearing import Bid, ClearingResult

from .registry import CLEARING_ALGORITHMS

__all__ = ["BaseClearingAlgorithm"]


class BaseClearingAlgorithm(abc.ABC):
    """Abstract clearing algorithm.

    Subclasses declare ``key`` (stored in ``Market.clearing_algorithm``,
    format ``^[A-Z][A-Z0-9_]{0,63}$``), ``label`` and ``description``, and
    implement :meth:`clear`. A concrete subclass that declares its own ``key``
    registers itself when its module is imported. Instances are created per
    run with no arguments and must not keep state between periods that
    changes the result of a period.
    """

    key: ClassVar[str] = ""
    label: ClassVar[str] = ""
    description: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register a concrete subclass that declares its own ``key``.

        Raises:
            django.core.exceptions.ImproperlyConfigured: If the key is invalid
                or already used by another class.
        """
        super().__init_subclass__(**kwargs)
        CLEARING_ALGORITHMS.register(cls)

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
