"""Non-learning comparison policies (PLAN 1.10, OQ-6).

Three reference bidding rules that operate in the same normalised action
space as the learning algorithms (ADR-003):

* :class:`RandomPolicy` draws ``u ~ U[-1, 1]^action_dim`` from the policy's
  own generator.
* :class:`MarginalCostPolicy` bids the variable cost (fuelCost) of every plant.
* :class:`FixedMarkupPolicy` bids ``(fuelCost + staticCost) * (1 + markup)``
  per plant (OQ-6, default ``markup = 0.10``).

Cost-based prices are clipped to ``[price_low, price_high]`` and converted to
actions with the inverse of the affine price map of
``abmem.services.agent.action_space``,
``u = 2 * (p - low) / (high - low) - 1``. The map is implemented locally so
the algorithm layer does not import the agent services. The runtime maps
``u`` back to a price quantised to ``0.01`` with ``action_space.to_price``.

None of these policies has a buffer or parameters. ``observe`` and ``update``
are no-ops (inherited from :class:`BaseAgentAlgorithm`, ``update`` returns
None because ``is_learning`` is False), ``exploration_scale`` returns None,
and the runtime never applies the fail-stack rule to them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from abmem.models.enums import AlgorithmType

from .algorithm_config import FixedMarkupConfig, MarginalCostConfig, RandomConfig
from .base import ActionResult, BaseAgentAlgorithm, PolicySpec


def price_to_action(
    prices: Sequence[float], price_low: float, price_high: float
) -> np.ndarray:
    """Map prices to normalised actions with the inverse affine price map.

    ``u_i = 2 * (clip(p_i, low, high) - low) / (high - low) - 1``. This is the
    inverse of ``p = low + (u + 1) / 2 * (high - low)`` used by
    ``abmem.services.agent.action_space.to_price``.

    Args:
        prices: Prices per plant (TL/MWh), finite.
        price_low: Lower bid bound.
        price_high: Upper bid bound, strictly greater than ``price_low``.

    Returns:
        float32 array of shape ``(len(prices),)`` with values in ``[-1, 1]``.

    Raises:
        ValueError: If ``price_high <= price_low`` or a price is not finite.
    """
    low = float(price_low)
    high = float(price_high)
    if not (np.isfinite(low) and np.isfinite(high) and high > low):
        raise ValueError(
            f"price bounds must be finite with high > low, got [{low}, {high}]"
        )
    values = np.asarray(prices, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"prices must be finite, got {values}")
    clipped = np.clip(values, low, high)
    actions = 2.0 * (clipped - low) / (high - low) - 1.0
    return np.clip(actions, -1.0, 1.0).astype(np.float32)


class RandomPolicy(BaseAgentAlgorithm):
    """Uniform random bidding in the normalised action space.

    Every call draws ``u ~ U[-1, 1]^action_dim`` from ``self.rng``, whether or
    not ``explore`` is set, so the policy stays random in the evaluation
    window as well. Two instances built with generators of the same seed
    produce identical action sequences.
    """

    algorithm_type = AlgorithmType.RANDOM
    is_learning = False
    is_on_policy = False
    config_class = RandomConfig

    def select_action(self, obs: np.ndarray, explore: bool) -> ActionResult:
        """Return a uniform random action.

        Args:
            obs: Normalised observation (unused).
            explore: Ignored. The draw is random in both modes.

        Returns:
            ``ActionResult`` with ``u ~ U[-1, 1]^action_dim`` as float32.
        """
        u = self.rng.uniform(-1.0, 1.0, size=self.spec.action_dim)
        return ActionResult(u=np.clip(u, -1.0, 1.0).astype(np.float32))


class _CostBasedPolicy(BaseAgentAlgorithm):
    """Common part of the deterministic cost-based baselines.

    The action is a constant computed once at construction from the plant
    costs in the spec, so ``select_action`` ignores both the observation and
    ``explore``.
    """

    is_learning = False
    is_on_policy = False

    def __init__(
        self,
        spec: PolicySpec,
        config: MarginalCostConfig | FixedMarkupConfig,
        rng: np.random.Generator,
    ) -> None:
        """Validate the arguments and precompute the constant action.

        Args:
            spec: Problem description with per-plant costs and bid bounds.
            config: Instance of ``config_class``.
            rng: Random generator owned by the policy (not used for actions).
        """
        super().__init__(spec, config, rng)
        prices = self.target_prices()
        self._action = price_to_action(prices, spec.price_low, spec.price_high)

    def target_prices(self) -> np.ndarray:
        """Return the unclipped bid price of every plant (TL/MWh).

        Returns:
            float64 array of shape ``(action_dim,)``.
        """
        raise NotImplementedError

    def select_action(self, obs: np.ndarray, explore: bool) -> ActionResult:
        """Return the precomputed cost-based action.

        Args:
            obs: Normalised observation (unused).
            explore: Ignored (no exploration).

        Returns:
            ``ActionResult`` with a copy of the constant action.
        """
        return ActionResult(u=self._action.copy())


class MarginalCostPolicy(_CostBasedPolicy):
    """Bid the variable cost of every plant.

    ``price_i = plant_variable_costs[i]`` (fuelCost, TL/MWh), clipped to
    ``[price_low, price_high]`` and mapped to ``u``.
    """

    algorithm_type = AlgorithmType.MARGINAL_COST
    config_class = MarginalCostConfig

    def target_prices(self) -> np.ndarray:
        """Return the variable cost of every plant.

        Returns:
            float64 array ``plant_variable_costs`` of shape ``(action_dim,)``.
        """
        return np.asarray(self.spec.plant_variable_costs, dtype=np.float64)


class FixedMarkupPolicy(_CostBasedPolicy):
    """Bid the full per-MWh cost of every plant plus a fixed relative markup.

    ``price_i = (plant_variable_costs[i] + plant_static_costs[i]) *
    (1 + markup)`` (OQ-6), clipped to ``[price_low, price_high]`` and mapped
    to ``u``.
    """

    algorithm_type = AlgorithmType.FIXED_MARKUP
    config_class = FixedMarkupConfig

    def target_prices(self) -> np.ndarray:
        """Return the marked-up total cost of every plant.

        Returns:
            float64 array of shape ``(action_dim,)``.
        """
        variable = np.asarray(self.spec.plant_variable_costs, dtype=np.float64)
        static = np.asarray(self.spec.plant_static_costs, dtype=np.float64)
        markup = float(self.config.markup)  # type: ignore[attr-defined]
        return (variable + static) * (1.0 + markup)
