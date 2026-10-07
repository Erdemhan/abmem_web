# Adding agent algorithms and market clearing algorithms

ABMEM finds both kinds of algorithm from the code (PLAN 2.10). A class that
subclasses the right base class, declares a `key` and lives in the right
package becomes selectable in the scenario wizard, the agent library, the HPO
page and the scenario YAML. No enum, table or template has to be edited.

## Rules shared by both kinds

- The key is stored in the database. It must match `^[A-Z][A-Z0-9_]{0,63}$`
  (for example `MY_ALGO`) and must not be used by another class. A clash stops
  the import with an `ImproperlyConfigured` error that names both classes.
- Only concrete classes that declare `key` themselves are registered. An
  abstract helper class, or a subclass that only inherits a key, is not
  listed.
- `label` is the name shown in the UI. `description` is one sentence shown as
  help text.
- The registry imports every module of the package the first time it is
  queried. Module names starting with `_` are skipped. A module that fails to
  import is reported with its traceback, so check the error page or the run
  error when a new algorithm does not appear.
- Do not rename or delete a key that is already stored. Agents and scenarios
  that still use it are shown as "unavailable", cannot be saved with it, and a
  run with it stops at validation.
- Research rule: an algorithm used for reported results needs its own entry
  in `FORMULATION.md` (equations, parameters and sources), approved by the
  user before the results are used. Agents and tools never edit
  `FORMULATION.md` without that approval.

## Agent algorithm

Location: a new module in `abmem/services/algorithms/`, for example
`abmem/services/algorithms/my_algo.py`. The `optimization/` subpackage is not
scanned.

A subclass of `BaseAgentAlgorithm` declares:

| Class variable | Meaning |
|---|---|
| `key`, `label`, `description` | identification (see above) |
| `is_learning` | False for a fixed bidding rule (a baseline). Baselines never use the fail-stack rule and are not saved as checkpoints. |
| `is_on_policy` | True for rollout-based learners such as PPO |
| `config_class` | a frozen dataclass derived from `AlgorithmConfig` with the hyperparameters |

`select_action(obs, explore)` returns an `ActionResult` with one value in
`[-1, 1]` per plant. Learning algorithms also implement `_update()` (see
`DDPG.py` for a complete example and `tests/algorithms/contract.py` for the
contract every algorithm must pass).

The HPO page lists an algorithm only when `config_class.default_search_bounds()`
returns at least one parameter. It maps a field name to
`(low, high, scale)` with `scale` one of `"linear"`, `"log"` or `"int"`.

Minimal example, a fixed rule that bids the middle of the bid bounds:

```python
from dataclasses import dataclass

import numpy as np

from .algorithm_config import AlgorithmConfig, SearchBounds
from .base import ActionResult, BaseAgentAlgorithm


@dataclass(frozen=True)
class MidPriceConfig(AlgorithmConfig):
    """No parameters besides the common ones."""

    @classmethod
    def default_search_bounds(cls) -> SearchBounds:
        """Nothing to search, so the algorithm is not offered on the HPO page."""
        return {}


class MidPricePolicy(BaseAgentAlgorithm):
    """Bids the middle of the bid bounds for every plant."""

    key = "MID_PRICE"
    label = "Mid price"
    description = "Bids the middle of the bid bounds (comparison baseline)."
    is_learning = False
    is_on_policy = False
    config_class = MidPriceConfig

    def select_action(self, obs: np.ndarray, explore: bool) -> ActionResult:
        """Return action 0, which maps to the middle of the bid bounds."""
        return ActionResult(u=np.zeros(self.spec.action_dim, dtype=np.float32))
```

Add a unit test that builds the policy with
`registry.build_policy("MID_PRICE", spec, None, rng)` and runs
`tests.algorithms.contract.assert_policy_contract` on it.

## Market clearing algorithm

Location: a new module in `abmem/services/market/clearing_algorithms/`, for
example `abmem/services/market/clearing_algorithms/my_clearing.py`.

A subclass of `BaseClearingAlgorithm` declares `key`, `label` and
`description` and implements

```python
def clear(self, bids: Sequence[Bid], demand: Decimal, quantum: Decimal) -> ClearingResult
```

It decides the accepted quantity of every bid and the clearing price (PTF).
It does not decide what accepted bids are paid: the settlement rule of the
scenario (pay-as-bid or pay-as-PTF, `Market.strategy`) is applied afterwards
for every clearing algorithm. The engine creates one instance per run, so the
class must not keep state between periods that changes a period's result.

The engine checks every result before it writes anything
(`abmem/services/market/clearing.py::validate_clearing_result`). A broken rule
stops the run with a `ClearingContractError` that names the key and the rule:

- `accepted` has an entry for exactly the keys of `bids`.
- Every accepted quantity is a finite `Decimal`, `0 <= accepted <= bid quantity`,
  and a multiple of `quantum`.
- `met_demand` equals the sum of the accepted quantities and is at most
  `min(demand, offered_volume)`.
- `offered_volume` equals the sum of all bid quantities.
- `ptf` is None exactly when `met_demand` is 0. Otherwise it is a finite
  `Decimal` within the bid bounds of the market.
- `marginal_group` is a tuple of bid keys without repeats.

Minimal example, pay-as-clear with the price of the last accepted bid and no
tie splitting:

```python
from collections.abc import Sequence
from decimal import Decimal

from abmem.services.market.clearing import Bid, ClearingResult

from .base import BaseClearingAlgorithm


class FirstComeClearing(BaseClearingAlgorithm):
    """Accepts bids in ascending price order, ties by key, without splitting ties."""

    key = "FIRST_COME"
    label = "Merit order without tie splitting"
    description = "Accepts bids in ascending price order, equal prices by bid order."

    def clear(
        self, bids: Sequence[Bid], demand: Decimal, quantum: Decimal
    ) -> ClearingResult:
        """Fill the demand from the cheapest bid upwards."""
        accepted = {bid.key: Decimal(0) for bid in bids}
        remaining = demand
        ptf = None
        for bid in sorted(bids, key=lambda b: (b.price, b.key)):
            if remaining <= 0 or bid.quantity <= 0:
                continue
            take = min(bid.quantity, remaining)
            accepted[bid.key] = take
            remaining -= take
            ptf = bid.price
        return ClearingResult(
            accepted=accepted,
            ptf=ptf,
            met_demand=sum(accepted.values(), Decimal(0)),
            offered_volume=sum((bid.quantity for bid in bids), Decimal(0)),
            marginal_group=tuple(b.key for b in bids if ptf is not None and b.price == ptf),
        )
```

Add a unit test that clears a few bid sets and passes every result through
`validate_clearing_result`, and a run test like
`tests/runtime/test_clearing_algorithm_run.py`.

## Where the key appears

- Scenario wizard: the agent algorithm selects and, in the market step, the
  "Clearing algorithm" select next to the pricing strategy.
- Scenario YAML: `agents[].algorithm` and the optional
  `market.clearing_algorithm` (`MERIT_ORDER` when missing, always written on
  export).
- HPO: the algorithm select (algorithms with search bounds only).
- Database: `Agent.algorithm_type`, `HPOStudy.algorithm_type`,
  `AlgorithmDefault.algorithm_type`, `PolicyCheckpoint.algorithm_type` and
  `Market.clearing_algorithm`.
