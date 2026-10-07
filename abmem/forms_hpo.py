"""Create form of the hyperparameter search UI (PLAN 2.4, spec hpo.ui).

The form checks what the user typed (types, ranges, bound order, known
parameter names, seed lists) and builds the input of
:func:`abmem.services.hpo.hpo_service.validate_study_input`, which remains the
authority for the remaining rules (dataclass validation ranges of the bounds,
both bound corners together).
"""

from __future__ import annotations

import dataclasses
import math
import typing
from collections.abc import Mapping
from typing import Any, Final

from django import forms

from abmem.models import Simulation
from abmem.services.algorithms import registry as algorithm_registry
from abmem.services.hpo.hpo_service import SEED_UPPER
from abmem.services.hpo.worker import FAILSTACK_SEARCH_KEYS

SCALES: Final[tuple[str, ...]] = ("linear", "log", "int")
INTEGER_PARAMS_EXTRA: Final[frozenset[str]] = frozenset({"failstack_limit"})


def searchable_algorithms() -> list[str]:
    """Return the algorithms that have at least one default search bound.

    Returns:
        Algorithm keys in the order of ``registry.available_algorithms``.
        Algorithms without search bounds (RANDOM, MARGINAL_COST) are left out.
    """
    return [
        info.key
        for info in algorithm_registry.available_algorithms()
        if algorithm_registry.get_config_class(info.key).default_search_bounds()
    ]


def supports_failstack(algorithm: str) -> bool:
    """Return True if the fail-stack rule applies to the algorithm.

    The runtime applies the rule only to learning algorithms
    (``SimulationContext`` skips it for non-learning algorithms).

    Args:
        algorithm: Algorithm key.

    Returns:
        Whether fail-stack settings have an effect.
    """
    return not algorithm_registry.is_baseline(algorithm)


def searchable_params(algorithm: str) -> tuple[set[str], set[str]]:
    """Return the scalar parameters of an algorithm's config class.

    Args:
        algorithm: Algorithm key.

    Returns:
        ``(all int or float field names, int field names)``.
    """
    config_cls = algorithm_registry.get_config_class(algorithm)
    hints = typing.get_type_hints(config_cls)
    fields = dataclasses.fields(config_cls)
    numeric = {f.name for f in fields if hints[f.name] in (int, float)}
    integers = {f.name for f in fields if hints[f.name] is int}
    return numeric, integers


def parse_seed_list(value: Any, label: str) -> list[int]:
    """Parse a comma-separated list of distinct seeds.

    Args:
        value: String such as ``"1, 2, 3"`` or a list of integers.
        label: Field label used in messages.

    Returns:
        The seeds in input order.

    Raises:
        forms.ValidationError: If the list is empty, contains a non-integer,
            a seed outside ``[0, 2**31)`` or duplicates.
    """
    if isinstance(value, str):
        items: list[Any] = [s.strip() for s in value.split(",")]
        if items == [""]:
            items = []
        if any(item == "" for item in items):
            raise forms.ValidationError(
                f"{label}: remove the empty entry between commas."
            )
    elif isinstance(value, (list, tuple)):
        items = list(value)
    elif value is None:
        items = []
    else:
        raise forms.ValidationError(
            f"{label} must be a comma-separated list of integers."
        )
    if not items:
        raise forms.ValidationError(f"{label}: enter at least one seed.")
    seeds: list[int] = []
    for item in items:
        if isinstance(item, bool):
            raise forms.ValidationError(f"{label}: {item!r} is not an integer.")
        try:
            seed = int(str(item).strip())
        except ValueError:
            raise forms.ValidationError(
                f"{label}: {item!r} is not an integer."
            ) from None
        if not 0 <= seed < SEED_UPPER:
            raise forms.ValidationError(
                f"{label}: seed {seed} is outside 0 .. {SEED_UPPER - 1}."
            )
        seeds.append(seed)
    duplicates = sorted({s for s in seeds if seeds.count(s) > 1})
    if duplicates:
        raise forms.ValidationError(f"{label}: duplicate seeds {duplicates}.")
    return seeds


def _parse_number(value: Any, what: str) -> float:
    """Parse a finite number.

    Raises:
        ValueError: With a user-facing message.
    """
    if isinstance(value, bool) or value is None or value == "":
        raise ValueError(f"{what} is required and must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number, got {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{what} must be finite")
    return number


def parse_bound_entry(name: str, entry: Any, integer: bool) -> dict[str, Any]:
    """Check one bound row ``{low, high, scale}``.

    Args:
        name: Parameter name.
        entry: The row as sent by the create page.
        integer: True if the parameter is an integer.

    Returns:
        ``{"low", "high", "scale"}`` with numbers (``int`` for ``scale='int'``).

    Raises:
        ValueError: With a user-facing message.
    """
    if not isinstance(entry, Mapping):
        raise ValueError("expected an object with min, max and scale")
    unknown = sorted(set(entry) - {"low", "high", "scale"})
    if unknown:
        raise ValueError(f"unknown keys {unknown}")
    scale = entry.get("scale") or ("int" if integer else "linear")
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {list(SCALES)}, got {scale!r}")
    if integer and scale != "int":
        raise ValueError("integer parameter, scale must be 'int'")
    if not integer and scale == "int":
        raise ValueError("not an integer parameter, scale 'int' is not allowed")
    low = _parse_number(entry.get("low"), "min")
    high = _parse_number(entry.get("high"), "max")
    if scale == "int" and (low != int(low) or high != int(high)):
        raise ValueError("min and max must be whole numbers")
    if not low < high:
        raise ValueError(f"min ({low:g}) must be smaller than max ({high:g})")
    if scale == "log" and low <= 0:
        raise ValueError("log scale needs min > 0")
    if scale == "int":
        return {"low": int(low), "high": int(high), "scale": scale}
    return {"low": low, "high": high, "scale": scale}


class HPOStudyForm(forms.Form):
    """Input of a new hyperparameter search.

    ``param_bounds`` is ``{name: {low, high, scale}}`` for the searched
    parameters only. ``failstack_limit`` and ``failstack_delta`` may appear in
    it when fail-stack is enabled, otherwise their fixed values are taken from
    the ``failstack_limit`` and ``failstack_delta`` fields.
    """

    name = forms.CharField(max_length=100, strip=True)
    scenario = forms.ModelChoiceField(
        queryset=Simulation.objects.filter(proxy=True).order_by("name", "id"),
        error_messages={
            "invalid_choice": "Select a scenario template (proxy simulation)."
        },
    )
    algorithm_type = forms.ChoiceField(choices=())
    failstack_enabled = forms.BooleanField(required=False)
    failstack_limit = forms.IntegerField(required=False, min_value=1)
    failstack_delta = forms.FloatField(required=False, min_value=0.0)
    param_bounds = forms.JSONField(
        error_messages={"required": "Select at least one parameter to search."}
    )
    colony_size = forms.IntegerField(min_value=2, max_value=1000)
    max_iter = forms.IntegerField(min_value=1, max_value=10000)
    limit = forms.IntegerField(min_value=1)
    search_seeds = forms.CharField(required=False, strip=True)
    evaluation_seeds = forms.CharField(required=False, strip=True)
    weight_agent = forms.FloatField()
    weight_market = forms.FloatField()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Create the form with the searchable algorithms as choices.

        Args:
            *args: Passed to ``forms.Form``.
            **kwargs: Passed to ``forms.Form``.
        """
        super().__init__(*args, **kwargs)
        self.fields["algorithm_type"].choices = [
            (key, algorithm_registry.label_for(key)) for key in searchable_algorithms()
        ]

    def clean_algorithm_type(self) -> str:
        """Return the selected algorithm key."""
        return str(self.cleaned_data["algorithm_type"])

    def clean_search_seeds(self) -> list[int]:
        """Parse the search seeds."""
        return parse_seed_list(self.cleaned_data.get("search_seeds"), "Search seeds")

    def clean_evaluation_seeds(self) -> list[int]:
        """Parse the evaluation seeds."""
        return parse_seed_list(
            self.cleaned_data.get("evaluation_seeds"), "Evaluation seeds"
        )

    def clean_weight_agent(self) -> float:
        """Reject NaN and infinite weights."""
        return self._finite("weight_agent")

    def clean_weight_market(self) -> float:
        """Reject NaN and infinite weights."""
        return self._finite("weight_market")

    def _finite(self, key: str) -> float:
        """Return a finite float field value.

        Raises:
            forms.ValidationError: If the value is NaN or infinite.
        """
        value = float(self.cleaned_data[key])
        if not math.isfinite(value):
            raise forms.ValidationError("Enter a finite number.")
        return value

    def clean(self) -> dict[str, Any]:
        """Check the bounds against the algorithm and the cross-field rules.

        Returns:
            The cleaned data.
        """
        data = super().clean()
        algorithm = data.get("algorithm_type")
        failstack = bool(data.get("failstack_enabled"))
        if isinstance(algorithm, str) and failstack:
            if not supports_failstack(algorithm):
                self.add_error(
                    "failstack_enabled",
                    f"Fail-stack applies only to learning algorithms, not to "
                    f"{algorithm}.",
                )
                failstack = False
        if isinstance(algorithm, str) and "param_bounds" in data:
            self._clean_bounds(data, algorithm, failstack)
        weights = (data.get("weight_agent"), data.get("weight_market"))
        if all(isinstance(w, float) for w in weights) and not any(weights):
            self.add_error("weight_market", "At least one weight must be non-zero.")
        search = data.get("search_seeds")
        evaluation = data.get("evaluation_seeds")
        if isinstance(search, list) and isinstance(evaluation, list):
            shared = sorted(set(search) & set(evaluation))
            if shared:
                self.add_error(
                    "evaluation_seeds",
                    f"Search and evaluation seeds must be disjoint, shared: {shared}.",
                )
        return data

    def _clean_bounds(
        self, data: dict[str, Any], algorithm: str, failstack: bool
    ) -> None:
        """Validate ``param_bounds`` rows and store the normalised mapping."""
        raw = data["param_bounds"]
        if not isinstance(raw, Mapping) or not raw:
            self.add_error("param_bounds", "Select at least one parameter to search.")
            return
        numeric, integers = searchable_params(algorithm)
        integers |= INTEGER_PARAMS_EXTRA
        allowed = numeric | (FAILSTACK_SEARCH_KEYS if failstack else set())
        bounds: dict[str, dict[str, Any]] = {}
        messages: list[str] = []
        for name, entry in raw.items():
            name = str(name)
            if name not in allowed:
                hint = (
                    " (enable fail-stack to search it)"
                    if name in FAILSTACK_SEARCH_KEYS
                    else ""
                )
                messages.append(f"{name}: unknown parameter for {algorithm}{hint}.")
                continue
            try:
                bounds[name] = parse_bound_entry(name, entry, name in integers)
            except ValueError as error:
                messages.append(f"{name}: {error}.")
                continue
            if name == "failstack_limit" and bounds[name]["low"] < 1:
                messages.append("failstack_limit: min must be >= 1.")
            if name == "failstack_delta" and bounds[name]["low"] < 0:
                messages.append("failstack_delta: min must be >= 0.")
        if messages:
            for message in messages:
                self.add_error("param_bounds", message)
            return
        data["param_bounds"] = bounds

    def to_service_input(self) -> dict[str, Any]:
        """Build the input of ``hpo_service.create_study``.

        Must be called after ``is_valid()`` returned True.

        Returns:
            Mapping in the shape documented by ``validate_study_input``.
        """
        data = self.cleaned_data
        fixed: dict[str, Any] = {"failstack_enabled": bool(data["failstack_enabled"])}
        if fixed["failstack_enabled"]:
            for key in ("failstack_limit", "failstack_delta"):
                if data.get(key) is not None and key not in data["param_bounds"]:
                    fixed[key] = data[key]
        return {
            "name": data["name"],
            "scenario": data["scenario"],
            "algorithm_type": data["algorithm_type"],
            "param_bounds": data["param_bounds"],
            "fixed_params": fixed,
            "colony_size": data["colony_size"],
            "max_iter": data["max_iter"],
            "limit": data["limit"],
            "search_seeds": data["search_seeds"],
            "evaluation_seeds": data["evaluation_seeds"],
            "objective": {
                "mode": "weighted",
                "weights": [data["weight_agent"], data["weight_market"]],
            },
        }


class HPOApplyForm(forms.Form):
    """Target of "apply best parameters" (OQ-9).

    ``target`` is ``default`` (``AlgorithmDefault`` of the study algorithm) or
    ``scenario`` (agents of the study scenario). With ``switch_algorithm``
    every scenario agent is set to the study algorithm, as in the evaluation
    runs (K7). Without it only the agents that already use the study
    algorithm are updated.
    """

    TARGET_DEFAULT: Final[str] = "default"
    TARGET_SCENARIO: Final[str] = "scenario"

    target = forms.ChoiceField(
        choices=(
            (TARGET_DEFAULT, "Algorithm default"),
            (TARGET_SCENARIO, "Scenario agents"),
        )
    )
    switch_algorithm = forms.BooleanField(required=False)
