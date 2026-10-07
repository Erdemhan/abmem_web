from typing import Any, Sequence

from django import forms
from django.core.exceptions import ValidationError

from .services.agent.library_service import (
    check_name_available,
    normalize_name,
)
from .services.agent.resource_rules import validate_agent_resources
from .services.algorithms import registry as algorithm_registry
from .models.models import Resource, Portfolio, Plant, Period, Offer, Agent
from django.forms import inlineformset_factory


class ResourceForm(forms.ModelForm):
    class Meta:
        model = Resource
        fields = [
            "name",
            "energyType",
            "emission",
            "fuelCost",
            "omCost",
            "investmentCost",
            "co2Cost",
        ]


class AgentForm(forms.ModelForm):
    """Library agent form (PLAN 2.3) with resource rule (2.2) and algorithm settings (2.1).

    The form edits library agents only (``proxy=True``, ``market=NULL``). The
    name is checked with ``library_service.check_name_available``, the same
    case-insensitive comparison as the database constraint, which stays the
    final guard. Resource rule violations are reported as non-field errors because the
    plants are not fields of this form. Views expose them under the
    ``plants`` key. The algorithm and fail-stack fields are optional: an
    omitted value keeps the model default (or the stored value on edit).
    Enabling the fail-stack rule for a baseline algorithm is a field error.
    """

    OPTIONAL_FIELDS: tuple[str, ...] = (
        "algorithm_type",
        "failstack_enabled",
        "failstack_limit",
        "failstack_delta",
        "failstack_min_margin",
    )

    class Meta:
        model = Agent
        fields = [
            "budget",
            "type",
            "name",
            "algorithm_type",
            "failstack_enabled",
            "failstack_limit",
            "failstack_delta",
            "failstack_min_margin",
        ]

    def __init__(
        self,
        *args: Any,
        plant_resources: Sequence[Resource] | None = None,
        **kwargs: Any,
    ) -> None:
        """Create the form.

        Args:
            *args: Positional arguments of ``forms.ModelForm``.
            plant_resources: Resources of the agent's plants. When given,
                ``clean()`` validates them against the selected agent type.
            **kwargs: Keyword arguments of ``forms.ModelForm``.
        """
        super().__init__(*args, **kwargs)
        if self.instance.pk is None:
            # New library agent: satisfies the constraint library_agent_is_proxy.
            self.instance.proxy = True
            self.instance.market = None
        self.plant_resources = plant_resources
        for name in self.OPTIONAL_FIELDS:
            self.fields[name].required = False

    def clean_name(self) -> str:
        """Normalise the name and reject a name already used by another library agent.

        Returns:
            The name without surrounding whitespace.

        Raises:
            ValidationError: Code ``required`` for an empty name, code
                ``name_taken`` when a library agent with the same name
                (case-insensitive) exists.
        """
        name = normalize_name(self.cleaned_data.get("name") or "")
        if not name:
            raise ValidationError("Agent name is required.", code="required")
        if not check_name_available(name, exclude_id=self.instance.pk):
            raise ValidationError(
                f"A library agent named '{name}' already exists "
                "(names are compared case-insensitively).",
                code="name_taken",
            )
        return name

    def clean(self) -> dict[str, Any]:
        """Validate the model fields and, if available, the plant resources.

        Returns:
            The cleaned data.

        Raises:
            ValidationError: Non-field error with the rule messages when a
                plant resource is not allowed for the agent type or no plant
                is given.
        """
        cleaned_data = super().clean()
        algorithm_type = cleaned_data.get("algorithm_type")
        if algorithm_type in (None, ""):
            algorithm_type = (
                self.instance.algorithm_type
                if self.instance.pk is not None
                else Agent._meta.get_field("algorithm_type").get_default()
            )
            cleaned_data["algorithm_type"] = algorithm_type
        limit = cleaned_data.get("failstack_limit")
        if limit is not None and limit < 1:
            self.add_error("failstack_limit", "Fail-stack limit L must be >= 1.")
        delta = cleaned_data.get("failstack_delta")
        if delta is not None and delta < 0:
            self.add_error("failstack_delta", "Fail-stack delta must be >= 0 USD/MWh.")
        if not algorithm_registry.is_known(algorithm_type):
            known = ", ".join(
                info.key for info in algorithm_registry.available_algorithms()
            )
            self.add_error(
                "algorithm_type",
                f"Algorithm {algorithm_type!r} is not available. "
                f"Known algorithms: {known}.",
            )
        elif cleaned_data.get("failstack_enabled") and algorithm_registry.is_baseline(
            algorithm_type
        ):
            self.add_error(
                "failstack_enabled",
                "The fail-stack rule cannot be enabled for the baseline "
                f"algorithm {algorithm_type}.",
            )
        agent_type = cleaned_data.get("type")
        if self.plant_resources is not None and agent_type is not None:
            try:
                validate_agent_resources(agent_type, self.plant_resources)
            except ValidationError as exc:
                raise ValidationError(exc.messages, code="invalid_plants") from exc
        return cleaned_data


class PortfolioForm(forms.ModelForm):
    class Meta:
        model = Portfolio
        fields = []  # Herhangi bir alan alınmayacak


class PlantForm(forms.ModelForm):
    class Meta:
        model = Plant
        fields = [
            "resource",
            "capacity",
        ]  # Buradaki fields Plant modelinde bulunan alanları temsil etmeli


# PlantFormSet, Portfolio ile Plant arasındaki ilişkiyi yönetmek için kullanılır
PlantFormSet = inlineformset_factory(
    Portfolio, Plant, form=PlantForm, extra=1, can_delete=True
)


class PeriodForm(forms.ModelForm):
    class Meta:
        model = Period
        fields = [
            "market",
            "periodNumber",
            "demand",
            "metDemand",
            "marketVolume",
            "ptf",
        ]


class OfferForm(forms.ModelForm):
    class Meta:
        model = Offer
        fields = [
            "period",
            "agent",
            "resource",
            "amount",
            "offerPrice",
            "acceptance",
            "acceptancePrice",
            "acceptanceAmount",
        ]
