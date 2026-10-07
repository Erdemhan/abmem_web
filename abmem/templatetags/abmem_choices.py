"""Template tags for the per-agent algorithm and fail-stack inputs (PLAN 2.1).

The templates render the algorithm select and the fail-stack inputs from
these tags, so which types are baselines and which defaults the inputs show
are defined once (``registry.BASELINE_TYPES`` and the ``Agent`` field
defaults) and never duplicated in JavaScript.
"""

from __future__ import annotations

from typing import Any

from django import template
from django.utils.html import format_html, format_html_join
from django.utils.safestring import SafeString, mark_safe

from abmem.models import Agent
from abmem.models.enums import AlgorithmType
from abmem.services.algorithms.registry import BASELINE_TYPES

register = template.Library()

FAILSTACK_FIELDS: tuple[str, ...] = (
    "failstack_limit",
    "failstack_delta",
    "failstack_min_margin",
)

# Delegated jQuery handlers shared by every page that renders algorithm_fields.
# A baseline algorithm disables and unchecks the fail-stack switch. The L,
# delta and m_min inputs are visible only while the switch is on.
_ALGORITHM_FIELDS_SCRIPT = """<script>
    // Per-agent algorithm and fail-stack inputs (PLAN 2.1), rendered by abmem_choices.algorithm_fields.
    function applyAlgorithmFields($container) {
        $container.find('.algorithm-fields').addBack('.algorithm-fields').each(function () {
            var $fields = $(this);
            var $selected = $fields.find('.agent-algorithm option:selected');
            var isBaseline = $selected.attr('data-baseline') === '1';
            var $switch = $fields.find('.agent-failstack-enabled');
            if (isBaseline) {
                $switch.prop('checked', false);
            }
            $switch.prop('disabled', isBaseline);
            $fields.find('.failstack-fields').toggle($switch.is(':checked'));
        });
    }

    // Reset the inputs to the model defaults stored in data-default.
    function resetAlgorithmFields($container) {
        $container.find('.algorithm-fields').addBack('.algorithm-fields').each(function () {
            var $fields = $(this);
            $fields.find('[data-default]').each(function () {
                var $input = $(this);
                if ($input.is(':checkbox')) {
                    $input.prop('checked', $input.attr('data-default') === '1');
                } else {
                    $input.val($input.attr('data-default'));
                }
            });
        });
        applyAlgorithmFields($container);
    }

    // Set stored values (edit of an existing agent), then apply the baseline rule.
    function setAlgorithmFields($container, values) {
        var $fields = $container.find('.algorithm-fields').addBack('.algorithm-fields').first();
        $fields.find('.agent-algorithm').val(String(values.algorithm_type));
        $fields.find('.agent-failstack-enabled').prop('checked', Boolean(values.failstack_enabled));
        $fields.find('.failstack-limit').val(values.failstack_limit);
        $fields.find('.failstack-delta').val(values.failstack_delta);
        $fields.find('.failstack-min-margin').val(values.failstack_min_margin);
        applyAlgorithmFields($container);
    }

    // Payload part of one agent: {algorithm_type, failstack: {enabled, limit, delta, min_margin}}.
    function readAlgorithmFields($container) {
        var $fields = $container.find('.algorithm-fields').first();
        return {
            algorithm_type: $fields.find('.agent-algorithm').val(),
            failstack: {
                enabled: $fields.find('.agent-failstack-enabled').is(':checked'),
                limit: $fields.find('.failstack-limit').val(),
                delta: $fields.find('.failstack-delta').val(),
                min_margin: $fields.find('.failstack-min-margin').val()
            }
        };
    }

    $(document).on('change', '.agent-algorithm, .agent-failstack-enabled', function () {
        applyAlgorithmFields($(this).closest('.algorithm-fields'));
    });
</script>"""


def _enum_value(value: Any) -> Any:
    """Return the integer value of an enum member, or ``value`` unchanged."""
    return getattr(value, "value", value)


@register.simple_tag
def algorithm_choices() -> list[tuple[int, str, bool]]:
    """Return ``(value, label, is_baseline)`` for every ``AlgorithmType``.

    Returns:
        One tuple per member in enum order. ``is_baseline`` is True for the
        members of ``registry.BASELINE_TYPES``.
    """
    return [
        (member.value, member.name, member in BASELINE_TYPES)
        for member in AlgorithmType
    ]


@register.simple_tag
def default_algorithm() -> int:
    """Return the value of the ``Agent.algorithm_type`` field default.

    Returns:
        The integer value of the default algorithm (DDPG).
    """
    return int(_enum_value(Agent._meta.get_field("algorithm_type").get_default()))


@register.simple_tag
def failstack_defaults() -> dict[str, Any]:
    """Return the model defaults of the fail-stack fields.

    Returns:
        Mapping ``{"failstack_limit": 6, "failstack_delta": "15",
        "failstack_min_margin": "0"}``. Decimal defaults are returned as
        strings so they render without a binary float expansion.
    """
    defaults: dict[str, Any] = {}
    for name in FAILSTACK_FIELDS:
        value = Agent._meta.get_field(name).get_default()
        defaults[name] = value if isinstance(value, int) else str(value)
    return defaults


@register.filter
def is_baseline_algorithm(value: Any) -> bool:
    """Return True when ``value`` names a baseline algorithm type.

    Args:
        value: An ``AlgorithmType`` member or its integer value.

    Returns:
        True for RANDOM, MARGINAL_COST and FIXED_MARKUP, else False.
    """
    return AlgorithmType.get(_enum_value(value)) in BASELINE_TYPES


@register.filter
def failstack_active(agent: Agent) -> bool:
    """Return True when the fail-stack rule applies to ``agent`` in a run.

    Args:
        agent: Agent instance.

    Returns:
        ``failstack_enabled`` and the algorithm is not a baseline.
    """
    return bool(agent.failstack_enabled) and not is_baseline_algorithm(
        agent.algorithm_type
    )


@register.simple_tag
def algorithm_fields(
    agent: Agent | None = None, form_names: bool = False
) -> SafeString:
    """Render the algorithm select, fail-stack switch and L, delta, m_min inputs.

    The markup contains no backtick and no ``${``, so it can be embedded in a
    JavaScript template literal.

    Args:
        agent: Agent whose stored values prefill the inputs. None renders the
            model defaults.
        form_names: Add ``name`` and ``id`` attributes matching the
            ``AgentForm`` fields (form POST), otherwise only classes are set
            (JSON payload pages with one row per agent).

    Returns:
        The HTML of one ``div.algorithm-fields``.
    """
    defaults = failstack_defaults()
    default_type = default_algorithm()
    if agent is None:
        selected = default_type
        enabled = False
        values = defaults
    else:
        selected = _enum_value(agent.algorithm_type)
        enabled = failstack_active(agent)
        values = {name: getattr(agent, name) for name in FAILSTACK_FIELDS}

    def attrs(name: str) -> SafeString:
        if not form_names:
            return mark_safe("")
        return format_html(' name="{}" id="{}"', name, name)

    options = format_html_join(
        "",
        '<option value="{}" data-baseline="{}"{}>{}</option>',
        (
            (
                value,
                "1" if is_baseline else "0",
                mark_safe(" selected") if value == selected else "",
                label,
            )
            for value, label, is_baseline in algorithm_choices()
        ),
    )
    is_baseline = AlgorithmType.get(selected) in BASELINE_TYPES
    return format_html(
        '<div class="algorithm-fields">'
        '<select class="form-control agent-algorithm"{} data-default="{}">{}</select>'
        '<div class="form-check form-switch mt-2">'
        '<label class="form-check-label">'
        '<input class="form-check-input agent-failstack-enabled" type="checkbox"'
        ' value="on"{} data-default="0"{}{}> Fail-stack</label>'
        "</div>"
        '<div class="failstack-fields"{}>'
        '<label class="mb-0">L<input type="number" min="1" step="1"'
        ' class="form-control failstack-limit"{} value="{}" data-default="{}"></label>'
        '<label class="mb-0">&delta; (TL/MWh)<input type="number" min="0" step="0.01"'
        ' class="form-control failstack-delta"{} value="{}" data-default="{}"></label>'
        '<label class="mb-0">m_min<input type="number" step="0.0001"'
        ' class="form-control failstack-min-margin"{} value="{}" data-default="{}"></label>'
        "</div>"
        "</div>",
        attrs("algorithm_type"),
        default_type,
        options,
        attrs("failstack_enabled"),
        mark_safe(" checked") if enabled else "",
        mark_safe(" disabled") if is_baseline else "",
        "" if enabled else mark_safe(' style="display: none;"'),
        attrs("failstack_limit"),
        values["failstack_limit"],
        defaults["failstack_limit"],
        attrs("failstack_delta"),
        values["failstack_delta"],
        defaults["failstack_delta"],
        attrs("failstack_min_margin"),
        values["failstack_min_margin"],
        defaults["failstack_min_margin"],
    )


@register.simple_tag
def algorithm_fields_script() -> SafeString:
    """Render the shared jQuery handlers of ``algorithm_fields``.

    Returns:
        A ``<script>`` element. jQuery must be loaded before it.
    """
    return mark_safe(_ALGORITHM_FIELDS_SCRIPT)
