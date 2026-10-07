"""Helpers for tests of the scenario wizard page (PLAN 2.6)."""

from __future__ import annotations

import json
import re
from typing import Any


def wizard_config(html: str) -> dict[str, Any]:
    """Return the decoded ``wizard-config`` JSON script of a wizard page.

    Args:
        html: Rendered page.

    Returns:
        The configuration object the page script reads.
    """
    match = re.search(
        r'<script id="wizard-config" type="application/json">(.*?)</script>',
        html,
        flags=re.S,
    )
    assert match, "wizard-config script missing"
    return json.loads(match.group(1))
