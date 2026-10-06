"""Creation of plants in a portfolio."""

from __future__ import annotations

from ...models.models import Plant, Portfolio, Resource


def createPlant(portfolio: Portfolio, resource: Resource, capacity: int) -> Plant:
    """Create and save a plant.

    Args:
        portfolio: Owning portfolio.
        resource: Resource of the plant.
        capacity: Capacity in MWh per period.

    Returns:
        The saved plant.
    """
    plant = Plant(portfolio=portfolio, resource=resource, capacity=capacity)
    plant.save()
    return plant
