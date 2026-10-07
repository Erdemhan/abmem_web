"""Verbatim pandas copy of the legacy ``compute_advanced_scores`` (reference for tests)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def reference_scores(df: pd.DataFrame) -> tuple[dict[str, float], float]:
    """Verbatim copy of the legacy ``compute_advanced_scores`` without prints."""
    price_cap = 200
    num_periods = df["period"].nunique()
    agents = df["agent"].unique()
    agent_scores = {}
    for agent in agents:
        adf = df[df["agent"] == agent]
        capacity = adf["amount"].iloc[0]
        initial_budget = adf.iloc[0]["budget"]
        final_budget = adf.iloc[-1]["budget"]
        raw_budget_change = final_budget - initial_budget
        max_possible_revenue = capacity * num_periods * price_cap
        norm_budget = (
            raw_budget_change / max_possible_revenue if max_possible_revenue > 0 else 0
        )
        norm_budget = max(-1.0, min(norm_budget, 1.0))
        accepted = adf[adf["accepted"]].copy()
        total_profit = raw_budget_change
        total_amount = accepted["amount"].sum()
        avg_profit_per_unit = total_profit / total_amount if total_amount > 0 else 0
        norm_profit = avg_profit_per_unit / price_cap if price_cap > 0 else 0
        norm_profit = max(-1.0, min(norm_profit, 1.0))
        rmse = np.sqrt(((adf["offer_price"] - adf["market_price"]) ** 2).mean())
        similarity = 1 - (rmse / price_cap)
        score = 0.4 * norm_budget + 0.4 * norm_profit + 0.2 * similarity
        score = max(0, min(score, 1))
        agent_scores[agent] = score
    mcp_avg = df["market_price"].mean()
    avg_offer_price = df["offer_price"].mean()
    mci = mcp_avg / price_cap
    mpi = mcp_avg / avg_offer_price if avg_offer_price > 0 else 0
    market_score = 0.6 * (1 - mci) + 0.4 * mpi
    return agent_scores, market_score
