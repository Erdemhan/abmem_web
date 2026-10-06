# Formulation and Parameter Register (FORMULATION.md)

> **LOCKED DOCUMENT.** Equations, symbols, explanations and parameter values in
> this file may only be written or changed with explicit user approval. If code
> and this file disagree, the code is corrected, not this file.
>
> **Status (2026-10-06): section skeleton only (PLAN 0.7).** No equation or value
> has been entered yet. Approved decisions that each section must formalise are
> listed under "To formalise" with their source (PLAN.md K1-K11,
> ARCHITECTURE.md section 11 OQ-1..OQ-19).

Entry format for equations:

```
### EQ-NN: <title>
- Formula:
- Source: <author, year, equation/page>
- Explanation:
- Code: <path>::<function>
```

---

## 1. Notation and units
To formalise: currency TL, energy MWh, prices TL/MWh, one period = 1 hour, 24 periods = 1 day.

## 2. Data
To formalise: `abmem/data/marketData.xlsx`, the 8808 valid DEMAND rows, demand scaling constants (mean and population standard deviation of all valid rows, OQ-2).

## 3. Market model
To formalise: bid format (one price-quantity bid per plant and period, quantity = capacity), bid bounds, merit order, PAYASBID and PAYASPTF acceptance prices, equal-share split of equal-price groups capped at bid quantity (OQ-18), supply shortfall, rejection of zero demand (OQ-12), met demand (B11).

## 4. Cost and budget equations
To formalise: static cost rate (omCost + investmentCost + co2Cost), variable cost rate (fuelCost), offer profit g, budget update.

## 5. Markov decision process
To formalise: agents, observation o_t with configurable lag set (default {1, 24}, optional 168, K2/OQ-1), action u_t in [-1, 1]^n and price mapping, reward r_t = Σ g_i / (Σ q_i · P_max) (K1), transition timing, termination and truncation, random actions before all lags exist (not stored, OQ-1).

## 6. Fail-stack rule
To formalise (K3): failure definition with the dimensionless profit margin and the zero-revenue case (OQ-13), consecutive-failure counter, intervention price interval, stored action, PPO masking, parameters L, δ, m_min.

## 7. Learning algorithms
To formalise: update equations of DDPG, TD3, SAC, PPO with GAE and RDPG as implemented, with citations.

## 8. Comparison policies
To formalise: RANDOM, MARGINAL_COST, FIXED_MARKUP with (fuelCost + staticCost) · (1 + m), default m = 0.10 (OQ-6).

## 9. Training and evaluation protocol
To formalise: evaluation window `evaluation_periods` (OQ-5), seeds, repetitions.

## 10. Evaluation metrics
To formalise: agent score and market score definitions (legacy definitions: `docs/legacy_rewards.md`, corrections B9).

## 11. Hyperparameter optimisation
To formalise: batch-synchronous parallel variant of ABC (Karaboga, 2005) (OQ-19), weighted single objective with per-study weights, Pareto front as information only (OQ-11), search and evaluation seeds.

## 12. Parameter register

| Symbol / parameter | Value | Unit / range | Source and explanation |
|---|---|---|---|
| | | | |
