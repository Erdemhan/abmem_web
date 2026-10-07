# Formulation and Parameter Register (FORMULATION.md)

> **LOCKED DOCUMENT.** Equations, symbols, explanations and parameter values in
> this file may only be written or changed with explicit user approval. If code
> and this file disagree, the code is corrected, not this file.
>
> **Status (2026-10-07):** skeleton plus the entries approved by the user on
> 2026-10-07 (EQ-02, experiment principles in section 9, EQ-11a, EQ-11b). All
> other sections are still to be formalised. Approved decisions that each section must formalise are
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
To formalise: `abmem/data/marketData.xlsx` source and the 8808 valid DEMAND rows.

### EQ-02: Demand scaling constants (approved 2026-10-07)
- Formula: $\tilde d_t = (d_t - \mu_d)/\sigma_d$ with $\mu_d = \frac{1}{N}\sum_{i=1}^{N} d_i = 38\,997.47348206176$ MWh and $\sigma_d = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(d_i-\mu_d)^2} = 6\,494.932135899229$ MWh (population standard deviation), $N = 8808$ valid DEMAND values of `marketData.xlsx`.
- Source: decision OQ-2 (ARCHITECTURE.md section 11).
- Explanation: fixed scaling constants for the demand feature of the observation, computed once from all valid rows. They only rescale the input and are not a fitted model. `tests/agent/test_observation.py::test_scaling_constants_match_market_data` recomputes them from the file.
- Code: `abmem/services/agent/observation.py::DEMAND_SCALING_MEAN`, `DEMAND_SCALING_STD`

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

### Principles of the comparative experiment (approved 2026-10-07)
1. Every learning algorithm is compared against the comparison policies RANDOM, MARGINAL_COST and FIXED_MARKUP on the same scenarios and seeds.
2. The period in which gradient updates start is equalised across algorithms (RDPG needs a full sequence window and PPO a full rollout, so a common start period is set for all), or any remaining difference is reported.
3. Results are reported over several seeds with confidence intervals.

Still open (user decision required before the experiment): number of seeds, the common start period, scenarios, run length and evaluation window, confidence-interval method.

## 10. Evaluation metrics
To formalise: agent score and market score definitions (legacy definitions: `docs/legacy_rewards.md`, corrections B9).

## 11. Hyperparameter optimisation
To formalise: weighted objective definition (agent score, market score, weights), search and evaluation seeds.

### EQ-11a: Batch-synchronous ABC variant (approved 2026-10-07)
- Formula: for food source $i$ and a random partner $k \ne i$ and dimension $j$, the candidate is $v_{ij} = x_{ij} + \phi_{ij}(x_{ij} - x_{kj})$ with $\phi_{ij} \sim U[-1, 1]$, clipped to the parameter bounds. Greedy selection keeps $v_i$ if its fitness is higher. Fitness of a maximised score $s$ with cost $c = -s$: $\mathrm{fit} = 1/(1+c)$ if $c \ge 0$, else $1+|c|$. Onlookers choose source $i$ with probability $p_i = \mathrm{fit}_i / \sum_n \mathrm{fit}_n$. The source with the largest trial counter is replaced by a uniform random source when its counter exceeds `limit`.
- Source: Karaboga (2005), Karaboga and Basturk (2007).
- Explanation: this project uses a batch-synchronous variant, not classic ABC. In each employed and onlooker phase all candidates are generated from the same snapshot of the food sources and evaluated as one parallel batch, and greedy selection is applied after the batch. In classic ABC each candidate is evaluated and accepted before the next one is generated. The variant allows parallel evaluation of the expensive simulations (OQ-19). The Pareto front of agent and market score is shown for information only and never used for selection (OQ-11).
- Code: `abmem/services/hpo/abc_optimizer.py::ArtificialBeeColony`, `fitness_from_score`, `onlooker_probabilities`

### EQ-11b: Score of failed evaluations (approved 2026-10-07)
- Formula: if the evaluation of a candidate fails on any of its search seeds, its score is $s = -\infty$, hence $\mathrm{fit} = 0$.
- Source: project rule (approved by the user).
- Explanation: a candidate with a failed simulation can never replace a food source or become the best candidate. Non-finite objectives are treated the same way. If the final evaluation of the best candidate fails, the study ends as FAILED.
- Code: `abmem/services/hpo/hpo_service.py`, `abmem/services/hpo/abc_optimizer.py::fitness_from_score`

## 12. Parameter register

| Symbol / parameter | Value | Unit / range | Source and explanation |
|---|---|---|---|
| $\mu_d$ (`DEMAND_SCALING_MEAN`) | 38997.47348206176 | MWh | EQ-02, mean of 8808 valid DEMAND values |
| $\sigma_d$ (`DEMAND_SCALING_STD`) | 6494.932135899229 | MWh | EQ-02, population std of the same values |
