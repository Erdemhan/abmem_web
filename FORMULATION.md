# Formulation and Parameter Register (FORMULATION.md)

> **LOCKED DOCUMENT.** Equations, symbols, explanations and parameter values in
> this file may only be written or changed with explicit user approval. If code
> and this file disagree, the code is corrected, not this file.
>
> **Status (2026-10-07):** skeleton plus the entries approved by the user on
> 2026-10-07 and 2026-10-12 (EQ-02, EQ-02b, EQ-02c, experiment principles and EQ-09a in section 9, EQ-11a, EQ-11b). All
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
To formalise: currency USD, energy MWh, prices USD/MWh, one period = 1 hour, 24 periods = 1 day.

## 2. Data
To formalise: `abmem/data/marketData.xlsx` source and the 8808 valid DEMAND rows.

### EQ-02: Demand scaling constants (approved 2026-10-07)
- Formula: $\tilde d_t = (d_t - \mu_d)/\sigma_d$ with $\mu_d = \frac{1}{N}\sum_{i=1}^{N} d_i = 38\,997.47348206176$ MWh and $\sigma_d = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(d_i-\mu_d)^2} = 6\,494.932135899229$ MWh (population standard deviation), $N = 8808$ valid DEMAND values of `marketData.xlsx`.
- Source: decision OQ-2 (ARCHITECTURE.md section 11).
- Explanation: fixed scaling constants for the demand feature of the observation, computed once from all valid rows. They only rescale the input and are not a fitted model. `tests/agent/test_observation.py::test_scaling_constants_match_market_data` recomputes them from the file.
- Code: `abmem/services/agent/observation.py::DEMAND_SCALING_MEAN`, `DEMAND_SCALING_STD`

### EQ-02c: Synthetic demand generation (approved 2026-10-07, ramps and seasons approved 2026-10-07)
- Formula: a latent stationary Gaussian AR(1) process $z_1 = \varepsilon_1$, $z_t = \phi z_{t-1} + \sqrt{1-\phi^2}\,\varepsilon_t$ with $\varepsilon_t \overset{iid}{\sim} N(0,1)$ and $\phi \in [0,1)$, so that $z_t \sim N(0,1)$ for every $t$. The uniform $u_t = \Phi(z_t)$ is clipped to $[10^{-12}, 1-10^{-12}]$. The demand of period $t$ is the comonotone mix $d_t = \sum_i a_{t,i}\, Q_i(u_t)$ of the quantile functions $Q_i$ of the segments, rounded to 0.001 MWh and clipped to $[\sum_i a_{t,i} a_i, \sum_i a_{t,i} b_i]$.
  - Segments: each segment $s$ has a marginal on $[a_s, b_s]$, which is Uniform on $[a_s, b_s]$, Normal $N(m_s, s_s^2)$ truncated to $[a_s, b_s]$, or Log-normal $\exp(Y)$ with $Y \sim N(\mu_{\ln}, \sigma_{\ln}^2)$, $\sigma_{\ln}^2 = \ln(1 + s_s^2/m_s^2)$, $\mu_{\ln} = \ln m_s - \sigma_{\ln}^2/2$, truncated to $[a_s, b_s]$. For the truncated Normal $Q(u) = m_s + s_s\,\Phi^{-1}(\Phi(\alpha) + u(\Phi(\beta)-\Phi(\alpha)))$ with $\alpha = (a_s - m_s)/s_s$, $\beta = (b_s - m_s)/s_s$, and analogously on the log scale for the Log-normal.
  - Transition ramp: $r(x, k) = 0$ for $x \le -k/2$, $1$ for $x \ge k/2$, and $\tfrac12\left(1 + \sin(\pi x / k)\right)$ otherwise ($k = 0$ is a step).
  - Day weight of hour $h = (t-1) \bmod 24$, evaluated at the hour centre $h + \tfrac12$, for a shape with day hours $[h_s, h_e)$ and ramp widths $k_m$ (morning) and $k_e$ (evening): for a day hour $w_h = \min\left(r(h + \tfrac12 - h_s, k_m),\ r(h_e - h - \tfrac12, k_e)\right)$, for a night hour $w_h = \max\left(r(-c', k_m),\ r(-a', k_e)\right)$ with $c' = (h_s - h - \tfrac12) \bmod 24$ and $a' = (h + \tfrac12 - h_e) \bmod 24$. Validity requires $k_m + k_e \le 2\min(\text{day hours}, \text{night hours})$ and $0 \le k_m, k_e \le 12$.
  - Without seasons: $a_t = (1 - w_h,\ w_h)$ over (night, day).
  - With seasons (summer end period $S \ge 1$, transition $W$ days, $12W \le S$): the winter weight is $v_t = r(t - \tfrac12 - S,\ 24W)$ and $a_t = \left((1-v_t)(1-w^S_h),\ (1-v_t)\,w^S_h,\ v_t(1-w^W_h),\ v_t\,w^W_h\right)$ over (summer night, summer day, winter night, winter day), where $w^S$ and $w^W$ are the day weights of the summer and winter shapes.
- Source: Gaussian copula construction and the comonotone (Fréchet upper bound) coupling (Nelsen, 2006, An Introduction to Copulas, 2nd ed., Sec. 2.3, 2.5 and 3.1), stationary AR(1) (Box, Jenkins, Reinsel and Ljung, 2015, Time Series Analysis, 5th ed., Sec. 3.2), inverse transform sampling of a truncated distribution (Devroye, 1986, Non-Uniform Random Variate Generation, Sec. II.2).
- Explanation: $\phi$ controls the correlation of consecutive hours. Outside transitions the marginal of each hour is exactly the chosen truncated segment. In a transition the demand moves continuously between the segments because all quantiles are read at the same $u_t$. With $k_m = k_e = 0$ and no seasons the weights are exactly 0 or 1 and the series equals the plain two-segment generator bit for bit. The process uses its own seed (the demand seed) and one innovation per period in period order, so the series of $m$ periods is a prefix of the series of $n \ge m$ periods. $m_s$ and $s_s$ are the mean and standard deviation entered for the segment before truncation.
- Code: `abmem/services/market/demand_generator.py::generate_demand`, `day_weights`, `_ramp`, `Seasons.winter_weight`, `DemandProfile.weights`, `Segment.quantile`

### EQ-02b: Demand scaling of a generated profile (approved 2026-10-07, generalised 2026-10-07)
- Formula: for a run with a stored demand profile the demand feature is $\tilde d_t = (d_t - \mu_g)/\sigma_g$. With $X_t = \sum_i a_{t,i} Q_i(U)$, $U \sim U(0,1)$ and $a_t$ of EQ-02c, $E[X_t] = a_t^\top \mu$ and $E[X_t^2] = a_t^\top G\, a_t$, where $\mu_i$ is the mean of segment $i$ and $G_{ij} = \int_0^1 Q_i(u) Q_j(u)\,du$. The diagonal is $G_{ii} = v_i + \mu_i^2$. Off-diagonal entries are computed by composite Gauss-Legendre quadrature (64 panels of 16 nodes), only for segment pairs that are mixed in some period. $\mu_g$ and $\sigma_g^2$ are the mean and variance of the equal-weight mixture of the rows $a_t$: $\mu_g = \frac1T \sum_t a_t^\top \mu$ and $\sigma_g^2 = \frac1T \sum_t a_t^\top G a_t - \mu_g^2$, over the 24 hours of a day ($T = 24$) without seasons and over the periods $1..T$ of the run with seasons.
  - Segment moments: Uniform $\mu = (a+b)/2$, $v = (b-a)^2/12$. Truncated Normal with $Z = \Phi(\beta)-\Phi(\alpha)$: $\mu = m + s\,(\varphi(\alpha)-\varphi(\beta))/Z$, $v = s^2\left[1 + (\alpha\varphi(\alpha) - \beta\varphi(\beta))/Z - ((\varphi(\alpha)-\varphi(\beta))/Z)^2\right]$. Truncated Log-normal with $\alpha = (\ln a - \mu_{\ln})/\sigma_{\ln}$, $\beta = (\ln b - \mu_{\ln})/\sigma_{\ln}$: $E[X^k] = \exp(k\mu_{\ln} + k^2\sigma_{\ln}^2/2)\,[\Phi(\beta - k\sigma_{\ln}) - \Phi(\alpha - k\sigma_{\ln})]/[\Phi(\beta)-\Phi(\alpha)]$, $\mu = E[X]$, $v = E[X^2] - \mu^2$.
  - Without ramps and seasons this reduces to $\mu_g = w_d\mu_d + w_n\mu_n$ and $\sigma_g^2 = w_d(v_d + \mu_d^2) + w_n(v_n + \mu_n^2) - \mu_g^2$ with $w_d = (h_e - h_s)/24$, $w_n = 1 - w_d$. Runs without a profile keep $\mu_d$, $\sigma_d$ of EQ-02.
- Source: truncated Normal moments (Johnson, Kotz and Balakrishnan, 1994, Continuous Univariate Distributions, Vol. 1, 2nd ed., Sec. 13.10.1), Log-normal partial moments (same, Sec. 14.3), moments of a finite mixture (Frühwirth-Schnatter, 2006, Finite Mixture and Markov Switching Models, Sec. 1.2.4), Gauss-Legendre quadrature (Abramowitz and Stegun, 1964, Handbook of Mathematical Functions, 25.4.29).
- Explanation: the file constants of EQ-02 do not match a generated series with a different level, so the demand feature is centred and scaled with the moments of the marginal of a generated period. They follow from the profile (and, with seasons, from the run length) alone, do not depend on the seed, and are the same in training and evaluation. The hour-to-hour dependence ($\phi > 0$) does not change these marginal moments. `tests/market/test_demand_scaling.py` and `tests/market/test_demand_ramps_seasons.py` check them against numerical integration and long generated series.
- Code: `abmem/services/market/demand_generator.py::Segment.moments`, `_quantile_cross_moment`, `DemandProfile.scaling`, `abmem/services/simulation/context.py::demand_scaling_for`, `abmem/services/agent/observation.py::ObservationNormalizer.for_market`

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

### EQ-09a: Common learning start (approved 2026-10-12)
- Formula: let $t_0$ be the first period with a complete observation (all lags available) and let a transition $(o_{t-1}, a_{t-1}, r_t, o_t)$ be stored once per training period $t > t_0$. Every learning agent $i$ makes its first gradient update after exactly $L$ stored transitions, that is in period $t_0 + L$, with
  $L = L_{	ext{scenario}}$ if the scenario sets it, otherwise $L = \max_i n_i$,
  where $n_i$ is the requirement of agent $i$: $n_i = \ell_i$ (`learning_starts`) for DDPG, TD3 and SAC, $n_i = \max(\ell_i, K_i)$ for RDPG with sequence length $K_i$, and $n_i = \max(\ell_i, R_i)$ for PPO with rollout length $R_i$. A scenario value must satisfy $n_i \le L_{	ext{scenario}} \le B_i$ for every learning agent, $B_i$ being the replay buffer capacity (no upper limit for PPO). PPO discards the first $L - R_i$ transitions when $L > R_i$, so its first rollout of $R_i$ transitions completes at the $L$-th transition. Later updates follow each algorithm's own schedule (every period for replay-buffer algorithms, every $R_i$ transitions for PPO).
- Source: principle 2 of this section. Fixed initial data collection before learning follows the start-up phase of TD3 (Fujimoto, van Hoof and Meger, 2018, Addressing Function Approximation Error in Actor-Critic Methods, ICML, Sec. 6.1). Equal experimental conditions across compared algorithms follow Henderson et al. (2018, Deep Reinforcement Learning that Matters, AAAI, Sec. "Experimental Analysis").
- Explanation: all learning agents of a run start updating in the same period, and runs of the same scenario use the same $L$ when the scenario sets it, so a comparison of algorithms does not depend on a different amount of pre-learning experience. With default parameters $\ell_i = R_i = 64$ and $K_i = 24$, so $L = 64$ for every algorithm. The transitions PPO discards come from its initial policy, so the first rollout stays on-policy. The value used by a run is recorded in its parameter snapshot.
- Code: `abmem/services/simulation/context.py::common_learning_starts`, `abmem/services/algorithms/algorithm_config.py::AlgorithmConfig.first_update_transitions`, `learning_starts_limit`, `abmem/services/algorithms/ppo.py::PPOAlgorithm.observe`, `ready_to_learn`, `abmem/models/simulation.py::Simulation.learning_starts`

Still open (user decision required before the experiment): number of seeds, the value of the common learning start if it is not automatic, scenarios, run length and evaluation window, confidence-interval method.

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
