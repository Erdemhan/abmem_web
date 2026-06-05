"""
Scenario runner — run the simulation across several scenarios sequentially and
collect the results into one comparison table.

A "scenario" here is one combination of:
  - algorithm  : RDPG | DDPG | TD3 | PPO | SAC
  - proxy_id   : id of a template (proxy) Simulation in the DB to clone
  - seed       : RNG seed for the run (lets you do multi-seed robustness sweeps)
  - params     : optional per-scenario hyperparameter overrides

It reuses the existing helpers from hyperparam_optimization:
  - create_simulation_from_proxy(proxy_id) -> fresh Simulation cloned from a template
  - evaluate_result(sim_id)                -> (agent_score, market_score)

Run it from the PROJECT ROOT (so 'abmem/data/' and 'abm_ddpg/sim_data' resolve,
while the script's own dir is still added to sys.path for the bare imports):
    python abmem/services/algorithms/optimization/scenario_runner.py

Output: a printed table plus scenario_results/scenarios_<timestamp>.{csv,json}.

IMPORTANT — keep this SEQUENTIAL.
AgentAlgorithm.__init__ reassigns module-level globals (ITERATION, DISCOUNT, TAU,
LEARNING_RATE_*, NOISE_*) from `params`. That shared mutable state is only safe
because one scenario fully finishes before the next starts. Do NOT parallelize
these runs in the same process without first removing those globals.
"""

import sys
import os
import json
import csv
import timeit
from datetime import datetime

# Match hyperparam_optimization.py: make the project root importable and configure Django.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web_project.settings")
import django
django.setup()

from abmem.services.simulation import simulation_service as SimulationService
from abmem.services.market import market_service as MarketService
from hyperparam_optimization import create_simulation_from_proxy, evaluate_result


# ---------------------------------------------------------------------------
# Defaults. AgentAlgorithm.__init__ indexes these keys directly (KeyError if
# missing), so every required key must be present here. Scenario-level overrides
# are merged on top per run.
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "ALGORITHM": "DDPG",
    # common (required by AgentAlgorithm.__init__)
    "ITERATION": 5,
    "LEARNING_RATE_ACTOR": 1e-5,
    "LEARNING_RATE_CRITIC": 1e-3,
    "TAU": 0.005,
    "DISCOUNT": 0.99,
    "NOISE_STD": 0.6,
    "NOISE_DECAY": 0.995,
    "FAILSTACK": 6,
    # PPO-specific (read via params.get with defaults; kept here for clarity)
    "PPO_GAMMA": 0.99,
    "PPO_GAE_LAMBDA": 0.95,
    "PPO_CLIP_PARAM": 0.2,
    "PPO_PPO_EPOCHS": 10,
    # SAC-specific
    "LEARNING_RATE_ALPHA": 1e-4,
    "INITIAL_ALPHA": 0.2,
}


# ---------------------------------------------------------------------------
# Define your scenarios here. `params` is optional and overrides DEFAULT_PARAMS.
# Examples below: compare the five algorithms on the same template + seed, and a
# small multi-seed robustness sweep for DDPG.
# ---------------------------------------------------------------------------
# Template Simulation to clone (was hardcoded as 529 in simulate_multi).
# Override with the TEMPLATE_PROXY_ID env var (the synthetic-demo seed prints the id to use).
TEMPLATE_PROXY_ID = int(os.environ.get("TEMPLATE_PROXY_ID", "529"))

SCENARIOS = [
    {"name": "ddpg_base", "algorithm": "DDPG", "proxy_id": TEMPLATE_PROXY_ID, "seed": 17081999},
    {"name": "rdpg_base", "algorithm": "RDPG", "proxy_id": TEMPLATE_PROXY_ID, "seed": 17081999},
    {"name": "td3_base",  "algorithm": "TD3",  "proxy_id": TEMPLATE_PROXY_ID, "seed": 17081999},
    {"name": "ppo_base",  "algorithm": "PPO",  "proxy_id": TEMPLATE_PROXY_ID, "seed": 17081999},
    {"name": "sac_base",  "algorithm": "SAC",  "proxy_id": TEMPLATE_PROXY_ID, "seed": 17081999},

    # multi-seed robustness example for one algorithm:
    # {"name": "ddpg_s2", "algorithm": "DDPG", "proxy_id": TEMPLATE_PROXY_ID, "seed": 42},
    # {"name": "ddpg_s3", "algorithm": "DDPG", "proxy_id": TEMPLATE_PROXY_ID, "seed": 7},

    # per-scenario hyperparameter override example:
    # {"name": "ddpg_fastlr", "algorithm": "DDPG", "proxy_id": TEMPLATE_PROXY_ID,
    #  "seed": 17081999, "params": {"LEARNING_RATE_ACTOR": 1e-4, "ITERATION": 10}},
]


def run_scenario(scn: dict) -> dict:
    """Run a single scenario and return a flat result record."""
    params = dict(DEFAULT_PARAMS)
    params["ALGORITHM"] = scn["algorithm"]
    params.update(scn.get("params", {}))  # per-scenario overrides win

    # market_service keeps a module-level `agents` list that init() appends to but
    # never clears. Running multiple scenarios in one process would otherwise let a
    # later scenario inherit earlier scenarios' agents. Reset it before each run.
    MarketService.agents = []

    sim = create_simulation_from_proxy(scn["proxy_id"])
    if sim is None:
        # create_simulation_from_proxy only clones when the source is a proxy or
        # not in CREATED state; otherwise it returns None.
        raise ValueError(
            f"Could not clone template id={scn['proxy_id']}. "
            f"Make sure it exists and is a proxy (or not in CREATED state)."
        )

    start = timeit.default_timer()
    SimulationService.run(sim, hyperparams=params, seed=scn.get("seed"))
    elapsed = timeit.default_timer() - start

    agent_score, market_score = evaluate_result(sim.id)

    return {
        "name": scn["name"],
        "algorithm": scn["algorithm"],
        "proxy_id": scn["proxy_id"],
        "seed": scn.get("seed"),
        "sim_id": sim.id,
        "agent_score": round(float(agent_score), 4),
        "market_score": round(float(market_score), 4),
        "elapsed_sec": round(elapsed, 2),
    }


def main():
    out_dir = "scenario_results"
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = []
    for i, scn in enumerate(SCENARIOS, 1):
        print(f"[{i}/{len(SCENARIOS)}] running '{scn['name']}' ({scn['algorithm']}, seed={scn.get('seed')}) ...")
        try:
            rec = run_scenario(scn)
            results.append(rec)
            print(f"    -> sim {rec['sim_id']}: agent={rec['agent_score']}, "
                  f"market={rec['market_score']}, {rec['elapsed_sec']}s")
        except Exception as e:  # one bad scenario shouldn't kill the whole sweep
            print(f"    !! scenario '{scn['name']}' failed: {e}")
            results.append({"name": scn["name"], "algorithm": scn["algorithm"],
                            "proxy_id": scn["proxy_id"], "seed": scn.get("seed"),
                            "sim_id": None, "agent_score": None,
                            "market_score": None, "elapsed_sec": None, "error": str(e)})

    # ---- write outputs ----
    json_path = os.path.join(out_dir, f"scenarios_{stamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(out_dir, f"scenarios_{stamp}.csv")
    fields = ["name", "algorithm", "proxy_id", "seed", "sim_id",
              "agent_score", "market_score", "elapsed_sec"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)

    # ---- print comparison table ----
    print("\n=== Scenario comparison ===")
    print(f"{'name':<14}{'algo':<7}{'seed':<12}{'sim':<7}{'agent':<9}{'market':<9}{'sec':<8}")
    for r in results:
        print(f"{str(r['name']):<14}{str(r['algorithm']):<7}{str(r['seed']):<12}"
              f"{str(r['sim_id']):<7}{str(r['agent_score']):<9}"
              f"{str(r['market_score']):<9}{str(r['elapsed_sec']):<8}")
    print(f"\nSaved: {csv_path}\n       {json_path}")


if __name__ == "__main__":
    main()
