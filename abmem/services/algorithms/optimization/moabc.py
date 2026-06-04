import numpy as np
import random
from typing import Callable, Dict, Tuple, List
from multiprocessing import Pool, cpu_count
import os
from datetime import datetime
import timeit
from abc_logger import ABCLogger

class MOABC:
    def __init__(
        self,
        fitness_func: Callable[[Dict[str, float]], Tuple[float, float, int]],
        param_bounds: Dict[str, Tuple[float, float]],
        colony_size: int = 10,
        limit: int = 5,
        max_iter: int = 30,
        log_path: str = "abc_log.json",
        seed: int = 42,
        score_weights: Tuple[float, float] = (1, 0), #w1 agent score, w2 market score
        algorithm_type: str = "DDPG"
    ):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.fitness_func = fitness_func
        self.param_bounds = param_bounds
        self.colony_size = colony_size
        self.limit = limit
        self.max_iter = max_iter
        self.log_path = log_path
        self.score_weights = score_weights
        self.algorithm_type = algorithm_type

        self.param_names = list(param_bounds.keys())
        self.lb = np.array([param_bounds[k][0] for k in self.param_names])
        self.ub = np.array([param_bounds[k][1] for k in self.param_names])

        self.food_sources = np.random.uniform(self.lb, self.ub, (colony_size, len(self.param_names)))
        self.trial = np.zeros(colony_size)
        self.sim_ids = [None] * colony_size

        self.logger = ABCLogger(
            algorithm_name=self.algorithm_type,
            log_dir=os.path.splitext(log_path)[0],
            meta_info={
                "colony_size": colony_size,
                "limit": limit,
                "max_iter": max_iter,
                "seed": seed,
                "score_weights": score_weights,
                "param_bounds": param_bounds
            }
        )

        self.logger.log_text(f"Starting evaluation of initial population at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        start_time = timeit.default_timer()
        self.fitness = self.evaluate_parallel(self.food_sources)
        self.logger.log_text(f"Initial population evaluation completed in {timeit.default_timer() - start_time:.2f} seconds")

        self.best_solution = None
        self.best_score = (-1, -1)
        for sol, score in zip(self.food_sources, self.fitness):
            if self.composite_score(score) > self.composite_score(self.best_score):
                self.best_score = score
                self.best_solution = sol.copy()

        initial_data = []
        for idx in range(self.colony_size):
            initial_data.append({
                "params": self.to_dict(self.food_sources[idx]),
                "agent_score": self.fitness[idx][0],
                "market_score": self.fitness[idx][1],
                "sim_id": self.fitness[idx][2]
            })
        self.logger.log_initial_population(initial_data)
        self.logger.log_text(f"📌 Initial Best: {self.to_dict(self.best_solution)} Score: {self.best_score}")

    def to_dict(self, position: np.ndarray) -> Dict[str, float]:
        return {
            name: (
                int(val) if 'ITERATION' in name.upper() or 'FAILSTACK' in name.upper() or "PPO_PPO_EPOCHS"in name.upper()
                else round(float(val), 5)
            )
            for name, val in zip(self.param_names, position)
        }

    def evaluate_parallel(self, positions: List[np.ndarray]) -> List[Tuple[float, float, int]]:
        param_list = [self.to_dict(pos) for pos in positions]

        for param in param_list:
            param["ALGORITHM"] = self.algorithm_type

        if not param_list:
            self.logger.log_text("⚠️ No individuals to evaluate in this batch (empty list).")
            return []
        else:
            self.logger.log_text(f"🔬 Evaluating batch of {len(param_list)} individuals at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            num_processes = min(cpu_count(), len(param_list))
            with Pool(processes=num_processes) as pool:
                results = pool.map(self.fitness_func, param_list)

        return results

    def dominates(self, a, b):
        return a[0] > b[0]

    def find_pareto_front(self):
        pareto = []
        for i, fi in enumerate(self.fitness):
            dominated = False
            for j, fj in enumerate(self.fitness):
                if i != j and self.dominates(fj[:2], fi[:2]):
                    dominated = True
                    break
            if not dominated:
                pareto.append((i, fi))
        return pareto

    def composite_score(self, score: Tuple[float, float]) -> float:
        w1, w2 = self.score_weights
        return w1 * score[0] + w2 * score[1]

    def select_with_probability(self):
        composite_scores = np.array([self.composite_score(f[:2]) for f in self.fitness])
        total = np.sum(composite_scores)
        probabilities = composite_scores / total if total > 0 else np.ones(len(composite_scores)) / len(composite_scores)
        return np.random.choice(range(self.colony_size), p=probabilities)

    def log_iteration(self, iteration):
        population_data = []
        for idx in range(self.colony_size):
            params = self.to_dict(self.food_sources[idx])
            score = self.fitness[idx]
            population_data.append({
                "params": params,
                "agent_score": score[0],
                "market_score": score[1],
                "sim_id": score[2]
            })
        self.logger.log_iteration(iteration, population_data)

    def optimize(self):
        for it in range(self.max_iter):
            self.logger.log_text(f"\n🔁 Iteration {it + 1}/{self.max_iter} started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            start_time = timeit.default_timer()
            np.random.seed(self.seed + it)
            random.seed(self.seed + it)

            for phase in ['employed', 'onlooker']:
                self.logger.log_text(f" -- Phase: {phase.title()} Bees")
                new_solutions = []
                indices = []
                for i in range(self.colony_size):
                    k = random.choice([x for x in range(self.colony_size) if x != i])
                    phi = np.random.uniform(-1, 1, len(self.param_names))
                    new_sol = self.food_sources[i] + phi * (self.food_sources[i] - self.food_sources[k])
                    new_sol = np.clip(new_sol, self.lb, self.ub)
                    new_solutions.append(new_sol)
                    indices.append(i)

                if new_solutions:
                    new_scores = self.evaluate_parallel(new_solutions)

                    for i, new_sol, new_score in zip(indices, new_solutions, new_scores):
                        if self.composite_score(new_score) > self.composite_score(self.best_score):
                            self.logger.log_text(f"✨ New global best found at iteration {it+1}: Agent={new_score[0]:.3f}, Market={new_score[1]:.3f}")
                            self.best_score = new_score
                            self.best_solution = new_sol.copy()

                        if self.dominates(new_score[:2], self.fitness[i][:2]):
                            self.food_sources[i] = new_sol
                            self.fitness[i] = new_score
                            self.trial[i] = 0
                        else:
                            self.trial[i] += 1
                else:
                    self.logger.log_text("⚠️ No new solutions generated in this phase.")

            scout_indices = [i for i in range(self.colony_size) if self.trial[i] >= self.limit]
            if scout_indices:
                scout_solutions = [np.random.uniform(self.lb, self.ub) for _ in scout_indices]
                scout_scores = self.evaluate_parallel(scout_solutions)

                for i, new_sol, new_score in zip(scout_indices, scout_solutions, scout_scores):
                    self.food_sources[i] = new_sol
                    self.fitness[i] = new_score
                    self.trial[i] = 0
                    if self.composite_score(new_score) > self.composite_score(self.best_score):
                        self.best_score = new_score
                        self.best_solution = new_sol.copy()
            else:
                self.logger.log_text("⚠️ No scouts triggered this round.")

            self.logger.log_text(f" -- Trial states: {[f'{i+1}- {int(self.trial[i])}/{self.limit}' for i in range(self.colony_size)]}")
            best_agent = max([s[0] for s in self.fitness])
            best_market = max([s[1] for s in self.fitness])
            self.logger.log_text(f"[Iter {it+1:02}/{self.max_iter}] Best Agent: {best_agent:.3f}, Market: {best_market:.3f}")
            self.log_iteration(it)
            self.logger.log_text(f"Iteration {it + 1} completed in {timeit.default_timer() - start_time:.2f} seconds")

        pareto_solutions = self.find_pareto_front()
        pareto_params = [self.to_dict(self.food_sources[i]) for i, _ in pareto_solutions]
        pareto_scores = [score[:2] for _, score in pareto_solutions]
        best_params = self.to_dict(self.best_solution)
        self.logger.log_text(f"\n✅ Global Best: {best_params} Score: {self.best_score}")
        return pareto_params, pareto_scores, best_params, self.best_score
