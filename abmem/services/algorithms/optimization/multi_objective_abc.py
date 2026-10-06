import json
import numpy as np
import random
from typing import Callable, Dict, Tuple, List
from multiprocessing import Pool, cpu_count
import os
from datetime import datetime
import timeit

class MultiObjectiveABC:
    def __init__(
        self,
        fitness_func: Callable[[Dict[str, float]], Tuple[float, float]],
        param_bounds: Dict[str, Tuple[float, float]],
        colony_size: int = 10,
        limit: int = 5,
        max_iter: int = 30,
        log_path: str = "abc_log.json",
        seed: int = 42
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

        self.param_names = list(param_bounds.keys())
        self.lb = np.array([param_bounds[k][0] for k in self.param_names])
        self.ub = np.array([param_bounds[k][1] for k in self.param_names])

        self.food_sources = np.random.uniform(self.lb, self.ub, (colony_size, len(self.param_names)))
        self.trial = np.zeros(colony_size)
        self.sim_ids = [None] * colony_size


        start_Time = timeit.default_timer()
        print(f"Starting evaluation of initial population at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.fitness = self.evaluate_parallel(self.food_sources)
        print(f"Initial population evaluation completed in {timeit.default_timer() - start_Time:.2f} seconds")
        print(f"Initial trials: {[f'{i+1}- {int(self.trial[i])}/{self.limit}' for i in range(self.colony_size)]}")

        self.log_data = []

        # Global en iyiyi başlangıçta ata
        self.best_solution = None
        self.best_score = (-np.inf, -np.inf)
        for sol, score in zip(self.food_sources, self.fitness):
            if score[0] > self.best_score[0]:
                self.best_score = score
                self.best_solution = sol.copy()

        print(f"Initial Best → Agent Score: {self.best_score[0]:.3f}, Market Score: {self.best_score[1]:.3f}")
        print(f"Best Params: {self.to_dict(self.best_solution)}")

    def to_dict(self, position: np.ndarray) -> Dict[str, float]:
        return {
            name: (
                int(val) if 'ITERATION' in name.upper() or 'FAILSTACK' in name.upper()
                else round(float(val), 5)
            )
            for name, val in zip(self.param_names, position)
        }

    def evaluate_parallel(self, positions: List[np.ndarray]) -> List[Tuple[float, float, int]]:
        param_list = [self.to_dict(pos) for pos in positions]
        print(f"> Running simulation batch of {len(param_list)} individuals at {datetime.now().strftime('%H:%M:%S')}")
        start = timeit.default_timer()
        with Pool(processes=min(cpu_count(), len(param_list))) as pool:
            results = pool.map(self.fitness_func, param_list)
        print(f"> Batch completed in {timeit.default_timer() - start:.2f} seconds")
        return results

    def dominates(self, a, b):
        return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))

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

    def log_iteration(self, iteration):
        if not hasattr(self, "_log_file_path"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.splitext(self.log_path)[0]
            os.makedirs(log_dir, exist_ok=True)
            self._log_file_path = os.path.join(log_dir, f"log_{timestamp}.json")

        entry = {"iteration": iteration, "population": []}
        for idx in range(self.colony_size):
            params = self.to_dict(self.food_sources[idx])
            score = self.fitness[idx]
            entry["population"].append({
                "params": params,
                "agent_score": score[0],
                "market_score": score[1],
                "sim_id": score[2]
            })

        self.log_data.append(entry)
        with open(self._log_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.log_data, f, indent=2, ensure_ascii=False)

    def optimize(self):
        for it in range(self.max_iter):
            print(f"\n🔁 Iteration {it + 1}/{self.max_iter} started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            start_time = timeit.default_timer()
            np.random.seed(self.seed + it)
            random.seed(self.seed + it)

            for phase in ['employed', 'onlooker']:
                print(f" -- Phase: {phase.title()} Bees")
                new_solutions = []
                indices = []
                for i in range(self.colony_size):
                    k = random.choice([x for x in range(self.colony_size) if x != i])
                    phi = np.random.uniform(-1, 1, len(self.param_names))
                    new_sol = self.food_sources[i] + phi * (self.food_sources[i] - self.food_sources[k])
                    new_sol = np.clip(new_sol, self.lb, self.ub)
                    new_solutions.append(new_sol)
                    indices.append(i)

                new_scores = self.evaluate_parallel(new_solutions)

                for i, new_sol, new_score in zip(indices, new_solutions, new_scores):
                    if new_score[0] > self.best_score[0]:
                        print(f"✨ New global best found at iteration {it+1}: Agent={new_score[0]:.3f}, Market={new_score[1]:.3f}")
                        self.best_score = new_score
                        self.best_solution = new_sol.copy()

                    if self.dominates(new_score[:2], self.fitness[i][:2]):
                        self.food_sources[i] = new_sol
                        self.fitness[i] = new_score
                        self.trial[i] = 0
                    else:
                        self.trial[i] += 1

            for i in range(self.colony_size):
                if self.trial[i] >= self.limit:
                    new_sol = np.random.uniform(self.lb, self.ub)
                    new_score = self.evaluate_parallel([new_sol])[0]
                    self.food_sources[i] = new_sol
                    self.fitness[i] = new_score
                    self.trial[i] = 0

                    if new_score[0] > self.best_score[0]:
                        print(f"✨ New global best found at iteration {it+1}: Agent={new_score[0]:.3f}, Market={new_score[1]:.3f}")
                        self.best_score = new_score
                        self.best_solution = new_sol.copy()

            print(f" -- Trial states: {[f'{i+1}- {int(self.trial[i])}/{self.limit}' for i in range(self.colony_size)]}")

            best_agent = max([s[0] for s in self.fitness])
            best_market = max([s[1] for s in self.fitness])
            print(f"[Iter {it+1:02}/{self.max_iter}] Best Agent: {best_agent:.3f}, Market: {best_market:.3f}")

            self.log_iteration(it)
            print(f"Iteration {it + 1} completed in {timeit.default_timer() - start_time:.2f} seconds")

        pareto_solutions = self.find_pareto_front()
        pareto_params = [self.to_dict(self.food_sources[i]) for i, _ in pareto_solutions]
        pareto_scores = [score[:2] for _, score in pareto_solutions]
        best_params = self.to_dict(self.best_solution)
        print(f"\n✅ Global Best: {best_params} Score: {self.best_score}")
        return pareto_params, pareto_scores, best_params, self.best_score