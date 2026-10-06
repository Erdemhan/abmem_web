"""Command-line entry point for the ABC hyperparameter search.

Run from the repository root:

    python -m abmem.services.algorithms.optimization.optimizer
"""

import timeit
from datetime import datetime

from abmem.bootstrap import setup_django

# Shared settings
COLONY_SIZE = 2
MAX_ITER = 2
LIMIT = 5
SEED = 17081999
ALGORITHMS = ["PPO"]  # "RDPG", "DDPG",  "TD3", "PPO", "SAC"


def main() -> None:
    """Run the hyperparameter search for every algorithm in ``ALGORITHMS``."""
    setup_django()
    from django.conf import settings

    from .abc_logger import ABCLogger
    from .hyperparam_optimization import optimize_hyperparams

    for algorithm in ALGORITHMS:
        logger = ABCLogger(
            algorithm_name=algorithm,
            log_dir=str(settings.BASE_DIR / "abc_logs"),
            meta_info={
                "colony_size": COLONY_SIZE,
                "max_iter": MAX_ITER,
                "limit": LIMIT,
                "seed": SEED,
            },
        )

        logger.log_text(
            f"\n>>> Starting {algorithm} optimization at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} <<<"
        )

        start = timeit.default_timer()
        optimize_hyperparams(
            colony_size=COLONY_SIZE,
            max_iter=MAX_ITER,
            limit=LIMIT,
            seed=SEED,
            algorithm_type=algorithm,
        )
        elapsed = timeit.default_timer() - start
        logger.log_text(
            f"✅ {algorithm} optimization completed in {elapsed:.2f} seconds"
        )


if __name__ == "__main__":
    main()
