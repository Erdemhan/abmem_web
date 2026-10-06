# ABMEM

Agent-based simulator of an hourly day-ahead electricity market. Producer agents bid with deep reinforcement learning policies (DDPG, RDPG, TD3, SAC, PPO), the market clears by merit order, and an Artificial Bee Colony search tunes the policy hyperparameters. A Django web interface manages resources, agents, scenarios and runs.

Project documents:
- `analiz.md` — defect analysis of the baseline code (Turkish)
- `PLAN.md` — work plan and approved decisions (Turkish)
- `ARCHITECTURE.md` — target architecture
- `specs/` — module specifications and `EXECUTION_ORDER.md`
- `FORMULATION.md` — locked formulation and parameter register

## Installation

Requirements: Python 3.11, PostgreSQL 16 (other recent versions should work).

```bash
python -m venv .venv            # or a conda env with Python 3.11
.venv/Scripts/activate          # Windows; use `source .venv/bin/activate` on Linux/macOS
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build for your machine
pip install -r requirements-dev.txt
```

Configuration is read from environment variables, optionally from a git-ignored `.env` file in the repository root:

```bash
cp .env.example .env            # then set DJANGO_SECRET_KEY and DB_PASSWORD
```

Create the database and apply the migrations:

```bash
createdb abmem                  # or create it with pgAdmin
python manage.py migrate
```

## Running

Web interface (pages live under `/dev/`):

```bash
python manage.py runserver
# open http://127.0.0.1:8000/dev/
```

Hyperparameter search from the command line (run from the repository root):

```bash
python -m abmem.services.algorithms.optimization.optimizer
```

## Tests

```bash
python -m pytest
```

pytest-django creates a temporary `test_abmem` database and applies the full migration chain. The database role therefore needs the `CREATEDB` privilege. If it is missing, grant it with `ALTER ROLE <user> CREATEDB;`.
