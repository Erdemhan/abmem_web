from pathlib import Path
from typing import Final

# Absolute path of abmem/data, independent of the current working directory.
DATA_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "data"
SIMULATION_DATA_PATH = "simulation.yml"
MARKET_DATA_PATH = "market.yml"
AGENT_DATA_PATH = "agents.yml"
RESOURCES_DATA_PATH = "resources.yml"
