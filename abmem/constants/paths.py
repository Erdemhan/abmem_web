from pathlib import Path
from typing import Final

# Absolute path of abmem/data, independent of the current working directory.
DATA_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "data"
