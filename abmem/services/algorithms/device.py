"""Single place for torch device selection of the algorithm layer."""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Return the torch device used by all policies.

    Returns:
        ``torch.device("cuda")`` when a CUDA device is available, otherwise
        ``torch.device("cpu")``.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
