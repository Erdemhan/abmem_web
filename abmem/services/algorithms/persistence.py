"""Checkpoint files of trained policies (finding A9).

A checkpoint is one ``torch.save`` file holding a plain dict of tensors and
plain Python values (dict, list, int, float, bool, str, None), so it can be
read back with ``torch.load(weights_only=True)``, which executes no arbitrary
pickle code. Replay and rollout buffers are not saved: a restored policy starts
with an empty buffer and learns again once ``learning_starts`` transitions are
stored.

File layout (``FORMAT_VERSION`` 1)::

    {
        "format_version": 1,
        "algorithm_type": int,          # AlgorithmType value
        "algorithm_name": str,
        "torch_version": str,
        "created_at": str,              # ISO 8601, UTC
        "config": dict,                 # AlgorithmConfig.to_dict()
        "spec": dict,                   # PolicySpec fields
        "meta": dict,                   # caller data (run, agent, normaliser)
        "state": dict,                  # BaseAgentAlgorithm.get_state()
    }
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import os
import pathlib
from collections.abc import Mapping
from typing import Any, Final

import torch

from abmem.models.enums import AlgorithmType

from .base import BaseAgentAlgorithm, to_plain

FORMAT_VERSION: Final[int] = 1


class CheckpointError(ValueError):
    """A checkpoint is missing, corrupted or does not fit the target policy."""


def state_dict_of(
    policy: BaseAgentAlgorithm, meta: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return the checkpoint dict of a learning policy.

    Args:
        policy: Learning policy.
        meta: Additional plain values stored under ``meta``.

    Returns:
        The checkpoint dict (see the module docstring).

    Raises:
        CheckpointError: If ``policy`` is not a learning policy.
    """
    if not policy.is_learning:
        raise CheckpointError(
            f"{type(policy).__name__} has no learned state to save"
        )
    return {
        "format_version": FORMAT_VERSION,
        "algorithm_type": int(policy.algorithm_type),
        "algorithm_name": policy.algorithm_type.name,
        "torch_version": str(torch.__version__),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": to_plain(policy.config.to_dict()),
        "spec": to_plain(dataclasses.asdict(policy.spec)),
        "meta": to_plain(dict(meta or {})),
        "state": policy.get_state(),
    }


def check_compatible(policy: BaseAgentAlgorithm, checkpoint: Mapping[str, Any]) -> None:
    """Check algorithm type and dimensions of a checkpoint against a policy.

    Args:
        policy: Target policy.
        checkpoint: Loaded checkpoint dict.

    Raises:
        CheckpointError: If the format version, the algorithm type,
            ``obs_dim`` or ``action_dim`` differ.
    """
    version = checkpoint.get("format_version")
    if version != FORMAT_VERSION:
        raise CheckpointError(
            f"checkpoint format version {version!r} is not supported "
            f"(expected {FORMAT_VERSION})"
        )
    saved_type = checkpoint.get("algorithm_type")
    if saved_type != int(policy.algorithm_type):
        saved_name = AlgorithmType.get(saved_type)
        raise CheckpointError(
            f"checkpoint algorithm {getattr(saved_name, 'name', saved_type)} "
            f"differs from {policy.algorithm_type.name}"
        )
    spec = checkpoint.get("spec") or {}
    for name in ("obs_dim", "action_dim"):
        expected = getattr(policy.spec, name)
        if spec.get(name) != expected:
            raise CheckpointError(
                f"checkpoint {name} {spec.get(name)!r} differs from {expected}"
            )


def load_state_into(
    policy: BaseAgentAlgorithm,
    checkpoint: Mapping[str, Any],
    *,
    restore_runtime: bool = True,
) -> None:
    """Load a checkpoint dict into a policy (strict, all or nothing).

    Args:
        policy: Target policy, built with the same algorithm and dimensions.
        checkpoint: Output of :func:`state_dict_of` or :func:`load_checkpoint`.
        restore_runtime: Also restore generators, frozen flag and acting
            history (exact round trip). False for a warm start.

    Raises:
        CheckpointError: If the checkpoint does not fit ``policy``. The policy
            is unchanged in that case.
    """
    check_compatible(policy, checkpoint)
    try:
        policy.set_state(checkpoint["state"], restore_runtime=restore_runtime)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise CheckpointError(f"checkpoint does not fit the policy: {error}") from error


def file_sha256(path: pathlib.Path) -> str:
    """Return the hex SHA-256 digest of a file.

    Args:
        path: File to hash.

    Returns:
        The digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    path: pathlib.Path,
    policy: BaseAgentAlgorithm,
    meta: Mapping[str, Any] | None = None,
) -> tuple[int, str]:
    """Write the checkpoint of ``policy`` to ``path`` atomically.

    The file is written to a temporary name in the same directory and renamed,
    so a crash never leaves a half-written checkpoint under ``path``.

    Args:
        path: Target file. Parent directories are created.
        policy: Learning policy.
        meta: Additional plain values stored under ``meta``.

    Returns:
        File size in bytes and hex SHA-256 digest.

    Raises:
        CheckpointError: If ``policy`` is not a learning policy.
        OSError: If the file cannot be written.
    """
    data = state_dict_of(policy, meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(data, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path.stat().st_size, file_sha256(path)


def load_checkpoint(
    path: pathlib.Path,
    *,
    expected_sha256: str | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Read a checkpoint file with ``torch.load(weights_only=True)``.

    Args:
        path: Checkpoint file.
        expected_sha256: When given, the file digest must match.
        device: Device the tensors are mapped to (``map_location``); a
            checkpoint written on CUDA can be read on CPU. Defaults to CPU.

    Returns:
        The checkpoint dict.

    Raises:
        CheckpointError: If the file is missing, its digest differs, it
            cannot be read, or its format version is not supported.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        raise CheckpointError(f"checkpoint file {path} does not exist")
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise CheckpointError(
            f"checkpoint file {path} is corrupted (SHA-256 mismatch)"
        )
    try:
        data = torch.load(
            path, weights_only=True, map_location=device or torch.device("cpu")
        )
    except Exception as error:  # torch raises several unrelated types here
        raise CheckpointError(f"checkpoint file {path} cannot be read: {error}") from error
    if not isinstance(data, dict):
        raise CheckpointError(f"checkpoint file {path} does not hold a dict")
    if data.get("format_version") != FORMAT_VERSION:
        raise CheckpointError(
            f"checkpoint file {path} has format version "
            f"{data.get('format_version')!r}, expected {FORMAT_VERSION}"
        )
    return data
