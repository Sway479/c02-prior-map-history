#!/usr/bin/env python3
"""Fail-closed path and permission helpers for protected C02 analyses."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


WORKSPACE_ENV = "C02_PRIVATE_WORKSPACE"
ALLOWED_REAL_ROOTS_ENV = "C02_ALLOWED_PRIVATE_REAL_ROOTS"
CODE_ROOT = Path(__file__).resolve().parents[1]


def _normalized_absolute(path: Path) -> Path:
    """Collapse relative components without following symlinks."""
    return Path(os.path.abspath(os.path.normpath(os.fspath(path.expanduser()))))


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _reject_code_path(path: Path, *, label: str) -> None:
    normalized = _normalized_absolute(path)
    resolved = normalized.resolve(strict=False)
    if _within(normalized, CODE_ROOT) or _within(resolved, CODE_ROOT):
        raise RuntimeError(f"{label} points inside public code: {normalized}")


def _validated_workspace(path: Path) -> Path:
    normalized = _normalized_absolute(path)
    _reject_code_path(normalized, label=WORKSPACE_ENV)
    return normalized


def configure_private_workspace(
    path: Path | str,
    allowed_real_roots: Iterable[Path | str] = (),
) -> Path:
    """Set the logical private root and explicit offload target allow-list."""
    workspace = _validated_workspace(Path(path))
    allowed: list[str] = []
    for item in allowed_real_roots:
        real_root = _normalized_absolute(Path(item)).resolve(strict=False)
        _reject_code_path(real_root, label=ALLOWED_REAL_ROOTS_ENV)
        allowed.append(str(real_root))
    os.environ[WORKSPACE_ENV] = str(workspace)
    os.environ[ALLOWED_REAL_ROOTS_ENV] = os.pathsep.join(allowed)
    os.umask(0o077)
    return workspace


def private_workspace_root() -> Path:
    """Return the configured private root or fail before any data access."""
    value = os.environ.get(WORKSPACE_ENV)
    if not value:
        raise RuntimeError(
            f"Set {WORKSPACE_ENV} to an access-controlled directory outside "
            "this public repository, or use run_pipeline.py with a config file."
        )
    os.umask(0o077)
    return _validated_workspace(Path(value))


def _allowed_real_roots(workspace: Path) -> tuple[Path, ...]:
    roots = [workspace.resolve(strict=False)]
    raw = os.environ.get(ALLOWED_REAL_ROOTS_ENV, "")
    for value in raw.split(os.pathsep):
        if not value:
            continue
        real_root = _normalized_absolute(Path(value)).resolve(strict=False)
        _reject_code_path(real_root, label=ALLOWED_REAL_ROOTS_ENV)
        roots.append(real_root)
    return tuple(dict.fromkeys(roots))


def require_private_path(path: Path, *, must_exist: bool = True) -> Path:
    """Validate both the logical path and its resolved symlink target."""
    workspace = private_workspace_root()
    logical = _normalized_absolute(path)
    if not _within(logical, workspace):
        raise RuntimeError(f"Protected path is outside {WORKSPACE_ENV}: {logical}")
    _reject_code_path(logical, label="Protected path")

    real = logical.resolve(strict=False)
    allowed = _allowed_real_roots(workspace)
    if not any(_within(real, root) for root in allowed):
        raise RuntimeError(
            "Protected path resolves outside approved private roots: "
            f"logical={logical}; resolved={real}"
        )
    if must_exist and not logical.exists():
        raise FileNotFoundError(logical)
    return logical


def secure_directory(path: Path) -> Path:
    """Create a private directory and re-check its resolved destination."""
    logical = require_private_path(path, must_exist=False)
    logical.mkdir(parents=True, exist_ok=True)
    logical = require_private_path(logical)
    try:
        logical.chmod(0o700)
    except OSError:
        # POSIX modes are not available on every supported platform.
        pass
    return logical


def protect_file(path: Path) -> Path:
    """Apply owner-only permissions to a newly written protected artifact."""
    logical = require_private_path(path)
    try:
        logical.chmod(0o600)
    except OSError:
        pass
    return logical
