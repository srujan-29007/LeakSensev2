"""Runtime configuration from environment variables.

``LEAKSENSE_DATA_DIR`` (required) points at the dataset's ``HanoiOK/`` folder, which lives
outside the repository and is only ever read. ``LEAKSENSE_ARTIFACTS_DIR`` (optional,
default ``./artifacts``) is where caches and checkpoints go.

Functions take an explicit ``environ`` mapping so tests never depend on the developer's
shell or ``.env``; only the CLI calls :func:`load_dotenv_file`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

DATA_DIR_VAR = "LEAKSENSE_DATA_DIR"
ARTIFACTS_DIR_VAR = "LEAKSENSE_ARTIFACTS_DIR"
DEFAULT_ARTIFACTS_DIR = "artifacts"

# Minimal entries that identify a valid HanoiOK folder (full contract: leaksense validate-data).
REQUIRED_DATA_ENTRIES = ("Hanoi.inp", "Labels.csv", "Scenario-1")


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def load_dotenv_file(path: str | os.PathLike[str] | None = None) -> None:
    """Load ``.env`` (default: current directory) without overriding existing variables."""
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=path or Path.cwd() / ".env", override=False)


def get_data_dir(environ: Mapping[str, str] | None = None) -> Path:
    """Return the validated dataset directory or raise :class:`ConfigError`."""
    env = os.environ if environ is None else environ
    raw = env.get(DATA_DIR_VAR, "").strip()
    if not raw:
        raise ConfigError(
            f"{DATA_DIR_VAR} is not set. Point it at the dataset's HanoiOK folder, e.g. in a "
            f".env file (see .env.example)."
        )
    path = Path(raw).expanduser()
    if not path.is_dir():
        raise ConfigError(f"{DATA_DIR_VAR}={raw!r} is not an existing directory.")
    missing = [name for name in REQUIRED_DATA_ENTRIES if not (path / name).exists()]
    if missing:
        raise ConfigError(
            f"{DATA_DIR_VAR}={raw!r} does not look like the HanoiOK folder; missing: "
            + ", ".join(missing)
        )
    return path.resolve()


def get_artifacts_dir(environ: Mapping[str, str] | None = None) -> Path:
    """Return the artifacts directory (not created here)."""
    env = os.environ if environ is None else environ
    raw = env.get(ARTIFACTS_DIR_VAR, "").strip() or DEFAULT_ARTIFACTS_DIR
    return Path(raw).expanduser().resolve()
