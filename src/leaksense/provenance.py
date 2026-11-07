"""What every report records: git commit, data-manifest hash, config and seed."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

from leaksense import __version__


def git_commit(cwd: str | Path | None = None) -> dict:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            # Generated outputs (reports/, splits/) do not make the code dirty.
            "dirty": bool(
                run(
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                    "--",
                    ".",
                    ":!reports",
                    ":!splits",
                )
            ),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def provenance(
    data_dir: str | Path | None,
    config: dict | None = None,
    seed: int | None = None,
    manifest_sha256: str | None = None,
) -> dict:
    """Pass ``data_dir`` to hash the dataset, or ``manifest_sha256`` when no data is read."""
    if manifest_sha256 is None and data_dir is not None:
        from leaksense.data.validate import manifest

        manifest_sha256 = manifest(Path(data_dir))["sha256"]
    return {
        "leaksense_version": __version__,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_commit(),
        "data_manifest_sha256": manifest_sha256,
        "config": config or {},
        "seed": seed,
    }


def file_sha256(path: str | Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()
