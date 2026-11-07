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
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def provenance(data_dir: str | Path, config: dict | None = None, seed: int | None = None) -> dict:
    from leaksense.data.validate import manifest

    m = manifest(Path(data_dir))
    return {
        "leaksense_version": __version__,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_commit(),
        "data_manifest_sha256": m["sha256"],
        "config": config or {},
        "seed": seed,
    }
