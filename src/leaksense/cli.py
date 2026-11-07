"""Command-line entry point: ``leaksense <command>``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from leaksense import __version__
from leaksense.config import DATA_DIR_VAR, ConfigError, get_data_dir, load_dotenv_file

EXIT_OK, EXIT_FAILED, EXIT_CONFIG = 0, 1, 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leaksense",
        description="Leak localization and sensor placement on the Hanoi network (LeakDB).",
    )
    parser.add_argument("--version", action="version", version=f"leaksense {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    v = sub.add_parser("validate-data", help="check the dataset against the data contract (D1-D10)")
    v.add_argument("--data-dir", help=f"dataset HanoiOK folder (default: ${DATA_DIR_VAR})")
    v.add_argument("--deep", action="store_true", help="also parse every pressure and leak file")
    v.add_argument("--out", default="reports/data_validation.json", help="JSON report path")
    v.add_argument("--contract", help="JSON file overriding the default data contract")
    v.add_argument("--quiet", action="store_true", help="no progress output")
    v.set_defaults(func=cmd_validate_data)
    return parser


def cmd_validate_data(args: argparse.Namespace) -> int:
    from leaksense.data.contract import DataContract
    from leaksense.data.validate import validate_dataset

    env = {DATA_DIR_VAR: args.data_dir} if args.data_dir else None
    data_dir = get_data_dir(env)
    contract = DataContract.from_json(args.contract) if args.contract else DataContract()
    t0 = time.perf_counter()
    report = validate_dataset(data_dir, contract, deep=args.deep, progress=not args.quiet)
    elapsed = time.perf_counter() - t0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload["elapsed_seconds"] = round(elapsed, 1)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    m = report.manifest
    print(
        f"dataset: {m['folder']}  files={m['n_files']}  bytes={m['total_bytes']:,}  "
        f"manifest={m['sha256'][:12]}  mode={'deep' if args.deep else 'fast'}"
    )
    for check in report.checks.values():
        print(f"  {check.id:<4} {check.status.upper():<5} {check.title}")
        for err in check.errors[:3]:
            print(f"         - {err}")
        if check.n_errors > 3:
            print(f"         ... {check.n_errors - 3} more (see {out})")
    print(f"{'OK' if report.ok else 'FAILED'} in {elapsed:.1f}s -> {out}")
    return EXIT_OK if report.ok else EXIT_FAILED


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return EXIT_OK
    load_dotenv_file()
    try:
        return args.func(args)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return EXIT_CONFIG


if __name__ == "__main__":
    sys.exit(main())
