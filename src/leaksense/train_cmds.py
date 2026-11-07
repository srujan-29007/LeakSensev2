"""CLI command for M8+: ``leaksense train``.

Trains the learned model named by the config's ``model.kind`` once per CV fold (the fold's
cases are never seen in training) and scores it through the same harness as ``eval``.
"""

from __future__ import annotations

import argparse
import time

from leaksense.config import ConfigError
from leaksense.eval_cmds import EXIT_OK, add_common, load_inputs, run_and_report


def learned_models() -> dict:
    from leaksense.models.gbm import MaskedGBM
    from leaksense.models.mlp import MaskedMLP

    return {"mlp": MaskedMLP, "gbm": MaskedGBM}


def add_parsers(sub) -> None:
    t = sub.add_parser(
        "train", help="train a learned model per CV fold and score it (reports/<name>.json, .md)"
    )
    add_common(t, default_config="configs/mlp_cpu.yaml")
    t.add_argument("--fold", type=int, help="score only this fold (a quick smoke run)")
    t.set_defaults(func=cmd_train)


def cmd_train(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    inp = load_inputs(args)
    kind = inp.cfg.model.get("kind")
    models = learned_models()
    if kind not in models:
        raise ConfigError(f"config model.kind must be one of {sorted(models)}, not {kind!r}")
    model = models[kind](inp.net, inp.cfg.model)
    all_folds = sorted(int(f) for f in inp.es.cases["fold"].unique())
    folds = None
    if args.fold is not None:
        if args.fold not in all_folds:
            raise ConfigError(f"--fold must be one of {all_folds}")
        folds = [args.fold]
    eval_folds = folds or all_folds
    name = args.name or (kind if folds is None else f"{kind}_fold{args.fold}")
    args.name = name
    print(f"training {kind} on {len(inp.es.cases)} CV cases, scoring folds {eval_folds}")

    def extra() -> dict:  # called after training, so the per-fold history is filled
        return {
            "command": "leaksense train",
            "folds_note": "5 folds pooled" if folds is None else f"fold {args.fold} only",
            "training": {**model.describe(), "eval_folds": eval_folds},
        }

    run_and_report(args, inp, model, kind, t0, folds=folds, extra=extra)
    return EXIT_OK
