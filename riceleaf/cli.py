"""Command-line interface: ``riceleaf {train,cv,evaluate,predict,summary}``."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import TrainConfig
from .data import IMAGE_EXTENSIONS


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="JSON config file (CLI flags override it)")
    p.add_argument("--data-dir")
    p.add_argument("--output-dir")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-tta", action="store_true")


def _config_from(args) -> TrainConfig:
    cfg = TrainConfig.from_json(args.config) if args.config else TrainConfig()
    for flag, attr in [("data_dir", "data_dir"), ("output_dir", "output_dir"), ("epochs", "epochs"),
                       ("batch_size", "batch_size"), ("lr", "lr"), ("seed", "seed")]:
        val = getattr(args, flag)
        if val is not None:
            setattr(cfg, attr, val)
    if args.no_augment:
        cfg.augment.enabled = False
    if args.no_tta:
        cfg.tta = False
    return cfg


def _setup_logging(output_dir: str | None, verbose: bool) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(output_dir) / "train.log", mode="w"))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S", force=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="riceleaf", description="Rice leaf disease classifier (pure NumPy CNN).")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train on a stratified train/val/test split")
    _add_train_args(p_train)

    p_cv = sub.add_parser("cv", help="stratified k-fold cross-validation")
    _add_train_args(p_cv)
    p_cv.add_argument("--folds", type=int, default=5)

    p_eval = sub.add_parser("evaluate", help="evaluate a checkpoint on a labelled folder")
    p_eval.add_argument("checkpoint")
    p_eval.add_argument("data_dir")
    p_eval.add_argument("--no-tta", action="store_true")

    p_pred = sub.add_parser("predict", help="predict one or more images (or a folder)")
    p_pred.add_argument("checkpoint")
    p_pred.add_argument("inputs", nargs="+")
    p_pred.add_argument("--threshold", type=float, default=0.6)
    p_pred.add_argument("--no-tta", action="store_true")

    p_sum = sub.add_parser("summary", help="print the architecture")
    p_sum.add_argument("--config")

    args = parser.parse_args(argv)

    if args.command in ("train", "cv"):
        cfg = _config_from(args)
        _setup_logging(cfg.output_dir, args.verbose)
        from .pipeline import cross_validate, train_holdout
        if args.command == "train":
            train_holdout(cfg)
        else:
            cross_validate(cfg, args.folds)
    elif args.command == "evaluate":
        _setup_logging(None, args.verbose)
        from .pipeline import evaluate_checkpoint
        m = evaluate_checkpoint(args.checkpoint, args.data_dir, tta=not args.no_tta)
        print(json.dumps({k: m[k] for k in ("accuracy", "macro_f1", "log_loss", "n_samples")}, indent=2))
    elif args.command == "predict":
        from .inference import Predictor
        files = []
        for item in args.inputs:
            path = Path(item)
            files += sorted(f for f in path.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS) if path.is_dir() else [path]
        predictor = Predictor(args.checkpoint, threshold=args.threshold, tta=not args.no_tta)
        print(json.dumps([p.to_dict() for p in predictor.predict_batch(files)], indent=2))
    elif args.command == "summary":
        import numpy as np
        from .nn import build_rice_leaf_net
        cfg = TrainConfig.from_json(args.config) if args.config else TrainConfig()
        model = build_rice_leaf_net(3, channels=cfg.model.channels, kernel_size=cfg.model.kernel_size,
                                    dropout=cfg.model.dropout, rng=np.random.default_rng(0))
        print(model.summary((3, *cfg.image_size)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
