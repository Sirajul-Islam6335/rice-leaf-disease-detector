"""Train a new model with the NumPy CNN and install it for the web app.

    python manage.py train_model --data-dir path/to/rice_leaf_diseases [--epochs 80] [--cv]
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from detector.models import Disease


class Command(BaseCommand):
    help = "Train RiceLeafNet (pure NumPy) on a folder of class sub-folders and install the checkpoint."

    def add_arguments(self, parser):
        parser.add_argument("--data-dir", required=True)
        parser.add_argument("--output-dir", default="runs/latest")
        parser.add_argument("--epochs", type=int)
        parser.add_argument("--seed", type=int)
        parser.add_argument("--cv", action="store_true", help="also run 5-fold cross-validation")
        parser.add_argument("--no-install", action="store_true", help="train only; don't replace the live model")

    def handle(self, *args, **opts):
        import logging
        from riceleaf.config import TrainConfig
        from riceleaf.pipeline import cross_validate, train_holdout

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
        cfg = TrainConfig(data_dir=opts["data_dir"], output_dir=opts["output_dir"])
        if opts["epochs"]:
            cfg.epochs = opts["epochs"]
        if opts["seed"] is not None:
            cfg.seed = opts["seed"]
        if not Path(cfg.data_dir).is_dir():
            raise CommandError(f"Data directory not found: {cfg.data_dir}")

        metrics = train_holdout(cfg)
        out = Path(cfg.output_dir)
        if opts["cv"]:
            cfg.output_dir = str(out / "cv")
            cross_validate(cfg, 5)

        names = json.loads((out / "metrics.json").read_text())["test"]["per_class"].keys()
        missing = set(names) - set(Disease.objects.values_list("name", flat=True))
        if missing:
            self.stdout.write(self.style.WARNING(f"Add Disease rows for {sorted(missing)} before installing."))
        if opts["no_install"] or missing:
            self.stdout.write(f"Checkpoint left in {out / 'model.npz'}")
            return
        target = settings.RICELEAF["MODEL_PATH"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out / "model.npz", target)
        if (out / "cv" / "cv_results.json").exists():
            shutil.copy2(out / "cv" / "cv_results.json", settings.RICELEAF["CV_RESULTS_PATH"])
        self.stdout.write(self.style.SUCCESS(
            f"Installed {target} (test accuracy {metrics['test']['accuracy']:.1%}). Restart the server to load it."))
