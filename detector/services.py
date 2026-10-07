"""Bridge between Django and the NumPy CNN.

The model is loaded once per process (lazily, thread-safe) and reused for every
request. Inference in eval mode is stateless, so concurrent requests are safe;
the lock only guards the one-time load.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from django.conf import settings
from django.core.files.base import ContentFile
from PIL import Image

from riceleaf.inference import Predictor

log = logging.getLogger(__name__)
_lock = threading.Lock()
_predictor: Predictor | None = None


class ModelUnavailable(RuntimeError):
    pass


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        with _lock:
            if _predictor is None:
                cfg = settings.RICELEAF
                path = cfg["MODEL_PATH"]
                if not path.exists():
                    raise ModelUnavailable(f"Model file not found: {path}")
                t0 = time.perf_counter()
                _predictor = Predictor(path, threshold=cfg["UNCERTAINTY_THRESHOLD"], tta=cfg["USE_TTA"])
                log.info("Loaded model %s in %.0f ms", path.name, (time.perf_counter() - t0) * 1000)
    return _predictor


@dataclass
class Diagnosis:
    label: str
    confidence: float
    probabilities: dict[str, float]
    uncertain: bool
    inference_ms: float
    model_version: str


def diagnose(image: Image.Image) -> Diagnosis:
    predictor = get_predictor()
    t0 = time.perf_counter()
    result = predictor.predict(image)
    ms = (time.perf_counter() - t0) * 1000
    return Diagnosis(result.label, result.confidence, result.probabilities, result.uncertain, round(ms, 1),
                     model_version())


def diagnose_file(uploaded_file) -> Diagnosis:
    """Run the model on a file-like object without storing anything."""
    uploaded_file.seek(0)
    with Image.open(uploaded_file) as img:
        img.load()
        return diagnose(img)


def diagnose_and_store(uploaded_file, original_name: str, request=None, source: str = "upload",
                       diagnosis: Diagnosis | None = None):
    """Run the model on an uploaded file and save the image + result as a Prediction row.

    The row is linked to the signed-in user; for anonymous visitors its id is kept in the
    session so they can view it (and claim it by signing up later).
    """
    from .models import Disease, Prediction

    if diagnosis is None:
        diagnosis = diagnose_file(uploaded_file)
    try:
        disease = Disease.objects.get(name=diagnosis.label)
    except Disease.DoesNotExist as exc:
        raise ModelUnavailable(f"No Disease row named '{diagnosis.label}'. Run `manage.py seed_diseases`.") from exc
    uploaded_file.seek(0)
    user = request.user if request is not None and request.user.is_authenticated else None
    prediction = Prediction(
        user=user,
        original_filename=original_name[:255],
        source=source,
        predicted_disease=disease,
        confidence=min(max(diagnosis.confidence, 0.0), 1.0),
        probabilities=diagnosis.probabilities,
        uncertain=diagnosis.uncertain,
        inference_ms=diagnosis.inference_ms,
        model_version=diagnosis.model_version,
    )
    prediction.image.save(original_name, ContentFile(uploaded_file.read()), save=False)
    prediction.save()
    if request is not None and user is None:
        ids = request.session.get("my_predictions", [])
        request.session["my_predictions"] = (ids + [str(prediction.pk)])[-200:]
    log.info("Prediction %s: %s (%.2f) in %.0f ms", prediction.pk, diagnosis.label, diagnosis.confidence,
             diagnosis.inference_ms)
    return prediction


def claim_session_predictions(request, user) -> int:
    """Attach predictions made before signing in to the user's account."""
    from .models import Prediction

    ids = request.session.pop("my_predictions", [])
    if not ids:
        return 0
    return Prediction.objects.filter(pk__in=ids, user__isnull=True).update(user=user)


@lru_cache(maxsize=1)
def model_version() -> str:
    meta = get_predictor().meta
    return f"RiceLeafNet {meta.get('package_version', '')}, trained {meta.get('created_utc', '')[:10]}"


@lru_cache(maxsize=1)
def model_card() -> dict:
    """Everything the 'Model' page shows, read from the checkpoint and CV results."""
    predictor = get_predictor()
    meta = predictor.meta
    names = meta["class_names"]
    test = meta["metrics"]["test"]
    cm = np.array(test["confusion_matrix"])
    card = {
        "class_names": names,
        "image_size": meta["image_size"],
        "architecture": meta["architecture"],
        "num_params": predictor.model.num_params(),
        "summary": predictor.model.summary((3, *meta["image_size"])),
        "best_epoch": meta["best_epoch"],
        "created": meta.get("created_utc", ""),
        "config": meta["config"],
        "test": test,
        "val": meta["metrics"]["val"],
        "test_matrix": _matrix_rows(cm, names),
        "threshold": settings.RICELEAF["UNCERTAINTY_THRESHOLD"],
        "tta": settings.RICELEAF["USE_TTA"],
        "cv": None,
    }
    cv_path = settings.RICELEAF["CV_RESULTS_PATH"]
    if cv_path.exists():
        cv = json.loads(cv_path.read_text())
        pooled = cv["pooled_out_of_fold"]
        card["cv"] = {
            "k": cv["k"], "folds": cv["folds"],
            "accuracy_mean": cv["accuracy_mean"], "accuracy_std": cv["accuracy_std"],
            "macro_f1_mean": cv["macro_f1_mean"], "macro_f1_std": cv["macro_f1_std"],
            "pooled": pooled,
            "matrix": _matrix_rows(np.array(pooled["confusion_matrix"]), names),
        }
    return card


def _matrix_rows(cm: np.ndarray, names: list[str]) -> list[dict]:
    """Confusion-matrix rows with a 0-1 shade per cell (share of the row) for the heat-map table."""
    rows = []
    for i, name in enumerate(names):
        total = max(int(cm[i].sum()), 1)
        rows.append({"name": name, "cells": [{"count": int(v), "share": v / total, "diagonal": i == j}
                                             for j, v in enumerate(cm[i])]})
    return rows
