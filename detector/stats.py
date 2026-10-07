"""Dashboard statistics written as plain, parameterised SQL.

The queries use only standard SQL (JOIN, LEFT JOIN, GROUP BY, CASE, COALESCE,
DATE) so they run unchanged on SQLite and PostgreSQL. Table names come from the
models' metadata, never from user input; every value is passed as a parameter.
The same queries are listed in sql/queries.sql for reference.
"""
from __future__ import annotations

from datetime import timedelta

from django.db import connection
from django.utils import timezone

from .models import Disease, Prediction

P = Prediction._meta.db_table   # detector_prediction
D = Disease._meta.db_table      # detector_disease


def _fetch(sql: str, params=()) -> list[dict]:
    with connection.cursor() as cur:
        cur.execute(sql, params)
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def overview() -> dict:
    """Headline numbers: volume, average confidence, review queue, feedback accuracy."""
    row = _fetch(f"""
        SELECT COUNT(*)                                                      AS total,
               COALESCE(AVG(confidence), 0)                                  AS avg_confidence,
               COALESCE(SUM(CASE WHEN uncertain THEN 1 ELSE 0 END), 0)        AS uncertain,
               COALESCE(SUM(CASE WHEN true_disease_id IS NOT NULL THEN 1 ELSE 0 END), 0) AS reviewed,
               COALESCE(SUM(CASE WHEN true_disease_id = predicted_disease_id THEN 1 ELSE 0 END), 0) AS confirmed,
               COUNT(DISTINCT user_id)                                       AS users
        FROM {P}
    """)[0]
    row["field_accuracy"] = row["confirmed"] / row["reviewed"] if row["reviewed"] else None
    row["uncertain_share"] = row["uncertain"] / row["total"] if row["total"] else 0
    return row


def by_disease() -> list[dict]:
    """One row per disease, including diseases that have no predictions yet (LEFT JOIN)."""
    rows = _fetch(f"""
        SELECT d.name, d.slug, d.color,
               COUNT(p.id)                                             AS predictions,
               AVG(p.confidence)                                       AS avg_confidence,
               COALESCE(SUM(CASE WHEN p.uncertain THEN 1 ELSE 0 END), 0) AS uncertain
        FROM {D} d
        LEFT JOIN {P} p ON p.predicted_disease_id = d.id
        GROUP BY d.id, d.name, d.slug, d.color, d.display_order
        ORDER BY d.display_order, d.name
    """)
    total = sum(r["predictions"] for r in rows) or 1
    for r in rows:
        r["share"] = r["predictions"] / total
    return rows


def daily_activity(days: int = 14) -> list[dict]:
    """Predictions per day for the last ``days`` days (UTC dates), with empty days filled in."""
    since = timezone.now() - timedelta(days=days - 1)
    since = since.replace(hour=0, minute=0, second=0, microsecond=0)
    rows = _fetch(f"""
        SELECT DATE(created_at) AS day, COUNT(*) AS n
        FROM {P}
        WHERE created_at >= %s
        GROUP BY DATE(created_at)
        ORDER BY day
    """, [since])
    counts = {str(r["day"]): r["n"] for r in rows}
    out, peak = [], max(counts.values(), default=0) or 1
    for i in range(days):
        day = (since + timedelta(days=i)).date()
        n = counts.get(str(day), 0)
        out.append({"day": day, "n": n, "height": n / peak})
    return out


def feedback_matrix() -> dict:
    """Confusion matrix from user feedback: model prediction vs. the disease the user confirmed."""
    names = list(Disease.objects.values_list("name", flat=True))
    rows = _fetch(f"""
        SELECT pd.name AS predicted, td.name AS actual, COUNT(*) AS n
        FROM {P} p
        JOIN {D} pd ON pd.id = p.predicted_disease_id
        JOIN {D} td ON td.id = p.true_disease_id
        GROUP BY pd.name, td.name
    """)
    cells = {(r["actual"], r["predicted"]): r["n"] for r in rows}
    matrix = []
    for actual in names:
        total = sum(cells.get((actual, pred), 0) for pred in names) or 1
        matrix.append({"name": actual, "cells": [
            {"count": cells.get((actual, pred), 0), "share": cells.get((actual, pred), 0) / total,
             "diagonal": actual == pred} for pred in names]})
    return {"names": names, "rows": matrix, "total": sum(cells.values())}


def user_summary(user_id: int) -> dict:
    """Per-user counts for the history page header."""
    return _fetch(f"""
        SELECT COUNT(*) AS total,
               COALESCE(SUM(CASE WHEN uncertain THEN 1 ELSE 0 END), 0) AS uncertain,
               COALESCE(SUM(CASE WHEN true_disease_id IS NOT NULL THEN 1 ELSE 0 END), 0) AS reviewed
        FROM {P}
        WHERE user_id = %s
    """, [user_id])[0]
