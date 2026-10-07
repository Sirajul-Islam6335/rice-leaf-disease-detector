"""Insert or update the Disease rows from ``disease_data.DISEASES`` (idempotent)."""

from __future__ import annotations

from .disease_data import DISEASES


def seed_diseases(disease_model) -> tuple[int, int]:
    created = updated = 0
    for order, d in enumerate(DISEASES.values()):
        _, was_created = disease_model.objects.update_or_create(
            name=d.name,
            defaults={
                "slug": d.slug,
                "pathogen": d.pathogen,
                "kind": d.kind.lower(),
                "color": d.color,
                "sample_image": d.sample,
                "summary": d.summary,
                "symptoms": "\n".join(d.symptoms),
                "favoured_by": "\n".join(d.favoured_by),
                "management": "\n".join(d.management),
                "looks_like": d.looks_like,
                "display_order": order,
            },
        )
        created += was_created
        updated += not was_created
    return created, updated
