from __future__ import annotations

from django.core.management.base import BaseCommand

from detector.models import Disease
from detector.seeding import seed_diseases


class Command(BaseCommand):
    help = "Insert or update the Disease reference rows (safe to run repeatedly)."

    def handle(self, *args, **options):
        created, updated = seed_diseases(Disease)
        self.stdout.write(self.style.SUCCESS(f"Diseases: {created} created, {updated} updated."))
