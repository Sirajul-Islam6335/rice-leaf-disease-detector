"""Database schema.

Two tables hold the application's data:

* ``detector_disease``    - reference data for each class the model can predict
                            (seeded by migration 0002, editable in the admin).
* ``detector_prediction`` - every diagnosed photo: who uploaded it, what the model
                            said, how confident it was, and the user's feedback.

User accounts use Django's built-in ``auth_user`` table.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import django
from django.conf import settings
from django.db import models
from django.db.models import F, Q
from django.db.models.signals import post_delete
from django.dispatch import receiver
from django.urls import reverse
from django.utils import timezone


def check_kwarg(q):
    """CheckConstraint's argument is `condition` from Django 5.1 and `check` before (Django 4.2 LTS)."""
    return {"condition": q} if django.VERSION >= (5, 1) else {"check": q}


class Disease(models.Model):
    class Kind(models.TextChoices):
        BACTERIAL = "bacterial", "Bacterial"
        FUNGAL = "fungal", "Fungal"
        VIRAL = "viral", "Viral"

    name = models.CharField(max_length=64, unique=True,
                            help_text="Must match a class name stored in the model checkpoint.")
    slug = models.SlugField(max_length=64, unique=True)
    pathogen = models.CharField(max_length=120)
    kind = models.CharField(max_length=16, choices=Kind.choices)
    color = models.CharField(max_length=7, help_text="Hex colour of the lesions, used to mark this disease in the UI.")
    sample_image = models.CharField(max_length=200, blank=True, help_text="Static path of a sample photo.")
    summary = models.TextField()
    symptoms = models.TextField(help_text="One symptom per line.")
    favoured_by = models.TextField(blank=True, help_text="One condition per line.")
    management = models.TextField(help_text="One practice per line.")
    looks_like = models.TextField(blank=True)
    display_order = models.PositiveSmallIntegerField(default=0)

    class Meta:
        ordering = ["display_order", "name"]

    def __str__(self) -> str:
        return self.name

    def get_absolute_url(self) -> str:
        return reverse("detector:diseases") + f"#{self.slug}"

    @staticmethod
    def _lines(text: str) -> list[str]:
        return [line.strip() for line in text.splitlines() if line.strip()]

    @property
    def symptom_list(self):
        return self._lines(self.symptoms)

    @property
    def favoured_by_list(self):
        return self._lines(self.favoured_by)

    @property
    def management_list(self):
        return self._lines(self.management)


def upload_path(instance, filename: str) -> str:
    ext = Path(filename).suffix.lower() or ".jpg"
    return f"uploads/{timezone.now():%Y/%m}/{instance.pk}{ext}"


class PredictionQuerySet(models.QuerySet):
    def visible_to(self, request):
        """Predictions this visitor may see: their own (account or this browser session); staff see all."""
        user = request.user
        if user.is_authenticated and user.is_staff:
            return self
        session_ids = request.session.get("my_predictions", [])
        q = Q(pk__in=session_ids)
        if user.is_authenticated:
            q |= Q(user=user)
        return self.filter(q)


class Prediction(models.Model):
    """One uploaded leaf photo and the model's diagnosis of it."""

    class Source(models.TextChoices):
        UPLOAD = "upload", "Uploaded photo"
        SAMPLE = "sample", "Sample photo"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL,
                             related_name="predictions")
    image = models.ImageField(upload_to=upload_path)
    original_filename = models.CharField(max_length=255, blank=True)
    source = models.CharField(max_length=10, choices=Source.choices, default=Source.UPLOAD)

    predicted_disease = models.ForeignKey(Disease, on_delete=models.PROTECT, related_name="predictions")
    confidence = models.FloatField(help_text="Probability of the predicted class (0-1)")
    probabilities = models.JSONField(help_text="Probability for every class")
    uncertain = models.BooleanField(default=False, help_text="Confidence below the review threshold")
    inference_ms = models.FloatField(default=0)
    model_version = models.CharField(max_length=80, blank=True)

    # Feedback: what the user (or an expert) says the leaf really has.
    true_disease = models.ForeignKey(Disease, null=True, blank=True, on_delete=models.SET_NULL,
                                     related_name="confirmed_predictions")
    feedback_note = models.CharField(max_length=500, blank=True)
    feedback_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    objects = PredictionQuerySet.as_manager()

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["predicted_disease", "created_at"], name="pred_disease_created_idx"),
            models.Index(fields=["user", "created_at"], name="pred_user_created_idx"),
        ]
        constraints = [
            models.CheckConstraint(**check_kwarg(Q(confidence__gte=0) & Q(confidence__lte=1)),
                                   name="prediction_confidence_0_1"),
        ]

    def __str__(self) -> str:
        return f"{self.predicted_disease} ({self.confidence:.0%}) - {self.created_at:%Y-%m-%d %H:%M}"

    def get_absolute_url(self) -> str:
        return reverse("detector:prediction_detail", args=[self.pk])

    @property
    def has_feedback(self) -> bool:
        return self.true_disease_id is not None

    @property
    def is_correct(self) -> bool | None:
        return None if self.true_disease_id is None else self.true_disease_id == self.predicted_disease_id

    def ranked_probabilities(self, colors: dict[str, str]) -> list[dict]:
        """Classes sorted by probability, with their display colour, for the bar chart."""
        return [{"name": name, "probability": p, "percent": round(p * 100, 1), "color": colors.get(name, "#6b7280")}
                for name, p in sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)]


@receiver(post_delete, sender=Prediction)
def delete_image_file(sender, instance, **kwargs):
    """Remove the stored photo when its prediction is deleted."""
    if instance.image:
        instance.image.delete(save=False)
