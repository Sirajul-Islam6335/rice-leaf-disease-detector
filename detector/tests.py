"""End-to-end tests: run with `python manage.py test`.

They use the real NumPy model in ml_models/model.npz and write uploads to a temporary folder.
"""

from __future__ import annotations

import io
import shutil
import tempfile

import numpy as np
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.db import IntegrityError, transaction
from django.test import TestCase, override_settings
from django.urls import reverse
from PIL import Image

from . import stats
from .models import Disease, Prediction

TMP_MEDIA = tempfile.mkdtemp(prefix="riceleaf-test-media-")


def leaf_photo(name="leaf.jpg", size=(300, 90), fmt="JPEG", seed=0):
    rng = np.random.default_rng(seed)
    arr = np.zeros((size[1], size[0], 3), np.uint8)
    arr[...] = (70, 140, 60)
    for _ in range(12):                                   # a few dark "lesions"
        y, x = rng.integers(0, size[1] - 8), rng.integers(0, size[0] - 12)
        arr[y:y + 8, x:x + 12] = (110, 70, 30)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, fmt)
    return SimpleUploadedFile(name, buf.getvalue(), content_type=f"image/{fmt.lower()}")


@override_settings(MEDIA_ROOT=TMP_MEDIA, STORAGES={
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"}})
class DetectorTests(TestCase):
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(TMP_MEDIA, ignore_errors=True)

    def upload(self, client=None, **kw):
        return (client or self.client).post(reverse("detector:home"), {"image": leaf_photo(**kw)})

    # --- schema & seed data --------------------------------------------------------
    def test_diseases_seeded_by_migration(self):
        self.assertEqual(list(Disease.objects.values_list("name", flat=True)),
                         ["Bacterial leaf blight", "Brown spot", "Leaf smut"])
        bs = Disease.objects.get(slug="brown-spot")
        self.assertGreater(len(bs.symptom_list), 2)

    def test_confidence_check_constraint(self):
        self.upload()
        p = Prediction.objects.get()
        with self.assertRaises(IntegrityError), transaction.atomic():
            Prediction.objects.filter(pk=p.pk).update(confidence=1.5)

    # --- diagnosis flow ----------------------------------------------------------------
    def test_home_page(self):
        r = self.client.get(reverse("detector:home"))
        self.assertContains(r, "Check a rice leaf for disease")
        self.assertContains(r, 'type="file"')

    def test_upload_creates_prediction_and_redirects(self):
        r = self.upload()
        p = Prediction.objects.get()
        self.assertRedirects(r, p.get_absolute_url())
        self.assertIn(p.predicted_disease.name, p.probabilities)
        self.assertAlmostEqual(sum(p.probabilities.values()), 1.0, places=5)
        self.assertEqual(p.predicted_disease.name, max(p.probabilities, key=p.probabilities.get))
        self.assertTrue(p.image.storage.exists(p.image.name))
        page = self.client.get(p.get_absolute_url())
        self.assertContains(page, p.predicted_disease.name)
        self.assertContains(page, "Was this right?")

    def test_rejects_non_image(self):
        bad = SimpleUploadedFile("notes.txt", b"not an image", content_type="text/plain")
        r = self.client.post(reverse("detector:home"), {"image": bad})
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "isn&#x27;t a readable image")
        self.assertFalse(Prediction.objects.exists())

    def test_rejects_tiny_image(self):
        r = self.upload(size=(20, 10))
        self.assertContains(r, "Upload one at least 32 pixels")
        self.assertFalse(Prediction.objects.exists())

    def test_sample_prediction(self):
        r = self.client.post(reverse("detector:predict_sample", args=["leaf-smut"]))
        p = Prediction.objects.get()
        self.assertRedirects(r, p.get_absolute_url())
        self.assertEqual(p.source, Prediction.Source.SAMPLE)

    # --- ownership ---------------------------------------------------------------------
    def test_anonymous_visitor_cannot_see_other_sessions(self):
        self.upload()
        p = Prediction.objects.get()
        other = self.client_class()
        self.assertEqual(other.get(p.get_absolute_url()).status_code, 404)
        self.assertEqual(other.post(reverse("detector:prediction_delete", args=[p.pk])).status_code, 404)
        self.assertNotContains(other.get(reverse("detector:history")), p.get_absolute_url())

    def test_users_only_see_their_own_staff_see_all(self):
        alice = User.objects.create_user("alice", password="pw-alice-123")
        User.objects.create_user("bob", password="pw-bob-123")
        User.objects.create_user("staff", password="pw-staff-123", is_staff=True)
        self.client.login(username="alice", password="pw-alice-123")
        self.upload()
        p = Prediction.objects.get()
        self.assertEqual(p.user, alice)
        bob = self.client_class(); bob.login(username="bob", password="pw-bob-123")
        self.assertEqual(bob.get(p.get_absolute_url()).status_code, 404)
        staff = self.client_class(); staff.login(username="staff", password="pw-staff-123")
        self.assertEqual(staff.get(p.get_absolute_url()).status_code, 200)

    def test_signup_claims_session_predictions(self):
        self.upload(); self.upload(seed=1)
        r = self.client.post(reverse("detector:signup"), {
            "username": "newgrower", "email": "", "password1": "Paddy-field-2026", "password2": "Paddy-field-2026"})
        self.assertRedirects(r, reverse("detector:history"))
        user = User.objects.get(username="newgrower")
        self.assertEqual(Prediction.objects.filter(user=user).count(), 2)

    def test_login_claims_session_predictions(self):
        User.objects.create_user("carol", password="pw-carol-123")
        self.upload()
        self.client.post(reverse("detector:login"), {"username": "carol", "password": "pw-carol-123"})
        self.assertEqual(Prediction.objects.get().user.username, "carol")

    def test_logout_requires_post(self):
        User.objects.create_user("dan", password="pw-dan-123")
        self.client.login(username="dan", password="pw-dan-123")
        self.assertEqual(self.client.get(reverse("detector:logout")).status_code, 405)
        self.client.post(reverse("detector:logout"))
        self.assertNotIn("_auth_user_id", self.client.session)

    # --- feedback, delete, history -------------------------------------------------------
    def test_feedback_is_saved(self):
        self.upload()
        p = Prediction.objects.get()
        other = Disease.objects.exclude(pk=p.predicted_disease_id).first()
        r = self.client.post(p.get_absolute_url(), {"true_disease": other.pk, "note": "checked in the field"})
        self.assertRedirects(r, p.get_absolute_url())
        p.refresh_from_db()
        self.assertEqual(p.true_disease, other)
        self.assertFalse(p.is_correct)
        self.assertEqual(p.feedback_note, "checked in the field")

    def test_delete_removes_row_and_file(self):
        self.upload()
        p = Prediction.objects.get()
        storage, name = p.image.storage, p.image.name
        self.client.post(reverse("detector:prediction_delete", args=[p.pk]))
        self.assertFalse(Prediction.objects.exists())
        self.assertFalse(storage.exists(name))

    def test_history_filters(self):
        self.upload(); self.upload(seed=3)
        disease = Prediction.objects.first().predicted_disease
        r = self.client.get(reverse("detector:history"), {"disease": disease.slug})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(all(p.predicted_disease == disease for p in r.context["page"].object_list))

    # --- SQL statistics ------------------------------------------------------------------
    def test_stats_queries(self):
        self.assertEqual(stats.overview()["total"], 0)
        self.assertEqual([r["predictions"] for r in stats.by_disease()], [0, 0, 0])   # LEFT JOIN keeps empty rows
        self.upload(); self.upload(seed=2)
        p = Prediction.objects.first()
        p.true_disease = p.predicted_disease; p.save()
        ov = stats.overview()
        self.assertEqual(ov["total"], 2)
        self.assertEqual(ov["reviewed"], 1)
        self.assertEqual(ov["field_accuracy"], 1.0)
        self.assertEqual(sum(r["predictions"] for r in stats.by_disease()), 2)
        daily = stats.daily_activity(14)
        self.assertEqual(len(daily), 14)
        self.assertEqual(daily[-1]["n"], 2)
        fm = stats.feedback_matrix()
        self.assertEqual(fm["total"], 1)
        self.assertEqual(sum(c["count"] for row in fm["rows"] for c in row["cells"] if c["diagonal"]), 1)
        self.assertEqual(self.client.get(reverse("detector:dashboard")).status_code, 200)

    # --- pages & API -----------------------------------------------------------------------
    def test_reference_pages(self):
        for name in ("diseases", "dashboard", "about", "history", "login", "signup"):
            self.assertEqual(self.client.get(reverse(f"detector:{name}")).status_code, 200, name)
        self.assertContains(self.client.get(reverse("detector:about")), "98,067")

    def test_api_predict(self):
        r = self.client.post(reverse("detector:api_predict"), {"image": leaf_photo()})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn(data["label"], data["probabilities"])
        self.assertFalse(Prediction.objects.exists())          # stateless

    def test_api_predict_rejects_bad_input(self):
        r = self.client.post(reverse("detector:api_predict"), {})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["error"], "invalid_image")

    def test_api_stats_and_health(self):
        self.upload()
        s = self.client.get(reverse("detector:api_stats")).json()
        self.assertEqual(s["total_predictions"], 1)
        h = self.client.get(reverse("detector:api_health")).json()
        self.assertEqual(h["status"], "ok")
        self.assertEqual(h["parameters"], 98067)
