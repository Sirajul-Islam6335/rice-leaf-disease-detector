"""Data migration: fill the detector_disease table with the three diseases the model knows."""
from django.db import migrations


def seed(apps, schema_editor):
    from detector.seeding import seed_diseases
    seed_diseases(apps.get_model("detector", "Disease"))


def unseed(apps, schema_editor):
    apps.get_model("detector", "Disease").objects.all().delete()


class Migration(migrations.Migration):
    dependencies = [("detector", "0001_initial")]
    operations = [migrations.RunPython(seed, unseed)]
