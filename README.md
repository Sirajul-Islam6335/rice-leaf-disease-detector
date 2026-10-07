# RiceLeaf: rice leaf disease detection web app

A Django web application that diagnoses **Bacterial leaf blight**, **Brown spot** and **Leaf smut** from a photo of a rice leaf. The model is a convolutional neural network written **from scratch in NumPy** (`riceleaf/`):

**Live project Link** : https://riceleaf-nywp.onrender.com


**Stack:** Python 3.8+ · Django 5.2 (4.2 LTS on Python 3.8/3.9) · HTML templates · Bootstrap 5 · custom CSS · vanilla JavaScript · SQL (SQLite by default, PostgreSQL supported) · NumPy, Convulation Neural Network

---

## What the app does

| Page | URL | What it does |
|---|---|---|
| Diagnose | `/` | Drag-and-drop or choose a leaf photo; or try one of three sample photos |
| Result | `/diagnosis/<id>/` | Predicted disease, probability for every class, low-confidence warning, symptoms and management advice, feedback form, delete |
| My diagnoses | `/history/` | Your diagnoses with photo strips, filters by disease and "needs review", pagination |
| Diseases | `/diseases/` | Profile of each disease, read from the `Disease` SQL table |
| Dashboard | `/dashboard/` | Usage statistics and a model-vs-user-feedback confusion matrix, computed with **raw SQL** |
| How it works | `/model/` | Architecture, measured accuracy, limits, API usage |
| Accounts | `/accounts/signup/`, `/accounts/login/` | Sign up / in. Diagnoses made before signing in are moved into the account |
| Admin | `/admin/` | Manage diseases, predictions and users |
| JSON API | `/api/predict/`, `/api/stats/`, `/api/health/` | Diagnose a photo, read statistics, check model status |

**Privacy rules.** Each diagnosis belongs to the signed-in user, or to the browser session for anonymous visitors. Nobody else can open or delete it; staff can see everything. Requests for someone else's diagnosis return 404, so their existence isn't revealed.

---

## Quick start (SQLite, no setup)

Works on **Python 3.8 to 3.13**. `pip` installs Django 5.2 on Python 3.10+ and Django 4.2 LTS on Python 3.8/3.9 automatically; the app and its tests run on both.

**Windows (Command Prompt or PowerShell)**

```bat
cd rice_leaf_webapp
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser   :: optional, for /admin/
python manage.py runserver
```

**macOS / Linux**

```bash
cd rice_leaf_webapp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python manage.py migrate           # creates the tables and fills the Disease table
python manage.py createsuperuser   # optional, for /admin/
python manage.py runserver
```

Open http://127.0.0.1:8000. A trained model is included at `ml_models/model.npz`, so diagnosis works immediately.

> **Seeing `Could not find a version that satisfies the requirement Django`?** Your `pip` is reading an old
> copy of `requirements.txt` that only allowed Django 5 (Python 3.10+). Use the one in this folder, which
> selects the right Django for your Python. Python 3.8 is past end-of-life, so for a long-lived deployment
> Python 3.12 is the better choice.

### With PostgreSQL

```bash
docker compose up --build          # PostgreSQL 16 + the app on http://localhost:8000
```

Or point the app at an existing server:

```bash
export DB_ENGINE=postgres DB_NAME=riceleaf DB_USER=riceleaf DB_PASSWORD=... DB_HOST=localhost DB_PORT=5432
python manage.py migrate
```

All configuration is through environment variables; see [`.env.example`](.env.example).

### Tests

```bash
python manage.py test detector     # 20 tests, ~10 s
```

The suite passes on Python 3.8 and 3.9 (Django 4.2) and Python 3.13 (Django 5.2) with SQLite, and on Python 3.13 against PostgreSQL 16.

The tests cover upload validation, the full diagnosis flow, ownership and privacy rules, sign-up and sign-in claiming earlier diagnoses, feedback, deletion (including the image file), every SQL statistics query, the database `CHECK` constraint, and the API.

---

## Project structure

```
rice_leaf_webapp/
├── manage.py
├── config/                    # Django project: settings (env-driven), urls, wsgi/asgi
├── detector/                  # the Django app
│   ├── models.py              # Disease and Prediction tables (FKs, indexes, CHECK constraint)
│   ├── migrations/            # 0001 schema, 0002 data migration that seeds the diseases
│   ├── disease_data.py        # seed content for the Disease table
│   ├── services.py            # loads the NumPy model once per process, runs + stores diagnoses
│   ├── stats.py               # dashboard statistics written in raw, parameterised SQL
│   ├── views.py, urls.py      # pages, accounts, JSON API, error handlers
│   ├── forms.py               # upload validation, feedback, sign-up
│   ├── admin.py               # admin with photo thumbnails and filters
│   ├── tests.py               # 20 end-to-end tests
│   ├── management/commands/   # seed_diseases, train_model
│   ├── templates/             # HTML (Bootstrap 5) - base, home, result, history, dashboard, ...
│   └── static/detector/       # css/style.css, js/app.js, sample photos, vendored Bootstrap + fonts
├── riceleaf/                  # the CNN engine in pure NumPy (layers, backprop, AdamW, training)
├── ml_models/                 # model.npz (trained weights + metadata), cv_results.json
├── sql/                       # schema_sqlite.sql, schema_postgres.sql, queries.sql
├── Dockerfile, docker-compose.yml, requirements.txt, .env.example
```

---

## The database

```
auth_user (Django)          detector_disease                     detector_prediction
──────────────────          ──────────────────────               ─────────────────────────────────────
id  PK  ◄──────────┐        id            PK  ◄────────┐         id                  UUID PK
username           │        name          UNIQUE       ├──────── predicted_disease_id FK NOT NULL
...                │        slug          UNIQUE       └──────── true_disease_id      FK NULL (user feedback)
                   └─────────────────────────────────────────── user_id             FK NULL
                            pathogen, kind, color                image, original_filename, source
                            summary, symptoms,                   confidence  CHECK (0 <= confidence <= 1)
                            favoured_by, management,             probabilities JSON, uncertain
                            looks_like, display_order            inference_ms, model_version
                                                                 feedback_note, feedback_at
                                                                 created_at  (indexed)
```

* **`detector_disease`** holds reference data; the class names stored in the model checkpoint must match `name`. It is filled by a data migration and can be edited in the admin.
* **`detector_prediction`** has one row per diagnosed photo. `true_disease_id` records what the user says the leaf really has, which turns real use into an accuracy measurement.
* There are composite indexes on `(predicted_disease_id, created_at)` and `(user_id, created_at)`, matching the history and dashboard queries.
* The full DDL for both databases is in [`sql/`](sql/).

**Raw SQL.** The dashboard queries in [`detector/stats.py`](detector/stats.py) use plain SQL through `connection.cursor()`: `LEFT JOIN`, `GROUP BY`, `CASE`, `COALESCE` and `DATE`. All values are passed as parameters, never formatted into the string. The same queries are in [`sql/queries.sql`](sql/queries.sql) for running in `sqlite3` or `psql`. Everything else uses the Django ORM.

---

## JSON API

```bash
curl -F "image=@leaf.jpg" http://127.0.0.1:8000/api/predict/
```

```json
{"label": "Brown spot", "confidence": 0.9961, "uncertain": false,
 "probabilities": {"Brown spot": 0.9961, "Leaf smut": 0.0028, "Bacterial leaf blight": 0.0011},
 "inference_ms": 88.4, "model_version": "RiceLeafNet 2.0.0, trained 2026-10-06"}
```

* `POST /api/predict/` is stateless: nothing is stored. It returns 400 for an invalid image and 503 if the model is missing.
* `GET /api/stats/` returns aggregate usage figures.
* `GET /api/health/` returns model status, class names, input size and parameter count.

---

## The model

* **Architecture:** `RiceLeafNet` has 4 blocks of [3×3 convolution → batch normalisation → ReLU → 2×2 max pooling], then global average pooling, dropout and a dense layer. That is **98,067 parameters**, and every forward and backward pass is hand-written in NumPy.
* **Input:** 80 × 256 pixels, which keeps the long shape of a leaf.
* **Accuracy:** 83.3 % on a held-out test set of 18 photos; **85.8 % ± 14.3 %** in 5-fold cross-validation over all 120 photos.
* **Inference:** about 70–90 ms per photo on a CPU, including flip test-time augmentation.
* **Low-confidence flag:** results below 60 % confidence are flagged for review. Set `RICELEAF_UNCERTAINTY_THRESHOLD` to change this.

To retrain on new data and install the new model:

```bash
python manage.py train_model --data-dir path/to/rice_leaf_diseases --cv
```

The command trains, evaluates, copies `model.npz` (and the cross-validation results) into `ml_models/`, and refuses to install a model whose classes have no matching `Disease` rows.

**Limits.**
* The model was trained on 120 photos of single leaves on plain backgrounds.
* It only knows three diseases, so a healthy leaf is still assigned to one of them, usually with low confidence.
* It is a screening aid; growers should confirm with their local extension service.

---

## Deployment notes

* Set `DJANGO_DEBUG=0`, a long random `DJANGO_SECRET_KEY`, `DJANGO_ALLOWED_HOSTS` and `DJANGO_CSRF_TRUSTED_ORIGINS`.
* With debug off, the app turns on secure cookies, SSL redirect and nosniff/frame-deny headers. To enable HSTS once the site is HTTPS-only, set `DJANGO_HSTS_SECONDS`.
* Static files are fingerprinted and served by WhiteNoise. Bootstrap, Bootstrap Icons and the IBM Plex fonts are self-hosted, so there are no CDN calls.
* Uploaded photos are served by Django (`DJANGO_SERVE_MEDIA=1`), which suits a single server. Behind nginx or with cloud storage, set it to `0` and serve `/media/` there.
* The Docker image runs migrations, then gunicorn with 2 workers. Each worker loads the model once; the weights are under 1 MB, so memory use is dominated by Python and NumPy themselves.

## Credits

Dataset: J. Shah, H. Prajapati, V. Dabhi (2017), *Rice Leaf Diseases*, UCI Machine Learning Repository, https://doi.org/10.24432/C5R013 (CC BY 4.0). Sample photos in `detector/static/detector/img/samples/` are resized copies of images from this dataset's held-out test split.

Bootstrap and Bootstrap Icons are MIT-licensed; IBM Plex is under the SIL Open Font License (licence files are included next to the vendored files).
