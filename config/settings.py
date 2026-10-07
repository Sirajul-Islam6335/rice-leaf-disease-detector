"""Django settings for the Rice Leaf Disease Detector.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(int(default))).strip().lower() in {"1", "true", "yes", "on"}


def env_list(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, default).split(",") if item.strip()]


# Render sets RENDER=true and RENDER_EXTERNAL_HOSTNAME (e.g. "riceleaf.onrender.com")
# on every service automatically.
ON_RENDER = "RENDER" in os.environ
RENDER_EXTERNAL_HOSTNAME = os.environ.get("RENDER_EXTERNAL_HOSTNAME", "")


# --- Core -----------------------------------------------------------------
# Debug is off by default on Render, so a missing env var never exposes debug pages.
DEBUG = env_bool("DJANGO_DEBUG", not ON_RENDER)
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "")
if not SECRET_KEY:
    if not DEBUG:
        raise RuntimeError("Set DJANGO_SECRET_KEY when DJANGO_DEBUG is off.")
    SECRET_KEY = "dev-only-insecure-key-change-me"

ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1],testserver")
CSRF_TRUSTED_ORIGINS = env_list("DJANGO_CSRF_TRUSTED_ORIGINS", "")

# Trust the service's own onrender.com address without having to set it by hand.
if RENDER_EXTERNAL_HOSTNAME:
    ALLOWED_HOSTS.append(RENDER_EXTERNAL_HOSTNAME)
    CSRF_TRUSTED_ORIGINS.append(f"https://{RENDER_EXTERNAL_HOSTNAME}")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.humanize",
    "detector.apps.DetectorConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "detector.context_processors.site",
            ],
        },
    },
]

# SQLite by default (zero setup); set DB_ENGINE=postgres to use PostgreSQL.
if os.environ.get("DB_ENGINE", "sqlite").lower() in {"postgres", "postgresql"}:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.environ.get("DB_NAME", "riceleaf"),
            "USER": os.environ.get("DB_USER", "riceleaf"),
            "PASSWORD": os.environ.get("DB_PASSWORD", ""),
            "HOST": os.environ.get("DB_HOST", "localhost"),
            "PORT": os.environ.get("DB_PORT", "5432"),
            "CONN_MAX_AGE": 60,
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": Path(os.environ.get("DJANGO_DB_PATH", BASE_DIR / "db.sqlite3")),
        }
    }

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-gb"
TIME_ZONE = os.environ.get("DJANGO_TIME_ZONE", "Europe/London")
USE_I18N = True
USE_TZ = True
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- Static & media ------------------------------------------------------------
STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {
        "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage" if DEBUG
        else "whitenoise.storage.CompressedManifestStaticFilesStorage"
    },
}
MEDIA_URL = "/media/"
MEDIA_ROOT = Path(os.environ.get("DJANGO_MEDIA_ROOT", BASE_DIR / "media"))
# Serve uploaded images through Django. Fine for a single-server deployment;
# behind nginx or with cloud storage, set this to 0 and serve /media/ there.
SERVE_MEDIA = env_bool("DJANGO_SERVE_MEDIA", True)

# Uploads: reject anything larger than this before it reaches the view.
MAX_UPLOAD_MB = int(os.environ.get("RICELEAF_MAX_UPLOAD_MB", "10"))
DATA_UPLOAD_MAX_MEMORY_SIZE = (MAX_UPLOAD_MB + 1) * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = (MAX_UPLOAD_MB + 1) * 1024 * 1024

# --- Model ----------------------------------------------------------------------
RICELEAF = {
    "MODEL_PATH": Path(os.environ.get("RICELEAF_MODEL_PATH", BASE_DIR / "ml_models" / "model.npz")),
    "CV_RESULTS_PATH": BASE_DIR / "ml_models" / "cv_results.json",
    # Below this top-class probability the result is flagged for expert review.
    "UNCERTAINTY_THRESHOLD": float(os.environ.get("RICELEAF_UNCERTAINTY_THRESHOLD", "0.6")),
    "USE_TTA": env_bool("RICELEAF_USE_TTA", True),
}

# --- Security (production) -----------------------------------------------------
if not DEBUG:
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    SECURE_SSL_REDIRECT = env_bool("DJANGO_SECURE_SSL_REDIRECT", True)
    # Secure cookies need HTTPS. Set DJANGO_SECURE_COOKIES=0 only for plain-HTTP local runs (e.g. docker compose).
    SESSION_COOKIE_SECURE = env_bool("DJANGO_SECURE_COOKIES", True)
    CSRF_COOKIE_SECURE = SESSION_COOKIE_SECURE
    SECURE_HSTS_SECONDS = int(os.environ.get("DJANGO_HSTS_SECONDS", "0"))
    SECURE_CONTENT_TYPE_NOSNIFF = True
    X_FRAME_OPTIONS = "DENY"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"simple": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "simple"}},
    "loggers": {
        "detector": {"handlers": ["console"], "level": os.environ.get("DJANGO_LOG_LEVEL", "INFO")},
        "django": {"handlers": ["console"], "level": "WARNING"},
    },
}

# --- Accounts -------------------------------------------------------------------
LOGIN_URL = "detector:login"
LOGIN_REDIRECT_URL = "detector:history"
LOGOUT_REDIRECT_URL = "detector:home"