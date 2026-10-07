from __future__ import annotations

import logging

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView
from django.contrib.staticfiles import finders
from django.core.paginator import Paginator
from django.http import Http404, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from . import stats
from .forms import FeedbackForm, LeafUploadForm, SignUpForm
from .models import Disease, Prediction
from .services import (ModelUnavailable, claim_session_predictions, diagnose_and_store, diagnose_file,
                       get_predictor, model_card)

log = logging.getLogger(__name__)
MODEL_MISSING = "The diagnosis model isn't installed on this server. Add ml_models/model.npz and restart."


def _colors() -> dict[str, str]:
    return dict(Disease.objects.values_list("name", "color"))


def _own_prediction(request, pk) -> Prediction:
    """Fetch a prediction the visitor is allowed to see, or 404 (never reveal that it exists)."""
    return get_object_or_404(Prediction.objects.visible_to(request).select_related("predicted_disease", "true_disease"),
                             pk=pk)


# --- Diagnosis ------------------------------------------------------------------
@require_http_methods(["GET", "POST"])
def home(request):
    form = LeafUploadForm(request.POST or None, request.FILES or None)
    if request.method == "POST" and form.is_valid():
        image = form.cleaned_data["image"]
        try:
            prediction = diagnose_and_store(image, image.name, request=request)
        except ModelUnavailable:
            log.exception("Model unavailable")
            messages.error(request, MODEL_MISSING)
        else:
            return redirect(prediction)
    return render(request, "detector/home.html", {
        "form": form,
        "diseases": Disease.objects.all(),
        "recent": Prediction.objects.visible_to(request).select_related("predicted_disease")[:3],
        "max_mb": settings.MAX_UPLOAD_MB,
    })


@require_POST
def predict_sample(request, slug):
    """Diagnose one of the bundled sample photos, so visitors can try the tool without their own image."""
    disease = get_object_or_404(Disease, slug=slug)
    path = finders.find(disease.sample_image) if disease.sample_image else None
    if not path:
        raise Http404("No sample photo for this disease")
    with open(path, "rb") as fh:
        try:
            prediction = diagnose_and_store(fh, f"sample-{slug}.jpg", request=request, source=Prediction.Source.SAMPLE)
        except ModelUnavailable:
            messages.error(request, MODEL_MISSING)
            return redirect("detector:home")
    return redirect(prediction)


@require_http_methods(["GET", "POST"])
def prediction_detail(request, pk):
    p = _own_prediction(request, pk)
    initial = {"true_disease": p.true_disease or p.predicted_disease, "note": p.feedback_note}
    form = FeedbackForm(request.POST or None, initial=initial)
    if request.method == "POST" and form.is_valid():
        p.true_disease = form.cleaned_data["true_disease"]
        p.feedback_note = form.cleaned_data["note"]
        p.feedback_at = timezone.now()
        p.save(update_fields=["true_disease", "feedback_note", "feedback_at"])
        verdict = "confirmed" if p.is_correct else f"corrected to {p.true_disease}"
        messages.success(request, f"Feedback saved: diagnosis {verdict}.")
        return redirect(p)
    return render(request, "detector/result.html", {
        "p": p,
        "disease": p.predicted_disease,
        "bars": p.ranked_probabilities(_colors()),
        "form": form,
        "threshold": settings.RICELEAF["UNCERTAINTY_THRESHOLD"],
    })


@require_POST
def prediction_delete(request, pk):
    p = _own_prediction(request, pk)
    p.delete()
    messages.success(request, "Diagnosis and photo deleted.")
    return redirect("detector:history")


def history(request):
    qs = Prediction.objects.visible_to(request).select_related("predicted_disease", "true_disease")
    slug = request.GET.get("disease", "")
    active = Disease.objects.filter(slug=slug).first() if slug else None
    if active:
        qs = qs.filter(predicted_disease=active)
    if request.GET.get("review") == "1":
        qs = qs.filter(uncertain=True)
    page = Paginator(qs, 12).get_page(request.GET.get("page"))
    return render(request, "detector/history.html", {
        "page": page,
        "active": active,
        "review_only": request.GET.get("review") == "1",
        "diseases": Disease.objects.all(),
        "summary": stats.user_summary(request.user.id) if request.user.is_authenticated else None,
    })


# --- Reference pages ------------------------------------------------------------
def diseases(request):
    return render(request, "detector/diseases.html", {"diseases": Disease.objects.all()})


def dashboard(request):
    return render(request, "detector/dashboard.html", {
        "overview": stats.overview(),
        "by_disease": stats.by_disease(),
        "daily": stats.daily_activity(14),
        "feedback": stats.feedback_matrix(),
        "review_queue": Prediction.objects.visible_to(request).filter(uncertain=True, true_disease__isnull=True)
                                         .select_related("predicted_disease")[:5],
    })


def about(request):
    try:
        card = model_card()
    except ModelUnavailable:
        card = None
    return render(request, "detector/about.html", {"card": card, "colors": _colors()})


# --- Accounts -------------------------------------------------------------------
class SignInView(LoginView):
    template_name = "registration/login.html"
    redirect_authenticated_user = True

    def form_valid(self, form):
        session_ids = self.request.session.get("my_predictions", [])
        response = super().form_valid(form)                    # logs in (session key is rotated)
        self.request.session["my_predictions"] = session_ids
        n = claim_session_predictions(self.request, form.get_user())
        if n:
            messages.info(self.request, f"{n} diagnos{'is' if n == 1 else 'es'} from this visit saved to your account.")
        return response


@require_POST
def signout(request):
    """Log out (POST only, so a link or image on another site can't sign people out)."""
    logout(request)
    messages.info(request, "You're signed out.")
    return redirect("detector:home")


@require_http_methods(["GET", "POST"])
def signup(request):
    if request.user.is_authenticated:
        return redirect("detector:history")
    form = SignUpForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        session_ids = request.session.get("my_predictions", [])
        user = form.save()
        login(request, user)                                   # rotates the session key
        request.session["my_predictions"] = session_ids
        n = claim_session_predictions(request, user)
        messages.success(request, f"Account created. {n} earlier diagnos{'is' if n == 1 else 'es'} saved to it."
                         if n else "Account created. Your diagnoses will now be saved to it.")
        return redirect("detector:history")
    return render(request, "registration/signup.html", {"form": form})


# --- JSON API ---------------------------------------------------------------------
@csrf_exempt
@require_POST
def api_predict(request):
    """POST multipart/form-data with an ``image`` field. Stateless: nothing is stored."""
    form = LeafUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        return JsonResponse({"error": "invalid_image", "details": form.errors.get_json_data()}, status=400)
    try:
        d = diagnose_file(form.cleaned_data["image"])
    except ModelUnavailable:
        return JsonResponse({"error": "model_unavailable"}, status=503)
    return JsonResponse({
        "label": d.label,
        "confidence": round(d.confidence, 4),
        "uncertain": d.uncertain,
        "probabilities": {k: round(v, 4) for k, v in d.probabilities.items()},
        "inference_ms": d.inference_ms,
        "model_version": d.model_version,
    })


@require_GET
def api_stats(request):
    ov = stats.overview()
    return JsonResponse({
        "total_predictions": ov["total"],
        "average_confidence": round(ov["avg_confidence"], 4),
        "flagged_for_review": ov["uncertain"],
        "with_feedback": ov["reviewed"],
        "feedback_accuracy": None if ov["field_accuracy"] is None else round(ov["field_accuracy"], 4),
        "by_disease": [{"disease": r["name"], "predictions": r["predictions"],
                        "average_confidence": None if r["avg_confidence"] is None else round(r["avg_confidence"], 4)}
                       for r in stats.by_disease()],
    })


@require_GET
def api_health(request):
    try:
        predictor = get_predictor()
    except ModelUnavailable as exc:
        return JsonResponse({"status": "error", "detail": str(exc)}, status=503)
    return JsonResponse({"status": "ok", "classes": predictor.class_names,
                         "input_size": list(predictor.size), "parameters": predictor.model.num_params()})


# --- Errors -----------------------------------------------------------------------
def error_404(request, exception):
    return render(request, "detector/error.html", {
        "code": 404, "title": "Page not found",
        "message": "This page doesn't exist, or the diagnosis was deleted or belongs to another account."}, status=404)


def error_500(request):
    return render(request, "detector/error.html", {
        "code": 500, "title": "Something went wrong",
        "message": "The server hit an error while handling this request. Try again, or diagnose a different photo."},
        status=500)
