from __future__ import annotations

from django import forms
from django.conf import settings
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

from .models import Disease

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
MIN_SIDE = 32


class LeafUploadForm(forms.Form):
    image = forms.ImageField(
        label="Leaf photo",
        widget=forms.ClearableFileInput(attrs={"accept": "image/jpeg,image/png,image/webp,image/bmp",
                                               "class": "dropzone-input"}),
        error_messages={
            "required": "Choose a photo of a rice leaf to diagnose.",
            "invalid_image": "That file isn't a readable image. Upload a JPG, PNG, WebP or BMP photo.",
        },
    )

    def clean_image(self):
        image = self.cleaned_data["image"]
        limit = settings.MAX_UPLOAD_MB * 1024 * 1024
        if image.size > limit:
            raise forms.ValidationError(
                f"The photo is {image.size / 1024 / 1024:.1f} MB. Upload one under {settings.MAX_UPLOAD_MB} MB.")
        pil = getattr(image, "image", None)  # set by Django's ImageField after Pillow verified the file
        if pil is not None:
            if pil.format not in ALLOWED_FORMATS:
                raise forms.ValidationError(f"{pil.format} images aren't supported. Upload a JPG, PNG, WebP or BMP.")
            if min(pil.size) < MIN_SIDE:
                raise forms.ValidationError(
                    f"The photo is only {pil.size[0]}×{pil.size[1]} pixels. Upload one at least {MIN_SIDE} pixels on each side.")
        return image


class FeedbackForm(forms.Form):
    true_disease = forms.ModelChoiceField(
        queryset=Disease.objects.all(), widget=forms.RadioSelect, empty_label=None,
        label="What does this leaf actually have?",
        error_messages={"required": "Choose the disease you believe the leaf has."},
    )
    note = forms.CharField(max_length=500, required=False, label="Note (optional)",
                           widget=forms.Textarea(attrs={"rows": 2, "class": "form-control",
                                                        "placeholder": "e.g. confirmed by the extension officer"}))


class SignUpForm(UserCreationForm):
    email = forms.EmailField(required=False, help_text="Optional. Only used to recover your account.")

    class Meta:
        model = User
        fields = ("username", "email")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs.setdefault("class", "form-control")
