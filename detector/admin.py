from __future__ import annotations

from django.contrib import admin
from django.utils.html import format_html

from .models import Disease, Prediction


@admin.register(Disease)
class DiseaseAdmin(admin.ModelAdmin):
    list_display = ("name", "swatch", "kind", "pathogen", "prediction_count", "display_order")
    list_editable = ("display_order",)
    prepopulated_fields = {"slug": ("name",)}
    search_fields = ("name", "pathogen")

    @admin.display(description="Colour")
    def swatch(self, obj):
        return format_html('<span style="display:inline-block;width:2.5rem;height:.9rem;border-radius:9px;'
                           'background:{}"></span>', obj.color)

    @admin.display(description="Predictions")
    def prediction_count(self, obj):
        return obj.predictions.count()


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ("thumbnail", "predicted_disease", "confidence_pct", "uncertain", "true_disease", "user",
                    "source", "created_at")
    list_filter = ("predicted_disease", "uncertain", "true_disease", "source", "created_at")
    list_select_related = ("predicted_disease", "true_disease", "user")
    search_fields = ("original_filename", "user__username", "feedback_note")
    readonly_fields = ("id", "preview", "predicted_disease", "confidence", "probabilities", "uncertain",
                       "inference_ms", "model_version", "source", "created_at")
    fields = ("id", "preview", "user", "original_filename", "source", "predicted_disease", "confidence",
              "probabilities", "uncertain", "inference_ms", "model_version", "true_disease", "feedback_note",
              "feedback_at", "created_at")
    date_hierarchy = "created_at"

    @admin.display(description="Photo")
    def thumbnail(self, obj):
        return format_html('<img src="{}" style="height:32px;width:110px;object-fit:cover;border-radius:16px">',
                           obj.image.url)

    @admin.display(description="Preview")
    def preview(self, obj):
        return format_html('<img src="{}" style="max-width:640px;width:100%">', obj.image.url)

    @admin.display(description="Confidence", ordering="confidence")
    def confidence_pct(self, obj):
        return f"{obj.confidence:.0%}"


admin.site.site_header = "RiceLeaf administration"
admin.site.site_title = "RiceLeaf admin"
