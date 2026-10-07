from __future__ import annotations

def site(request):
    """Values every template needs (site name, active navigation item)."""
    match = getattr(request, "resolver_match", None)
    nav = match.url_name if match else ""
    if nav == "prediction_detail":
        nav = "history"
    return {"site_name": "RiceLeaf", "nav": nav}
