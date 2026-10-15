"""Sample panel for host tests. It is not registered as an installed entry point."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

router = APIRouter()


@router.get("/ping")
def ping() -> dict:
    return {"ok": True, "panel": "sample_panel"}


class SamplePanel:
    name = "sample_panel"
    version = "0.1.0"
    api_version = "0.2.0"
    title = "Sample notes"
    slot = "more"
    order = 10
    route = "#/sample"
    icon = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="none" '
        'stroke="currentColor" stroke-width="1.6"><path d="M5 4h10v12H5z"/>'
        '<path d="M8 8h4M8 11h4"/></svg>'
    )
    assets = Path(__file__).resolve().parent / "sample_panel_assets"
    api_prefix = "/api/plugins/sample_panel"
    router = router

    def status(self) -> dict:
        return {
            "state": "ready",
            "checks": [
                {
                    "id": "local",
                    "label": "Sample files",
                    "ok": True,
                    "detail": "Bundled with the tests",
                    "fix": "",
                }
            ],
        }


sample_panel = SamplePanel()
