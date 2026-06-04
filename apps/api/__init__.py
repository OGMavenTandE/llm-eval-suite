"""Local FastAPI shell for the Offline AI Evaluation Workbench."""

from apps.api.main import app, create_app, run_server

__all__ = ["app", "create_app", "run_server"]
