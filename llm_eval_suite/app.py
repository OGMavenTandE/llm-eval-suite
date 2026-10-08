"""Local web app. Bind to 127.0.0.1 and serve the static UI from this package."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from llm_eval.offline import apply_startup_offline

apply_startup_offline()

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from llm_eval.models.nanogpt_convert import convert_nanogpt_to_hf
from llm_eval.offline import (
    load_settings,
    offline_status,
    refuse_remote_http,
    write_settings_offline,
)
from llm_eval_suite.connections import (
    ConnectionStore,
    build_model,
    detect_path,
    judge_max_tokens,
    judge_timeout,
    test_connection,
)
from llm_eval_suite.presets import demo_pair, list_presets
from llm_eval_suite.runs import RunManager
from llm_eval_suite.scoring import PASS_BAR
from llm_eval_suite.suites import validate_factcheck_text

STATIC_DIR = Path(__file__).parent / "static"
REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATASET = REPO_ROOT / "datasets" / "sample_factcheck_50.jsonl"


class ConnectionIn(BaseModel):
    id: str | None = None
    name: str = ""
    type: str = "openai"
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    mode: str = "chat"
    max_context: int | None = None
    folder: str = ""
    cloud: bool = False
    hub: bool = False
    max_new_tokens: int | None = None
    precision: str = ""
    thinking_max_tokens: int | None = None
    trust_remote_code: bool = False
    use_chat_template: bool | str | None = None


class TestIn(BaseModel):
    profile_id: str | None = None
    profile: ConnectionIn | None = None
    prompt: str | None = None


class DetectIn(BaseModel):
    path: str


class ConvertIn(BaseModel):
    ckpt_path: str
    out_dir: str


class DatasetIn(BaseModel):
    filename: str = "upload.jsonl"
    text: str


class JudgesIn(BaseModel):
    chairman: str = ""
    judges: list[dict] = Field(default_factory=list)
    timeout: int | None = None
    max_tokens: int | None = None


class OfflineIn(BaseModel):
    offline: bool = False


class RunIn(BaseModel):
    connection_id: str
    preset: str
    dataset_id: str = "sample"
    resume_run_id: str | None = None


class DemoIn(BaseModel):
    folder: str | None = None
    preset: str | None = None


def create_app(
    data_dir: str | Path | None = None,
    runs_dir: str | Path | None = None,
    model_factory=None,
    sample_dataset: str | Path | None = None,
) -> FastAPI:
    data = Path(data_dir or os.environ.get("LLM_EVAL_DATA_DIR") or "data")
    runs = Path(runs_dir or os.environ.get("LLM_EVAL_RUNS_DIR") or "runs")
    sample = Path(sample_dataset) if sample_dataset else SAMPLE_DATASET
    app = FastAPI(title="LLM Eval Suite", version="0.3.0")
    load_settings(data)
    app.state.store = ConnectionStore(data)
    app.state.runs = RunManager(runs, model_factory=model_factory, timing_path=data / "timing.json")
    app.state.sample_dataset = sample
    app.state.datasets = data / "datasets"
    app.state.datasets.mkdir(parents=True, exist_ok=True)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index():
        page = STATIC_DIR / "index.html"
        if not page.is_file():
            raise HTTPException(status_code=404, detail="UI is missing")
        return FileResponse(page)

    @app.get("/favicon.ico")
    def favicon():
        icon = STATIC_DIR / "favicon.ico"
        if not icon.is_file():
            raise HTTPException(status_code=404, detail="Favicon is missing")
        return FileResponse(icon, media_type="image/x-icon")

    @app.get("/api/dow/export")
    def dow_export():
        from dow_bench.export import write_export

        destination = runs / "dow_leaderboard.csv"
        paths = write_export(runs, destination)
        return FileResponse(paths["csv"], media_type="text/csv", filename="dow_leaderboard.csv")

    @app.get("/api/health")
    def health():
        return {"status": "ok", **offline_status(app.state.store.data_dir)}

    @app.get("/api/offline")
    def get_offline():
        return offline_status(app.state.store.data_dir)

    @app.put("/api/offline")
    def put_offline(body: OfflineIn):
        try:
            return write_settings_offline(app.state.store.data_dir, body.offline)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/presets")
    def presets(dataset_id: str = "sample", connection_id: str | None = None):
        rows = _dataset_rows(app, dataset_id)
        profile = app.state.store.get(connection_id) if connection_id else None
        suite_rates = {}
        if profile is not None and app.state.runs.timing is not None:
            for suite_name in ("garak", "factcheck", "robustness", "consistency"):
                rate = app.state.runs.timing.rate_for(profile, suite_name)
                if rate is not None:
                    suite_rates[suite_name] = rate
        return {
            "presets": list_presets(
                dataset_rows=rows,
                suite_rates=suite_rates,
            )
        }

    @app.get("/api/demo")
    def demo():
        saved = _read_demo(app)
        pair = demo_pair(folder=saved.get("folder"))
        return pair

    @app.post("/api/demo/start")
    def start_demo(body: DemoIn):
        pair = demo_pair(folder=body.folder)
        preset_id = body.preset or pair["preset"]
        _write_demo(app, pair["model_a"].get("folder") or "")
        dataset_path = _dataset_path(app, "sample")
        started = []
        for model in (pair["model_a"], pair["model_b"]):
            saved = app.state.store.save(
                {
                    "name": model.get("name") or model.get("model"),
                    "type": model.get("type") or "hf",
                    "model": model.get("model") or "",
                    "folder": model.get("folder") or "",
                    "mode": model.get("mode") or "completions",
                    "max_context": model.get("max_context") or 1024,
                    "hub": bool(model.get("hub")),
                    "base_url": "",
                }
            )
            profile = app.state.store.get(saved["id"])
            run = app.state.runs.start(
                connection=profile,
                preset_id=preset_id,
                dataset_path=str(dataset_path),
                background=True,
                offline=bool(pair.get("offline")),
            )
            started.append({"connection_id": saved["id"], "run_id": run["run_id"], "model": profile.get("model")})
        return {
            "preset": preset_id,
            "label": pair["label"],
            "model_a": pair["model_a"],
            "model_b": pair["model_b"],
            "runs": started,
        }

    @app.get("/api/connections")
    def connections():
        return {"connections": app.state.store.list_profiles()}

    @app.post("/api/connections")
    def save_connection(body: ConnectionIn):
        try:
            saved = app.state.store.save(body.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return saved

    @app.delete("/api/connections/{profile_id}")
    def delete_connection(profile_id: str):
        if not app.state.store.delete(profile_id):
            raise HTTPException(status_code=404, detail="Connection not found")
        return {"ok": True}

    @app.post("/api/connections/test")
    def test_connection_route(body: TestIn):
        profile = _resolve_profile(app, body.profile_id, body.profile)
        model = None
        if app.state.runs.model_factory is not None:
            model = app.state.runs.model_factory(profile)
        return test_connection(profile, model=model, prompt=body.prompt)

    @app.post("/api/connections/detect")
    def detect(body: DetectIn):
        return detect_path(body.path)

    @app.post("/api/connections/convert")
    def convert(body: ConvertIn):
        try:
            destination = convert_nanogpt_to_hf(body.ckpt_path, body.out_dir)
        except (RuntimeError, ValueError, OSError) as exc:
            report_path = Path(body.out_dir) / "conversion_report.json"
            detail = str(exc)
            if report_path.is_file():
                detail = f"{detail} Report: {report_path}"
            raise HTTPException(status_code=400, detail=detail) from exc
        report_path = destination / "conversion_report.json"
        report = None
        if report_path.is_file():
            import json

            report = json.loads(report_path.read_text(encoding="utf-8"))
        return {"ok": True, "folder": str(destination), "report": report}

    @app.get("/api/ollama/tags")
    def ollama_tags(base_url: str = "http://127.0.0.1:11434"):
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        try:
            refuse_remote_http(root, what="remote Ollama")
            response = requests.get(f"{root}/api/tags", timeout=5)
            response.raise_for_status()
            payload = response.json()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail=f"Could not list Ollama models: {exc}") from exc
        names = []
        for row in payload.get("models") or []:
            name = row.get("name") or row.get("model")
            if name:
                names.append(name)
        return {"models": names}

    @app.get("/api/datasets/sample")
    def sample_info():
        path = app.state.sample_dataset
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Sample fact-check set is missing")
        rows = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        return {"dataset_id": "sample", "path": str(path), "row_count": rows}

    @app.post("/api/datasets/validate")
    def validate_dataset(body: DatasetIn):
        dataset_id = _dataset_id(body.filename)
        destination = app.state.datasets / f"{dataset_id}{Path(body.filename).suffix.lower()}"
        result = validate_factcheck_text(body.filename, body.text, destination)
        result["dataset_id"] = dataset_id if result.get("ok") else None
        return result

    @app.get("/api/judges")
    def get_judges():
        return app.state.store.judges()

    @app.put("/api/judges")
    def put_judges(body: JudgesIn):
        return app.state.store.save_judges(body.model_dump())

    @app.post("/api/runs")
    def start_run(body: RunIn):
        profile = app.state.store.get(body.connection_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Connection not found")
        dataset_path = _dataset_path(app, body.dataset_id)
        try:
            return app.state.runs.start(
                connection=profile,
                preset_id=body.preset,
                dataset_path=str(dataset_path),
                resume_run_id=body.resume_run_id,
                background=True,
            )
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/runs")
    def list_runs():
        return {
            "runs": app.state.runs.list_runs(),
            "pass_bar": PASS_BAR,
            "pass_bar_percent": round(PASS_BAR * 100, 1),
        }

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        try:
            return app.state.runs.get(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc

    @app.get("/api/runs/{run_id}/items")
    def get_items(run_id: str):
        try:
            return {"items": app.state.runs.items(run_id)}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str):
        try:
            return app.state.runs.cancel(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc

    @app.get("/api/runs/{run_id}/report")
    def report(run_id: str, download: int = 0):
        try:
            html = app.state.runs.report_html(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        headers = {}
        if download:
            headers["Content-Disposition"] = f'attachment; filename="eval-report-{run_id}.html"'
        return HTMLResponse(html, headers=headers)

    @app.post("/api/runs/{run_id}/analyze")
    def analyze(run_id: str):
        settings = app.state.store.judges()
        default_timeout = judge_timeout(settings)
        judges = []
        for row in settings.get("judges") or []:
            prepared = dict(row)
            prepared["timeout"] = int(row.get("timeout") or default_timeout)
            judges.append(prepared)
        try:
            return app.state.runs.analyze(
                run_id,
                judges,
                chairman=settings.get("chairman"),
                max_tokens=judge_max_tokens(settings),
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc

    @app.get("/api/compare")
    def compare(left: str, right: str):
        try:
            return app.state.runs.compare(left, right)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


def _resolve_profile(app: FastAPI, profile_id: str | None, profile: ConnectionIn | None) -> dict:
    if profile_id:
        stored = app.state.store.get(profile_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="Connection not found")
        return stored
    if profile is None:
        raise HTTPException(status_code=400, detail="Provide a saved connection or a profile.")
    return profile.model_dump()


def _dataset_id(filename: str) -> str:
    import uuid

    stem = Path(filename).stem or "upload"
    safe = "".join(ch for ch in stem if ch.isalnum() or ch in {"-", "_"})[:40] or "upload"
    return f"{safe}-{uuid.uuid4().hex[:8]}"


def _dataset_rows(app: FastAPI, dataset_id: str) -> int:
    try:
        path = _dataset_path(app, dataset_id)
    except (FileNotFoundError, HTTPException):
        return 50
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _demo_path(app: FastAPI) -> Path:
    return Path(app.state.store.data_dir) / "demo.json"


def _read_demo(app: FastAPI) -> dict:
    path = _demo_path(app)
    if not path.is_file():
        return {}
    import json

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_demo(app: FastAPI, folder: str) -> None:
    import json

    path = _demo_path(app)
    path.write_text(json.dumps({"folder": folder}, indent=2) + "\n", encoding="utf-8")


def _dataset_path(app: FastAPI, dataset_id: str) -> Path:
    if dataset_id == "sample":
        path = Path(app.state.sample_dataset)
        if not path.is_file():
            raise FileNotFoundError("Sample dataset is missing")
        return path
    matches = list(Path(app.state.datasets).glob(f"{dataset_id}.*"))
    if not matches:
        raise FileNotFoundError("Dataset not found")
    return matches[0]


app = create_app()


def open_app_browser(url: str) -> bool:
    """Open the local app in one tab per launch.

    A second call in this process, or a child that inherited the flag, does nothing.
    ``run_app.bat`` does not open a browser of its own.
    """
    if os.environ.get("LLM_EVAL_BROWSER_OPENED") == "1":
        return False
    os.environ["LLM_EVAL_BROWSER_OPENED"] = "1"
    import sys
    import webbrowser

    if sys.platform == "win32":
        try:
            os.startfile(url)  # noqa: S606 - local app URL only
            return True
        except OSError:
            pass
    webbrowser.open(url, new=0)
    return True


def main() -> None:
    import threading
    import time

    import uvicorn

    url = "http://127.0.0.1:8765"

    def _open() -> None:
        time.sleep(0.6)
        open_app_browser(url)

    threading.Thread(target=_open, daemon=True).start()
    print(f"LLM Eval Suite is at {url}")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")


if __name__ == "__main__":
    main()
