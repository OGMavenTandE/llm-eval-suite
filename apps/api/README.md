# Local API (Milestone 2)

FastAPI shell for the Offline AI Evaluation Workbench.

## Run locally

```bash
pip install -e ".[api]"
llm-eval-api
```

Or:

```bash
uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | App and API health |
| GET | `/system/status` | Local install and discovery summary |
| GET | `/profiles` | Available YAML evaluation profiles |
| GET | `/datasets` | Discoverable datasets |
| GET | `/models` | Models referenced by profiles |
| POST | `/runs` | Create a run (returns 202, executes in background) |
| GET | `/runs` | List recent runs |
| GET | `/runs/{run_id}` | Run summary |
| GET | `/runs/{run_id}/results` | Loaded result summaries |
| GET | `/runs/{run_id}/audit` | Persisted audit JSON |
| GET | `/runs/{run_id}/artifacts` | Known artifact file paths |

Routes call `llm_eval.core` services only. Background execution uses an in-process thread pool via `RunJobManager`; persisted run index and audit files remain the source of truth after completion.
