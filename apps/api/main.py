from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.error_handlers import register_error_handlers
from apps.api.router import api_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="LLM Eval Suite API",
        description="Local offline API for the AI Evaluation Workbench.",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_error_handlers(app)
    app.include_router(api_router)
    return app


app = create_app()


def run_server() -> None:
    import uvicorn

    uvicorn.run("apps.api.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    run_server()
