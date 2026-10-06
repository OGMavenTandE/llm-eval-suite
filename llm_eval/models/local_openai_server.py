"""Tiny OpenAI-compatible server bound to 127.0.0.1.

Used so a folder model (or any in-process adapter) can be the endpoint that
garak's ``openai.OpenAICompatible`` generator calls. No API key is required.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from llm_eval.models.base import BaseModel


class LocalOpenAIServer:
    def __init__(self, model: BaseModel):
        self.model = model
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: Thread | None = None

    @property
    def base_url(self) -> str:
        if self._httpd is None:
            raise RuntimeError("server is not started")
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    def start(self) -> str:
        model = self.model

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:
                return

            def _send(self, code: int, payload: dict) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if self.path.rstrip("/").endswith("/models"):
                    self._send(200, {"data": [{"id": model.name}]})
                    return
                self._send(404, {"error": {"message": "not found"}})

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    body = json.loads(raw.decode("utf-8") or "{}")
                except json.JSONDecodeError:
                    self._send(400, {"error": {"message": "invalid JSON"}})
                    return
                path = self.path.rstrip("/")
                if path.endswith("/chat/completions"):
                    messages = body.get("messages") or []
                    prompt = ""
                    if messages:
                        prompt = messages[-1].get("content") or ""
                    mode = "chat"
                elif path.endswith("/completions"):
                    prompt = body.get("prompt") or ""
                    mode = "completions"
                else:
                    self._send(404, {"error": {"message": "not found"}})
                    return
                try:
                    result = model.generate(prompt, max_tokens=body.get("max_tokens"))
                except Exception as exc:  # surface model errors to the caller
                    self._send(500, {"error": {"message": str(exc)}})
                    return
                usage = {
                    "prompt_tokens": None,
                    "completion_tokens": result.tokens_used,
                    "total_tokens": result.tokens_used,
                }
                if mode == "completions":
                    payload = {
                        "model": model.name,
                        "choices": [{"text": result.text, "finish_reason": "stop"}],
                        "usage": usage,
                    }
                else:
                    payload = {
                        "model": model.name,
                        "choices": [
                            {
                                "message": {"role": "assistant", "content": result.text},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": usage,
                    }
                self._send(200, payload)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self.base_url

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
