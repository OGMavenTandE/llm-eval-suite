"""Playwright check for the shell: a reserved slot and a sample panel."""

from __future__ import annotations

import socket
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from tests.ui.test_screens import _launch


def test_shell_shows_not_installed_and_mounts_the_sample_panel(tmp_path: Path):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    import uvicorn

    from llm_eval_suite.app import create_app
    from tests.unit.test_panel_host import _entries, sample_panel

    app = create_app(
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        panel_entries=_entries(("sample_panel", sample_panel)),
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}/"
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "api/health", timeout=0.5) as response:
                if response.status == 200:
                    break
        except Exception:
            time.sleep(0.1)
    else:
        server.should_exit = True
        pytest.fail("app did not start")

    playwright, browser = _launch()
    errors: list[str] = []
    try:
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("pageerror", lambda err: errors.append(f"pageerror: {err}"))
        page.on("console", lambda msg: errors.append(f"console: {msg.text}") if msg.type == "error" else None)
        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_selector("#test-connection")
        page.click('.nav-item[data-slot="knowledge"]')
        page.wait_for_selector("#install-command")
        card = page.locator("#view-panel").inner_text()
        assert "Knowledge is not installed" in card
        assert "runs on this computer only" in card
        assert "pip install --no-index --find-links /opt/ai-toolkit/wheels airgap-ai-toolkit[rag]" in card
        assert "Check again" in card
        assert "What else is available" in card
        page.locator('.nav-item[data-slot="more"]').click()
        page.wait_for_function("() => document.body.innerText.includes('Sample panel route answered.')")
        assert page.locator("#sample-ping").inner_text() == "Sample panel route answered."
        page.click('.nav-item[data-slot="evaluate"]')
        page.click('button[data-tab="connect"]')
        assert "hidden" not in (page.locator("#connect").get_attribute("class") or "").split()
        assert errors == [], errors
    finally:
        browser.close()
        playwright.stop()
        server.should_exit = True
        thread.join(timeout=5)
