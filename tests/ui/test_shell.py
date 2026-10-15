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
        page.wait_for_function("() => document.querySelector('#env-status').innerText.includes('Ollama')")
        rail = page.locator(".rail").bounding_box()
        assert rail is not None and 210 <= rail["width"] <= 230
        label_box = page.locator('.nav-item[data-slot="finetune"] .nav-label').bounding_box()
        tag_box = page.locator('.nav-item[data-slot="finetune"] .nav-tag').bounding_box()
        assert label_box is not None and tag_box is not None
        assert label_box["x"] + label_box["width"] <= tag_box["x"] + 1
        banner = page.locator("#class-banner-top").bounding_box()
        assert banner is not None and banner["height"] == 24
        toggle_y = page.locator(".toggle-row").bounding_box()["y"]
        actions_y = page.locator("#connect .actions-primary").bounding_box()["y"]
        assert toggle_y < actions_y
        assert "Offline mode" in page.locator(".toggle-row").inner_text()
        page.click('.nav-item[data-slot="knowledge"]')
        page.wait_for_selector("#install-command")
        card = page.locator("#view-panel").inner_text()
        assert "comes with the Air-Gap AI Toolkit" in card
        assert "runs on this computer only" in card
        assert "Install from the local wheel folder" in card
        assert "pip install --no-index --find-links /opt/ai-toolkit/wheels airgap-ai-toolkit[rag]" in card
        assert "Check again" in card
        assert "Last checked" in card and "Found: none." in card
        assert "What else is available" in card
        assert "Needs airgap-ai-toolkit[safety]" in card
        assert "Needs airgap-ai-toolkit[finetune]" in card
        assert "Evaluate" not in card
        assert "Settings" not in card
        page.locator('.nav-item[data-slot="more"]').click()
        page.wait_for_function("() => document.body.innerText.includes('Sample panel route answered.')")
        assert page.locator("#sample-ping").inner_text() == "Sample panel route answered."
        assert page.locator("#plugin-mount h2").count() == 0
        assert page.locator("#view-panel h2").inner_text() == "Sample notes"
        page.click('.nav-item[data-slot="settings"]')
        page.wait_for_selector("#settings-class-text")
        page.fill("#settings-class-text", "UNCLASSIFIED")
        page.locator("#settings-class-enabled").check()
        main_y = page.locator("#main").bounding_box()["y"]
        page.click("#settings-save")
        page.wait_for_function("() => document.querySelector('#class-banner-top').classList.contains('is-on')")
        assert page.locator("#class-banner-top").inner_text() == "UNCLASSIFIED"
        assert page.locator("#main").bounding_box()["y"] == main_y
        page.click('.nav-item[data-slot="evaluate"]')
        page.click('button[data-tab="connect"]')
        assert "hidden" not in (page.locator("#connect").get_attribute("class") or "").split()
        assert errors == [], errors
    finally:
        browser.close()
        playwright.stop()
        server.should_exit = True
        thread.join(timeout=5)
