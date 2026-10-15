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
        rail_info = page.evaluate(
            """() => {
              const shell = document.querySelector('.shell');
              const rail = document.querySelector('.rail');
              return {
                cols: getComputedStyle(shell).gridTemplateColumns,
                offset: rail.offsetWidth,
                top: rail.getBoundingClientRect().top,
              };
            }"""
        )
        assert rail is not None and rail_info["offset"] == 220
        assert rail_info["cols"].startswith("220px")
        assert rail_info["top"] == 0
        assert rail["height"] >= 890
        for slot, title in (("models", "Models"), ("knowledge", "Knowledge"), ("finetune", "Fine-tune"), ("safety", "Safety")):
            info = page.evaluate(
                """(slot) => {
                  const button = document.querySelector('.nav-item[data-slot="' + slot + '"]');
                  const label = button.querySelector('.nav-label');
                  const tag = button.querySelector('.nav-tag');
                  const rail = button.closest('.rail').getBoundingClientRect();
                  return {
                    label: label.textContent.trim(),
                    truncated: label.scrollWidth > label.clientWidth + 1,
                    pastRail: button.getBoundingClientRect().right > rail.right + 1,
                    overlap: label.getBoundingClientRect().right > tag.getBoundingClientRect().left + 0.5,
                    phrase: (tag.getAttribute('title') || '') + ' ' + tag.textContent,
                  };
                }""",
                slot,
            )
            assert info["label"] == title
            assert info["truncated"] is False
            assert info["pastRail"] is False
            assert info["overlap"] is False
            assert "Not installed" in info["phrase"]
        banner = page.locator("#class-banner-top").bounding_box()
        assert banner is not None and banner["height"] == 0
        switch = page.locator("#offline-toggle").bounding_box()
        assert switch is not None
        assert round(switch["width"]) == 36 and round(switch["height"]) == 20
        knob = page.evaluate(
            """() => {
              const el = document.getElementById('offline-toggle');
              const after = getComputedStyle(el, '::after');
              const fields = document.querySelector('#connect .fields').getBoundingClientRect();
              const box = el.getBoundingClientRect();
              return {
                w: after.width,
                h: after.height,
                bg: getComputedStyle(el).backgroundColor,
                rightDelta: Math.abs(box.right - fields.right),
              };
            }"""
        )
        assert knob["w"] == "16px" and knob["h"] == "16px"
        assert knob["bg"] == "rgb(107, 115, 128)"
        assert knob["rightDelta"] <= 1
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
        main_box = page.locator("#main").bounding_box()
        page.click("#settings-save")
        page.wait_for_function("() => document.querySelector('#class-banner-top').classList.contains('is-on')")
        assert page.locator("#class-banner-top").inner_text() == "UNCLASSIFIED"
        assert page.locator("#class-banner-top").bounding_box()["height"] == 24
        main_on = page.locator("#main").bounding_box()
        assert main_on["x"] == main_box["x"]
        assert abs(main_on["y"] - (main_box["y"] + 24)) <= 1
        page.click('.nav-item[data-slot="evaluate"]')
        page.click('button[data-tab="connect"]')
        assert "hidden" not in (page.locator("#connect").get_attribute("class") or "").split()
        assert errors == [], errors
    finally:
        browser.close()
        playwright.stop()
        server.should_exit = True
        thread.join(timeout=5)
