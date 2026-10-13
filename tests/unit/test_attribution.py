"""Package owner, the frontend license, and the Llama attribution."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_project_author_and_owner_match_the_license():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Copyright (c) 2026 AI Eval Corporation" in license_text
    assert 'authors = [{ name = "AI Eval Corporation" }]' in text
    assert 'maintainers = [{ name = "AI Eval Corporation" }]' in text
    assert 'owner = "AI Eval Corporation"' in text
    assert "OG Maven" not in text


def test_frontend_package_is_mit():
    payload = json.loads((ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
    assert payload["license"] == "MIT"
    assert payload["author"] == "AI Eval Corporation"


def test_notice_and_vendored_mpl_files_are_in_the_repo():
    assert (ROOT / "NOTICE").read_text(encoding="utf-8").strip() == "Built with Llama."
    page = (ROOT / "llm_eval_suite" / "static" / "index.html").read_text(encoding="utf-8")
    assert "Built with Llama." in page
    for package in ("certifi", "tqdm"):
        folder = ROOT / "third_party" / package
        notice = (folder / "LICENSE").read_text(encoding="utf-8")
        mpl = (folder / "MPL-2.0").read_text(encoding="utf-8")
        assert "MPL" in notice or "Mozilla Public License" in notice
        assert mpl.startswith("Mozilla Public License Version 2.0")
