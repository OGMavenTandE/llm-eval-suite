"""One-time dataset fetch. This is the only path that opens a network connection.

The runner never calls this module. A closed machine copies the directory
``fetch`` wrote and checks the manifest hashes.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from standard_bench.catalog import SUITES, SuiteSpec
from standard_bench.load import jsonl_lines
from standard_bench.manifest import file_record, write_manifest
from standard_bench.scorers.code import mbpp_entry_point, mbpp_function_header, rewrite_evalplus_test

_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 5

USER_AGENT = "llm-eval-suite-standard-bench"
_TOKEN_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


class FetchError(RuntimeError):
    pass


def token_from_env() -> str:
    for key in _TOKEN_ENV:
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    return ""


def auth_headers(token: str | None = None) -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT}
    secret = token if token is not None else token_from_env()
    if secret:
        headers["Authorization"] = f"Bearer {secret}"
    return headers


def _retry_delay(attempt: int, exc: urllib.error.HTTPError) -> float:
    header = ""
    if exc.headers is not None:
        header = str(exc.headers.get("Retry-After") or "").strip()
    if header.isdigit():
        return min(float(header), 30.0)
    return min(30.0, 0.5 * (2**attempt))


def _request(url: str, token: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers=auth_headers(token))
    host = url.split("?")[0]
    for attempt in range(_MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                raise FetchError(
                    f"Hugging Face refused {host} ({exc.code}). "
                    "GPQA is gated. Set HF_TOKEN in the environment. The token is not stored."
                ) from exc
            if exc.code in _RETRY_STATUS and attempt + 1 < _MAX_ATTEMPTS:
                time.sleep(_retry_delay(attempt, exc))
                continue
            raise FetchError(f"Download failed ({exc.code}) for {host}.") from exc
        except urllib.error.URLError as exc:
            raise FetchError(f"Download failed for {host}: {exc.reason}.") from exc
    raise FetchError(f"Download failed for {host}.")


def live_revision(spec: SuiteSpec, token: str) -> str:
    url = f"https://huggingface.co/api/datasets/{spec.repo}"
    payload = json.loads(_request(url, token).decode("utf-8"))
    return str(payload.get("sha") or "")


def download_text(repo: str, revision: str, path: str, token: str) -> str:
    quoted = urllib.parse.quote(path)
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{quoted}"
    return _request(url, token, timeout=120).decode("utf-8")


def download_rows(repo: str, config: str, split: str, token: str) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    total = None
    while True:
        query = urllib.parse.urlencode(
            {
                "dataset": repo,
                "config": config,
                "split": split,
                "offset": offset,
                "length": 100,
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{query}"
        payload = json.loads(_request(url, token, timeout=120).decode("utf-8"))
        batch = payload.get("rows") or []
        if total is None:
            total = payload.get("num_rows_total")
        if not batch:
            break
        for item in batch:
            row = item.get("row") if isinstance(item, dict) else None
            if isinstance(row, dict):
                rows.append(row)
        offset += len(batch)
        if total is not None and offset >= int(total):
            break
        if len(batch) < 100:
            break
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _test_imports(row: dict) -> list[str]:
    raw = row.get("test_imports") or []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if str(item).strip()]


def _loads_jsonl(text: str) -> list:
    return [json.loads(line) for line in jsonl_lines(text)]


def normalize_humaneval(base_rows: list[dict], plus_rows: list[dict]) -> list[dict]:
    plus_by_id = {str(row.get("task_id")): row for row in plus_rows}
    normalized = []
    for row in base_rows:
        task_id = str(row.get("task_id") or "")
        plus = plus_by_id.get(task_id) or {}
        plus_test = str(plus.get("test") or "")
        plus_prompt = str(plus.get("prompt") or row.get("prompt") or "")
        solution = str(plus.get("canonical_solution") or "")
        normalized.append(
            {
                "task_id": task_id,
                "prompt": row.get("prompt") or "",
                "entry_point": row.get("entry_point") or "",
                "canonical_solution": plus_prompt + solution if solution else "",
                "base_test": row.get("test") or "",
                "plus_test": rewrite_evalplus_test(plus_test) if plus_test else "",
            }
        )
    return normalized


def wrap_module_as_check(source: str, entry: str) -> str:
    """Put a module-level EvalPlus body inside ``def check(candidate)``.

    MBPP+ ``test`` values call the entry point at module level and have no
    ``def check``. Renaming ``entry(`` to ``candidate(`` and indenting the
    whole body keeps the inputs, results, and assert loop. A body that already
    defines ``check`` is only renamed.
    """
    text = (source or "").replace("\r\n", "\n")
    if entry:
        text = text.replace(f"{entry}(", "candidate(")
    if re.search(r"(?m)^def check\(", text):
        return text if text.endswith("\n") else text + "\n"
    lines = []
    for line in text.strip("\n").split("\n"):
        lines.append(("    " + line) if line.strip() else "")
    if not lines:
        lines = ["    pass"]
    return "def check(candidate):\n" + "\n".join(lines) + "\n"


def normalize_mbpp(rows: list[dict]) -> list[dict]:
    normalized = []
    for row in rows:
        code = str(row.get("code") or "")
        tests = [str(item) for item in (row.get("test_list") or [])]
        entry = mbpp_entry_point(code, tests)
        base_test = "def check(candidate):\n    pass\n"
        if entry:
            base_test = (
                "def check(candidate):\n"
                + "\n".join(
                    "    " + item.replace(f"{entry}(", "candidate(") for item in tests
                )
                + "\n"
            )
        plus_source = str(row.get("test") or "")
        plus_test = ""
        if plus_source.strip():
            plus_test = rewrite_evalplus_test(wrap_module_as_check(plus_source, entry))
        normalized.append(
            {
                "task_id": str(row.get("task_id")),
                "prompt": row.get("prompt") or "",
                "entry_point": entry,
                "canonical_solution": code,
                "function_header": mbpp_function_header(code, entry),
                "test_imports": _test_imports(row),
                "test_list": tests,
                "base_test": base_test,
                "plus_test": plus_test,
            }
        )
    return normalized


def _gpqa_rows(text: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for index, raw in enumerate(reader):
        lowered = {str(key or "").strip().lower(): value for key, value in raw.items()}

        def take(*names: str) -> str:
            for name in names:
                if name in lowered and lowered[name] is not None:
                    return str(lowered[name])
            return ""

        record_id = take("record id", "record_id", "id") or str(index)
        rows.append(
            {
                "id": record_id,
                "question": take("question"),
                "correct": take("correct answer", "correct_answer"),
                "incorrect": [
                    take("incorrect answer 1", "incorrect_answer_1"),
                    take("incorrect answer 2", "incorrect_answer_2"),
                    take("incorrect answer 3", "incorrect_answer_3"),
                ],
            }
        )
    return rows


def _check_revision(spec: SuiteSpec, token: str, allow_moved: bool) -> str:
    live = live_revision(spec, token)
    if not live:
        raise FetchError(f"Could not read a revision for {spec.repo}.")
    if live != spec.revision and not allow_moved:
        raise FetchError(
            f"{spec.repo} is at {live}, not the pinned revision {spec.revision}. "
            "Refusing to fetch a different set. Pass --allow-moved to pin the new revision."
        )
    return live


def fetch_suite(spec: SuiteSpec, dest: Path, token: str, *, allow_moved: bool) -> dict:
    revision = _check_revision(spec, token, allow_moved)
    relative = f"{spec.name}/test.jsonl"
    target = dest / relative
    if spec.name == "mmlu_pro":
        test_rows = download_rows(spec.repo, "default", "test", token)
        val_rows = download_rows(spec.repo, "default", "validation", token)
        _write_jsonl(target, test_rows)
        val_relative = f"{spec.name}/validation.jsonl"
        _write_jsonl(dest / val_relative, val_rows)
        files = [file_record(dest, relative), file_record(dest, val_relative)]
        count = len(test_rows)
    elif spec.name == "gpqa_diamond":
        if not token:
            raise FetchError(
                "GPQA is gated. Set HF_TOKEN to your own Hugging Face token. "
                "The token is not written to the manifest or the dataset."
            )
        text = download_text(spec.repo, revision, spec.direct_file, token)
        rows = _gpqa_rows(text)
        _write_jsonl(target, rows)
        files = [file_record(dest, relative)]
        count = len(rows)
    elif spec.name == "gsm8k":
        rows = download_rows(spec.repo, "main", "test", token)
        _write_jsonl(target, rows)
        files = [file_record(dest, relative)]
        count = len(rows)
    elif spec.name == "math500":
        text = download_text(spec.repo, revision, spec.direct_file, token)
        rows = _loads_jsonl(text)
        _write_jsonl(target, rows)
        files = [file_record(dest, relative)]
        count = len(rows)
    elif spec.name == "ifeval":
        text = download_text(spec.repo, revision, spec.direct_file, token)
        rows = _loads_jsonl(text)
        _write_jsonl(target, rows)
        files = [file_record(dest, relative)]
        count = len(rows)
    elif spec.name == "humaneval_plus":
        plus_text = download_text(spec.repo, revision, "test.jsonl", token)
        plus_rows = _loads_jsonl(plus_text)
        base_rows = download_rows(
            "openai/openai_humaneval",
            "openai_humaneval",
            "test",
            token,
        )
        rows = normalize_humaneval(base_rows, plus_rows)
        _write_jsonl(target, rows)
        files = [file_record(dest, relative)]
        count = len(rows)
    elif spec.name == "mbpp_plus":
        raw_rows = download_rows(spec.repo, "default", "test", token)
        rows = normalize_mbpp(raw_rows)
        _write_jsonl(target, rows)
        files = [file_record(dest, relative)]
        count = len(rows)
    else:
        raise FetchError(f"No fetch path for {spec.name}.")
    return {
        "repo": spec.repo,
        "revision": revision,
        "license": spec.license_id,
        "rows": count,
        "full_size": spec.full_size,
        "files": files,
        "gated": spec.gated,
    }


def fetch(dest: str | Path, suites: list[str] | None = None, *, allow_moved: bool = False) -> dict:
    from llm_eval.offline import offline_active

    if offline_active():
        raise FetchError(
            "Offline mode is on. fetch needs a network and will not run. Nothing was downloaded."
        )
    root = Path(dest)
    root.mkdir(parents=True, exist_ok=True)
    token = token_from_env()
    names = list(suites or SUITES)
    unknown = [name for name in names if name not in SUITES]
    if unknown:
        raise FetchError(f"Unknown suites: {', '.join(unknown)}.")
    existing = {}
    manifest_file = root / "manifest.json"
    if manifest_file.is_file():
        try:
            existing = json.loads(manifest_file.read_text(encoding="utf-8")).get("suites") or {}
        except json.JSONDecodeError:
            existing = {}
    for name in names:
        entry = fetch_suite(SUITES[name], root, token, allow_moved=allow_moved)
        existing[name] = entry
        # Do not print question text. Row counts and hashes only.
        digest = entry["files"][0]["sha256"]
        print(f"{name}: {entry['rows']} rows revision {entry['revision']} sha256 {digest}")
    payload = {"version": 1, "suites": existing}
    write_manifest(root, payload)
    # The token must not land in the manifest.
    blob = manifest_file.read_text(encoding="utf-8")
    if token and token in blob:
        manifest_file.unlink()
        raise FetchError("Refusing to write a manifest that contains the HF token.")
    return payload


def manifest_contains_secret(payload: dict, secret: str) -> bool:
    if not secret:
        return False
    return secret in json.dumps(payload)
