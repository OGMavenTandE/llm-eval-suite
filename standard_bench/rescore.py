"""Re-score a finished standard-bench run from saved responses. No model is loaded.

Each saved item is joined to its dataset row by the stable id stored on the
item (suite plus task id, question id, or question hash). Rescore does not
re-run subset selection and does not zip items to rows by position.

An id that is missing, duplicated, or only partly present fails the rescore.
A run written before those ids existed is aligned by reproducing the subset
recorded in run.json (subset name, seed, per-suite count) and checking the
stored prompt against that row. Any mismatch aborts.

Code suites need the pinned dataset so the current plus tests can run. Other
suites use that dataset when it is passed, and otherwise the gold already
stored on the item. GPQA question text stays redacted. When the dataset is
loaded, the stored letter is graded against the row that id points at.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

from llm_eval.models.context import strip_think_blocks

from standard_bench import SCORER_VERSION, SUITE_NAMES, SUITE_VERSION
from standard_bench.export import render_csv, render_json, rows_for_run
from standard_bench.identity import ItemIdError, index_rows
from standard_bench.load import jsonl_lines, load_file, load_suite, select_rows
from standard_bench.manifest import HashMismatch, ManifestError
from standard_bench.protocol import QUICK_SEED, subset_limits
from standard_bench.reference import exclusions_for
from standard_bench.runner import _aggregate, _prompt_for, _score
from standard_bench.sandbox import RunLimits
from standard_bench.scorers.gpqa import score_gpqa, shuffle_options


class RescoreError(RuntimeError):
    pass


def _scorer_stamp() -> str:
    """Package version plus the scorer version. Used when git cannot answer."""
    version = "0.1.0"
    try:
        found = package_version("llm-eval-suite")
    except PackageNotFoundError:
        found = ""
    if found:
        version = found
    return f"{version}+scorer{SCORER_VERSION}"


def scorer_commit() -> str:
    """HEAD of this checkout, or the package stamp when git cannot answer."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return _scorer_stamp()
    sha = out.strip()
    return sha or _scorer_stamp()


def _read_items(path: Path) -> list[dict]:
    rows = []
    for line in jsonl_lines(path.read_text(encoding="utf-8")):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


_NEEDS_ROW = frozenset({"ifeval", "humaneval_plus", "mbpp_plus"})
_MISSING = object()


def _load_suites(data_dir: str | Path | None, suites: set[str]) -> dict[str, dict]:
    """Load each suite once. ``loaded`` is false when the dataset was not used.

    A hash or manifest failure on a code suite or on IFEval stops the rescore.
    Other suites then keep the gold already stored on the item.
    """
    if not data_dir:
        return {}
    root = Path(data_dir)
    loaded: dict[str, dict] = {}
    for suite in sorted(suites):
        if suite not in SUITE_NAMES:
            continue
        try:
            rows, _entry = load_suite(root, suite)
        except (ManifestError, HashMismatch) as exc:
            if suite in _NEEDS_ROW:
                raise RescoreError(str(exc)) from exc
            loaded[suite] = {"rows": [], "index": {}, "validation": [], "loaded": False}
            continue
        try:
            index = index_rows(suite, rows)
        except ItemIdError as exc:
            raise RescoreError(str(exc)) from exc
        validation: list[dict] = []
        if suite == "mmlu_pro":
            validation = load_file(root, suite, "validation.jsonl")
        loaded[suite] = {
            "rows": rows,
            "index": index,
            "validation": validation,
            "loaded": True,
        }
    return loaded


def _short_id(item: dict) -> str:
    task_id = str(item.get("id") or "")
    return task_id.split(":", 1)[1] if ":" in task_id else task_id


def _suite_record(record: dict, suite: str) -> dict:
    rows = [
        row
        for row in (record.get("suites") or [])
        if isinstance(row, dict) and row.get("name") == suite
    ]
    if len(rows) > 1:
        raise RescoreError(f"{suite} is listed more than once in run.json.")
    return rows[0] if rows else {}


def _pick(source: dict, key: str):
    if key not in source:
        return _MISSING
    return source[key]


def _agree(items: list[dict], key: str, suite: str):
    present = [item for item in items if key in item]
    if not present:
        return _MISSING
    if len(present) != len(items):
        raise RescoreError(f"{suite} items do not all record {key}.")
    first = present[0][key]
    if any(item[key] != first for item in present[1:]):
        raise RescoreError(f"{suite} items disagree on {key}.")
    return first


def _first_present(*values):
    for value in values:
        if value is not _MISSING:
            return value
    return _MISSING


def _recorded_selection(record: dict, items: list[dict], suite: str) -> tuple[str, int | None, int | None]:
    """Subset name, seed, and cap as run.json recorded them.

    Item fields fill a gap. They do not override run.json. quick50 with no
    recorded seed uses the protocol seed. Any other missing subset aborts.
    """
    suite_row = _suite_record(record, suite)
    metrics = suite_row.get("metrics") if isinstance(suite_row.get("metrics"), dict) else {}
    subset = _first_present(
        _pick(suite_row, "subset"),
        _pick(metrics, "subset"),
        _agree(items, "subset", suite),
    )
    seed = _first_present(
        _pick(suite_row, "seed"),
        _pick(metrics, "seed"),
        _agree(items, "seed", suite),
    )
    limit = _first_present(
        _pick(metrics, "limit"),
        _pick(suite_row, "subset_limit"),
        _pick(metrics, "subset_limit"),
        _agree(items, "subset_limit", suite),
    )
    if subset is _MISSING or subset in ("", None):
        raise RescoreError(
            f"{suite} items have no stable dataset id, and run.json does not record a subset. "
            "Refusing to grade by position."
        )
    subset = str(subset)
    try:
        named = subset_limits(subset)
    except KeyError as exc:
        raise RescoreError(str(exc)) from exc
    if seed is _MISSING:
        seed = QUICK_SEED if subset == "quick50" else None
    elif seed is not None:
        seed = int(seed)
    if limit is _MISSING or limit is None:
        limit = None if named is None else named.get(suite)
    else:
        limit = int(limit)
    recorded = metrics.get("n_items") if isinstance(metrics, dict) else None
    if recorded is not None and int(recorded) != len(items):
        raise RescoreError(
            f"{suite} run.json records {recorded} items and the run saved {len(items)}. "
            "Refusing to grade by position."
        )
    return subset, seed, limit


def _rows_for_selection(suite: str, rows: list[dict]) -> list[dict]:
    dropped = exclusions_for(suite)
    if not dropped:
        return list(rows)
    return [row for row in rows if str(row.get("task_id")) not in dropped]


def _prompt_matches(item: dict, suite: str, row: dict, validation: list[dict]) -> bool:
    rebuilt = _prompt_for(suite, row, validation)[0]
    stored = item.get("prompt")
    digest = hashlib.sha256(rebuilt.encode("utf-8")).hexdigest()
    recorded = str(item.get("prompt_sha256") or "")
    if isinstance(stored, str) and stored and stored == rebuilt:
        return True
    return bool(recorded) and recorded == digest


def _legacy_rows(suite: str, items: list[dict], data: dict, record: dict) -> list[dict]:
    """Reproduce the recorded subset and require every stored prompt to match."""
    subset, seed, limit = _recorded_selection(record, items, suite)
    chosen = select_rows(_rows_for_selection(suite, data["rows"]), limit, seed)
    if len(items) > len(chosen):
        raise RescoreError(
            f"{suite} legacy rescore reproduced {len(chosen)} rows for subset {subset} "
            f"seed {seed} limit {limit}, but the run saved {len(items)} items. "
            "Refusing to grade by position."
        )
    validation = data.get("validation") or []
    for index, item in enumerate(items):
        row = chosen[index]
        if _prompt_matches(item, suite, row, validation):
            continue
        label = item.get("id") or "<missing>"
        raise RescoreError(
            f"{suite} item {label} does not match the reproduced subset row at position {index}. "
            "Prompt text does not match. Refusing to grade by position."
        )
    return chosen[: len(items)]


def _align_suite(suite: str, items: list[dict], data: dict, record: dict) -> dict[int, dict]:
    """Map ``id(item)`` to its dataset row. Never joins by position when an id matches."""
    if not data.get("loaded"):
        return {}
    pending = [item for item in items if _short_id(item) not in exclusions_for(suite)]
    if not pending:
        return {}
    index = data["index"]
    resolved: list[dict] = []
    unresolved: list[dict] = []
    seen: set[str] = set()
    for item in pending:
        item_id = str(item.get("id") or "")
        if not item_id:
            unresolved.append(item)
            continue
        if item_id in seen:
            raise RescoreError(
                f"{suite} item id {item_id} is ambiguous: it is stored more than once."
            )
        seen.add(item_id)
        if item_id in index:
            resolved.append(item)
        else:
            unresolved.append(item)
    if resolved and unresolved:
        sample = ", ".join(str(item.get("id") or "<missing>") for item in unresolved[:5])
        raise RescoreError(
            f"{suite} rescore cannot align every item by id ({sample}). "
            "Refusing to grade by position."
        )
    if unresolved:
        chosen = _legacy_rows(suite, pending, data, record)
        return {id(item): row for item, row in zip(pending, chosen)}
    return {id(item): index[str(item.get("id"))] for item in pending}


def _private(suite: str, item: dict, row: dict | None) -> dict:
    if suite == "mmlu_pro":
        gold = ""
        if row is not None:
            gold = str(row.get("answer") or "")
        return {"gold": gold or str(item.get("expected") or "")}
    if suite == "gsm8k":
        gold = str(row.get("answer") or "") if row is not None else ""
        return {"gold": gold or str(item.get("expected") or "")}
    if suite == "math500":
        gold = str(row.get("answer") or "") if row is not None else ""
        return {"gold": gold or str(item.get("expected") or "")}
    if suite == "gpqa_diamond":
        if row is None:
            return {"gold": str(item.get("expected") or "")}
        _options, letter = shuffle_options(
            str(row.get("id") or ""),
            str(row.get("correct") or ""),
            list(row.get("incorrect") or []),
        )
        return {"gold": letter}
    if suite == "ifeval":
        if row is None:
            raise RescoreError(
                f"IFEval item {item.get('id')} needs the dataset row. Pass --data or --fixtures."
            )
        return {"row": row}
    if suite in {"humaneval_plus", "mbpp_plus"}:
        if row is None:
            raise RescoreError(
                f"{suite} item {item.get('id')} needs the dataset row. Pass --data or --fixtures."
            )
        return {"row": row}
    raise RescoreError(f"No scorer for suite {suite}.")


def _answer_text(item: dict) -> str:
    raw = item.get("raw_response")
    if raw is None or raw == "":
        raw = item.get("response") or ""
    suite = str(item.get("suite") or "")
    return strip_think_blocks(str(raw), strip_edges=suite not in {"humaneval_plus", "mbpp_plus"})


def _apply_score(item: dict, scored: dict, commit: str) -> None:
    for key, value in scored.items():
        item[key] = value
    item["scorer_version"] = SCORER_VERSION
    item["scorer_commit"] = commit
    item["rescored"] = True


def _backup_file(path: Path) -> Path:
    """Copy ``path`` to a ``.bak`` beside it. A second backup is numbered."""
    bak = path.with_name(path.name + ".bak")
    index = 2
    while bak.exists():
        bak = path.with_name(f"{path.name}.bak.{index}")
        index += 1
    shutil.copy2(path, bak)
    return bak


def _rescored_dir(source: Path) -> Path:
    """A new sibling directory. The source run is left as it was."""
    candidate = source.parent / f"{source.name}-rescored"
    index = 2
    while candidate.exists():
        candidate = source.parent / f"{source.name}-rescored-{index}"
        index += 1
    candidate.mkdir(parents=True)
    return candidate


def rescore_run_dir(
    run_dir: str | Path,
    *,
    data_dir: str | Path | None = None,
    in_place: bool = False,
) -> dict:
    """Score saved responses with the current scorer. Does not load a model.

    The default writes a new sibling directory and leaves the source run
    untouched. ``in_place`` copies ``items.jsonl`` and ``run.json`` to ``.bak``
    files, then overwrites them.
    """
    source = Path(run_dir)
    items_path = source / "items.jsonl"
    run_path = source / "run.json"
    if not items_path.is_file() or not run_path.is_file():
        raise FileNotFoundError("rescore needs items.jsonl and run.json in the run directory")
    items = _read_items(items_path)
    record = json.loads(run_path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = {}
    for item in items:
        suite = str(item.get("suite") or "")
        if suite in SUITE_NAMES:
            grouped.setdefault(suite, []).append(item)
    loaded = _load_suites(data_dir, set(grouped))
    row_for: dict[int, dict] = {}
    for suite, suite_items in grouped.items():
        data = loaded.get(suite) or {"loaded": False}
        row_for.update(_align_suite(suite, suite_items, data, record))
    if in_place:
        _backup_file(items_path)
        _backup_file(run_path)
        directory = source
    else:
        directory = _rescored_dir(source)
    commit = scorer_commit()
    limits = RunLimits()
    updated = []
    for item in items:
        suite = str(item.get("suite") or "")
        if suite not in SUITE_NAMES:
            updated.append(item)
            continue
        data = loaded.get(suite) or {"loaded": False}
        stored = str(item.get("raw_response") or item.get("response") or "").strip()
        row = row_for.get(id(item))
        kept_gpqa = item.get("redacted") or (suite == "gpqa_diamond" and not stored)
        if kept_gpqa and (not data.get("loaded") or row is None):
            item["scorer_version"] = SCORER_VERSION
            item["scorer_commit"] = commit
            updated.append(item)
            continue
        if item.get("prompt_over_budget"):
            item["scorer_version"] = SCORER_VERSION
            item["scorer_commit"] = commit
            updated.append(item)
            continue
        short_id = _short_id(item)
        if short_id in exclusions_for(suite):
            item["reference_excluded"] = True
            item["passed"] = False
            item["score"] = None
            item["scorer_version"] = SCORER_VERSION
            item["scorer_commit"] = commit
            item["rescored"] = True
            updated.append(item)
            continue
        private = _private(suite, item, row)
        needs_gold = suite in {"mmlu_pro", "gsm8k", "math500", "gpqa_diamond"}
        if needs_gold and row is None and not str(private.get("gold") or "").strip():
            raise RescoreError(
                f"{suite} item {item.get('id')} has no stored gold and no dataset row. Pass --data or --fixtures."
            )
        if suite == "gpqa_diamond" and row is not None and not stored:
            predicted = str(item.get("predicted") or "")
            reply = f"Answer: {predicted}" if predicted else ""
            scored = score_gpqa(str(private.get("gold") or ""), reply)
        else:
            scored = _score(suite, _answer_text(item), private, limits)
        _apply_score(item, scored, commit)
        if item.get("redacted"):
            item["expected"] = ""
            item["prompt"] = ""
            item["response"] = ""
            item["raw_response"] = ""
        updated.append(item)

    out_items = directory / "items.jsonl"
    out_run = directory / "run.json"
    out_items.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in updated),
        encoding="utf-8",
    )
    record["scorer_version"] = SCORER_VERSION
    record["scorer_commit"] = commit
    by_suite: dict[str, list[dict]] = {}
    for item in updated:
        suite = str(item.get("suite") or "")
        if suite in SUITE_NAMES:
            by_suite.setdefault(suite, []).append(item)
    suite_rows = record.get("suites")
    if not isinstance(suite_rows, list):
        suite_rows = []
        record["suites"] = suite_rows
    seen = set()
    for suite_row in suite_rows:
        if not isinstance(suite_row, dict):
            continue
        name = str(suite_row.get("name") or "")
        if name not in by_suite:
            continue
        seen.add(name)
        _refresh_suite_row(suite_row, name, by_suite[name], commit)
    for name, suite_items in by_suite.items():
        if name in seen:
            continue
        suite_row = {"name": name, "metrics": {}}
        _refresh_suite_row(suite_row, name, suite_items, commit)
        suite_rows.append(suite_row)

    from llm_eval_suite.scoring import scorecard

    card = scorecard(updated)
    card["scorer_version"] = SCORER_VERSION
    card["scorer_commit"] = commit
    public = card.get("public_benchmarks")
    if isinstance(public, dict):
        public["scorer_version"] = SCORER_VERSION
        public["scorer_commit"] = commit
    record["scorecard"] = card
    out_run.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    csv_path = directory / "public_benchmarks.csv"
    json_path = directory / "public_benchmarks.json"
    exported = rows_for_run(directory)
    csv_path.write_text(render_csv(exported), encoding="utf-8")
    json_path.write_text(render_json(exported), encoding="utf-8")
    return {
        "run_dir": str(directory),
        "csv": str(csv_path),
        "json": str(json_path),
        "items": len(updated),
        "scorer_version": SCORER_VERSION,
        "scorer_commit": commit,
        "scorecard": card,
    }


def _refresh_suite_row(suite_row: dict, name: str, suite_items: list[dict], commit: str) -> None:
    fresh = _aggregate(name, suite_items)
    metrics = dict(suite_row.get("metrics") or {})
    metrics.update(fresh)
    if name in {"humaneval_plus", "mbpp_plus"}:
        metrics["reference_exclusions"] = [
            {"task_id": task_id, "reason": reason}
            for task_id, reason in sorted(exclusions_for(name).items())
        ]
    metrics["scorer_version"] = SCORER_VERSION
    metrics["scorer_commit"] = commit
    suite_row["metrics"] = metrics
    suite_row["pass_rate"] = fresh.get("score")
    suite_row["suite_version"] = suite_row.get("suite_version") or SUITE_VERSION
