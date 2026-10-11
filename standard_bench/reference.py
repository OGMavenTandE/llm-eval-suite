"""Score stored canonical solutions. Offline. No network.

HumanEval+ stores the plus prompt plus ``canonical_solution``. MBPP+ stores
the dataset ``code`` field. ``verify-reference`` requires every one of those
to pass, apart from the exclusions below.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from standard_bench.load import load_suite
from standard_bench.sandbox import RunLimits
from standard_bench.scorers.code import score_code_task

CODE_SUITES = ("humaneval_plus", "mbpp_plus")

# task_id -> why the reference solution is not required to pass.
# These still fail after the helper, import, and 20 s / 1 GB fixes.
# Platform-dependent expected values are excluded on every OS.
# The pinned HumanEval+ and MBPP+ rows were scanned for sys.getsizeof, id(,
# hash(, and memory-address expectations. The only match is MBPP+ 596.
# HumanEval/39 calls random inside its reference primality test. Its expected
# values are fixed integers, so it stays scored.
EXCLUSIONS: dict[str, dict[str, str]] = {
    "humaneval_plus": {
        "HumanEval/32": (
            "The plus test calls _poly(*candidate(*inp), inp). "
            "find_zero returns a float, so that star raises TypeError. "
            "The reference solution passes the base tests."
        ),
    },
    "mbpp_plus": {
        "255": (
            "The plus test builds combinations_with_replacement for a length-77 sequence. "
            "The reference solution does the same and exceeds the 1 GB address space. "
            "The base tests pass."
        ),
        "596": (
            "The plus results are frozen sys.getsizeof integers. "
            "The same object is 88 bytes on Linux and 96 bytes on Windows. "
            "The base tests call sys.getsizeof on both sides, so they match. "
            "The row stays in the file and is not scored."
        ),
    },
}


def exclusions_for(suite: str) -> dict[str, str]:
    return {str(task_id): reason for task_id, reason in EXCLUSIONS.get(suite, {}).items()}


def _one(suite: str, row: dict, limits: RunLimits | None) -> dict:
    task_id = str(row.get("task_id"))
    code = str(row.get("canonical_solution") or "")
    if not code.strip():
        return {
            "task_id": task_id,
            "passed": False,
            "base_pass": False,
            "plus_pass": False,
            "error": "missing canonical solution",
        }
    scored = score_code_task(row, "```python\n" + code + "\n```", suite=suite, limits=limits)
    return {
        "task_id": task_id,
        "passed": bool(scored.get("plus_pass")),
        "base_pass": bool(scored.get("base_pass")),
        "plus_pass": bool(scored.get("plus_pass")),
        "error": str(scored.get("error") or ""),
    }


def verify_rows(
    rows: list[dict],
    suite: str,
    *,
    limits: RunLimits | None = None,
    workers: int = 1,
) -> dict:
    """Score canonical solutions for one suite. Excluded tasks are reported, not failed."""
    excluded = exclusions_for(suite)
    skipped = []
    pending = []
    for row in rows:
        task_id = str(row.get("task_id"))
        if task_id in excluded:
            skipped.append({"task_id": task_id, "reason": excluded[task_id]})
        else:
            pending.append(row)
    bounds = limits or RunLimits()
    if pending and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda row: _one(suite, row, bounds), pending))
    else:
        results = [_one(suite, row, bounds) for row in pending]
    failures = [item for item in results if not item["passed"]]
    passed = len(results) - len(failures)
    return {
        "suite": suite,
        "rows": len(rows),
        "checked": len(results),
        "passed": passed,
        "excluded": skipped,
        "failures": failures,
        "ok": not failures,
    }


def verify_reference(dest: str | Path, *, workers: int = 8) -> dict:
    """Load pinned HumanEval+ and MBPP+ and score every canonical solution."""
    suites = {}
    for name in CODE_SUITES:
        rows, _entry = load_suite(dest, name)
        suites[name] = verify_rows(rows, name, workers=workers)
    return {"ok": all(item["ok"] for item in suites.values()), "suites": suites}
