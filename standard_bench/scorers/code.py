"""HumanEval+ and MBPP+ prompting and pass@1.

One greedy sample. Base tests and plus tests are scored separately.
A plus pass requires the base tests and the plus tests.
"""

from __future__ import annotations

import re

from standard_bench.sandbox import RunLimits, run_code

HUMANEVAL_STOPS = ("\nclass", "\ndef", "\n#", "\nif", "\nprint")
MBPP_STOPS = ('\n"""', "\nassert", "\nclass", "\ndef")

_STDLIB_HEADER = '''
import math

def is_floats(x):
    if isinstance(x, float) and not isinstance(x, bool):
        return True
    if isinstance(x, (list, tuple)):
        return all(isinstance(i, float) and not isinstance(i, bool) for i in x)
    return False

def _close(out, exp, rtol, atol):
    if isinstance(out, (set, frozenset)) or isinstance(exp, (set, frozenset)):
        return out == exp
    if isinstance(out, (list, tuple)) and isinstance(exp, (list, tuple)):
        if len(out) != len(exp):
            return False
        return all(_close(a, b, rtol, atol) for a, b in zip(out, exp))
    try:
        return abs(float(out) - float(exp)) <= (atol + rtol * abs(float(exp)))
    except (TypeError, ValueError):
        return False

def assertion(out, exp, atol):
    exact = out == exp
    if atol == 0 and is_floats(exp):
        atol = 1e-6
    if not exact and atol != 0:
        assert _close(out, exp, 1e-07, atol)
    else:
        assert exact
'''

_SET_HEADER = _STDLIB_HEADER.replace(
    "def assertion(out, exp, atol):",
    "def assertion(out, exp, atol):\n"
    "    out = set(out)\n"
    "    exp = set(exp)",
)


def rewrite_evalplus_test(source: str) -> str:
    """Replace the numpy assertion helper. Inputs and expected values stay."""
    text = source or ""
    marker = "def check("
    index = text.find(marker)
    if index < 0:
        raise ValueError("EvalPlus test has no def check.")
    body = text[index:]
    header = _SET_HEADER if "out = set(out)" in text[:index] else _STDLIB_HEADER
    rewritten = header + "\n" + body
    rewritten = re.sub(r"(?m)^[ \t]*import numpy(?:\s+as\s+\w+)?\s*$", "", rewritten)
    rewritten = re.sub(r"(?m)^[ \t]*from numpy\s+import\s+.+$", "", rewritten)
    rewritten = rewritten.replace("np.allclose", "_close")
    # A copied helper may still mention the numpy array type. The stdlib
    # header does not, and the run does not import numpy.
    rewritten = rewritten.replace("isinstance(x, np.ndarray)", "False")
    rewritten = rewritten.replace("np.float64", "float")
    rewritten = rewritten.replace("np.float32", "float")
    return rewritten


def extract_fenced(text: str) -> str:
    """The first fenced block, or the raw completion when there is no fence."""
    fenced = re.search(r"```(?:python)?\s*\n(.*?)```", text or "", flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)
    return text or ""


def apply_stops(source: str, stops: tuple[str, ...]) -> str:
    cut = len(source)
    for stop in stops:
        index = source.find(stop)
        if index != -1:
            cut = min(cut, index)
    return source[:cut]


def _entry_match(source: str, entry_point: str) -> re.Match[str] | None:
    if not entry_point:
        return None
    return re.search(
        rf"^([ \t]*)def\s+{re.escape(entry_point)}\b[^\n]*",
        source,
        re.MULTILINE,
    )


def _function_end(source: str, match: re.Match[str]) -> int:
    """Index just after the entry function. Nested helpers stay inside it."""
    def_indent = len(match.group(1).expandtabs(4))
    pos = match.end()
    end = pos
    for line in source[pos:].splitlines(keepends=True):
        if not line.strip():
            pos += len(line)
            end = pos
            continue
        indent = len(line[: len(line) - len(line.lstrip(" \t"))].expandtabs(4))
        if indent <= def_indent:
            break
        pos += len(line)
        end = pos
    return end


def assemble_program(prompt: str, completion: str, entry_point: str, stops: tuple[str, ...]) -> str:
    """Build the program that the tests call.

    A fenced reply is unwrapped first. When it defines ``entry_point``, imports
    and helpers before that function stay, helpers after it stay, and stop
    strings apply only to the text after the function. ``\\ndef`` and
    ``\\nclass`` are not cuts there, so a following helper is kept. Other stops
    still drop a trailing ``print`` or assert.

    A reply with no ``def entry_point`` is a bare body. The prompt (HumanEval)
    or the function header (MBPP) is the prefix, and every stop applies to the
    body, which drops a second function the model continued into.
    """
    source = extract_fenced(completion)
    match = _entry_match(source, entry_point)
    if match is None:
        return (prompt or "") + apply_stops(source, stops)
    end = _function_end(source, match)
    prefix = source[: match.start()]
    function = source[match.start() : end]
    # Keep helpers after the entry point. Still cut prose stops such as print.
    suffix_stops = tuple(stop for stop in stops if stop not in {"\ndef", "\nclass"})
    suffix = apply_stops(source[end:], suffix_stops)
    return prefix + function + suffix


def _program(code: str, test: str, entry_point: str) -> str:
    return code.rstrip() + "\n\n" + test.rstrip() + f"\n\ncheck({entry_point})\n"


def run_tests(program: str, test: str, entry_point: str, limits: RunLimits | None = None) -> dict:
    if not str(test or "").strip():
        return {"ok": False, "error": "missing test", "timed_out": False}
    result = run_code(_program(program, test, entry_point), limits=limits)
    return {
        "ok": result.ok,
        "error": result.error,
        "timed_out": result.timed_out,
        "isolation": result.isolation,
        "blocked_network": result.blocked_network,
    }


def humaneval_prompt(row: dict) -> str:
    return (
        "Complete the following Python function. Reply with the code only.\n\n"
        + str(row.get("prompt") or "")
    )


def mbpp_prompt(row: dict) -> str:
    tests = "\n".join(str(item) for item in (row.get("test_list") or []))
    statement = str(row.get("prompt") or "").strip()
    return (
        "You are an expert Python programmer, and here is your task: "
        f"{statement} Your code should pass these tests:\n\n{tests}\n"
    )


def score_code_task(row: dict, completion: str, *, suite: str, limits: RunLimits | None = None) -> dict:
    entry = str(row.get("entry_point") or "")
    stops = HUMANEVAL_STOPS if suite == "humaneval_plus" else MBPP_STOPS
    prompt = str(row.get("prompt") or "")
    # HumanEval stores the function header as the prompt. MBPP stores prose,
    # and the function header is separate when the dataset has one.
    header = str(row.get("function_header") or "")
    if suite == "mbpp_plus":
        program = assemble_program(header, completion, entry, stops)
    else:
        program = assemble_program(prompt, completion, entry, stops)
    base = run_tests(program, str(row.get("base_test") or ""), entry, limits)
    if base["ok"]:
        plus = run_tests(program, str(row.get("plus_test") or ""), entry, limits)
    else:
        plus = {"ok": False, "error": "base failed", "timed_out": False, "isolation": base.get("isolation", "")}
    base_pass = bool(base["ok"])
    plus_pass = bool(base_pass and plus["ok"])
    error = ""
    if not plus_pass:
        error = str(base.get("error") or plus.get("error") or "")
    return {
        "predicted": "",
        "expected": entry,
        "passed": plus_pass,
        "score": 1.0 if plus_pass else 0.0,
        "metric": "pass_at_1_plus",
        "base_pass": base_pass,
        "plus_pass": plus_pass,
        "timed_out": bool(base.get("timed_out") or plus.get("timed_out")),
        "isolation": plus.get("isolation") or base.get("isolation") or "",
        "error": error,
    }
