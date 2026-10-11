"""HumanEval+ and MBPP+ prompting and pass@1.

One greedy sample. Base tests and plus tests are scored separately.
A plus pass requires the base tests and the plus tests.
"""

from __future__ import annotations

import cmath
import inspect
import math
import re

from standard_bench.sandbox import RunLimits, run_code

HUMANEVAL_STOPS = ("\nclass", "\ndef", "\n#", "\nif", "\nprint")
MBPP_STOPS = ('\n"""', "\nassert", "\nclass", "\ndef")

# EvalPlus uses atol 1e-6 and rtol 1e-7. The same pair applies to every
# float and complex, including ones nested in tuples, lists, and dicts.
FLOAT_ABS_TOL = 1e-6
FLOAT_REL_TOL = 1e-7


def values_close(out, exp, rel_tol, abs_tol):
    """Recursive compare. Floats and complexes are close. Everything else is exact.

    Bool is not a float. ``True`` compared with ``1.0000001`` stays exact, so a
    tolerance cannot turn that pair into a pass. Integers stay exact, so a
    platform-specific ``sys.getsizeof`` pair such as 88 and 96 does not pass.
    A list does not match a tuple. Sets stay exact.
    """
    if isinstance(out, bool) or isinstance(exp, bool):
        return out == exp
    if isinstance(out, complex) or isinstance(exp, complex):
        if isinstance(out, (int, float, complex)) and isinstance(exp, (int, float, complex)):
            return cmath.isclose(complex(out), complex(exp), rel_tol=rel_tol, abs_tol=abs_tol)
        return False
    if isinstance(out, float) or isinstance(exp, float):
        if isinstance(out, (int, float)) and isinstance(exp, (int, float)):
            return math.isclose(float(out), float(exp), rel_tol=rel_tol, abs_tol=abs_tol)
        return False
    if isinstance(out, dict) and isinstance(exp, dict):
        if set(out) != set(exp):
            return False
        return all(values_close(out[key], exp[key], rel_tol, abs_tol) for key in out)
    if isinstance(out, list) and isinstance(exp, list):
        if len(out) != len(exp):
            return False
        return all(values_close(left, right, rel_tol, abs_tol) for left, right in zip(out, exp))
    if isinstance(out, tuple) and isinstance(exp, tuple):
        if len(out) != len(exp):
            return False
        return all(values_close(left, right, rel_tol, abs_tol) for left, right in zip(out, exp))
    return out == exp


def _assertion_header() -> str:
    """Stdlib assertion injected into a plus test. The sandbox does not import this module."""
    # One implementation. The child process gets this source, not the module object.
    body = inspect.getsource(values_close)
    return (
        "import math\n"
        "import cmath\n"
        "\n"
        f"{body}\n"
        "def _close(out, exp, rtol, atol):\n"
        "    return values_close(out, exp, rtol, atol)\n"
        "\n"
        "def assertion(out, exp, atol):\n"
        f"    abs_tol = {FLOAT_ABS_TOL!r} if atol == 0 else float(atol)\n"
        f"    assert values_close(out, exp, {FLOAT_REL_TOL!r}, abs_tol)\n"
    )


_STDLIB_HEADER = "\n" + _assertion_header()

_SET_HEADER = _STDLIB_HEADER.replace(
    "def assertion(out, exp, atol):",
    "def assertion(out, exp, atol):\n"
    "    out = set(out)\n"
    "    exp = set(exp)",
)


def _drop_numpy_helpers(prefix: str) -> str:
    """Drop numpy imports and ``is_floats`` / ``assertion`` helpers at any indent.

    Fetch indents an MBPP+ plus body into ``check``, so those helpers are
    nested there and the inner ``assertion`` would shadow the tolerant one.
    The same drop runs on the text before ``check`` and on the ``check`` body.
    Other code stays. HumanEval/14, 15, 32, 83, 100, 130, and 139 define
    ``ref_func`` or ``_poly`` before ``def check``, and the plus tests call
    those names.
    """
    lines = (prefix or "").replace("\r\n", "\n").split("\n")
    kept: list[str] = []
    skipping = False
    skip_indent = 0
    for line in lines:
        if skipping:
            if not line.strip():
                continue
            indent = len(line[: len(line) - len(line.lstrip(" \t"))])
            if indent > skip_indent:
                continue
            skipping = False
        stripped = line.strip()
        if re.match(r"^(?:import|from)\s+numpy\b", stripped):
            continue
        if re.match(r"^[ \t]*def\s+(?:is_floats|assertion|_close|values_close)\b", line):
            skipping = True
            skip_indent = len(line[: len(line) - len(line.lstrip(" \t"))])
            continue
        kept.append(line)
    return "\n".join(kept).strip("\n")


def prepare_plus_test(source: str) -> str:
    """Install the current assertion helper before a plus test runs.

    Pinned rows already contain whatever helper fetch wrote, including a copy
    nested inside ``check``. Scoring strips that copy and rewrites the helper
    so a kit fetched earlier uses the tolerant compare.
    """
    text = str(source or "")
    if "def check(" not in text:
        return _drop_numpy_helpers(text) if text.strip() else text
    try:
        return rewrite_evalplus_test(text)
    except ValueError:
        return _drop_numpy_helpers(text)


def rewrite_evalplus_test(source: str) -> str:
    """Replace the numpy assertion helper. Inputs, expected values, and other helpers stay.

    ``is_floats`` and ``assertion`` are removed before ``check`` and inside it.
    """
    text = source or ""
    marker = "def check("
    index = text.find(marker)
    if index < 0:
        raise ValueError("EvalPlus test has no def check.")
    # The set marker sits in the assertion helper. After fetch that helper is
    # inside check, so the search covers the whole test.
    header = _SET_HEADER if "out = set(out)" in text else _STDLIB_HEADER
    prefix = _drop_numpy_helpers(text[:index])
    body = _drop_numpy_helpers(text[index:])
    rewritten = header.strip() + "\n\n"
    if prefix:
        rewritten += prefix + "\n\n"
    rewritten += body.strip() + "\n"
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


# MBPP+ task 56 defines ``check``. A test function of the same name replaces
# that solution, and the call then fails with ``'int' object is not callable``.
TEST_FUNCTION = "_standard_bench_check"


def test_imports_block(row: dict) -> str:
    """Module-level imports from the row. Base asserts such as ``math.isclose`` need these."""
    raw = row.get("test_imports") or []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return ""
    lines = [str(item).strip() for item in raw if str(item).strip()]
    return "\n".join(lines)


def _bind_test(test: str) -> str:
    return re.sub(r"(?m)^def check\(", f"def {TEST_FUNCTION}(", test, count=1)


def _program(code: str, test: str, entry_point: str) -> str:
    body = _bind_test(test.rstrip())
    return code.rstrip() + "\n\n" + body + f"\n\n{TEST_FUNCTION}({entry_point})\n"


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
    imports = test_imports_block(row)
    if imports:
        program = imports + "\n" + program
    base = run_tests(program, str(row.get("base_test") or ""), entry, limits)
    if base["ok"]:
        plus = run_tests(program, prepare_plus_test(str(row.get("plus_test") or "")), entry, limits)
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
