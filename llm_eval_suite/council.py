"""Council summary, adapted from the llm-council pattern.

Stage 1: each judge reviews the results JSON.
Stage 2: each judge ranks the other reviews, labeled Review A/B/C.
Stage 3: a chairman writes the plain-English summary.

One judge skips stage 2. Zero judges, a failed number guard, or a judge that
is the model under test falls back to the template summary from PR #8.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from llm_eval.reporting.executive_summary import (
    build_executive_summary,
    narrative_from_summary,
)

NUMBER_RE = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?%?)(?![A-Za-z0-9])")
_META_NOTE_RE = re.compile(
    r"(?i)("
    r"\bi omitted\b|"
    r"\bi have omitted\b|"
    r"\bomitted numbers\b|"
    r"\bnumbers (?:that )?(?:were|was) not\b|"
    r"\bi (?:only )?used numbers (?:that|which) appear\b|"
    r"\bno numbers were (?:invented|added)\b"
    r")"
)
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def extract_numbers(text: str) -> list[str]:
    return NUMBER_RE.findall(text or "")


def strip_meta_notes(text: str) -> str:
    """Drop sentences that talk about the number check instead of the results."""
    kept_paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text or ""):
        sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
        clean = [sentence for sentence in sentences if sentence and not _META_NOTE_RE.search(sentence)]
        if clean:
            kept_paragraphs.append(" ".join(clean))
    return "\n\n".join(kept_paragraphs).strip()


def _walk_numbers(value, found: list[float]) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        found.append(float(value))
        return
    if isinstance(value, dict):
        for item in value.values():
            _walk_numbers(item, found)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _walk_numbers(item, found)


def _close(target: float, value: float) -> bool:
    tolerance = 0.05 if abs(target) >= 2 else 0.006
    return abs(target - value) <= tolerance


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# A failed count is derived only from keys on the same object.
_TOTAL_KEYS = {"sample_count", "item_count", "total", "live_item_count"}
_PASSED_KEYS = {"passed_count", "passed", "passes", "pass_count"}
_RATE_KEYS = {"pass_rate", "rate", "overall_pass_rate"}


def _percent_complements(value, found: list[float]) -> None:
    """``100 - p`` for a value stored under a percent key, such as 68.4."""
    if isinstance(value, dict):
        for key, item in value.items():
            if _is_number(item) and "percent" in str(key).lower():
                number = float(item)
                if 0 < number <= 100:
                    found.append(100.0 - number)
            else:
                _percent_complements(item, found)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _percent_complements(item, found)


def _decimal_places(number: float) -> int:
    """Digits after the decimal in the shortest round-trip text."""
    text = str(number)
    if "e" in text or "E" in text:
        from decimal import Decimal

        text = format(Decimal(text), "f").rstrip("0").rstrip(".")
    if "." not in text:
        return 0
    return len(text.split(".", 1)[1])


def _rate_half_step(number: float, *, percent: bool) -> float:
    """Half a step of this stored rate, as a fraction of 1.

    A percent from ``round(rate * 100, 1)`` is off by at most 0.05 percentage
    points, which is 0.0005. A pass rate from ``round(rate, 4)`` uses half of
    0.0001. Digits finer than that known step use the finer step.
    """
    places = _decimal_places(number)
    if percent:
        places = max(places, 1)
        return 0.5 * (10 ** (-places)) / 100.0
    # pass_rate and overall_pass_rate are stored to 4 decimal places.
    places = max(places, 4)
    return 0.5 * (10 ** (-places))


def _allow_rate_count(total: float, rate: float, half_step: float, found: list[float]) -> None:
    """Allow integer counts within rounding distance of ``total * rate``.

    ``|n - total * rate| <= 0.5 + total * half_step``. The 0.5 is half a
    count. ``half_step`` is half the stored rate's last digit, so a one-decimal
    percent uses 0.0005. ``76 * 0.316`` is 24.016 and the slack is about 0.54,
    which accepts 24 and rejects 23 and 25.
    """
    if total <= 0 or rate < 0 or half_step < 0:
        return
    product = total * rate
    tolerance = 0.5 + total * half_step
    start = max(0, int(product - tolerance))
    end = int(product + tolerance) + 1
    for count in range(start, end + 1):
        if abs(count - product) <= tolerance:
            found.append(float(count))


def _rate_counts(value, found: list[float]) -> None:
    """Passed and failed counts from a total and a rate on the same object.

    ``count = total * rate`` and ``count = total * (100 - percent) / 100``,
    within half a count plus that rate's own rounding step. A rate on one
    category is not applied to another category's total.
    """
    if isinstance(value, dict):
        totals = []
        rates = []
        percents = []
        for key, item in value.items():
            if _is_number(item) and key in _TOTAL_KEYS and float(item) > 0:
                totals.append(float(item))
            elif _is_number(item) and key in _RATE_KEYS and 0 <= float(item) <= 1:
                rates.append((float(item), _rate_half_step(float(item), percent=False)))
            elif _is_number(item) and "percent" in str(key).lower() and 0 <= float(item) <= 100:
                percents.append((float(item), _rate_half_step(float(item), percent=True)))
            elif isinstance(item, (dict, list, tuple)):
                _rate_counts(item, found)
        for total in totals:
            for rate, step in rates:
                _allow_rate_count(total, rate, step, found)
                _allow_rate_count(total, 1.0 - rate, step, found)
            for percent, step in percents:
                _allow_rate_count(total, percent / 100.0, step, found)
                _allow_rate_count(total, (100.0 - percent) / 100.0, step, found)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _rate_counts(item, found)


def _count_complements(value, found: list[float]) -> None:
    """Failed count when one object gives both the total and the passed count."""
    if isinstance(value, dict):
        totals = [float(item) for key, item in value.items() if key in _TOTAL_KEYS and _is_number(item)]
        passed = [float(item) for key, item in value.items() if key in _PASSED_KEYS and _is_number(item)]
        for total in totals:
            for count in passed:
                found.append(abs(total - count))
        for item in value.values():
            if isinstance(item, (dict, list, tuple)):
                _count_complements(item, found)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _count_complements(item, found)


def _failure_rate_percents(value, found: list[float]) -> None:
    """Overall failure percent, and its complement, from one object's counts.

    ``27 / 130`` is 20.769, so 20.8% is allowed. The fraction itself is not
    added: a bare 21 would otherwise match 0.2077 through the percent rule.
    """
    if isinstance(value, dict):
        failed = value.get("failure_count")
        total = value.get("item_count")
        if _is_number(failed) and _is_number(total) and float(total) > 0:
            percent = float(failed) / float(total) * 100.0
            found.append(percent)
            found.append(100.0 - percent)
        for item in value.values():
            if isinstance(item, (dict, list, tuple)):
                _failure_rate_percents(item, found)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _failure_rate_percents(item, found)


def _shown_score_counts(value, found: list[float]) -> None:
    """How many shown failures share a score, such as 7 at 0.0 and 1 at 0.5."""
    if isinstance(value, dict):
        sample = value.get("failures_sample")
        if isinstance(sample, list):
            buckets: dict[float, int] = {}
            for item in sample:
                if isinstance(item, dict) and _is_number(item.get("score")):
                    score = float(item["score"])
                    buckets[score] = buckets.get(score, 0) + 1
            for count in buckets.values():
                found.append(float(count))
        for item in value.values():
            if isinstance(item, (dict, list, tuple)):
                _shown_score_counts(item, found)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _shown_score_counts(item, found)


def derived_figures(results) -> list[float]:
    """Whitelist of figures a summary may use beyond the raw input numbers.

    Allowed: the complement of an input percent (``100 - p``), the complement
    of a rate stored between 0 and 1, ``total - passed`` when those two counts
    are keys on the same object, ``total * rate`` or ``total * (100 - percent) / 100``
    when that total and that rate are on the same object, the overall failure
    percent ``failure_count / item_count`` on that same object, and the count
    of shown failures at one score. A derived count must land within half a
    count plus the stored rate's rounding step. Sums and differences of
    unrelated figures are not allowed.
    """
    values: list[float] = []
    _walk_numbers(results, values)
    derived: list[float] = []
    _percent_complements(results, derived)
    _count_complements(results, derived)
    _rate_counts(results, derived)
    _failure_rate_percents(results, derived)
    _shown_score_counts(results, derived)
    for value in values:
        if 0 <= value <= 1:
            derived.append(1.0 - value)
        # A non-integer in (1, 100] is a percent even when the key name is not.
        # Integer counts in that range are not percents, so 100 - 76 is not 24.
        if isinstance(value, float) and not value.is_integer() and 1 < value <= 100:
            derived.append(100.0 - value)
    return derived


def number_allowed(token: str, values: list[float]) -> bool:
    """Match a narrative number to one of the figures in ``values``.

    A bare count below 10 is not treated as a percent, so "2 items" does not
    match a score of 0.02. "80" and "80%" both match 0.8. Rounding uses the
    same tolerance as the rest of the guard.
    """
    percent = token.endswith("%")
    try:
        raw = float(token[:-1] if percent else token)
    except ValueError:
        return False
    targets = [raw]
    if percent or raw >= 10:
        targets.append(raw / 100.0)
    for target in targets:
        for value in values:
            if _close(target, value):
                return True
    return False


_OF_RE = re.compile(r"(?<![\d.])(\d+)\s+(?:out of|of)\s+(\d+)(?!\d)", re.IGNORECASE)
_COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_SCORE_COUNT_RE = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
    r"(?:responses?|samples?|items?|failures?|prompts?)?\s*"
    r"scored\s+(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _category_rows(value) -> list[dict]:
    rows: list[dict] = []
    if isinstance(value, dict):
        categories = value.get("categories")
        if isinstance(categories, list):
            rows.extend(row for row in categories if isinstance(row, dict))
        for item in value.values():
            if isinstance(item, (dict, list, tuple)):
                rows.extend(_category_rows(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            rows.extend(_category_rows(item))
    return rows


def _split_counts(category: dict) -> set[int]:
    """Passed and failed counts for one category, within the rate tolerance."""
    total = category.get("sample_count")
    if not _is_number(total) or float(total) <= 0:
        return set()
    total_f = float(total)
    allowed: set[int] = set()
    rates: list[tuple[float, float]] = []
    rate = category.get("pass_rate")
    if _is_number(rate) and 0 <= float(rate) <= 1:
        rates.append((float(rate), _rate_half_step(float(rate), percent=False)))
    for key, item in category.items():
        if _is_number(item) and "percent" in str(key).lower() and 0 <= float(item) <= 100:
            rates.append((float(item) / 100.0, _rate_half_step(float(item), percent=True)))
    passed = category.get("passed_count")
    if _is_number(passed):
        allowed.add(int(round(float(passed))))
        allowed.add(int(round(abs(total_f - float(passed)))))
    for share_rate, step in rates:
        for share in (share_rate, 1.0 - share_rate):
            product = total_f * share
            tolerance = 0.5 + total_f * step
            start = max(0, int(product - tolerance))
            end = int(product + tolerance) + 1
            for count in range(start, end + 1):
                if abs(count - product) <= tolerance:
                    allowed.add(count)
    return allowed


def _category_claim_tokens(narrative: str, results) -> list[str]:
    """``X out of N`` must be that category's failed or passed count.

    Both numbers can be real on their own. 27 is the overall failure count
    and 76 is the Security total, but "27 out of 76" is still wrong.
    """
    allowed_by_total: dict[int, set[int]] = {}
    for row in _category_rows(results):
        total = row.get("sample_count")
        if not _is_number(total) or float(total) != int(total):
            continue
        allowed_by_total.setdefault(int(total), set()).update(_split_counts(row))
    bad = []
    for match in _OF_RE.finditer(narrative or ""):
        total = int(match.group(2))
        if total not in allowed_by_total:
            continue
        if int(match.group(1)) not in allowed_by_total[total]:
            bad.append(match.group(1))
    return bad


def _failure_samples(value) -> list[dict] | None:
    if isinstance(value, dict):
        sample = value.get("failures_sample")
        if isinstance(sample, list):
            return [item for item in sample if isinstance(item, dict)]
        for item in value.values():
            if isinstance(item, (dict, list, tuple)):
                found = _failure_samples(item)
                if found is not None:
                    return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, (dict, list, tuple)):
                found = _failure_samples(item)
                if found is not None:
                    return found
    return None


def _score_claim_tokens(narrative: str, results) -> list[str]:
    """``one response scored 0.0`` has to match the shown failures at 0.0."""
    sample = _failure_samples(results)
    if not sample:
        return []
    buckets: dict[float, int] = {}
    for item in sample:
        if _is_number(item.get("score")):
            score = float(item["score"])
            buckets[score] = buckets.get(score, 0) + 1
    bad = []
    for match in _SCORE_COUNT_RE.finditer(narrative or ""):
        raw = match.group(1)
        word = _COUNT_WORDS.get(raw.lower())
        claimed = word if word is not None else int(raw)
        target = float(match.group(2))
        actual = sum(count for score, count in buckets.items() if abs(score - target) <= 1e-9)
        if claimed != actual:
            bad.append(raw)
    return bad


def unmatched_numbers(narrative: str, results: dict) -> list[str]:
    """A narrative number must be an input figure or one whitelisted derivation.

    Rounding and percent-versus-fraction are allowed. A percent complement, a
    total-minus-passed count, a total times that same object's pass rate or
    failing share (within half a count plus that rate's rounding step), and
    the overall failure percent from the failed count and the item total are
    allowed. Any other sum or difference is rejected.
    ``X out of N`` is rejected when N is a category total and X is not that
    category's failed or passed count. A count of shown failures at a score
    is rejected when it does not match those rows.
    Numbers glued to words, such as a model name, are ignored.
    """
    blob = json.dumps(results, default=str)
    present = set(extract_numbers(blob))
    values: list[float] = []
    _walk_numbers(results, values)
    allowed = values + derived_figures(results)
    bad = []
    seen = set()

    def add(token: str) -> None:
        if token not in seen:
            seen.add(token)
            bad.append(token)

    for token in extract_numbers(narrative):
        if token in present or token.rstrip("%") in present:
            continue
        if number_allowed(token, allowed):
            continue
        add(token)
    for token in _category_claim_tokens(narrative, results):
        add(token)
    for token in _score_claim_tokens(narrative, results):
        add(token)
    return bad


def drop_unsupported_sentences(text: str, results: dict) -> str:
    """Drop sentences whose numbers are not supported by the results."""
    kept_paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text or ""):
        sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
        clean = [sentence for sentence in sentences if sentence and not unmatched_numbers(sentence, results)]
        if clean:
            kept_paragraphs.append(" ".join(clean))
    return "\n\n".join(kept_paragraphs).strip()


def _lacks_terminal_punctuation(text: str) -> bool:
    stripped = (text or "").rstrip()
    return not stripped or stripped[-1] not in ".!?"


def complete_sentences(text: str) -> str:
    """Keep text through the last complete sentence. Drop a trailing fragment."""
    raw = (text or "").strip()
    if not raw:
        return ""
    last = -1
    for index, char in enumerate(raw):
        if char not in ".!?":
            continue
        if char == "." and index > 0 and raw[index - 1].isdigit():
            nxt = raw[index + 1] if index + 1 < len(raw) else ""
            if nxt.isdigit():
                continue
        last = index
    if last < 0:
        return ""
    return raw[: last + 1].strip()


def _present_narrative(text: str) -> str:
    """Strip meta notes and never leave a sentence that was cut off."""
    cleaned = strip_meta_notes(text or "")
    if cleaned and _lacks_terminal_punctuation(cleaned):
        cleaned = complete_sentences(cleaned)
    return cleaned.strip()


def _needs_more_room(text: str, done_reason: str | None) -> bool:
    if done_reason in {"length", "max_tokens"}:
        return True
    if not (text or "").strip():
        return False
    return _lacks_terminal_punctuation(text)


def anonymize_reviews(reviews: list[dict]) -> list[dict]:
    hidden = []
    for index, review in enumerate(reviews):
        label = f"Review {LABELS[index]}"
        hidden.append({"label": label, "text": review.get("text") or ""})
    return hidden


def build_rank_prompt(hidden_reviews: list[dict]) -> str:
    blocks = []
    for review in hidden_reviews:
        blocks.append(f"{review['label']}\n{review['text']}")
    body = "\n\n".join(blocks)
    return (
        "TASK: rank\n"
        "Rank the other judges' reviews for accuracy and insight. "
        "The reviews are anonymous. Refer to them only as Review A, Review B, and so on.\n"
        "Reply with one line starting with RANKING: and the labels from best to worst, separated by >.\n"
        "Example: RANKING: Review B > Review A\n\n"
        f"{body}"
    )


def parse_ranking(text: str, labels: list[str]) -> list[str]:
    line = ""
    for raw in (text or "").splitlines():
        if "RANKING:" in raw:
            line = raw.split("RANKING:", 1)[1]
            break
    ordered = []
    if line:
        pieces = re.split(r">|,|\n", line)
        for piece in pieces:
            cleaned = " ".join(piece.split())
            for label in labels:
                if label.lower() in cleaned.lower() and label not in ordered:
                    ordered.append(label)
    if not ordered:
        for label in labels:
            if label in (text or "") and label not in ordered:
                ordered.append(label)
    for label in labels:
        if label not in ordered:
            ordered.append(label)
    return ordered


def aggregate_rankings(rankings: list[list[str]]) -> list[dict]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        size = len(ranking)
        for place, label in enumerate(ranking):
            scores[label] = scores.get(label, 0.0) + (size - place)
    ordered = sorted(scores, key=lambda label: (-scores[label], label))
    return [{"label": label, "points": scores[label]} for label in ordered]


def judge_payload(run: dict, items: list[dict], *, failure_cap: int = 8) -> dict:
    failures = []
    for item in items:
        if item.get("passed"):
            continue
        if item.get("counts_toward_score") is False:
            continue
        if item.get("source") != "live":
            continue
        failures.append(
            {
                "id": item.get("id"),
                "category": item.get("category"),
                "prompt": _clip(item.get("prompt") or ""),
                "response": _clip(item.get("response") or ""),
                "expected": item.get("expected"),
                "score": item.get("score"),
                "source": item.get("source"),
            }
        )
        if len(failures) >= failure_cap:
            break
    card = run.get("scorecard") or {}
    return {
        "run_id": run.get("run_id"),
        "preset": run.get("preset"),
        "model": (run.get("connection") or {}).get("model"),
        "overall_pass_rate": card.get("overall_pass_rate"),
        "overall_pass_percent": card.get("overall_pass_percent"),
        "failure_count": card.get("failure_count"),
        "failures_shown": len(failures),
        "item_count": card.get("item_count"),
        "live_item_count": card.get("live_item_count"),
        "categories": card.get("categories") or [],
        "failures_sample": failures,
        "sample_scores": [
            item.get("score")
            for item in items
            if item.get("score") is not None
        ][:30],
        "suites": [
            {"name": suite.get("name"), "source": suite.get("source"), "label": suite.get("label")}
            for suite in run.get("suites") or []
        ],
    }


def _clip(text: str, limit: int = 400) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def template_narrative(run: dict, items: list[dict]) -> str:
    """No-LLM summary. Reuses the PR #8 executive-summary builder."""
    payload = judge_payload(run, items, failure_cap=8)
    detailed = []
    for index, item in enumerate(payload["failures_sample"]):
        score = item.get("score")
        detailed.append(
            {
                "sample_idx": index,
                "prompt": item.get("prompt") or "",
                "evaluations": [
                    {
                        "score": 0.0 if score is None else float(score),
                        "passed": False,
                        "metric_name": item.get("category") or "score",
                    }
                ],
            }
        )
    if not detailed:
        for index, item in enumerate(items[:3]):
            if item.get("counts_toward_score") is False:
                continue
            score = item.get("score")
            detailed.append(
                {
                    "sample_idx": index,
                    "prompt": item.get("prompt") or "",
                    "evaluations": [
                        {
                            "score": 0.0 if score is None else float(score),
                            "passed": bool(item.get("passed")),
                            "metric_name": item.get("category") or "score",
                        }
                    ],
                }
            )
    summary_rows = []
    for row in payload["categories"]:
        if row.get("pass_rate") is None:
            continue
        summary_rows.append(
            {
                "metric_name": row["category"],
                "mean_score": row["pass_rate"],
                "pass_rate": row["pass_rate"],
                "sample_count": row["sample_count"],
            }
        )
    model_name = (run.get("connection") or {}).get("model") or "the selected model"
    summary = build_executive_summary(
        run_id=str(run.get("run_id") or ""),
        run_name=str(run.get("preset") or "evaluation"),
        dataset_path=(run.get("dataset") or {}).get("path"),
        model_names=[model_name],
        evaluator_names=["correctness"],
        model_results=[{"summary": summary_rows, "detailed": detailed}],
    )
    card = run.get("scorecard") or {}
    if "failure_count" in card:
        count = int(card.get("failure_count") or 0)
    else:
        from llm_eval_suite.scoring import live_failures

        count = len(live_failures(items))
    noun = "prompt" if count == 1 else "prompts"
    if count == 0:
        summary.overall_outcome = "The evaluation finished with no failing live prompts."
    else:
        summary.overall_outcome = f"The evaluation finished with {count} failing live {noun}."
    summary.needs_human_review_count = count
    return narrative_from_summary(summary)


_CHAIR_TOKEN_CAP = 2400


def _ollama_judge_call(profile: dict, prompt: str, max_tokens: int) -> tuple[str, str | None]:
    from llm_eval.models.ollama_model import ollama_chat

    base = (profile.get("base_url") or "http://127.0.0.1:11434").rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    reply = ollama_chat(
        base,
        profile.get("model") or "model",
        prompt,
        max_tokens=max_tokens,
        think=False if profile.get("think") is None else bool(profile.get("think")),
        timeout=int(profile.get("timeout") or 120),
    )
    return reply.get("text") or "", reply.get("done_reason")


def judge_generate(judge: dict, *, max_tokens: int):
    """One judge callable.

    Ollama judges call native ``/api/chat`` with ``think`` off and a context
    window sized to the prompt. The ``/v1`` chat route often ignores ``think``
    and leaves the reply in a thinking field, so ``content`` comes back empty.
    That thinking text is not the summary. A reply stopped for length, or one
    that does not end a sentence, is asked once more with a higher token cap
    and then trimmed to the last complete sentence.
    """
    from llm_eval.models.context import strip_think_blocks
    from llm_eval_suite.connections import build_model

    prepared = dict(judge)
    if prepared.get("think") is None:
        prepared["think"] = False

    if (prepared.get("type") or "") == "ollama":

        def _generate(prompt: str) -> str:
            text, reason = _ollama_judge_call(prepared, prompt, max_tokens)
            if _needs_more_room(text, reason):
                larger = min(_CHAIR_TOKEN_CAP, max_tokens * 2)
                if larger > max_tokens:
                    text, reason = _ollama_judge_call(prepared, prompt, larger)
            text = strip_think_blocks(text)
            if text.strip() and (_needs_more_room(text, reason) or _lacks_terminal_punctuation(text)):
                text = complete_sentences(text)
            return text

        return _generate

    model = build_model(prepared)

    def _generate(prompt: str) -> str:
        result = model.generate(prompt, max_tokens=max_tokens)
        text = result if isinstance(result, str) else result.text
        return strip_think_blocks(text)

    return _generate


def _ask(generate: Callable, prompt: str) -> str:
    result = generate(prompt)
    if isinstance(result, str):
        return result
    return result.text


def run_council(
    run: dict,
    items: list[dict],
    judges: list[dict],
    *,
    under_test: dict | None = None,
    chairman_name: str | None = None,
    generate_for: Callable | None = None,
    max_tokens: int = 1200,
) -> dict:
    """Run the council.

    ``generate_for(judge)`` returns a callable ``prompt -> text``. Tests pass
    stub callables. When omitted, each judge is built as a connection profile.
    """
    from llm_eval_suite.connections import same_model

    under_test = under_test or run.get("connection") or {}
    eligible = []
    excluded = []
    seen_models = set()
    for judge in judges:
        if same_model(judge, under_test):
            excluded.append(judge.get("model"))
            continue
        model_name = (judge.get("model") or "").strip()
        if model_name and model_name in seen_models:
            continue
        if model_name:
            seen_models.add(model_name)
        eligible.append(judge)

    payload = judge_payload(run, items)
    payload_json = json.dumps(payload, indent=2, default=str)
    template = template_narrative(run, items)

    if generate_for is None:
        def generate_for(judge):  # noqa: A001 - local factory
            return judge_generate(judge, max_tokens=max_tokens)

    if not eligible:
        return _fallback(
            template,
            payload,
            reason="No judge was used. The model under test is not allowed to grade itself, and no other judge is configured.",
            excluded=excluded,
            number_guard="pass",
        )

    # Drop judges that cannot be called. A single remaining judge skips ranking.
    usable = []
    errors = []
    for judge in eligible:
        try:
            usable.append((judge, generate_for(judge)))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{judge.get('model')}: {exc}")
    if not usable:
        return _fallback(
            template,
            payload,
            reason="No judge could be reached. " + "; ".join(errors),
            excluded=excluded,
            number_guard="pass",
        )

    review_prompt = (
        "TASK: review\n"
        "Review this evaluation results JSON. Write a short plain-English review. "
        "Use only numbers that appear in the JSON. Do not invent counts or rates.\n\n"
        f"{payload_json}"
    )
    reviews = []
    reviewed = []
    for judge, generate in usable:
        try:
            text = _ask(generate, review_prompt)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{judge.get('model')}: {exc}")
            continue
        reviews.append({"author": judge.get("model"), "text": text, "cloud": bool(judge.get("cloud"))})
        reviewed.append((judge, generate))

    if not reviews:
        return _fallback(
            template,
            payload,
            reason="Judges did not return reviews.",
            excluded=excluded,
            number_guard="pass",
        )

    # A single judge's review is the summary, so the guard below can retry it.
    # Two or more reviews are checked before they are shown to the chair.
    # A sentence with a number the results do not support is dropped, and a
    # review that then has nothing left is not passed on.
    if len(reviews) >= 2:
        kept_reviews = []
        kept_reviewed = []
        dropped_for_numbers = False
        for review, pair in zip(reviews, reviewed):
            cleaned = drop_unsupported_sentences(review.get("text") or "", payload)
            if cleaned:
                kept_reviews.append({**review, "text": cleaned})
                kept_reviewed.append(pair)
            elif (review.get("text") or "").strip():
                dropped_for_numbers = True
                errors.append(f"{review.get('author')}: review dropped; numbers were not in the results")
            else:
                kept_reviews.append(review)
                kept_reviewed.append(pair)
        reviews = kept_reviews
        reviewed = kept_reviewed
        if not reviews and dropped_for_numbers:
            return _fallback(
                template,
                payload,
                reason="The judge summary failed the number check, so the template summary was used.",
                excluded=excluded,
                errors=errors,
                number_guard="fallback",
            )

    rankings = []
    hidden = anonymize_reviews(reviews)
    mode = "single_judge"
    if len(reviews) >= 2:
        mode = "council"
        rank_prompt = build_rank_prompt(hidden)
        labels = [row["label"] for row in hidden]
        for judge, generate in reviewed:
            try:
                ranking_text = _ask(generate, rank_prompt)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{judge.get('model')} rank: {exc}")
                continue
            rankings.append(parse_ranking(ranking_text, labels))
    aggregate = aggregate_rankings(rankings) if rankings else []

    chairman = _pick_chairman(reviewed, chairman_name, under_test)
    # Reviews are checked against the results before they get here. The chair
    # is checked only against the results and the ranking points, never against
    # review text, so a wrong figure in a review cannot become allowed.
    chair_sources = payload
    if mode == "single_judge":
        narrative = _present_narrative(reviews[0]["text"])
        chair_label = reviews[0]["author"]
        chair_generate = reviewed[0][1]
    else:
        chair_judge, chair_generate = chairman
        chair_sources = {
            "results": payload,
            "aggregate_ranking": aggregate,
        }
        chair_prompt = (
            "TASK: chair\n"
            "Write a plain-English summary of this evaluation for a non-technical reader. "
            "Use the reviews for the prose and the aggregate ranking for order. "
            "Use only numbers from the results JSON or the aggregate ranking, "
            "a failing percent that is 100 minus an input percent, "
            "the overall failure rate from the failed count and the item total, "
            "or a failed count from the total and the pass rate on that same category. "
            "Do not copy a number that appears only in a review. "
            "Do not add or subtract any other figures. "
            "Finish the last sentence. "
            "Do not add a note about omitted numbers or about these instructions.\n\n"
            f"Aggregate ranking: {json.dumps(aggregate)}\n\n"
            f"Reviews:\n{json.dumps(hidden, indent=2)}\n\n"
            f"Results JSON:\n{payload_json}"
        )
        # An empty chairman reply is not asked again on the same model.
        # The other council judge writes the summary before the template does.
        narrative = ""
        chair_label = None
        candidates = [chairman] + [pair for pair in reviewed if pair[0] is not chairman[0]]
        for index, (judge, generate) in enumerate(candidates):
            if index >= 2:
                break
            try:
                narrative = _present_narrative(_ask(generate, chair_prompt))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"chairman: {exc}")
                narrative = ""
            chair_judge = judge
            chair_generate = generate
            chair_label = judge.get("model")
            if narrative.strip():
                break
            errors.append(f"{chair_label or 'chairman'}: empty summary")
        chairman = (chair_judge, chair_generate)

    narrative = _present_narrative(narrative)
    guard = unmatched_numbers(narrative, chair_sources)
    number_guard = "pass"
    if guard:
        retry_prompt = (
            "TASK: chair\n"
            "Your previous summary used numbers that are not in the inputs you were given. "
            "Rewrite it using only numbers from the results JSON or the aggregate ranking, "
            "a failing percent that is 100 minus an input percent, "
            "the overall failure rate from the failed count and the item total, "
            "or a failed count from the total and the pass rate on that same category. "
            "Do not copy a number that appears only in a review. "
            "Do not add or subtract any other figures. "
            "Finish the last sentence. "
            "Do not add a note about omitted numbers or about these instructions. "
            f"Unmatched numbers: {', '.join(guard)}.\n\n"
            f"Aggregate ranking: {json.dumps(aggregate)}\n\n"
            f"Results JSON:\n{payload_json}"
        )
        try:
            narrative = _present_narrative(_ask(chair_generate, retry_prompt))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"retry: {exc}")
            narrative = ""
        guard = unmatched_numbers(narrative, chair_sources)
        number_guard = "retry_pass" if not guard else "fallback"
    if guard or not narrative.strip():
        return _fallback(
            template,
            payload,
            reason="The judge summary failed the number check, so the template summary was used.",
            excluded=excluded,
            reviews=reviews,
            rankings=aggregate,
            hidden=hidden,
            errors=errors,
            number_guard="fallback",
        )

    clouds = [row["author"] for row in reviews if row.get("cloud")]
    locality = "cloud" if clouds else "local"
    if mode == "council":
        source_label = f"Council: {len(reviews)} judges, {locality}"
    else:
        source_label = f"Single judge, {locality}"
    return {
        "mode": mode,
        "source_label": source_label,
        "number_guard": number_guard,
        "narrative": narrative.strip(),
        "reviews": reviews,
        "anonymized_reviews": hidden,
        "rankings": aggregate,
        "chairman": chair_label,
        "excluded_judges": excluded,
        "errors": errors,
        "unmatched_numbers": [],
    }


def _pick_chairman(usable, chairman_name, under_test):
    from llm_eval_suite.connections import same_model

    if chairman_name:
        for judge, generate in usable:
            if judge.get("model") == chairman_name and not same_model(judge, under_test):
                return judge, generate
    return usable[0]


def _fallback(
    template,
    payload,
    *,
    reason,
    excluded,
    reviews=None,
    rankings=None,
    hidden=None,
    errors=None,
    number_guard="fallback",
):
    guard = unmatched_numbers(template, payload)
    # The template is deterministic. If a formatted score still fails the guard,
    # keep the template and record the mismatch rather than looping.
    recorded = "fallback" if guard and number_guard == "pass" else number_guard
    return {
        "mode": "template",
        "source_label": "Template",
        "number_guard": recorded,
        "narrative": template,
        "reviews": reviews or [],
        "anonymized_reviews": hidden or [],
        "rankings": rankings or [],
        "chairman": None,
        "excluded_judges": excluded,
        "errors": errors or [],
        "unmatched_numbers": guard,
        "reason": reason,
    }
