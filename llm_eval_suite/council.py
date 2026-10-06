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
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def extract_numbers(text: str) -> list[str]:
    return NUMBER_RE.findall(text or "")


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


def number_allowed(token: str, values: list[float]) -> bool:
    """Match a narrative number to a JSON value, including rounding and percents.

    A bare count below 10 is not treated as a percent, so "2 items" does not
    match a score of 0.02. "80" and "80%" both match 0.8.
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
        tolerance = 0.05 if abs(target) >= 2 else 0.006
        for value in values:
            if abs(target - value) <= tolerance:
                return True
    return False


def unmatched_numbers(narrative: str, results: dict) -> list[str]:
    """A narrative number must show up in the results JSON, or match a value there.

    Rounding and percent-versus-fraction are allowed. Numbers glued to words,
    such as a model name, are ignored.
    """
    blob = json.dumps(results, default=str)
    present = set(extract_numbers(blob))
    values: list[float] = []
    _walk_numbers(results, values)
    bad = []
    for token in extract_numbers(narrative):
        if token in present or token.rstrip("%") in present:
            continue
        if number_allowed(token, values):
            continue
        bad.append(token)
    return bad


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


def judge_generate(judge: dict, *, max_tokens: int):
    """One judge callable. ``think`` is off, and any thinking trace is removed."""
    from llm_eval.models.context import strip_think_blocks
    from llm_eval_suite.connections import build_model

    prepared = dict(judge)
    if prepared.get("think") is None:
        prepared["think"] = False
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
    for judge, generate in usable:
        try:
            text = _ask(generate, review_prompt)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{judge.get('model')}: {exc}")
            continue
        reviews.append({"author": judge.get("model"), "text": text, "cloud": bool(judge.get("cloud"))})

    if not reviews:
        return _fallback(
            template,
            payload,
            reason="Judges did not return reviews.",
            excluded=excluded,
            number_guard="pass",
        )

    rankings = []
    hidden = anonymize_reviews(reviews)
    mode = "single_judge"
    if len(reviews) >= 2:
        mode = "council"
        rank_prompt = build_rank_prompt(hidden)
        labels = [row["label"] for row in hidden]
        for judge, generate in usable[: len(reviews)]:
            try:
                ranking_text = _ask(generate, rank_prompt)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{judge.get('model')} rank: {exc}")
                continue
            rankings.append(parse_ranking(ranking_text, labels))
    aggregate = aggregate_rankings(rankings) if rankings else []

    chairman = _pick_chairman(usable, chairman_name, under_test)
    # The guard allows every number we actually handed the chairman, including
    # ranking points. It does not allow a number that was not in that input.
    chair_sources = payload
    if mode == "single_judge":
        narrative = reviews[0]["text"]
        chair_label = reviews[0]["author"]
    else:
        chair_judge, chair_generate = chairman
        chair_sources = {
            "results": payload,
            "aggregate_ranking": aggregate,
            "reviews": hidden,
        }
        chair_prompt = (
            "TASK: chair\n"
            "Write a plain-English summary of this evaluation for a non-technical reader. "
            "Use the reviews, the aggregate ranking, and the results JSON. "
            "Use only numbers that appear in those inputs.\n\n"
            f"Aggregate ranking: {json.dumps(aggregate)}\n\n"
            f"Reviews:\n{json.dumps(hidden, indent=2)}\n\n"
            f"Results JSON:\n{payload_json}"
        )
        try:
            narrative = _ask(chair_generate, chair_prompt)
            chair_label = chair_judge.get("model")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"chairman: {exc}")
            narrative = ""
            chair_label = None

    guard = unmatched_numbers(narrative, chair_sources)
    number_guard = "pass"
    if guard:
        retry_prompt = (
            "TASK: chair\n"
            "Your previous summary used numbers that are not in the inputs you were given. "
            "Rewrite it using only numbers from those inputs. "
            f"Unmatched numbers: {', '.join(guard)}.\n\n"
            f"Aggregate ranking: {json.dumps(aggregate)}\n\n"
            f"Results JSON:\n{payload_json}"
        )
        try:
            narrative = _ask(chairman[1], retry_prompt)
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
