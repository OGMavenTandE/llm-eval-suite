"""Deterministic executive summary.

Ported from the unmerged Milestone 4A branch (PR #8,
``cursor/milestone-4a-executive-summary-6205``). The click-through app uses
this as the no-LLM fallback when a council or single judge is unavailable,
or when the number guard rejects a judge's narrative.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from llm_eval.schemas.executive_summary import ExecutiveSummary, NotableMetric

REVIEW_THRESHOLD = 0.8
METRIC_LABELS = {
    "correctness": "Answer accuracy",
    "exact_match": "Exact match rate",
    "latency": "Response time",
    "cost": "Cost",
    "robustness": "Robustness",
    "pass_rate": "Pass rate",
    "mean_score": "Average score",
}


def _metric_label(metric_name: str) -> str:
    return METRIC_LABELS.get(metric_name, metric_name.replace("_", " ").title())


def _basename(path: str | None) -> str:
    if not path:
        return "the selected dataset"
    return Path(path).name


def _truncate(text: str, limit: int = 120) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 3]}..."


def _collect_review_items(
    detailed_rows: list[dict],
    threshold: float = REVIEW_THRESHOLD,
) -> list[dict]:
    items: list[dict] = []
    for sample in detailed_rows:
        for evaluation in sample.get("evaluations", []):
            score = float(evaluation.get("score", 0))
            passed = bool(evaluation.get("passed", False))
            if not passed or score < threshold:
                items.append(
                    {
                        "sample_idx": sample.get("sample_idx", 0),
                        "prompt": sample.get("prompt", ""),
                        "score": score,
                        "metric_name": evaluation.get("metric_name", "metric"),
                    }
                )
    return sorted(items, key=lambda item: item["score"])


def _sample_highlights(
    detailed_rows: list[dict],
    *,
    best: bool,
    limit: int = 3,
) -> list[str]:
    if not detailed_rows:
        return []

    scored = []
    for sample in detailed_rows:
        evaluations = sample.get("evaluations") or []
        if not evaluations:
            continue
        score = float(evaluations[0].get("score", 0))
        scored.append((score, sample))

    scored.sort(key=lambda item: item[0], reverse=best)
    highlights: list[str] = []
    for score, sample in scored[:limit]:
        prompt = _truncate(str(sample.get("prompt", "Unknown question")))
        highlights.append(f"Question: \"{prompt}\" scored {score:.2f}.")
    return highlights


def _build_notable_metrics(summary_rows: list[dict]) -> list[NotableMetric]:
    metrics: list[NotableMetric] = []
    for row in summary_rows:
        metric_name = row.get("metric_name", "metric")
        mean_score = row.get("mean_score", "—")
        pass_rate = row.get("pass_rate", "—")
        sample_count = row.get("sample_count", "—")
        metrics.append(
            NotableMetric(
                label=_metric_label(str(metric_name)),
                value=str(mean_score),
                context=f"Pass rate {pass_rate} across {sample_count} questions",
            )
        )
    return metrics


def build_executive_summary(
    *,
    run_id: str,
    run_name: str,
    dataset_path: str | None,
    model_names: list[str],
    evaluator_names: list[str],
    model_results: list[dict],
    review_threshold: float = REVIEW_THRESHOLD,
) -> ExecutiveSummary:
    """Build a plain-language executive summary from loaded run results."""
    primary = model_results[0] if model_results else {}
    summary_rows = primary.get("summary") or []
    detailed_rows = primary.get("detailed") or []

    dataset_name = _basename(dataset_path)
    models_label = ", ".join(model_names) if model_names else "the selected model"
    evaluators_label = ", ".join(evaluator_names) if evaluator_names else "configured scoring rules"

    review_items = _collect_review_items(detailed_rows, review_threshold)
    strengths = _sample_highlights(detailed_rows, best=True)
    weaknesses = _sample_highlights(detailed_rows, best=False)
    notable_metrics = _build_notable_metrics(summary_rows)

    evaluation_purpose = (
        f"This evaluation tested {models_label} against {dataset_name} "
        f"using {evaluators_label} for run \"{run_name}\"."
    )

    if review_items:
        overall_outcome = (
            f"The evaluation finished with {len(review_items)} item"
            f"{'' if len(review_items) == 1 else 's'} that may need a closer look."
        )
        recommended_next_step = (
            "Review the flagged examples below before sharing results or deciding whether to rerun."
        )
        notes = "Results are based on automated scoring. Manual review is recommended where items were flagged."
    else:
        overall_outcome = "The evaluation finished without flagged items below the review threshold."
        recommended_next_step = (
            "Use the score overview and examples below to decide whether to accept the model, rerun, or compare runs."
        )
        notes = "Results are based on automated scoring on the configured dataset."

    if not strengths:
        strengths = ["No standout strong answers were identified in the saved detailed results."]
    if not weaknesses and review_items:
        weaknesses = [
            "Some answers did not meet the pass threshold. See the review section for details.",
        ]
    elif not weaknesses:
        weaknesses = ["No major weak answers were highlighted in the saved detailed results."]

    return ExecutiveSummary(
        run_id=run_id,
        generated_at=datetime.now(timezone.utc),
        evaluation_purpose=evaluation_purpose,
        overall_outcome=overall_outcome,
        recommended_next_step=recommended_next_step,
        key_strengths=strengths,
        key_weaknesses=weaknesses,
        needs_human_review_count=len(review_items),
        notable_metrics=notable_metrics,
        notes=notes,
    )


def narrative_from_summary(summary: ExecutiveSummary) -> str:
    """Join the summary fields into the paragraph the number guard checks."""
    parts = [
        summary.evaluation_purpose,
        summary.overall_outcome,
        summary.recommended_next_step,
        *summary.key_strengths,
        *summary.key_weaknesses,
    ]
    for metric in summary.notable_metrics:
        parts.append(f"{metric.label}: {metric.value}. {metric.context}")
    if summary.notes:
        parts.append(summary.notes)
    return "\n".join(part for part in parts if part)
