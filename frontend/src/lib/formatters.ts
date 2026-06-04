export function formatStatus(status: string): string {
  return status.replaceAll("_", " ");
}

export function formatDate(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

export function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function basename(path?: string | null): string {
  if (!path) return "—";
  const parts = path.split(/[/\\]/);
  return parts[parts.length - 1] || path;
}

export function isTerminalStatus(status?: string | null): boolean {
  if (!status) return false;
  return !["pending", "running"].includes(status);
}

export function isSuccessStatus(status?: string | null): boolean {
  return status === "completed" || status === "validated";
}

export function isFailureStatus(status?: string | null): boolean {
  return status === "failed_runtime" || status === "failed_validation";
}

export function buildReviewItems(
  detailed: import("./types").DetailedSample[] | undefined,
  threshold = 0.8,
): import("./types").ReviewItem[] {
  if (!detailed) return [];
  const items: import("./types").ReviewItem[] = [];

  for (const sample of detailed) {
    for (const evaluation of sample.evaluations) {
      if (!evaluation.passed || evaluation.score < threshold) {
        items.push({
          sample_idx: sample.sample_idx,
          prompt: sample.prompt,
          expected: sample.expected,
          model_response: sample.model_response,
          score: evaluation.score,
          passed: evaluation.passed,
          metric_name: evaluation.metric_name,
        });
      }
    }
  }

  return items.sort((a, b) => a.score - b.score);
}

export function summarizeMetrics(
  summaryRows: Array<Record<string, string>> | undefined,
): Array<{ metric: string; meanScore: string; passRate: string; samples: string }> {
  if (!summaryRows) return [];
  return summaryRows.map((row) => ({
    metric: formatMetricLabel(row.metric_name ?? "Metric"),
    meanScore: row.mean_score ?? "—",
    passRate: row.pass_rate ?? "—",
    samples: row.sample_count ?? "—",
  }));
}

export function formatMetricLabel(metricName: string): string {
  const labels: Record<string, string> = {
    correctness: "Answer accuracy",
    exact_match: "Exact match rate",
    pass_rate: "Pass rate",
    mean_score: "Average score",
  };
  return labels[metricName] ?? metricName.replaceAll("_", " ");
}

const ARTIFACT_LABELS: Record<string, string> = {
  audit: "Run audit record",
  summary: "Score summary",
  detailed: "Sample-by-sample results",
  comparison_summary: "Comparison summary",
  comparison_detailed: "Comparison details",
};

const ARTIFACT_GROUPS: Record<string, string> = {
  audit: "Audit",
  summary: "Results",
  detailed: "Detailed Samples",
  comparison_summary: "Comparison",
  comparison_detailed: "Comparison",
};

export function getArtifactLabel(file: import("./types").ArtifactFile): string {
  const base = ARTIFACT_LABELS[file.kind] ?? file.kind.replaceAll("_", " ");
  if (file.model_name && (file.kind === "summary" || file.kind === "detailed")) {
    return `${base} (${file.model_name})`;
  }
  return base;
}

export function getArtifactGroupLabel(kind: string): string {
  return ARTIFACT_GROUPS[kind] ?? "Other Files";
}

export interface ReportSummary {
  headline: string;
  evaluated: string;
  outcome: string;
  nextAction: string;
}

export function buildReportSummary(
  run: import("./types").RunDetail,
  resultsReady: boolean,
  reviewCount: number,
): ReportSummary {
  const runType = run.dry_run ? "validation check" : "full evaluation";
  const datasetName = basename(run.dataset_path);
  const models = run.model_names.join(", ") || "the selected model";

  if (!resultsReady) {
    return {
      headline: "Report not ready yet",
      evaluated: `This run is testing ${models} against ${datasetName}.`,
      outcome: "Results are still being prepared.",
      nextAction: "Open the progress page and return here when the run finishes.",
    };
  }

  if (isFailureStatus(run.status)) {
    return {
      headline: "Evaluation did not finish successfully",
      evaluated: `This ${runType} tested ${models} on ${datasetName}.`,
      outcome: run.error_message || "The run stopped before producing a complete report.",
      nextAction: "Review the error details, adjust your setup if needed, and start a new run.",
    };
  }

  if (run.dry_run) {
    return {
      headline: "Validation completed successfully",
      evaluated: `This validation checked ${models} against ${datasetName} without scoring every sample.`,
      outcome: "Configuration, dataset, and model connections look usable.",
      nextAction: "If this matches your intent, run a full evaluation to generate scored results.",
    };
  }

  if (reviewCount > 0) {
    return {
      headline: "Evaluation completed with items to review",
      evaluated: `This evaluation scored ${models} on ${datasetName}.`,
      outcome: `${reviewCount} sample${reviewCount === 1 ? "" : "s"} scored below the review threshold.`,
      nextAction: "Review the flagged examples below before sharing results or rerunning.",
    };
  }

  return {
    headline: "Evaluation completed successfully",
    evaluated: `This evaluation scored ${models} on ${datasetName}.`,
    outcome: "No samples were flagged for manual review.",
    nextAction: "Use the summary below to decide whether to accept the model, rerun, or compare runs.",
  };
}
