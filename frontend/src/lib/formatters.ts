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
  executive_summary: "Executive summary",
  report_manifest: "Report manifest",
  summary: "Score summary",
  detailed: "Sample-by-sample results",
  comparison_summary: "Comparison summary",
  comparison_detailed: "Comparison details",
};

const ARTIFACT_GROUPS: Record<string, string> = {
  audit: "Audit",
  executive_summary: "Reports",
  report_manifest: "Reports",
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

export function groupArtifactsBySection(
  files: import("./types").ArtifactFile[],
): Record<string, import("./types").ArtifactFile[]> {
  const groups: Record<string, import("./types").ArtifactFile[]> = {};

  for (const file of files) {
    const group = getArtifactGroupLabel(file.kind);
    if (!groups[group]) groups[group] = [];
    groups[group].push(file);
  }

  return groups;
}

