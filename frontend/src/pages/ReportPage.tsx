import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "../lib/api";
import {
  basename,
  buildReviewItems,
  formatDate,
  isFailureStatus,
  isSuccessStatus,
  summarizeMetrics,
} from "../lib/formatters";
import { LoadingState, ErrorState, EmptyState } from "../components/feedback/FeedbackStates";
import { MetricCard } from "../components/cards/MetricCard";
import { StatusBadge } from "../components/status/StatusBadge";

export function ReportPage() {
  const { runId = "" } = useParams();

  const runQuery = useQuery({ queryKey: ["run", runId], queryFn: () => api.getRun(runId), enabled: Boolean(runId) });
  const resultsQuery = useQuery({
    queryKey: ["run-results", runId],
    queryFn: () => api.getRunResults(runId),
    enabled: Boolean(runId),
  });
  const auditQuery = useQuery({
    queryKey: ["run-audit", runId],
    queryFn: () => api.getRunAudit(runId),
    enabled: Boolean(runId),
  });
  const artifactsQuery = useQuery({
    queryKey: ["run-artifacts", runId],
    queryFn: () => api.getRunArtifacts(runId),
    enabled: Boolean(runId),
  });

  if (runQuery.isLoading) {
    return <LoadingState message="Loading report..." />;
  }

  if (runQuery.error || !runQuery.data) {
    return <ErrorState message={(runQuery.error as Error | undefined)?.message || "Run not found."} />;
  }

  const run = runQuery.data;
  const primaryResults = resultsQuery.data?.model_results[0];
  const metrics = summarizeMetrics(primaryResults?.summary);
  const detailed = primaryResults?.detailed ?? [];
  const reviewItems = buildReviewItems(detailed);
  const strongest = [...detailed]
    .map((sample) => ({
      sample,
      score: sample.evaluations[0]?.score ?? 0,
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 3);
  const weakest = [...detailed]
    .map((sample) => ({
      sample,
      score: sample.evaluations[0]?.score ?? 1,
    }))
    .sort((a, b) => a.score - b.score)
    .slice(0, 3);

  const verdict = isSuccessStatus(run.status)
    ? run.dry_run
      ? "Validation completed successfully."
      : "Evaluation completed successfully."
    : isFailureStatus(run.status)
      ? "This run did not complete successfully."
      : "This run is still in progress.";

  return (
    <>
      <header className="page-header">
        <h1>{run.run_name}</h1>
        <p>{verdict}</p>
      </header>

      <div className="card-grid">
        <MetricCard label="Status" value={run.status.replaceAll("_", " ")} />
        <MetricCard label="Dataset" value={basename(run.dataset_path)} />
        <MetricCard label="Models" value={run.model_names.join(", ") || "—"} />
        <MetricCard label="Completed" value={formatDate(run.completed_at || run.started_at)} />
      </div>

      <p>
        <StatusBadge status={run.status} /> {run.dry_run ? "Dry run" : "Full evaluation"}
      </p>

      {!resultsQuery.data?.ready ? (
        <div className="feedback-box">
          {resultsQuery.data?.message || "Results are not ready yet."}{" "}
          <Link to={`/runs/${run.run_id}/progress`}>View progress</Link>
        </div>
      ) : null}

      {resultsQuery.data?.ready && metrics.length ? (
        <>
          <h2 className="section-title">Summary metrics</h2>
          <div className="metric-grid">
            {metrics.map((metric) => (
              <MetricCard
                key={metric.metric}
                label={metric.metric}
                value={metric.meanScore}
                hint={`Pass rate ${metric.passRate} · ${metric.samples} samples`}
              />
            ))}
          </div>
        </>
      ) : null}

      {resultsQuery.data?.ready ? (
        <>
          <h2 className="section-title">Strongest examples</h2>
          {strongest.length ? (
            <div className="review-list">
              {strongest.map(({ sample, score }) => (
                <div className="review-item" key={`strong-${sample.sample_idx}`}>
                  <h4>Sample {sample.sample_idx + 1}</h4>
                  <p>
                    <strong>Prompt:</strong> {sample.prompt}
                  </p>
                  <p>
                    <strong>Model answer:</strong> {sample.model_response}
                  </p>
                  <p>
                    <strong>Score:</strong> {score.toFixed(2)}
                  </p>
                </div>
              ))}
            </div>
          ) : (
            <EmptyState message="No detailed examples were available for this run." />
          )}

          <h2 className="section-title">Weakest examples</h2>
          {weakest.length ? (
            <div className="review-list">
              {weakest.map(({ sample, score }) => (
                <div className="review-item" key={`weak-${sample.sample_idx}`}>
                  <h4>Sample {sample.sample_idx + 1}</h4>
                  <p>
                    <strong>Prompt:</strong> {sample.prompt}
                  </p>
                  <p>
                    <strong>Expected:</strong> {sample.expected}
                  </p>
                  <p>
                    <strong>Model answer:</strong> {sample.model_response}
                  </p>
                  <p>
                    <strong>Score:</strong> {score.toFixed(2)}
                  </p>
                </div>
              ))}
            </div>
          ) : (
            <EmptyState message="No weak examples were identified." />
          )}

          <h2 className="section-title">What needs review</h2>
          {reviewItems.length ? (
            <div className="review-list">
              {reviewItems.map((item) => (
                <div className="review-item" key={`${item.sample_idx}-${item.metric_name}`}>
                  <h4>
                    Sample {item.sample_idx + 1} · {item.metric_name}
                  </h4>
                  <p>
                    <strong>Prompt:</strong> {item.prompt}
                  </p>
                  <p>
                    <strong>Expected:</strong> {item.expected}
                  </p>
                  <p>
                    <strong>Model answer:</strong> {item.model_response}
                  </p>
                  <p>
                    <strong>Score:</strong> {item.score.toFixed(2)} ·{" "}
                    {item.passed ? "Passed" : "Needs review"}
                  </p>
                  <p style={{ color: "#52606d" }}>Reviewer notes will be supported in a later milestone.</p>
                </div>
              ))}
            </div>
          ) : (
            <EmptyState message="No flagged examples require review." />
          )}
        </>
      ) : null}

      <h2 className="section-title">Audit and artifacts</h2>
      <div className="card">
        {auditQuery.data?.ready && auditQuery.data.audit ? (
          <p>
            <strong>Config hash:</strong> {String(auditQuery.data.audit.config_hash || "—")}
          </p>
        ) : (
          <p>{auditQuery.data?.message || "Audit metadata not available yet."}</p>
        )}
        {artifactsQuery.data?.files?.length ? (
          <ul>
            {artifactsQuery.data.files.map((file) => (
              <li key={`${file.kind}-${file.path}`}>
                {file.kind}: {file.path}
              </li>
            ))}
          </ul>
        ) : (
          <p>{artifactsQuery.data?.message || "No artifact files recorded yet."}</p>
        )}
      </div>

      <div className="button-row">
        <Link className="button secondary" to="/runs">
          Back to runs
        </Link>
        <Link className="button secondary" to={`/runs/${run.run_id}/progress`}>
          View progress
        </Link>
      </div>
    </>
  );
}
