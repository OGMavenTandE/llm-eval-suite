import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "../lib/api";
import {
  basename,
  buildReportSummary,
  buildReviewItems,
  formatDate,
  formatMetricLabel,
  summarizeMetrics,
} from "../lib/formatters";
import { LoadingState, ErrorState, EmptyState, NotReadyState } from "../components/feedback/FeedbackStates";
import { MetricCard } from "../components/cards/MetricCard";
import { SectionCard } from "../components/cards/SectionCard";
import { StatusBadge } from "../components/status/StatusBadge";
import { ArtifactList } from "../components/artifacts/ArtifactList";

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
    return <LoadingState title="Loading report" message="Gathering run results and audit details..." />;
  }

  if (runQuery.error || !runQuery.data) {
    return (
      <ErrorState
        title="Report unavailable"
        message={(runQuery.error as Error | undefined)?.message || "This run could not be found."}
      />
    );
  }

  const run = runQuery.data;
  const resultsReady = Boolean(resultsQuery.data?.ready);
  const primaryResults = resultsQuery.data?.model_results[0];
  const metrics = summarizeMetrics(primaryResults?.summary);
  const detailed = primaryResults?.detailed ?? [];
  const reviewItems = buildReviewItems(detailed);
  const reportSummary = buildReportSummary(run, resultsReady, reviewItems.length);

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

  return (
    <>
      <header className="page-header">
        <h1>{run.run_name}</h1>
        <p>Evaluation report and recommended next steps.</p>
      </header>

      <section className="report-summary card">
        <h2>{reportSummary.headline}</h2>
        <dl className="report-summary-list">
          <div>
            <dt>What was evaluated</dt>
            <dd>{reportSummary.evaluated}</dd>
          </div>
          <div>
            <dt>What this means</dt>
            <dd>{reportSummary.outcome}</dd>
          </div>
          <div>
            <dt>Recommended next step</dt>
            <dd>{reportSummary.nextAction}</dd>
          </div>
        </dl>
      </section>

      <div className="card-grid">
        <MetricCard label="Run status" value={run.status.replaceAll("_", " ")} />
        <MetricCard label="Dataset" value={basename(run.dataset_path)} />
        <MetricCard label="Models tested" value={run.model_names.join(", ") || "—"} />
        <MetricCard label="Finished" value={formatDate(run.completed_at || run.started_at)} />
      </div>

      <p>
        <StatusBadge status={run.status} /> {run.dry_run ? "Validation run" : "Full evaluation"}
      </p>

      {!resultsReady ? (
        <NotReadyState
          title="Results still preparing"
          message={
            <>
              {resultsQuery.data?.message || "Scored results are not ready yet."}{" "}
              <Link to={`/runs/${run.run_id}/progress`}>View progress</Link>
            </>
          }
        />
      ) : null}

      {resultsReady && metrics.length ? (
        <SectionCard
          title="Score overview"
          description="High-level scoring results across all tested samples."
        >
          <div className="metric-grid">
            {metrics.map((metric) => (
              <MetricCard
                key={metric.metric}
                label={metric.metric}
                value={metric.meanScore}
                hint={`Samples passing: ${metric.passRate} · ${metric.samples} questions tested`}
              />
            ))}
          </div>
        </SectionCard>
      ) : null}

      {resultsReady ? (
        <>
          <SectionCard
            title="Strongest answers"
            description="Examples where the model performed best."
          >
            {strongest.length ? (
              <div className="review-list">
                {strongest.map(({ sample, score }) => (
                  <div className="review-item" key={`strong-${sample.sample_idx}`}>
                    <h4>Question {sample.sample_idx + 1}</h4>
                    <p>
                      <strong>Question:</strong> {sample.prompt}
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
          </SectionCard>

          <SectionCard
            title="Weakest answers"
            description="Examples where the model struggled or missed the expected answer."
          >
            {weakest.length ? (
              <div className="review-list">
                {weakest.map(({ sample, score }) => (
                  <div className="review-item" key={`weak-${sample.sample_idx}`}>
                    <h4>Question {sample.sample_idx + 1}</h4>
                    <p>
                      <strong>Question:</strong> {sample.prompt}
                    </p>
                    <p>
                      <strong>Expected answer:</strong> {sample.expected}
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
          </SectionCard>

          <SectionCard
            title="What needs review"
            description="Samples that did not meet the pass threshold and may need a human decision."
          >
            {reviewItems.length ? (
              <div className="review-list">
                {reviewItems.map((item) => (
                  <div className="review-item" key={`${item.sample_idx}-${item.metric_name}`}>
                    <h4>
                      Question {item.sample_idx + 1} · {formatMetricLabel(item.metric_name)}
                    </h4>
                    <p>
                      <strong>Question:</strong> {item.prompt}
                    </p>
                    <p>
                      <strong>Expected answer:</strong> {item.expected}
                    </p>
                    <p>
                      <strong>Model answer:</strong> {item.model_response}
                    </p>
                    <p>
                      <strong>Score:</strong> {item.score.toFixed(2)} ·{" "}
                      {item.passed ? "Passed" : "Needs review"}
                    </p>
                    <p className="review-note">Reviewer notes will be supported in a later milestone.</p>
                  </div>
                ))}
              </div>
            ) : (
              <EmptyState
                title="No review needed"
                message="All scored samples met the pass threshold for this run."
              />
            )}
          </SectionCard>
        </>
      ) : null}

      <SectionCard
        title="Audit trail and output files"
        description="Supporting records saved with this run for traceability and offline review."
      >
        <div className="card nested-card">
          {auditQuery.data?.ready && auditQuery.data.audit ? (
            <p>
              <strong>Configuration fingerprint:</strong> {String(auditQuery.data.audit.config_hash || "—")}
            </p>
          ) : (
            <p>{auditQuery.data?.message || "Audit metadata not available yet."}</p>
          )}
          <ArtifactList
            files={artifactsQuery.data?.files ?? []}
            ready={Boolean(artifactsQuery.data?.ready)}
            message={artifactsQuery.data?.message}
            outputDir={artifactsQuery.data?.output_dir}
          />
        </div>
      </SectionCard>

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
