import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../lib/api";
import { formatDate, isFailureStatus, isSuccessStatus, isTerminalStatus } from "../lib/formatters";
import { LoadingState, ErrorState } from "../components/feedback/FeedbackStates";
import { StatusBadge } from "../components/status/StatusBadge";

export function RunProgressPage() {
  const { runId = "" } = useParams();
  const navigate = useNavigate();

  const runQuery = useQuery({
    queryKey: ["run", runId],
    queryFn: () => api.getRun(runId),
    enabled: Boolean(runId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status && isTerminalStatus(status) ? false : 2000;
    },
  });

  useEffect(() => {
    const status = runQuery.data?.status;
    if (status && isSuccessStatus(status)) {
      const timer = window.setTimeout(() => navigate(`/runs/${runId}`), 1200);
      return () => window.clearTimeout(timer);
    }
    return undefined;
  }, [runQuery.data?.status, navigate, runId]);

  if (runQuery.isLoading) {
    return <LoadingState message="Loading run status..." />;
  }

  if (runQuery.error || !runQuery.data) {
    return <ErrorState message={(runQuery.error as Error | undefined)?.message || "Run not found."} />;
  }

  const run = runQuery.data;

  return (
    <>
      <header className="page-header">
        <h1>Evaluation in progress</h1>
        <p>We will refresh this page automatically while the run is pending or running.</p>
      </header>

      <div className="card">
        <p>
          <strong>Run ID:</strong> {run.run_id}
        </p>
        <p>
          <strong>Run name:</strong> {run.run_name}
        </p>
        <p>
          <strong>Status:</strong> <StatusBadge status={run.status} />
        </p>
        <p>
          <strong>Started:</strong> {formatDate(run.started_at || run.created_at)}
        </p>
        <p>
          <strong>Last updated:</strong> {formatDate(run.completed_at || run.started_at || run.created_at)}
        </p>
        {run.message ? <p>{run.message}</p> : null}
        {run.error_message ? <div className="feedback-box error">{run.error_message}</div> : null}
      </div>

      {isFailureStatus(run.status) ? (
        <div className="button-row">
          <Link className="button secondary" to="/new">
            Start another evaluation
          </Link>
        </div>
      ) : null}

      {isSuccessStatus(run.status) ? (
        <div className="button-row">
          <Link className="button" to={`/runs/${run.run_id}`}>
            Open report
          </Link>
        </div>
      ) : (
        <div className="feedback-box">Your evaluation is still running. This page will update every few seconds.</div>
      )}
    </>
  );
}
