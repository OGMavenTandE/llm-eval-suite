import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../lib/api";
import { formatDate, isFailureStatus, isSuccessStatus, isTerminalStatus } from "../lib/formatters";
import { LoadingState, ErrorState, InProgressState } from "../components/feedback";
import { SectionCard } from "../components/cards/SectionCard";
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
    return <LoadingState title="Loading run status" message="Fetching the latest progress for this evaluation..." />;
  }

  if (runQuery.error || !runQuery.data) {
    return (
      <ErrorState
        title="Run not found"
        message="We could not load this evaluation."
        detail={(runQuery.error as Error | undefined)?.message || "This evaluation could not be loaded."}
      />
    );
  }

  const run = runQuery.data;
  const isRunning = !isTerminalStatus(run.status);

  return (
    <>
      <header className="page-header">
        <h1>Evaluation in progress</h1>
        <p>This page refreshes automatically while the run is pending or running.</p>
      </header>

      <SectionCard title="Run details" description="Current status for this evaluation.">
        <div className="card">
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
          {run.error_message ? (
            <ErrorState
              title="Evaluation stopped with an error"
              message="This run did not complete successfully."
              detail={run.error_message}
            />
          ) : null}
        </div>
      </SectionCard>

      {isRunning ? (
        <InProgressState
          title="Evaluation running"
          message="Your evaluation is still running. This page checks for updates every few seconds."
        />
      ) : null}

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
      ) : null}
    </>
  );
}
