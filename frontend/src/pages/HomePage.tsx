import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { RunsTable } from "../components/tables/RunsTable";
import { LoadingState, ErrorState, EmptyState } from "../components/feedback/FeedbackStates";

export function HomePage() {
  const runsQuery = useQuery({
    queryKey: ["runs"],
    queryFn: async () => (await api.listRuns()).runs.slice(0, 5),
  });

  return (
    <>
      <header className="page-header">
        <h1>Welcome to the Evaluation Workbench</h1>
        <p>
          This tool helps you run structured AI evaluations on your local system, review results in
          plain language, and keep an audit trail without editing configuration files by hand.
        </p>
      </header>

      <div className="button-row">
        <Link className="button" to="/new">
          Run new evaluation
        </Link>
        <Link className="button secondary" to="/runs">
          View previous runs
        </Link>
        <Link className="button secondary" to="/status">
          Check system status
        </Link>
      </div>

      <h2 className="section-title">Recent runs</h2>
      {runsQuery.isLoading ? <LoadingState title="Loading recent runs" message="Fetching your latest evaluations..." /> : null}
      {runsQuery.error ? (
        <ErrorState title="Unable to load runs" message={(runsQuery.error as Error).message} />
      ) : null}
      {runsQuery.data?.length === 0 ? (
        <EmptyState
          title="No runs yet"
          message="Start your first evaluation to see results here."
        />
      ) : null}
      {runsQuery.data && runsQuery.data.length > 0 ? <RunsTable runs={runsQuery.data} /> : null}
    </>
  );
}
