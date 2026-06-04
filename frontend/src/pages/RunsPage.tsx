import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { RunsTable } from "../components/tables/RunsTable";
import { LoadingState, ErrorState, EmptyState } from "../components/feedback/FeedbackStates";

export function RunsPage() {
  const runsQuery = useQuery({
    queryKey: ["runs"],
    queryFn: async () => (await api.listRuns()).runs,
  });

  return (
    <>
      <header className="page-header">
        <h1>Previous runs</h1>
        <p>Browse recent evaluations, check their status, and reopen completed reports.</p>
      </header>
      {runsQuery.isLoading ? <LoadingState title="Loading runs" message="Fetching previous evaluations..." /> : null}
      {runsQuery.error ? (
        <ErrorState title="Unable to load runs" message={(runsQuery.error as Error).message} />
      ) : null}
      {runsQuery.data?.length === 0 ? (
        <EmptyState title="No runs yet" message="Completed evaluations will appear in this list." />
      ) : null}
      {runsQuery.data && runsQuery.data.length > 0 ? <RunsTable runs={runsQuery.data} /> : null}
    </>
  );
}
