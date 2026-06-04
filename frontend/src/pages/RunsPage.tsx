import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { RunsTable } from "../components/tables/RunsTable";
import { LoadingState, ErrorState } from "../components/feedback/FeedbackStates";

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
      {runsQuery.isLoading ? <LoadingState message="Loading runs..." /> : null}
      {runsQuery.error ? <ErrorState message={(runsQuery.error as Error).message} /> : null}
      {runsQuery.data ? <RunsTable runs={runsQuery.data} /> : null}
    </>
  );
}
