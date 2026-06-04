import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { MetricCard } from "../components/cards/MetricCard";
import { LoadingState, ErrorState } from "../components/feedback/FeedbackStates";

export function StatusPage() {
  const healthQuery = useQuery({ queryKey: ["health"], queryFn: api.getHealth });
  const statusQuery = useQuery({ queryKey: ["system-status"], queryFn: api.getSystemStatus });

  if (healthQuery.isLoading || statusQuery.isLoading) {
    return <LoadingState message="Checking system readiness..." />;
  }

  if (healthQuery.error || statusQuery.error) {
    return (
      <ErrorState
        message={
          (healthQuery.error as Error | undefined)?.message ||
          (statusQuery.error as Error | undefined)?.message ||
          "Unable to reach the local API."
        }
      />
    );
  }

  const status = statusQuery.data!;

  return (
    <>
      <header className="page-header">
        <h1>System status</h1>
        <p>
          Confirm the workbench is installed correctly and the resources needed for an evaluation
          are available on this machine.
        </p>
      </header>

      <div className="card-grid">
        <MetricCard label="API health" value={healthQuery.data!.api_status} />
        <MetricCard label="App version" value={status.app_version} />
        <MetricCard label="Profiles available" value={String(status.profiles_count)} />
        <MetricCard label="Datasets discovered" value={String(status.datasets_count)} />
        <MetricCard label="Models referenced" value={String(status.models_count)} />
      </div>

      <div className="card">
        <h3>Storage locations</h3>
        <p>
          <strong>Output directory:</strong> {status.output_dir}
        </p>
        <p>
          <strong>Run index:</strong> {status.run_index_path}
        </p>
        <p>
          <strong>Model providers:</strong> {status.model_providers.join(", ") || "None detected"}
        </p>
      </div>

      {status.warnings.length ? (
        <div className="card" style={{ marginTop: "1rem" }}>
          <h3>Warnings</h3>
          <ul>
            {status.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </div>
      ) : (
        <div className="feedback-box" style={{ marginTop: "1rem" }}>
          No major readiness warnings were detected.
        </div>
      )}
    </>
  );
}
