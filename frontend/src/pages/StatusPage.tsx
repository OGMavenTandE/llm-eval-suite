import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { assessReadiness } from "../lib/readiness";
import { MetricCard } from "../components/cards/MetricCard";
import { SectionCard } from "../components/cards/SectionCard";
import { LoadingState, ErrorState } from "../components/feedback/FeedbackStates";

export function StatusPage() {
  const healthQuery = useQuery({ queryKey: ["health"], queryFn: api.getHealth });
  const statusQuery = useQuery({ queryKey: ["system-status"], queryFn: api.getSystemStatus });

  if (healthQuery.isLoading || statusQuery.isLoading) {
    return <LoadingState title="Checking readiness" message="Confirming API access and local resources..." />;
  }

  if (healthQuery.error || statusQuery.error) {
    return (
      <ErrorState
        title="Status check failed"
        message={
          (healthQuery.error as Error | undefined)?.message ||
          (statusQuery.error as Error | undefined)?.message ||
          "Unable to reach the local API."
        }
      />
    );
  }

  const status = statusQuery.data!;
  const readiness = assessReadiness(healthQuery.data, status);

  return (
    <>
      <header className="page-header">
        <h1>System readiness</h1>
        <p>
          Use this page as a go/no-go check before starting an evaluation on this machine.
        </p>
      </header>

      <section className={`readiness-banner readiness-${readiness.level}`}>
        <h2>{readiness.headline}</h2>
        <p>{readiness.detail}</p>
        <p>
          <strong>Next step:</strong> {readiness.action}
        </p>
        {readiness.level === "ready" ? (
          <div className="button-row">
            <Link className="button" to="/new">
              Start new evaluation
            </Link>
          </div>
        ) : null}
      </section>

      <SectionCard title="Connection and version" description="Basic service health on this machine.">
        <div className="card-grid">
          <MetricCard
            label="API reachable"
            value={healthQuery.data!.api_status === "ok" ? "Yes" : "No"}
            hint={`App status: ${healthQuery.data!.app_status}`}
          />
          <MetricCard label="Workbench version" value={status.app_version} />
          <MetricCard label="Last checked" value={new Date(status.timestamp).toLocaleString()} />
        </div>
      </SectionCard>

      <SectionCard
        title="Evaluation resources"
        description="Counts of profiles, datasets, and models available for the setup wizard."
      >
        <div className="card-grid">
          <MetricCard label="Profiles available" value={String(status.profiles_count)} />
          <MetricCard label="Datasets discovered" value={String(status.datasets_count)} />
          <MetricCard label="Models detected" value={String(status.models_count)} />
          <MetricCard label="Model providers" value={status.model_providers.join(", ") || "None detected"} />
        </div>
      </SectionCard>

      <SectionCard title="Storage locations" description="Where run outputs and the run index are stored locally.">
        <div className="card">
          <p>
            <strong>Output directory:</strong> {status.output_dir}
          </p>
          <p>
            <strong>Run index file:</strong> {status.run_index_path}
          </p>
        </div>
      </SectionCard>

      {status.warnings.length ? (
        <SectionCard
          title="Warnings"
          description="These issues may block or reduce the usefulness of an evaluation."
        >
          <div className="card warning-card">
            <ul>
              {status.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          </div>
        </SectionCard>
      ) : (
        <div className="feedback-box success">
          <strong>No blocking warnings detected</strong>
          <p>Required resources were found and no major readiness warnings were reported.</p>
        </div>
      )}
    </>
  );
}
