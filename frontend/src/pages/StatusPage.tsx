import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { assessReadiness, buildReadinessChecklist } from "../lib/readiness";
import { MetricCard } from "../components/cards/MetricCard";
import { SectionCard } from "../components/cards/SectionCard";
import { ErrorState, LoadingState, SuccessState } from "../components/feedback";

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
        message="We could not confirm whether the workbench is ready."
        detail={
          (healthQuery.error as Error | undefined)?.message ||
          (statusQuery.error as Error | undefined)?.message ||
          "Unable to reach the local API."
        }
      />
    );
  }

  const health = healthQuery.data!;
  const status = statusQuery.data!;
  const readiness = assessReadiness(health, status);
  const checklist = buildReadinessChecklist(health, status);

  return (
    <>
      <header className="page-header">
        <h1>System readiness</h1>
        <p>Use this page as a go/no-go check before starting an evaluation on this machine.</p>
      </header>

      <section className={`readiness-banner readiness-${readiness.level}`}>
        <h2>{readiness.headline}</h2>
        <p>{readiness.detail}</p>
        <p>
          <strong>Can I start an evaluation now?</strong> {readiness.canStartLabel}
        </p>
        <p>
          <strong>Recommended next step:</strong> {readiness.action}
        </p>
        <div className="button-row">
          {readiness.canStartNow !== "no" ? (
            <Link className="button" to="/new">
              Start new evaluation
            </Link>
          ) : null}
          <button className="button secondary" type="button" onClick={() => window.location.reload()}>
            Refresh status
          </button>
        </div>
      </section>

      <SectionCard
        title="Readiness checklist"
        description="Quick view of what is available and what may block an evaluation."
      >
        <ul className="readiness-checklist">
          {checklist.map((item) => (
            <li key={item.label} className={`readiness-check readiness-check-${item.status}`}>
              <strong>{item.label}</strong>
              <span>{item.detail}</span>
            </li>
          ))}
        </ul>
      </SectionCard>

      <SectionCard title="Connection and version" description="Basic service health on this machine.">
        <div className="card-grid">
          <MetricCard
            label="API reachable"
            value={health.api_status === "ok" ? "Yes" : "No"}
            hint={`Service status: ${health.app_status}`}
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
        <SuccessState
          title="No blocking warnings detected"
          message="Required resources were found and no major readiness warnings were reported."
        />
      )}
    </>
  );
}
