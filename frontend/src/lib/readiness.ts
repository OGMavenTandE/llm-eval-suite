import { HealthResponse, SystemStatusResponse } from "../lib/types";

export type ReadinessLevel = "ready" | "warning" | "not_ready";

export interface ReadinessSummary {
  level: ReadinessLevel;
  headline: string;
  detail: string;
  action: string;
}

export function assessReadiness(
  health: HealthResponse | undefined,
  status: SystemStatusResponse | undefined,
): ReadinessSummary {
  if (!health || health.api_status !== "ok") {
    return {
      level: "not_ready",
      headline: "The workbench is not reachable",
      detail: "The local API did not respond successfully.",
      action: "Confirm the evaluation service is running, then refresh this page.",
    };
  }

  if (!status) {
    return {
      level: "not_ready",
      headline: "System details unavailable",
      detail: "The API responded, but readiness details could not be loaded.",
      action: "Refresh this page or restart the local evaluation service.",
    };
  }

  const missingResources =
    status.profiles_count === 0 || status.datasets_count === 0 || status.models_count === 0;
  const hasWarnings = status.warnings.length > 0;

  if (missingResources) {
    const gaps: string[] = [];
    if (status.profiles_count === 0) gaps.push("evaluation profiles");
    if (status.datasets_count === 0) gaps.push("datasets");
    if (status.models_count === 0) gaps.push("models");
    return {
      level: "not_ready",
      headline: "Not ready to run evaluations",
      detail: `Missing required resources: ${gaps.join(", ")}.`,
      action: "Add the missing configuration or data files, then check status again.",
    };
  }

  if (hasWarnings) {
    return {
      level: "warning",
      headline: "Ready with warnings",
      detail: `${status.warnings.length} warning${status.warnings.length === 1 ? "" : "s"} may affect evaluation quality.`,
      action: "Review the warnings below before starting a new evaluation.",
    };
  }

  return {
    level: "ready",
    headline: "Ready to run evaluations",
    detail: "The API is reachable and the required profiles, datasets, and models were detected.",
    action: "You can start a new evaluation from the home page.",
  };
}
