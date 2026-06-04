import { HealthResponse, SystemStatusResponse } from "../lib/types";

export type ReadinessLevel = "ready" | "warning" | "not_ready";

export interface ReadinessSummary {
  level: ReadinessLevel;
  headline: string;
  detail: string;
  action: string;
  canStartNow: "yes" | "with_caution" | "no";
  canStartLabel: string;
}

export interface ReadinessCheckItem {
  label: string;
  status: "ok" | "missing" | "warning";
  detail: string;
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
      canStartNow: "no",
      canStartLabel: "No. The workbench cannot be used until the API is reachable.",
    };
  }

  if (!status) {
    return {
      level: "not_ready",
      headline: "System details unavailable",
      detail: "The API responded, but readiness details could not be loaded.",
      action: "Refresh this page or restart the local evaluation service.",
      canStartNow: "no",
      canStartLabel: "No. Readiness details are unavailable.",
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
      canStartNow: "no",
      canStartLabel: "No. Required setup items are missing.",
    };
  }

  if (hasWarnings) {
    return {
      level: "warning",
      headline: "Ready with warnings",
      detail: `${status.warnings.length} warning${status.warnings.length === 1 ? "" : "s"} may affect evaluation quality.`,
      action: "Review the warnings below before starting a new evaluation.",
      canStartNow: "with_caution",
      canStartLabel: "Yes, but review warnings first.",
    };
  }

  return {
    level: "ready",
    headline: "Ready to run evaluations",
    detail: "The API is reachable and the required profiles, datasets, and models were detected.",
    action: "You can start a new evaluation now.",
    canStartNow: "yes",
    canStartLabel: "Yes. You can start a new evaluation now.",
  };
}

export function buildReadinessChecklist(
  health: HealthResponse,
  status: SystemStatusResponse,
): ReadinessCheckItem[] {
  const items: ReadinessCheckItem[] = [
    {
      label: "API reachable",
      status: health.api_status === "ok" ? "ok" : "missing",
      detail: health.api_status === "ok" ? "The local service responded successfully." : "The API did not respond.",
    },
    {
      label: "Evaluation profiles",
      status: status.profiles_count > 0 ? "ok" : "missing",
      detail:
        status.profiles_count > 0
          ? `${status.profiles_count} profile${status.profiles_count === 1 ? "" : "s"} available.`
          : "No profiles were found.",
    },
    {
      label: "Datasets",
      status: status.datasets_count > 0 ? "ok" : "missing",
      detail:
        status.datasets_count > 0
          ? `${status.datasets_count} dataset${status.datasets_count === 1 ? "" : "s"} discovered.`
          : "No datasets were found.",
    },
    {
      label: "Models and providers",
      status: status.models_count > 0 ? "ok" : "missing",
      detail:
        status.models_count > 0
          ? `${status.models_count} model${status.models_count === 1 ? "" : "s"} across ${status.model_providers.join(", ") || "unknown providers"}.`
          : "No models were detected.",
    },
  ];

  if (status.warnings.length) {
    items.push({
      label: "Warnings",
      status: "warning",
      detail: `${status.warnings.length} warning${status.warnings.length === 1 ? "" : "s"} reported.`,
    });
  }

  return items;
}
