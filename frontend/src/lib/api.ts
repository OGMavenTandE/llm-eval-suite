const API_BASE = import.meta.env.VITE_API_BASE ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const payload = await response.json();
      message = payload.message ?? message;
    } catch {
      // ignore parse errors
    }
    throw new Error(message);
  }

  return response.json() as Promise<T>;
}

export const api = {
  getHealth: () => request<import("./types").HealthResponse>("/health"),
  getSystemStatus: () => request<import("./types").SystemStatusResponse>("/system/status"),
  listProfiles: () =>
    request<{ profiles: import("./types").ProfileSummary[]; count: number }>("/profiles"),
  getProfile: (profileId: string) =>
    request<import("./types").ProfileDetail>(`/profiles/${profileId}`),
  listDatasets: () =>
    request<{ datasets: import("./types").DatasetSummary[]; count: number }>("/datasets"),
  listModels: () =>
    request<{ models: import("./types").ModelSummary[]; count: number; providers: string[] }>(
      "/models",
    ),
  listRuns: () => request<{ runs: import("./types").RunSummary[]; count: number }>("/runs"),
  getRun: (runId: string) => request<import("./types").RunDetail>(`/runs/${runId}`),
  getRunResults: (runId: string) =>
    request<import("./types").RunResultsResponse>(`/runs/${runId}/results`),
  getRunAudit: (runId: string) =>
    request<import("./types").RunAuditResponse>(`/runs/${runId}/audit`),
  getRunArtifacts: (runId: string) =>
    request<import("./types").RunArtifactsResponse>(`/runs/${runId}/artifacts`),
  getExecutiveSummary: (runId: string) =>
    request<import("./types").ExecutiveSummaryResponse>(`/runs/${runId}/reports/executive-summary`),
  createRun: (body: {
    config: Record<string, unknown>;
    dry_run?: boolean;
    run_name?: string;
    compare?: boolean;
  }) =>
    request<import("./types").CreateRunResponse>("/runs", {
      method: "POST",
      body: JSON.stringify(body),
    }),
};
