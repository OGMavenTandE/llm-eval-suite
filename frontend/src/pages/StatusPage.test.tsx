import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { StatusPage } from "./StatusPage";

vi.mock("../lib/api", () => ({
  api: {
    getHealth: vi.fn().mockResolvedValue({
      api_status: "ok",
      app_status: "ok",
      package_version: "0.1.0",
      timestamp: new Date().toISOString(),
    }),
    getSystemStatus: vi.fn().mockResolvedValue({
      app_version: "0.1.0",
      output_dir: "/workspace/results",
      run_index_path: "/workspace/results/.llm_eval_runs.json",
      profiles_count: 2,
      datasets_count: 1,
      models_count: 1,
      model_providers: ["ollama"],
      warnings: [],
      timestamp: new Date().toISOString(),
    }),
  },
}));

describe("StatusPage readiness", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows a ready summary when resources are available", async () => {
    const client = new QueryClient();
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter>
          <StatusPage />
        </MemoryRouter>
      </QueryClientProvider>,
    );

    expect(await screen.findByText(/Ready to run evaluations/)).toBeInTheDocument();
    expect(screen.getByText(/Can I start an evaluation now\?/)).toBeInTheDocument();
    expect(screen.getByText("Readiness checklist")).toBeInTheDocument();
    expect(screen.getByText("Profiles available")).toBeInTheDocument();
  });
});
