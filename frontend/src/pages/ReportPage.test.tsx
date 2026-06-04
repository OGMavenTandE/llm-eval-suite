import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReportPage } from "./ReportPage";

vi.mock("../lib/api", () => ({
  api: {
    getRun: vi.fn().mockResolvedValue({
      run_id: "run123",
      run_name: "Test Run",
      status: "completed",
      dry_run: false,
      dataset_path: "/datasets/sample.jsonl",
      model_names: ["test-model"],
      completed_at: new Date().toISOString(),
    }),
    getExecutiveSummary: vi.fn().mockResolvedValue({
      run_id: "run123",
      ready: true,
      summary: {
        run_id: "run123",
        generated_at: new Date().toISOString(),
        evaluation_purpose: "This evaluation tested test-model against sample.jsonl.",
        overall_outcome: "The evaluation finished without flagged items below the review threshold.",
        recommended_next_step: "Use the score overview below to decide next steps.",
        key_strengths: ["Question: \"What is 2+2?\" scored 1.00."],
        key_weaknesses: ["No major weak answers were highlighted in the saved detailed results."],
        needs_human_review_count: 0,
        notable_metrics: [
          { label: "Answer accuracy", value: "1.00", context: "Pass rate 100% across 1 questions" },
        ],
        notes: "Results are based on automated scoring on the configured dataset.",
      },
    }),
    getRunResults: vi.fn().mockResolvedValue({
      run_id: "run123",
      ready: true,
      model_results: [{ summary: [], detailed: [] }],
    }),
    getRunAudit: vi.fn().mockResolvedValue({ run_id: "run123", ready: true, audit: {} }),
    getRunArtifacts: vi.fn().mockResolvedValue({ run_id: "run123", ready: true, files: [] }),
  },
}));

describe("ReportPage executive summary", () => {
  it("renders the executive summary when available", async () => {
    const client = new QueryClient();

    render(
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={["/runs/run123"]}>
          <Routes>
            <Route path="/runs/:runId" element={<ReportPage />} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    );

    expect(await screen.findByLabelText("Executive summary")).toBeInTheDocument();
    expect(screen.getByText(/without flagged items below the review threshold/)).toBeInTheDocument();
    expect(screen.getByText("Main strengths")).toBeInTheDocument();
    expect(screen.getByText("Answer accuracy:")).toBeInTheDocument();
  });
});
