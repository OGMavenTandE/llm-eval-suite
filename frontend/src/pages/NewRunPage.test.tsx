import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { NewRunPage } from "./NewRunPage";

const mockNavigate = vi.fn();

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom");
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

vi.mock("../lib/api", () => ({
  api: {
    listModels: vi.fn().mockResolvedValue({
      models: [{ provider: "ollama", name: "test-model", available: true, connection_status: "reachable" }],
      count: 1,
      providers: ["ollama"],
    }),
    listDatasets: vi.fn().mockResolvedValue({
      datasets: [
        {
          dataset_id: "sample",
          name: "Sample",
          path: "/datasets/sample.jsonl",
          format: "jsonl",
          sample_count: 2,
          valid: true,
        },
      ],
      count: 1,
    }),
    listProfiles: vi.fn().mockResolvedValue({
      profiles: [
        {
          profile_id: "test_profile",
          name: "Test Profile",
          path: "/config/test_profile.yaml",
          evaluators: ["correctness"],
          model_names: ["test-model"],
          available: true,
          valid: true,
        },
      ],
      count: 1,
    }),
    getProfile: vi.fn().mockResolvedValue({
      profile_id: "test_profile",
      name: "Test Profile",
      path: "/config/test_profile.yaml",
      evaluators: [{ name: "correctness", mode: "exact_match", threshold: 0.8 }],
      run_name: "test-run",
      output_dir: "results/",
      valid: true,
      model_names: ["test-model"],
      available: true,
    }),
    createRun: vi.fn().mockResolvedValue({
      run_id: "abc123",
      status: "pending",
      created_at: new Date().toISOString(),
      output_dir: "results/",
      dry_run: true,
    }),
  },
}));

describe("NewRunPage wizard", () => {
  beforeEach(() => {
    mockNavigate.mockReset();
  });

  it("navigates through steps and submits a dry run", async () => {
    const user = userEvent.setup();
    const client = new QueryClient();

    render(
      <QueryClientProvider client={client}>
        <MemoryRouter>
          <NewRunPage />
        </MemoryRouter>
      </QueryClientProvider>,
    );

    expect(await screen.findByText(/Choose model/)).toBeInTheDocument();
    await user.click(await screen.findByText("test-model"));
    await user.click(screen.getByRole("button", { name: "Continue" }));

    await user.click(await screen.findByText("Sample"));
    await user.click(screen.getByRole("button", { name: "Continue" }));

    await user.click(await screen.findByText("Test Profile"));
    await user.click(screen.getByRole("button", { name: "Continue" }));

    await user.click(screen.getByRole("button", { name: "Start evaluation" }));
    expect(mockNavigate).toHaveBeenCalledWith("/runs/abc123/progress");
  });
});
