import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "../lib/queryClient";
import { HomePage } from "../pages/HomePage";

vi.mock("../lib/api", () => ({
  api: {
    listRuns: vi.fn().mockResolvedValue({ runs: [], count: 0 }),
  },
}));

describe("HomePage", () => {
  it("renders primary actions", async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter>
          <HomePage />
        </MemoryRouter>
      </QueryClientProvider>,
    );

    expect(screen.getByText("Run new evaluation")).toBeInTheDocument();
    expect(screen.getByText("View previous runs")).toBeInTheDocument();
    expect(await screen.findByText("Recent runs")).toBeInTheDocument();
  });
});
