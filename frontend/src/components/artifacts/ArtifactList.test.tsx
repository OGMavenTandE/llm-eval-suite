import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ArtifactList } from "./ArtifactList";

describe("ArtifactList", () => {
  it("groups artifacts into human-readable sections", () => {
    render(
      <ArtifactList
        ready
        files={[
          { kind: "audit", path: "/tmp/audit.json", exists: true },
          { kind: "summary", path: "/tmp/summary.csv", exists: true, model_name: "llama3" },
          { kind: "detailed", path: "/tmp/detailed.csv", exists: true, model_name: "llama3" },
        ]}
      />,
    );

    expect(screen.getByText("Audit")).toBeInTheDocument();
    expect(screen.getByText("Results")).toBeInTheDocument();
    expect(screen.getByText("Detailed Samples")).toBeInTheDocument();
    expect(screen.getByText("Run audit record")).toBeInTheDocument();
    expect(screen.getByText("Score summary (llama3)")).toBeInTheDocument();
  });

  it("offers a copy path action for each artifact", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.spyOn(navigator.clipboard, "writeText").mockImplementation(writeText);

    render(
      <ArtifactList
        ready
        files={[{ kind: "audit", path: "/tmp/audit.json", exists: true }]}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Copy file path" }));
    expect(writeText).toHaveBeenCalledWith("/tmp/audit.json");
  });

  it("explains when artifacts are not ready", () => {
    render(<ArtifactList ready={false} files={[]} message="Run still in progress." />);
    expect(screen.getByText("Output files not ready yet")).toBeInTheDocument();
    expect(screen.getByText("Run still in progress.")).toBeInTheDocument();
  });
});
