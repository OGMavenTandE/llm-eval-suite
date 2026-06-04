import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
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
});
