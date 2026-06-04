import { describe, expect, it } from "vitest";
import { getArtifactGroupLabel, getArtifactLabel } from "./formatters";
import { ArtifactFile } from "./types";

describe("artifact labeling", () => {
  it("maps known artifact kinds to readable labels", () => {
    const file: ArtifactFile = { kind: "audit", path: "/tmp/audit.json", exists: true };
    expect(getArtifactLabel(file)).toBe("Run audit record");
    expect(getArtifactGroupLabel("audit")).toBe("Audit");
  });

  it("includes model name for per-model artifacts", () => {
    const file: ArtifactFile = {
      kind: "detailed",
      path: "/tmp/detailed.csv",
      exists: true,
      model_name: "llama3",
    };
    expect(getArtifactLabel(file)).toBe("Sample-by-sample results (llama3)");
    expect(getArtifactGroupLabel("detailed")).toBe("Detailed Samples");
  });
});
