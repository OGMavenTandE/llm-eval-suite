import { describe, expect, it } from "vitest";
import { assessReadiness } from "./readiness";

describe("assessReadiness", () => {
  it("returns not ready when API is unreachable", () => {
    const result = assessReadiness({ api_status: "error", app_status: "ok", package_version: "0.1", timestamp: "" }, undefined);
    expect(result.level).toBe("not_ready");
  });

  it("returns ready when resources are present and no warnings", () => {
    const result = assessReadiness(
      { api_status: "ok", app_status: "ok", package_version: "0.1", timestamp: "" },
      {
        app_version: "0.1",
        output_dir: "/tmp",
        run_index_path: "/tmp/index.json",
        profiles_count: 2,
        datasets_count: 1,
        models_count: 1,
        model_providers: ["ollama"],
        warnings: [],
        timestamp: new Date().toISOString(),
      },
    );
    expect(result.level).toBe("ready");
    expect(result.headline).toMatch(/Ready to run evaluations/);
  });

  it("returns warning when resources exist but warnings are present", () => {
    const result = assessReadiness(
      { api_status: "ok", app_status: "ok", package_version: "0.1", timestamp: "" },
      {
        app_version: "0.1",
        output_dir: "/tmp",
        run_index_path: "/tmp/index.json",
        profiles_count: 1,
        datasets_count: 1,
        models_count: 1,
        model_providers: ["ollama"],
        warnings: ["Model connection could not be verified."],
        timestamp: new Date().toISOString(),
      },
    );
    expect(result.level).toBe("warning");
  });
});
