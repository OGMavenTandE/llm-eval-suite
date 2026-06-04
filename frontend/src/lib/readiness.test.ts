import { describe, expect, it } from "vitest";
import { assessReadiness, buildReadinessChecklist } from "./readiness";

describe("assessReadiness", () => {
  it("returns not ready when API is unreachable", () => {
    const result = assessReadiness({ api_status: "error", app_status: "ok", package_version: "0.1", timestamp: "" }, undefined);
    expect(result.level).toBe("not_ready");
    expect(result.canStartNow).toBe("no");
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
    expect(result.canStartNow).toBe("yes");
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
    expect(result.canStartNow).toBe("with_caution");
  });
});

describe("buildReadinessChecklist", () => {
  it("marks missing resources clearly", () => {
    const checklist = buildReadinessChecklist(
      { api_status: "ok", app_status: "ok", package_version: "0.1", timestamp: "" },
      {
        app_version: "0.1",
        output_dir: "/tmp",
        run_index_path: "/tmp/index.json",
        profiles_count: 0,
        datasets_count: 1,
        models_count: 0,
        model_providers: [],
        warnings: [],
        timestamp: new Date().toISOString(),
      },
    );

    expect(checklist.find((item) => item.label === "Evaluation profiles")?.status).toBe("missing");
    expect(checklist.find((item) => item.label === "Models and providers")?.status).toBe("missing");
  });
});
