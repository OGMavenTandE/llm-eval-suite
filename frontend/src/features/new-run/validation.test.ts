import { describe, expect, it } from "vitest";
import { canAdvanceWizardStep } from "./validation";

describe("canAdvanceWizardStep", () => {
  const model = { provider: "ollama", name: "test", available: true, connection_status: "ok" };
  const dataset = {
    dataset_id: "d1",
    name: "Sample",
    path: "/datasets/sample.jsonl",
    valid: true,
  };
  const profileDetail = {
    profile_id: "p1",
    name: "Profile",
    path: "/config/profile.yaml",
    evaluators: [],
    model_names: ["test"],
    valid: true,
  };

  it("requires a model on step 0", () => {
    expect(canAdvanceWizardStep(0, null, null, null, undefined)).toBe(false);
    expect(canAdvanceWizardStep(0, model, null, null, undefined)).toBe(true);
  });

  it("requires a valid dataset on step 1", () => {
    expect(canAdvanceWizardStep(1, model, null, null, undefined)).toBe(false);
    expect(canAdvanceWizardStep(1, model, { ...dataset, valid: false }, null, undefined)).toBe(false);
    expect(canAdvanceWizardStep(1, model, dataset, null, undefined)).toBe(true);
  });

  it("requires a valid profile on step 2", () => {
    expect(canAdvanceWizardStep(2, model, dataset, null, undefined)).toBe(false);
    expect(canAdvanceWizardStep(2, model, dataset, "p1", { ...profileDetail, valid: false })).toBe(false);
    expect(canAdvanceWizardStep(2, model, dataset, "p1", profileDetail)).toBe(true);
  });
});
