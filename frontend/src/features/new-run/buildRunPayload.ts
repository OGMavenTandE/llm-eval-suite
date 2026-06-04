import { DatasetSummary, ModelSummary, ProfileDetail } from "../../lib/types";

export function buildRunConfig(
  profile: ProfileDetail,
  selectedModel: ModelSummary,
  selectedDataset: DatasetSummary,
  runName: string,
) {
  return {
    run_name: runName || profile.run_name || "workbench-run",
    output_dir: profile.output_dir,
    dataset: selectedDataset.path,
    models: [{ name: selectedModel.name, provider: selectedModel.provider }],
    evaluators: profile.evaluators,
  };
}

export interface CreateRunPayload {
  config: ReturnType<typeof buildRunConfig>;
  dry_run: boolean;
  run_name?: string;
  compare: boolean;
}

export function buildCreateRunPayload(
  profile: ProfileDetail,
  selectedModel: ModelSummary,
  selectedDataset: DatasetSummary,
  runName: string,
  dryRun: boolean,
  compare: boolean,
): CreateRunPayload {
  const resolvedRunName = runName || profile.run_name || "workbench-run";
  return {
    config: buildRunConfig(profile, selectedModel, selectedDataset, resolvedRunName),
    dry_run: dryRun,
    run_name: runName || undefined,
    compare,
  };
}
