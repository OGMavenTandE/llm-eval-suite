import { DatasetSummary, ModelSummary, ProfileDetail } from "../../lib/types";

export function canAdvanceWizardStep(
  step: number,
  selectedModel: ModelSummary | null,
  selectedDataset: DatasetSummary | null,
  selectedProfileId: string | null,
  profileDetail: ProfileDetail | undefined,
): boolean {
  if (step === 0) return Boolean(selectedModel);
  if (step === 1) return Boolean(selectedDataset?.valid);
  if (step === 2) return Boolean(selectedProfileId && profileDetail?.valid);
  return true;
}
