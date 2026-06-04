import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { api } from "../../lib/api";
import { DatasetSummary, ModelSummary } from "../../lib/types";
import { buildCreateRunPayload } from "./buildRunPayload";
import { canAdvanceWizardStep } from "./validation";

export const WIZARD_STEPS = ["Choose model", "Choose dataset", "Choose profile", "Review and start"];

export function useNewRunWizard() {
  const navigate = useNavigate();
  const [step, setStep] = useState(0);
  const [selectedModel, setSelectedModel] = useState<ModelSummary | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<DatasetSummary | null>(null);
  const [selectedProfileId, setSelectedProfileId] = useState<string | null>(null);
  const [runName, setRunName] = useState("");
  const [dryRun, setDryRun] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [compare, setCompare] = useState(false);

  const modelsQuery = useQuery({ queryKey: ["models"], queryFn: async () => (await api.listModels()).models });
  const datasetsQuery = useQuery({
    queryKey: ["datasets"],
    queryFn: async () => (await api.listDatasets()).datasets,
  });
  const profilesQuery = useQuery({
    queryKey: ["profiles"],
    queryFn: async () => (await api.listProfiles()).profiles,
  });
  const profileDetailQuery = useQuery({
    queryKey: ["profile", selectedProfileId],
    queryFn: () => api.getProfile(selectedProfileId!),
    enabled: Boolean(selectedProfileId),
  });

  useEffect(() => {
    if (profileDetailQuery.data?.run_name && !runName) {
      setRunName(profileDetailQuery.data.run_name);
    }
  }, [profileDetailQuery.data, runName]);

  const createRunMutation = useMutation({
    mutationFn: api.createRun,
    onSuccess: (response) => navigate(`/runs/${response.run_id}/progress`),
  });

  const canContinue = useMemo(
    () =>
      canAdvanceWizardStep(
        step,
        selectedModel,
        selectedDataset,
        selectedProfileId,
        profileDetailQuery.data,
      ),
    [step, selectedModel, selectedDataset, selectedProfileId, profileDetailQuery.data],
  );

  const handleStart = () => {
    if (!selectedModel || !selectedDataset || !profileDetailQuery.data) return;
    createRunMutation.mutate(
      buildCreateRunPayload(
        profileDetailQuery.data,
        selectedModel,
        selectedDataset,
        runName,
        dryRun,
        compare,
      ),
    );
  };

  const isLoading = modelsQuery.isLoading || datasetsQuery.isLoading || profilesQuery.isLoading;
  const loadError =
    (modelsQuery.error as Error | undefined) ||
    (datasetsQuery.error as Error | undefined) ||
    (profilesQuery.error as Error | undefined);

  return {
    step,
    setStep,
    selectedModel,
    setSelectedModel,
    selectedDataset,
    setSelectedDataset,
    selectedProfileId,
    setSelectedProfileId,
    runName,
    setRunName,
    dryRun,
    setDryRun,
    showAdvanced,
    setShowAdvanced,
    compare,
    setCompare,
    models: modelsQuery.data ?? [],
    datasets: datasetsQuery.data ?? [],
    profiles: profilesQuery.data ?? [],
    profileDetail: profileDetailQuery.data,
    createRunMutation,
    canContinue,
    handleStart,
    isLoading,
    loadError,
  };
}
