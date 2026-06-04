import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import { WizardSteps } from "../components/forms/WizardSteps";
import { LoadingState, ErrorState } from "../components/feedback/FeedbackStates";
import { StatusBadge } from "../components/status/StatusBadge";
import { ModelSummary, ProfileDetail, DatasetSummary } from "../lib/types";

const STEPS = ["Choose model", "Choose dataset", "Choose profile", "Review and start"];

export function NewRunPage() {
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

  const canContinue = useMemo(() => {
    if (step === 0) return Boolean(selectedModel);
    if (step === 1) return Boolean(selectedDataset?.valid);
    if (step === 2) return Boolean(selectedProfileId && profileDetailQuery.data?.valid);
    return true;
  }, [step, selectedModel, selectedDataset, selectedProfileId, profileDetailQuery.data]);

  const buildConfig = (profile: ProfileDetail) => ({
    run_name: runName || profile.run_name || "workbench-run",
    output_dir: profile.output_dir,
    dataset: selectedDataset!.path,
    models: [{ name: selectedModel!.name, provider: selectedModel!.provider }],
    evaluators: profile.evaluators,
  });

  const handleStart = () => {
    if (!selectedModel || !selectedDataset || !profileDetailQuery.data) return;
    createRunMutation.mutate({
      config: buildConfig(profileDetailQuery.data),
      dry_run: dryRun,
      run_name: runName || undefined,
      compare,
    });
  };

  if (modelsQuery.isLoading || datasetsQuery.isLoading || profilesQuery.isLoading) {
    return <LoadingState message="Loading wizard options..." />;
  }

  if (modelsQuery.error || datasetsQuery.error || profilesQuery.error) {
    return (
      <ErrorState
        message={
          (modelsQuery.error as Error | undefined)?.message ||
          (datasetsQuery.error as Error | undefined)?.message ||
          (profilesQuery.error as Error | undefined)?.message ||
          "Unable to load wizard data."
        }
      />
    );
  }

  return (
    <>
      <header className="page-header">
        <h1>New evaluation</h1>
        <p>Follow the steps below to configure and start a local evaluation run.</p>
      </header>

      <WizardSteps steps={STEPS} currentStep={step} />

      {step === 0 ? (
        <section className="option-list">
          {modelsQuery.data!.map((model) => (
            <div
              key={`${model.provider}:${model.name}`}
              className={`option-card ${selectedModel?.name === model.name ? "selected" : ""}`}
              onClick={() => setSelectedModel(model)}
            >
              <strong>{model.name}</strong>
              <div>Provider: {model.provider}</div>
              <div>Connection: {model.connection_status}</div>
              {model.warning_message ? <div>{model.warning_message}</div> : null}
            </div>
          ))}
        </section>
      ) : null}

      {step === 1 ? (
        <section className="option-list">
          {datasetsQuery.data!.map((dataset) => (
            <div
              key={dataset.dataset_id}
              className={`option-card ${selectedDataset?.dataset_id === dataset.dataset_id ? "selected" : ""} ${dataset.valid ? "" : "disabled"}`}
              onClick={() => dataset.valid && setSelectedDataset(dataset)}
            >
              <strong>{dataset.name}</strong>
              <div>Samples: {dataset.sample_count ?? "Unknown"}</div>
              <div>Format: {dataset.format}</div>
              {!dataset.valid && dataset.validation_message ? <div>{dataset.validation_message}</div> : null}
            </div>
          ))}
        </section>
      ) : null}

      {step === 2 ? (
        <section className="option-list">
          {profilesQuery.data!.map((profile) => (
            <div
              key={profile.profile_id}
              className={`option-card ${selectedProfileId === profile.profile_id ? "selected" : ""} ${profile.valid ? "" : "disabled"}`}
              onClick={() => profile.valid && setSelectedProfileId(profile.profile_id)}
            >
              <strong>{profile.name}</strong>
              <div>Evaluators: {profile.evaluators.join(", ") || "None listed"}</div>
              {!profile.valid && profile.validation_message ? <div>{profile.validation_message}</div> : null}
            </div>
          ))}
        </section>
      ) : null}

      {step === 3 ? (
        <section className="card">
          <h3>Review your evaluation</h3>
          <p>
            <strong>Model:</strong> {selectedModel?.name} ({selectedModel?.provider})
          </p>
          <p>
            <strong>Dataset:</strong> {selectedDataset?.name}
          </p>
          <p>
            <strong>Profile:</strong> {profileDetailQuery.data?.name}
          </p>
          <p>
            <strong>Evaluators:</strong>{" "}
            {profileDetailQuery.data?.evaluators
              .map((item) => String(item.name ?? "evaluator"))
              .join(", ") || "—"}
          </p>

          <div className="field">
            <label htmlFor="run-name">Run name (optional)</label>
            <input
              id="run-name"
              value={runName}
              onChange={(event) => setRunName(event.target.value)}
              placeholder="Example: Q2 model check"
            />
          </div>

          <label>
            <input type="checkbox" checked={dryRun} onChange={(event) => setDryRun(event.target.checked)} /> Run
            validation only (recommended first)
          </label>

          <div className="advanced-panel">
            <button className="button secondary" type="button" onClick={() => setShowAdvanced((value) => !value)}>
              {showAdvanced ? "Hide advanced" : "Show advanced"}
            </button>
            {showAdvanced ? (
              <div style={{ marginTop: "0.75rem" }}>
                <label>
                  <input type="checkbox" checked={compare} onChange={(event) => setCompare(event.target.checked)} /> Enable
                  comparison mode
                </label>
              </div>
            ) : null}
          </div>

          {createRunMutation.error ? (
            <div className="feedback-box error" style={{ marginTop: "1rem" }}>
              {(createRunMutation.error as Error).message}
            </div>
          ) : null}
        </section>
      ) : null}

      <div className="button-row">
        <button className="button secondary" type="button" disabled={step === 0} onClick={() => setStep((value) => value - 1)}>
          Back
        </button>
        {step < 3 ? (
          <button className="button" type="button" disabled={!canContinue} onClick={() => setStep((value) => value + 1)}>
            Continue
          </button>
        ) : (
          <button className="button" type="button" disabled={createRunMutation.isPending} onClick={handleStart}>
            {createRunMutation.isPending ? "Starting..." : "Start evaluation"}
          </button>
        )}
        <Link className="button secondary" to="/">
          Cancel
        </Link>
      </div>

      {selectedModel && step > 0 ? (
        <p>
          Selected model: {selectedModel.name} <StatusBadge status={selectedModel.available ? "completed" : "failed_validation"} />
        </p>
      ) : null}
    </>
  );
}
