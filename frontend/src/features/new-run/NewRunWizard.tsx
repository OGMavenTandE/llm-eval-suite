import { Link } from "react-router-dom";
import { WizardSteps } from "../../components/forms/WizardSteps";
import { LoadingState, ErrorState } from "../../components/feedback/FeedbackStates";
import { StatusBadge } from "../../components/status/StatusBadge";
import { useNewRunWizard, WIZARD_STEPS } from "./useNewRunWizard";
import { ModelStep } from "./steps/ModelStep";
import { DatasetStep } from "./steps/DatasetStep";
import { ProfileStep } from "./steps/ProfileStep";
import { ReviewStep } from "./steps/ReviewStep";

export function NewRunWizard() {
  const wizard = useNewRunWizard();

  if (wizard.isLoading) {
    return <LoadingState title="Preparing wizard" message="Loading models, datasets, and profiles..." />;
  }

  if (wizard.loadError) {
    return (
      <ErrorState
        title="Unable to start a new evaluation"
        message={wizard.loadError.message || "Unable to load wizard data."}
      />
    );
  }

  return (
    <>
      <WizardSteps steps={WIZARD_STEPS} currentStep={wizard.step} />

      {wizard.step === 0 ? (
        <ModelStep
          models={wizard.models}
          selectedModel={wizard.selectedModel}
          onSelect={wizard.setSelectedModel}
        />
      ) : null}

      {wizard.step === 1 ? (
        <DatasetStep
          datasets={wizard.datasets}
          selectedDataset={wizard.selectedDataset}
          onSelect={wizard.setSelectedDataset}
        />
      ) : null}

      {wizard.step === 2 ? (
        <ProfileStep
          profiles={wizard.profiles}
          selectedProfileId={wizard.selectedProfileId}
          onSelect={wizard.setSelectedProfileId}
        />
      ) : null}

      {wizard.step === 3 ? (
        <ReviewStep
          selectedModel={wizard.selectedModel}
          selectedDataset={wizard.selectedDataset}
          profileDetail={wizard.profileDetail}
          runName={wizard.runName}
          onRunNameChange={wizard.setRunName}
          dryRun={wizard.dryRun}
          onDryRunChange={wizard.setDryRun}
          showAdvanced={wizard.showAdvanced}
          onToggleAdvanced={() => wizard.setShowAdvanced((value) => !value)}
          compare={wizard.compare}
          onCompareChange={wizard.setCompare}
          submitError={(wizard.createRunMutation.error as Error | null) ?? null}
        />
      ) : null}

      <div className="button-row">
        <button
          className="button secondary"
          type="button"
          disabled={wizard.step === 0}
          onClick={() => wizard.setStep((value) => value - 1)}
        >
          Back
        </button>
        {wizard.step < 3 ? (
          <button
            className="button"
            type="button"
            disabled={!wizard.canContinue}
            onClick={() => wizard.setStep((value) => value + 1)}
          >
            Continue
          </button>
        ) : (
          <button
            className="button"
            type="button"
            disabled={wizard.createRunMutation.isPending}
            onClick={wizard.handleStart}
          >
            {wizard.createRunMutation.isPending ? "Starting..." : "Start evaluation"}
          </button>
        )}
        <Link className="button secondary" to="/">
          Cancel
        </Link>
      </div>

      {wizard.selectedModel && wizard.step > 0 ? (
        <p className="selection-reminder">
          Selected model: {wizard.selectedModel.name}{" "}
          <StatusBadge status={wizard.selectedModel.available ? "completed" : "failed_validation"} />
        </p>
      ) : null}
    </>
  );
}
