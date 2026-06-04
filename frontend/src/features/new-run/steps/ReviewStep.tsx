import { ModelSummary, ProfileDetail, DatasetSummary } from "../../../lib/types";

interface ReviewStepProps {
  selectedModel: ModelSummary | null;
  selectedDataset: DatasetSummary | null;
  profileDetail: ProfileDetail | undefined;
  runName: string;
  onRunNameChange: (value: string) => void;
  dryRun: boolean;
  onDryRunChange: (value: boolean) => void;
  showAdvanced: boolean;
  onToggleAdvanced: () => void;
  compare: boolean;
  onCompareChange: (value: boolean) => void;
  submitError: Error | null;
}

export function ReviewStep({
  selectedModel,
  selectedDataset,
  profileDetail,
  runName,
  onRunNameChange,
  dryRun,
  onDryRunChange,
  showAdvanced,
  onToggleAdvanced,
  compare,
  onCompareChange,
  submitError,
}: ReviewStepProps) {
  return (
    <section className="card" aria-label="Review and start">
      <h3>Review your evaluation</h3>
      <p className="step-helper">Confirm your choices before starting. You can run a quick validation first.</p>
      <dl className="review-summary">
        <div>
          <dt>Model</dt>
          <dd>
            {selectedModel?.name} ({selectedModel?.provider})
          </dd>
        </div>
        <div>
          <dt>Dataset</dt>
          <dd>{selectedDataset?.name}</dd>
        </div>
        <div>
          <dt>Profile</dt>
          <dd>{profileDetail?.name}</dd>
        </div>
        <div>
          <dt>Scoring rules</dt>
          <dd>
            {profileDetail?.evaluators.map((item) => String(item.name ?? "evaluator")).join(", ") || "—"}
          </dd>
        </div>
      </dl>

      <div className="field">
        <label htmlFor="run-name">Run name (optional)</label>
        <input
          id="run-name"
          value={runName}
          onChange={(event) => onRunNameChange(event.target.value)}
          placeholder="Example: Q2 model check"
        />
      </div>

      <label className="checkbox-row">
        <input type="checkbox" checked={dryRun} onChange={(event) => onDryRunChange(event.target.checked)} />
        Run validation only (recommended first)
      </label>

      <div className="advanced-panel">
        <button className="button secondary" type="button" onClick={onToggleAdvanced}>
          {showAdvanced ? "Hide advanced" : "Show advanced"}
        </button>
        {showAdvanced ? (
          <div className="advanced-options">
            <label className="checkbox-row">
              <input type="checkbox" checked={compare} onChange={(event) => onCompareChange(event.target.checked)} />
              Enable comparison mode
            </label>
          </div>
        ) : null}
      </div>

      {submitError ? <div className="feedback-box error">{submitError.message}</div> : null}
    </section>
  );
}
