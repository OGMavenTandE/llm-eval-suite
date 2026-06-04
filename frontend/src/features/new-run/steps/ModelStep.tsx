import { ModelSummary } from "../../../lib/types";

interface ModelStepProps {
  models: ModelSummary[];
  selectedModel: ModelSummary | null;
  onSelect: (model: ModelSummary) => void;
}

export function ModelStep({ models, selectedModel, onSelect }: ModelStepProps) {
  return (
    <section className="option-list" aria-label="Choose a model">
      <p className="step-helper">
        Select the AI model you want to evaluate. Choose one that is reachable on this machine.
      </p>
      {models.map((model) => (
        <div
          key={`${model.provider}:${model.name}`}
          className={`option-card ${selectedModel?.name === model.name ? "selected" : ""}`}
          onClick={() => onSelect(model)}
          role="button"
          tabIndex={0}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") onSelect(model);
          }}
        >
          <strong>{model.name}</strong>
          <div>Provider: {model.provider}</div>
          <div>Connection: {model.connection_status}</div>
          {model.warning_message ? <div className="option-note">{model.warning_message}</div> : null}
        </div>
      ))}
    </section>
  );
}
