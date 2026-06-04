import { DatasetSummary } from "../../../lib/types";

interface DatasetStepProps {
  datasets: DatasetSummary[];
  selectedDataset: DatasetSummary | null;
  onSelect: (dataset: DatasetSummary) => void;
}

export function DatasetStep({ datasets, selectedDataset, onSelect }: DatasetStepProps) {
  return (
    <section className="option-list" aria-label="Choose a dataset">
      <p className="step-helper">
        Pick the question-and-answer set to test the model against. Only valid datasets can be selected.
      </p>
      {datasets.map((dataset) => (
        <div
          key={dataset.dataset_id}
          className={`option-card ${selectedDataset?.dataset_id === dataset.dataset_id ? "selected" : ""} ${dataset.valid ? "" : "disabled"}`}
          onClick={() => dataset.valid && onSelect(dataset)}
          role="button"
          tabIndex={dataset.valid ? 0 : -1}
          onKeyDown={(event) => {
            if ((event.key === "Enter" || event.key === " ") && dataset.valid) onSelect(dataset);
          }}
        >
          <strong>{dataset.name}</strong>
          <div>Sample questions: {dataset.sample_count ?? "Unknown"}</div>
          <div>Format: {dataset.format}</div>
          {!dataset.valid && dataset.validation_message ? (
            <div className="option-note">{dataset.validation_message}</div>
          ) : null}
        </div>
      ))}
    </section>
  );
}
