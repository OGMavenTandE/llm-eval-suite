import { NotableMetric, ExecutiveSummary } from "../../lib/types";

interface ExecutiveSummarySectionProps {
  summary: ExecutiveSummary;
}

export function ExecutiveSummarySection({ summary }: ExecutiveSummarySectionProps) {
  return (
    <section className="report-summary card" aria-label="Executive summary">
      <h2>{summary.overall_outcome}</h2>
      <dl className="report-summary-list">
        <div>
          <dt>What was evaluated</dt>
          <dd>{summary.evaluation_purpose}</dd>
        </div>
        <div>
          <dt>Recommended next step</dt>
          <dd>{summary.recommended_next_step}</dd>
        </div>
        <div>
          <dt>Items needing further review</dt>
          <dd>
            {summary.needs_human_review_count}{" "}
            {summary.needs_human_review_count === 1 ? "item" : "items"}
          </dd>
        </div>
      </dl>

      {summary.notable_metrics.length ? (
        <>
          <h3 className="executive-subheading">Key scores</h3>
          <ul className="executive-list">
            {summary.notable_metrics.map((metric: NotableMetric) => (
              <li key={`${metric.label}-${metric.value}`}>
                <strong>{metric.label}:</strong> {metric.value}
                <span className="executive-meta"> · {metric.context}</span>
              </li>
            ))}
          </ul>
        </>
      ) : null}

      {summary.key_strengths.length ? (
        <>
          <h3 className="executive-subheading">Main strengths</h3>
          <ul className="executive-list">
            {summary.key_strengths.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </>
      ) : null}

      {summary.key_weaknesses.length ? (
        <>
          <h3 className="executive-subheading">Main weaknesses</h3>
          <ul className="executive-list">
            {summary.key_weaknesses.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </>
      ) : null}

      {summary.notes ? <p className="executive-note">{summary.notes}</p> : null}
    </section>
  );
}
