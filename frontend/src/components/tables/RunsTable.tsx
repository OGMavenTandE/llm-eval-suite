import { Link } from "react-router-dom";
import { RunSummary } from "../../lib/types";
import { basename, formatDate } from "../../lib/formatters";
import { StatusBadge } from "../status/StatusBadge";

export function RunsTable({ runs }: { runs: RunSummary[] }) {
  if (!runs.length) {
    return <div className="feedback-box empty">No runs yet. Start a new evaluation to see results here.</div>;
  }

  return (
    <table className="table">
      <thead>
        <tr>
          <th>Run</th>
          <th>Status</th>
          <th>Models</th>
          <th>Dataset</th>
          <th>Started</th>
          <th>Type</th>
        </tr>
      </thead>
      <tbody>
        {runs.map((run) => (
          <tr key={run.run_id}>
            <td>
              <Link to={`/runs/${run.run_id}`}>{run.run_name || run.run_id}</Link>
              <div style={{ color: "#52606d", fontSize: "0.85rem" }}>{run.run_id}</div>
            </td>
            <td>
              <StatusBadge status={run.status} />
            </td>
            <td>{run.model_names.join(", ") || "—"}</td>
            <td>{basename(run.dataset_path)}</td>
            <td>{formatDate(run.started_at || run.created_at)}</td>
            <td>{run.dry_run ? "Dry run" : "Full run"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
