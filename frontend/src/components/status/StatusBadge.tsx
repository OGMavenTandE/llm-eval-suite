import { formatStatus } from "../../lib/formatters";

const statusClass = (status: string) => {
  if (["completed", "validated"].includes(status)) return "success";
  if (["pending", "running"].includes(status)) return "warning";
  if (status.startsWith("failed")) return "danger";
  return "neutral";
};

export function StatusBadge({ status }: { status: string }) {
  return <span className={`status-badge ${statusClass(status)}`}>{formatStatus(status)}</span>;
}
