import { ArtifactFile } from "../../lib/types";
import { basename, getArtifactGroupLabel, getArtifactLabel } from "../../lib/formatters";
import { EmptyState, NotReadyState } from "../feedback/FeedbackStates";

interface ArtifactListProps {
  files: ArtifactFile[];
  ready: boolean;
  message?: string | null;
  outputDir?: string | null;
}

const GROUP_ORDER = ["Audit", "Results", "Detailed Samples", "Comparison", "Other Files"];

export function ArtifactList({ files, ready, message, outputDir }: ArtifactListProps) {
  if (!ready) {
    return <NotReadyState title="Files not ready yet" message={message || "Artifact files will appear when the run finishes."} />;
  }

  if (!files.length) {
    return <EmptyState title="No files recorded" message={message || "This run did not produce downloadable files."} />;
  }

  const grouped = groupArtifactsBySection(files);

  return (
    <div className="artifact-groups">
      {GROUP_ORDER.filter((group) => grouped[group]?.length).map((group) => (
        <div className="artifact-group" key={group}>
          <h4>{group}</h4>
          <ul className="artifact-list">
            {grouped[group].map((file) => (
              <li key={`${file.kind}-${file.path}`} className="artifact-item">
                <div className="artifact-primary">
                  <strong>{getArtifactLabel(file)}</strong>
                  {file.model_name ? <span className="artifact-meta">Model: {file.model_name}</span> : null}
                </div>
                <div className="artifact-secondary">
                  <span className="artifact-path" title={file.path}>
                    {basename(file.path)}
                  </span>
                  {!file.exists ? <span className="artifact-missing">File not found on disk</span> : null}
                </div>
              </li>
            ))}
          </ul>
        </div>
      ))}
      {outputDir ? (
        <p className="artifact-footnote">
          Files are stored under <code>{outputDir}</code>
        </p>
      ) : null}
    </div>
  );
}

function groupArtifactsBySection(files: ArtifactFile[]): Record<string, ArtifactFile[]> {
  const groups: Record<string, ArtifactFile[]> = {};

  for (const file of files) {
    const group = getArtifactGroupLabel(file.kind);
    if (!groups[group]) groups[group] = [];
    groups[group].push(file);
  }

  return groups;
}
