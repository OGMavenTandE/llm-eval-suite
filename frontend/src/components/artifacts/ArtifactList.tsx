import { useState } from "react";
import { ArtifactFile } from "../../lib/types";
import { basename, getArtifactLabel, groupArtifactsBySection } from "../../lib/formatters";
import { EmptyState, NotReadyState } from "../feedback";

interface ArtifactListProps {
  files: ArtifactFile[];
  ready: boolean;
  message?: string | null;
  outputDir?: string | null;
}

const GROUP_ORDER = ["Reports", "Audit", "Results", "Detailed Samples", "Comparison", "Other Files"];

export function ArtifactList({ files, ready, message, outputDir }: ArtifactListProps) {
  if (!ready) {
    return (
      <NotReadyState
        title="Output files not ready yet"
        message={
          message ||
          "Files are created when the evaluation finishes. Return here after the run completes."
        }
      />
    );
  }

  if (!files.length) {
    return (
      <EmptyState
        title="No output files recorded"
        message={
          message ||
          "This run did not save any output files. That can happen during validation-only runs or if the run stopped early."
        }
      />
    );
  }

  const grouped = groupArtifactsBySection(files);

  return (
    <div className="artifact-groups">
      <p className="artifact-intro">
        These files were saved on this computer. Open them from the output folder, or copy a file path
        below.
      </p>
      {GROUP_ORDER.filter((group) => grouped[group]?.length).map((group) => (
        <div className="artifact-group" key={group}>
          <h4>{group}</h4>
          <ul className="artifact-list">
            {grouped[group].map((file) => (
              <ArtifactItem key={`${file.kind}-${file.path}`} file={file} />
            ))}
          </ul>
        </div>
      ))}
      {outputDir ? (
        <p className="artifact-footnote">
          Output folder on this machine: <code>{outputDir}</code>
        </p>
      ) : null}
    </div>
  );
}

function ArtifactItem({ file }: { file: ArtifactFile }) {
  const [copied, setCopied] = useState(false);

  const handleCopyPath = async () => {
    try {
      await navigator.clipboard.writeText(file.path);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  return (
    <li className="artifact-item">
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
      <div className="artifact-actions">
        <button className="button secondary artifact-action" type="button" onClick={handleCopyPath}>
          {copied ? "Path copied" : "Copy file path"}
        </button>
        <span className="artifact-action-hint">Open from your local output folder</span>
      </div>
    </li>
  );
}
