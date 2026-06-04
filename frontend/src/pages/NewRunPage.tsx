import { NewRunWizard } from "../features/new-run/NewRunWizard";

export function NewRunPage() {
  return (
    <>
      <header className="page-header">
        <h1>New evaluation</h1>
        <p>Follow the steps below to configure and start a local evaluation run.</p>
      </header>
      <NewRunWizard />
    </>
  );
}
