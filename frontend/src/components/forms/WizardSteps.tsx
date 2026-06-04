export function WizardSteps({
  steps,
  currentStep,
}: {
  steps: string[];
  currentStep: number;
}) {
  return (
    <div className="wizard-steps">
      {steps.map((label, index) => {
        const className =
          index === currentStep ? "wizard-step active" : index < currentStep ? "wizard-step done" : "wizard-step";
        return (
          <div key={label} className={className}>
            {index + 1}. {label}
          </div>
        );
      })}
    </div>
  );
}
