export function LoadingState({ message = "Loading..." }: { message?: string }) {
  return <div className="feedback-box">{message}</div>;
}

export function ErrorState({ message }: { message: string }) {
  return <div className="feedback-box error">{message}</div>;
}

export function EmptyState({ message }: { message: string }) {
  return <div className="feedback-box empty">{message}</div>;
}
