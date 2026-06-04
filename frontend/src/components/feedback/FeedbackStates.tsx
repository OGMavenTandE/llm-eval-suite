import { ReactNode } from "react";

export interface FeedbackProps {
  title?: string;
  message: ReactNode;
  detail?: ReactNode;
}

export function LoadingState({
  title = "Loading",
  message = "Please wait while we load this page.",
}: FeedbackProps) {
  return (
    <div className="feedback-box loading" role="status" aria-live="polite">
      <strong>{title}</strong>
      <p>{message}</p>
    </div>
  );
}

export function ErrorState({
  title = "Something went wrong",
  message,
  detail,
}: FeedbackProps) {
  return (
    <div className="feedback-box error" role="alert">
      <strong>{title}</strong>
      <p>{message}</p>
      {detail ? <p className="feedback-detail">{detail}</p> : null}
    </div>
  );
}

export function EmptyState({
  title = "Nothing to show yet",
  message,
}: FeedbackProps) {
  return (
    <div className="feedback-box empty" role="status">
      <strong>{title}</strong>
      <p>{message}</p>
    </div>
  );
}

export function InProgressState({
  title = "Still in progress",
  message,
}: FeedbackProps) {
  return (
    <div className="feedback-box in-progress" role="status" aria-live="polite">
      <strong>{title}</strong>
      <p>{message}</p>
    </div>
  );
}

export function NotReadyState({
  title = "Not ready yet",
  message,
}: FeedbackProps) {
  return (
    <div className="feedback-box not-ready" role="status">
      <strong>{title}</strong>
      <p>{message}</p>
    </div>
  );
}

export function SuccessState({
  title = "All set",
  message,
}: FeedbackProps) {
  return (
    <div className="feedback-box success" role="status">
      <strong>{title}</strong>
      <p>{message}</p>
    </div>
  );
}
