import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ErrorState, LoadingState } from "./index";

describe("feedback components", () => {
  it("renders loading and error states with title and message", () => {
    render(<LoadingState title="Loading data" message="Please wait." />);
    expect(screen.getByText("Loading data")).toBeInTheDocument();
    expect(screen.getByText("Please wait.")).toBeInTheDocument();

    render(
      <ErrorState
        title="Request failed"
        message="We could not complete that action."
        detail="Network error"
      />,
    );
    expect(screen.getByText("Request failed")).toBeInTheDocument();
    expect(screen.getByText("Network error")).toBeInTheDocument();
  });
});
