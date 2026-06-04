import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { AppShell } from "../components/layout/AppShell";
import { HomePage } from "../pages/HomePage";
import { NewRunPage } from "../pages/NewRunPage";
import { NotFoundPage } from "../pages/NotFoundPage";
import { ReportPage } from "../pages/ReportPage";
import { RunProgressPage } from "../pages/RunProgressPage";
import { RunsPage } from "../pages/RunsPage";
import { StatusPage } from "../pages/StatusPage";

const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <HomePage /> },
      { path: "status", element: <StatusPage /> },
      { path: "new", element: <NewRunPage /> },
      { path: "runs", element: <RunsPage /> },
      { path: "runs/:runId", element: <ReportPage /> },
      { path: "runs/:runId/progress", element: <RunProgressPage /> },
    ],
  },
  { path: "*", element: <NotFoundPage /> },
]);

export function AppRouter() {
  return <RouterProvider router={router} />;
}
