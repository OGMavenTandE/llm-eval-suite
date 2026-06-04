import { NavLink, Outlet } from "react-router-dom";

export function AppShell() {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <h1>AI Evaluation Workbench</h1>
        <p>Run and review local model evaluations with confidence.</p>
        <nav>
          <NavLink className="nav-link" to="/">
            Home
          </NavLink>
          <NavLink className="nav-link" to="/new">
            New evaluation
          </NavLink>
          <NavLink className="nav-link" to="/runs">
            Previous runs
          </NavLink>
          <NavLink className="nav-link" to="/status">
            System status
          </NavLink>
        </nav>
      </aside>
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  );
}
