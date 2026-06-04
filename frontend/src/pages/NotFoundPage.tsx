import { Link } from "react-router-dom";

export function NotFoundPage() {
  return (
    <main className="main-content">
      <header className="page-header">
        <h1>Page not found</h1>
        <p>The page you requested is not part of this workbench.</p>
      </header>
      <Link className="button" to="/">
        Return home
      </Link>
    </main>
  );
}
