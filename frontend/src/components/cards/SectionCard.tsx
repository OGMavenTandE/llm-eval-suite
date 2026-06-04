import { ReactNode } from "react";

interface SectionCardProps {
  title: string;
  description?: string;
  children: ReactNode;
  className?: string;
}

export function SectionCard({ title, description, children, className = "" }: SectionCardProps) {
  return (
    <section className={`section-card ${className}`.trim()}>
      <header className="section-card-header">
        <h2 className="section-title">{title}</h2>
        {description ? <p className="section-description">{description}</p> : null}
      </header>
      {children}
    </section>
  );
}
