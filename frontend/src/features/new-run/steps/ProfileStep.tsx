import { ProfileSummary } from "../../../lib/types";

interface ProfileStepProps {
  profiles: ProfileSummary[];
  selectedProfileId: string | null;
  onSelect: (profileId: string) => void;
}

export function ProfileStep({ profiles, selectedProfileId, onSelect }: ProfileStepProps) {
  return (
    <section className="option-list" aria-label="Choose an evaluation profile">
      <p className="step-helper">
        Profiles define how answers are scored. Pick the profile that matches your review standards.
      </p>
      {profiles.map((profile) => (
        <div
          key={profile.profile_id}
          className={`option-card ${selectedProfileId === profile.profile_id ? "selected" : ""} ${profile.valid ? "" : "disabled"}`}
          onClick={() => profile.valid && onSelect(profile.profile_id)}
          role="button"
          tabIndex={profile.valid ? 0 : -1}
          onKeyDown={(event) => {
            if ((event.key === "Enter" || event.key === " ") && profile.valid) onSelect(profile.profile_id);
          }}
        >
          <strong>{profile.name}</strong>
          <div>Scoring rules: {profile.evaluators.join(", ") || "None listed"}</div>
          {!profile.valid && profile.validation_message ? (
            <div className="option-note">{profile.validation_message}</div>
          ) : null}
        </div>
      ))}
    </section>
  );
}
