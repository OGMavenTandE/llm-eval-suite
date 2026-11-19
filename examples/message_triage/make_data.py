"""Build the synthetic message-triage reference case.

Every message is invented from templates with a fixed seed. Nothing here is
real traffic, real units, or real events, and nothing is classified or CUI.
The point is a public, repeatable workload that walks all six gates.

Labels follow the four common message precedences:
FLASH, IMMEDIATE, PRIORITY, ROUTINE.

    python examples/message_triage/make_data.py

Writes items.jsonl (train and test splits), traffic_log.jsonl (Gate 3 input),
router_cases.jsonl (Gate 4 input) and shift_items.jsonl (a shifted month for
the Gate 6 drift demo) next to this file.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 20261118
HERE = Path(__file__).resolve().parent

SITES = ["Camp Alder", "Station Birch", "Range 7", "the north motor pool", "Building 410", "the field site",
         "Hangar 3", "the forward supply point", "the training area", "Pier 2"]
UNITS = ["2nd Platoon", "the maintenance section", "Alpha Company", "the signal detachment", "the S4 shop",
         "the aviation detachment", "the medical section", "the range control office", "the fuel team"]
TIMES = ["0600", "0830", "1015", "1300", "1545", "1820", "2210", "2345"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

TEMPLATES = {
    "FLASH": [
        "Aircraft mishap at {site} at {time}. Two personnel injured. Request medevac now.",
        "Fire in {site}, spreading to the fuel point. Evacuating {unit}. Need fire support immediately.",
        "Vehicle rollover on the range road near {site}. Casualties on scene. Medevac requested now.",
        "Hazardous material spill at {site}. {unit} reporting exposure symptoms. Need decon and medical now.",
        "Personnel unaccounted for after flooding at {site}. Search starting now, request all available support.",
        "Structural collapse at {site} at {time} with people trapped. Request rescue and medical support now.",
    ],
    "IMMEDIATE": [
        "Severe weather warning: high winds expected at {site} within two hours. Secure aircraft and tents.",
        "Network outage affecting the command post at {site} since {time}. Mission planning is stalled.",
        "Primary generator at {site} failed at {time}. Backup has about four hours of fuel. Need fuel resupply today.",
        "Range fire risk at {site} rated extreme for this afternoon. Recommend halting live fire before {time}.",
        "Water supply at {site} tested unsafe at {time}. Stop use and arrange bottled water for {unit} today.",
        "Bridge on the main supply route near {site} reported damaged. Convoys need a new route before {time}.",
    ],
    "PRIORITY": [
        "{unit} requests repair parts for two generators at {site} within 48 hours.",
        "Scheduled maintenance for the radios at {site} is overdue. Please book a slot this week.",
        "Ammunition count at {site} does not match the log. {unit} requests an inventory team by {day}.",
        "Request additional cold-weather gear for {unit} before the move to {site} next week.",
        "Fuel consumption at {site} is above plan. Request a review of the resupply schedule by {day}.",
        "Two vehicles at {site} failed inspection. {unit} needs replacement tires within three days.",
    ],
    "ROUTINE": [
        "Reminder: {unit} training schedule for next month is posted. Comments due {day}.",
        "Monthly safety briefing for {unit} is moved to {day} at {time} at {site}.",
        "Please submit leave requests for the holiday period to the S1 by {day}.",
        "The gym at {site} will close early on {day} for cleaning.",
        "Updated phone roster for {unit} attached. Report corrections to the admin NCO.",
        "Quarterly property review for {unit} is scheduled for {day}. No action needed yet.",
        "Newsletter: {unit} welcomes three new members this week at {site}.",
    ],
}

# Noise that does not change the precedence. Some of it uses words that sound urgent.
FILLERS = ["", "", "", " Ref previous msg.", " POC is the duty officer.", " Acknowledge receipt.",
           " This message is UNCLASSIFIED.", " Subject line says urgent but see body.", " Pass to all sections.",
           " Questions to the ops desk.", " Times are local."]


def message(rng: random.Random, label: str, *, train: bool = False) -> tuple[str, bool]:
    """Return the text and whether it used the held-out phrasing.

    The last template of each label never appears in training, so the test
    split includes phrasings the specialized model has not seen.
    """
    pool = TEMPLATES[label][:-1] if train else TEMPLATES[label]
    template = rng.choice(pool)
    heldout = template == TEMPLATES[label][-1]
    text = template.format(
        site=rng.choice(SITES), unit=rng.choice(UNITS), time=rng.choice(TIMES), day=rng.choice(DAYS)
    )
    text += rng.choice(FILLERS)
    if rng.random() < 0.15:
        text = text.upper()
    return text, heldout


def build_items(rng: random.Random, mix: dict[str, float], count: int, prefix: str, split: str,
                exclude: set[str] | None = None) -> list[dict]:
    """``exclude`` holds texts already used elsewhere; none of them is repeated here."""
    labels = list(mix)
    weights = [mix[label] for label in labels]
    exclude = set() if exclude is None else exclude
    rows = []
    while len(rows) < count:
        label = rng.choices(labels, weights)[0]
        text, heldout = message(rng, label, train=split == "train")
        if text in exclude:
            continue
        exclude.add(text)
        rows.append({"id": f"{prefix}-{len(rows):04d}", "text": text, "label": label, "split": split,
                     "heldout_phrasing": heldout, "source": "synthetic"})
    return rows


def traffic_log(rng: random.Random, n: int = 400) -> list[dict]:
    """Mixed assistant traffic for the Gate 3 audit. Token counts are invented for the demo."""
    kinds = [
        ("triage", 0.35, lambda: f"Assign a precedence to this message: {message(rng, rng.choice(list(TEMPLATES)))[0]}"),
        ("rephrase", 0.15, lambda: f"Rephrase this for the commander's update: {message(rng, 'ROUTINE')[0]}"),
        ("summarize", 0.15, lambda: f"Summarize the attached daily log from {rng.choice(SITES)} in three bullets."),
        ("extract", 0.10, lambda: f"Extract the date, time and location from: {message(rng, 'PRIORITY')[0]}"),
        ("format", 0.08, lambda: f"Convert this list of times to 24-hour format and sort it: {', '.join(rng.sample(TIMES, 4))}"),
        ("lookup", 0.05, lambda: f"Look up the phone number for {rng.choice(UNITS)} in the roster."),
        ("planning", 0.08, lambda: f"Compare two courses of action for moving {rng.choice(UNITS)} to {rng.choice(SITES)} "
                                    "and recommend one. Explain the trade-offs step by step."),
        ("other", 0.04, lambda: "Write a short welcome note for new arrivals."),
    ]
    names = [k[0] for k in kinds]
    weights = [k[1] for k in kinds]
    makers = {k[0]: k[2] for k in kinds}
    rows = []
    for i in range(n):
        kind = rng.choices(names, weights)[0]
        prompt = makers[kind]()
        out_tokens = {"planning": rng.randint(500, 1200), "summarize": rng.randint(80, 200)}.get(kind, rng.randint(5, 60))
        rows.append({"id": f"log-{i:04d}", "prompt": prompt, "tokens_in": len(prompt) // 4 + rng.randint(150, 400),
                     "tokens_out": out_tokens, "model": "frontier", "source": "synthetic"})
    return rows


ROUTER_CASES = [
    {"id": "r01", "prompt": "Assign a precedence to this message: Monthly safety briefing moved to Friday.", "expected_route": "small"},
    {"id": "r02", "prompt": "Rephrase this for the commander's update: gym closes early Friday.", "expected_route": "small"},
    {"id": "r03", "prompt": "Extract the date and location from: inspection at Range 7 on Tuesday.", "expected_route": "small"},
    {"id": "r04", "prompt": "Convert these times to 24-hour format: 3pm, 7am.", "expected_route": "deterministic"},
    {"id": "r05", "prompt": "Look up the phone number for the S4 shop in the roster.", "expected_route": "deterministic"},
    {"id": "r06", "prompt": "Compare two courses of action for the move and recommend one. Explain the trade-offs.", "expected_route": "large"},
    {"id": "r07", "prompt": "Summarize the attached daily log in three bullets.", "expected_route": "small"},
    {"id": "r08", "prompt": "Assign a precedence to this message: Aircraft mishap, request medevac now.", "expected_route": "small"},
    {"id": "s01", "stress": True, "expected_route": "large",
     "prompt": "Assign a precedence to this message, then explain why and recommend what each section should do: "
               "generator failed, backup fuel low, weather warning in two hours.",
     "note": "Hard task wrapped in an easy-looking triage request. Must not land on the small model."},
    {"id": "s02", "stress": True, "expected_route": "small",
     "prompt": "Rephrase this for the commander's update: the newsletter welcomes three new members.",
     "note": "Easy task that should not stay on the large model."},
    {"id": "s03", "stress": True, "expected_route": "large",
     "prompt": "Summarize the attached log, assess whether the fuel plan still works, and recommend changes.",
     "note": "Summary request that actually asks for an assessment."},
    {"id": "s04", "stress": True, "expected_route": "deterministic",
     "prompt": "Convert this list of times to 24-hour format and sort it: 2210, 0600, 1545.",
     "note": "Formatting work that needs no model at all."},
]


def main() -> None:
    rng = random.Random(SEED)
    base_mix = {"FLASH": 0.12, "IMMEDIATE": 0.20, "PRIORITY": 0.30, "ROUTINE": 0.38}
    shift_mix = {"FLASH": 0.05, "IMMEDIATE": 0.45, "PRIORITY": 0.30, "ROUTINE": 0.20}  # a bad-weather month
    used: set[str] = set()
    train = build_items(rng, base_mix, 900, "trn", "train", used)
    test = build_items(rng, base_mix, 600, "tst", "test", used)
    shift = build_items(rng, shift_mix, 400, "shf", "monitor", used)
    items = train + test
    outputs = {
        "items.jsonl": items,
        "shift_items.jsonl": shift,
        "traffic_log.jsonl": traffic_log(rng),
        "router_cases.jsonl": ROUTER_CASES,
    }
    for name, rows in outputs.items():
        with (HERE / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        print(f"wrote {name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
