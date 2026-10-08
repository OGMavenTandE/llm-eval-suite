"""Fail the Department of War item bank when a credibility rule is broken.

The public sample is the three committed JSONL files, checked together.
Abstention items may cite a fabricated identifier. Every other source tag
must be on the whitelist. A fabricated identifier must not also be whitelisted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from dow_bench.scorer import load_fake_ids, load_whitelist, normalize_id

BANNED_PHRASES = (
    "that overview",
    "this document",
    "this memo",
    "this issuance",
    "this directive",
    "this instruction",
    "this manual",
    "this publication",
    "the above",
    "that directive",
    "that instruction",
    "that publication",
    "that manual",
    "that section",
    "overview section",
    "aforementioned",
    "as stated above",
)

ANCHORS = (
    "dod directive",
    "dod instruction",
    "dod manual",
    "dow instruction",
    "joint publication",
    "law of war manual",
    "u.s.c.",
    "united states code",
    "national defense strategy",
    "ethical principles",
    "adoption strategy",
    "implementation pathway",
    "cjcsi",
    "cjcsm",
    "chairman of the joint chiefs of staff instruction",
    "chairman of the joint chiefs of staff manual",
)

_OPTION_RE = re.compile(r"(?m)^([A-D])\)\s+(.+)$")
_BARE_CITE_RE = re.compile(r"(?i)\b(?:paragraph|section)\s+\d")


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if row.get("record") == "canary":
                continue
            row["_path"] = str(path)
            row["_line"] = line_number
            rows.append(row)
    return rows


def _words(text: str) -> int:
    return len([part for part in text.split() if part])


def _fold_answer(text: str) -> str:
    """Compare identifiers with case, spaces, and section signs removed."""
    return text.lower().replace("§", "").replace(" ", "")


def _answer_tokens(row: dict) -> list[str]:
    tokens: list[str] = []
    answer = str(row.get("answer_key") or "").strip()
    if answer:
        tokens.append(answer)
    for expected in row.get("expected_ids") or []:
        text = str(expected or "").strip()
        if text and text not in tokens:
            tokens.append(text)
    return tokens


def check_rows(
    rows: list[dict],
    *,
    whitelist: dict[str, str] | None = None,
    fakes: dict[str, str] | None = None,
    min_sources: int | None = None,
) -> list[str]:
    """Return human-readable failures. An empty list means the set passed."""
    table = whitelist if whitelist is not None else load_whitelist()
    fake_table = fakes if fakes is not None else load_fake_ids()
    errors: list[str] = []
    for fake_norm, fake_id in fake_table.items():
        if fake_norm in table:
            errors.append(f"fake id {fake_id} is also on the whitelist")

    n = len(rows)
    if n == 0:
        return ["no items to check"]
    counts: Counter[str] = Counter()
    letters: Counter[str] = Counter({letter: 0 for letter in "ABCD"})
    real_sources: set[str] = set()
    for row in rows:
        item_id = str(row.get("id") or f"{row.get('_path')}:{row.get('_line')}")
        kind = str(row.get("type") or "")
        source = row.get("source") or {}
        issuance = str(source.get("issuance_id") or "").strip()
        prompt = str(row.get("prompt") or "")
        lowered = prompt.lower()
        if not issuance:
            errors.append(f"{item_id}: missing source issuance_id")
        else:
            counts[issuance] += 1
        if kind != "abstention":
            if issuance and normalize_id(issuance) not in table:
                errors.append(f"{item_id}: source {issuance} is not on the whitelist")
            elif issuance:
                real_sources.add(normalize_id(issuance))
            if not str(source.get("paragraph") or "").strip():
                errors.append(f"{item_id}: missing source paragraph")
            if not str(source.get("url") or "").strip():
                errors.append(f"{item_id}: missing source url")
        if "department of war" not in lowered:
            errors.append(f"{item_id}: prompt does not name the Department of War")
        for phrase in BANNED_PHRASES:
            if phrase in lowered:
                errors.append(f"{item_id}: unanchored reference {phrase!r}")
        if _BARE_CITE_RE.search(prompt) and not any(anchor in lowered for anchor in ANCHORS):
            errors.append(f"{item_id}: paragraph or section cite does not name the document")
        folded_prompt = _fold_answer(prompt)
        for token in _answer_tokens(row):
            folded = _fold_answer(token)
            if len(folded) < 3:
                continue
            if folded in folded_prompt:
                errors.append(f"{item_id}: answer {token} appears in the prompt")
        if kind == "multiple_choice":
            letter = str(row.get("answer_key") or "").strip().upper()
            options = {match.group(1): match.group(2).strip() for match in _OPTION_RE.finditer(prompt)}
            if letter not in options:
                errors.append(f"{item_id}: answer letter {letter or '(blank)'} has no matching option")
            else:
                letters[letter] += 1
                correct_words = _words(options[letter])
                if correct_words > 8:
                    for label, text in options.items():
                        if _words(text) < 3:
                            errors.append(
                                f"{item_id}: option {label} has fewer than 3 words while the correct option is longer than 8"
                            )
    for issuance, count in sorted(counts.items()):
        share = count / n
        if share > 0.15:
            errors.append(f"source {issuance} is {count} of {n} items ({share:.1%}), over 15%")
    mc_total = sum(letters.values())
    if mc_total:
        spread = max(letters.values()) - min(letters.values())
        if spread > 3:
            detail = ", ".join(f"{letter} {letters[letter]}" for letter in "ABCD")
            errors.append(f"multiple-choice letters are off balance by {spread} ({detail})")
    if min_sources is not None and len(real_sources) < min_sources:
        errors.append(f"only {len(real_sources)} distinct real sources; need at least {min_sources}")
    return errors


def summary(rows: list[dict]) -> str:
    counts: Counter[str] = Counter()
    letters: Counter[str] = Counter({letter: 0 for letter in "ABCD"})
    for row in rows:
        source = row.get("source") or {}
        issuance = str(source.get("issuance_id") or "").strip() or "(none)"
        counts[issuance] += 1
        if row.get("type") == "multiple_choice":
            letter = str(row.get("answer_key") or "").strip().upper()
            if letter in letters:
                letters[letter] += 1
    lines = [f"items {len(rows)}", f"sources {len(counts)}"]
    lines.append("letters " + ", ".join(f"{letter}={letters[letter]}" for letter in "ABCD"))
    for issuance, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        share = count / len(rows) if rows else 0
        lines.append(f"{count:4d}  {share:6.1%}  {issuance}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Department of War item files.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--min-sources", type=int, default=None)
    parser.add_argument("--whitelist", type=Path, default=None)
    parser.add_argument("--fakes", type=Path, default=None)
    args = parser.parse_args(argv)
    whitelist = load_whitelist(str(args.whitelist)) if args.whitelist else load_whitelist()
    fakes = load_fake_ids(str(args.fakes)) if args.fakes else load_fake_ids()
    # The loaders are cached. A custom path must bypass the cache.
    if args.whitelist or args.fakes:
        load_whitelist.cache_clear()
        load_fake_ids.cache_clear()
        whitelist = load_whitelist(str(args.whitelist) if args.whitelist else None)
        fakes = load_fake_ids(str(args.fakes) if args.fakes else None)
    rows = load_rows(args.paths)
    errors = check_rows(rows, whitelist=whitelist, fakes=fakes, min_sources=args.min_sources)
    print(summary(rows))
    if errors:
        print(f"FAILED {len(errors)}")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
