"""GSM8K final-number exact match.

The eight shots are the public gsm8k_cot exemplars. They are training items,
not the test split.
"""

from __future__ import annotations

import re

_STRICT = re.compile(r"The answer is (-?[0-9][0-9,]*(?:\.[0-9]+)?)")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# Eight public exemplars from lm-evaluation-harness gsm8k_cot (version 3.0).
SHOTS: tuple[tuple[str, str], ...] = (
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6. The answer is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. "
        "After eating 35, they had 74 - 35 = 39. The answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
        "How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
        "So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. "
        "5 + 4 = 9. The answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers were installed each day, "
        "from monday to thursday. How many computers are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more computers were added. "
        "So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. "
        "After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    ),
)


def normalize_number(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip().replace(",", "").replace("$", "").rstrip(".")
    if not text:
        return None
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        return None
    if "." in text:
        whole, frac = text.split(".", 1)
        if set(frac) <= {"0"}:
            text = whole
    if text in {"-0", "-0.0"}:
        text = "0"
    return text


def extract_gold(answer: str) -> str | None:
    tail = answer.split("####")[-1] if "####" in (answer or "") else (answer or "")
    numbers = _NUMBER.findall(tail)
    if not numbers:
        return None
    return normalize_number(numbers[-1])


def extract_prediction(response: str) -> str | None:
    match = _STRICT.search(response or "")
    if match:
        return normalize_number(match.group(1))
    numbers = _NUMBER.findall(response or "")
    if not numbers:
        return None
    return normalize_number(numbers[-1])


def build_prompt(question: str) -> str:
    lines = []
    for shot_q, shot_a in SHOTS:
        lines.append(f"Q: {shot_q}\nA: {shot_a}")
    lines.append(f"Q: {question}\nA:")
    return "\n\n".join(lines)


def score_gsm8k(gold_answer: str, response: str) -> dict:
    expected = extract_gold(gold_answer)
    predicted = extract_prediction(response)
    passed = expected is not None and predicted == expected
    return {
        "predicted": predicted or "",
        "expected": expected or "",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "metric": "exact_match",
    }
