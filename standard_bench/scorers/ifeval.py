"""IFEval instruction checks.

The 25 instruction ids and the strict and loose rules follow
google-research/instruction_following_eval. Sentence boundaries use that
package's regex splitter. Language checks use scripts and stopwords instead
of langdetect. See the suite deviation note in the catalog.
"""

from __future__ import annotations

import json
import re
from collections import Counter

INSTRUCTION_IDS = (
    "keywords:existence",
    "keywords:frequency",
    "keywords:forbidden_words",
    "keywords:letter_frequency",
    "language:response_language",
    "length_constraints:number_sentences",
    "length_constraints:number_paragraphs",
    "length_constraints:number_words",
    "length_constraints:nth_paragraph_first_word",
    "detectable_content:number_placeholders",
    "detectable_content:postscript",
    "detectable_format:number_bullet_lists",
    "detectable_format:constrained_response",
    "detectable_format:number_highlighted_sections",
    "detectable_format:multiple_sections",
    "detectable_format:json_format",
    "detectable_format:title",
    "combination:two_responses",
    "combination:repeat_prompt",
    "startend:end_checker",
    "startend:quotation",
    "change_case:capital_word_frequency",
    "change_case:english_capital",
    "change_case:english_lowercase",
    "punctuation:no_comma",
)

_RELATIONS = ("less than", "at least")
_CONSTRAINED = (
    "My answer is yes.",
    "My answer is no.",
    "My answer is maybe.",
)

_ALPHABETS = "([A-Za-z])"
_PREFIXES = "(Mr|St|Mrs|Ms|Dr)[.]"
_SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
_STARTERS = (
    r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|"
    r"Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
)
_ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = "[.](com|net|org|io|gov|edu|me)"
_DIGITS = "([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"

# Scripts that identify one language on their own.
_UNIQUE_SCRIPTS = {
    "ko": re.compile(r"[\uac00-\ud7af]"),
    "th": re.compile(r"[\u0e00-\u0e7f]"),
    "bn": re.compile(r"[\u0980-\u09ff]"),
    "gu": re.compile(r"[\u0a80-\u0aff]"),
    "pa": re.compile(r"[\u0a00-\u0a7f]"),
    "te": re.compile(r"[\u0c00-\u0c7f]"),
    "kn": re.compile(r"[\u0c80-\u0cff]"),
    "ta": re.compile(r"[\u0b80-\u0bff]"),
}
_SHARED_SCRIPTS = {
    "devanagari": re.compile(r"[\u0900-\u097f]"),
    "arabic": re.compile(r"[\u0600-\u06ff]"),
    "cyrillic": re.compile(r"[\u0400-\u04ff]"),
}
_SCRIPT_GROUP = {
    "hi": "devanagari",
    "mr": "devanagari",
    "ne": "devanagari",
    "ar": "arabic",
    "fa": "arabic",
    "ur": "arabic",
    "ru": "cyrillic",
    "bg": "cyrillic",
}
_STOPWORDS = {
    "en": {"the", "and", "of", "to", "is", "a", "in"},
    "de": {"der", "die", "und", "nicht", "ein", "das"},
    "pt": {"não", "uma", "para", "com", "que", "os"},
    "it": {"che", "non", "una", "per", "con", "del"},
    "vi": {"không", "một", "các", "của", "và", "là"},
    "fi": {"ja", "on", "ei", "että", "se", "oli"},
    "sw": {"na", "ya", "kwa", "ni", "wa", "katika"},
    "hi": {"है", "के", "की", "और", "में", "का"},
    "mr": {"आहे", "आणि", "या", "ते", "की"},
    "ne": {"छ", "र", "हो", "लाई", "को"},
    "ar": {"في", "من", "على", "أن", "هذا"},
    "fa": {"در", "به", "از", "که", "را"},
    "ur": {"ہے", "کے", "میں", "اور", "کا"},
    "ru": {"и", "в", "не", "что", "на"},
    "bg": {"и", "на", "за", "се", "да", "не"},
}


def split_into_sentences(text: str) -> list[str]:
    """Regex sentence splitter matching instructions_util.split_into_sentences."""
    cleaned = " " + (text or "") + "  "
    cleaned = cleaned.replace("\n", " ")
    cleaned = re.sub(_PREFIXES, r"\1<prd>", cleaned)
    cleaned = re.sub(_WEBSITES, r"<prd>\1", cleaned)
    cleaned = re.sub(_DIGITS + "[.]" + _DIGITS, r"\1<prd>\2", cleaned)
    cleaned = re.sub(
        _MULTIPLE_DOTS,
        lambda match: "<prd>" * len(match.group(0)) + "<stop>",
        cleaned,
    )
    if "Ph.D" in cleaned:
        cleaned = cleaned.replace("Ph.D.", "Ph<prd>D<prd>")
    cleaned = re.sub(r"\s" + _ALPHABETS + "[.] ", r" \1<prd> ", cleaned)
    cleaned = re.sub(_ACRONYMS + " " + _STARTERS, r"\1<stop> \2", cleaned)
    cleaned = re.sub(
        _ALPHABETS + "[.]" + _ALPHABETS + "[.]" + _ALPHABETS + "[.]",
        r"\1<prd>\2<prd>\3<prd>",
        cleaned,
    )
    cleaned = re.sub(_ALPHABETS + "[.]" + _ALPHABETS + "[.]", r"\1<prd>\2<prd>", cleaned)
    cleaned = re.sub(" " + _SUFFIXES + "[.] " + _STARTERS, r" \1<stop> \2", cleaned)
    cleaned = re.sub(" " + _SUFFIXES + "[.]", r" \1<prd>", cleaned)
    cleaned = re.sub(" " + _ALPHABETS + "[.]", r" \1<prd>", cleaned)
    if "”" in cleaned:
        cleaned = cleaned.replace(".”", "”.")
    if '"' in cleaned:
        cleaned = cleaned.replace('."', '".')
    if "!" in cleaned:
        cleaned = cleaned.replace('!"', '"!')
    if "?" in cleaned:
        cleaned = cleaned.replace('?"', '"?')
    cleaned = cleaned.replace(".", ".<stop>")
    cleaned = cleaned.replace("?", "?<stop>")
    cleaned = cleaned.replace("!", "!<stop>")
    cleaned = cleaned.replace("<prd>", ".")
    sentences = [part.strip() for part in cleaned.split("<stop>")]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences


def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def _relation(count: int, relation: str, threshold: int) -> bool:
    if relation == "less than":
        return count < threshold
    if relation == "at least":
        return count >= threshold
    return False


def _require(kwargs: dict, *keys: str) -> bool:
    return all(kwargs.get(key) is not None for key in keys)


def _stopword_hits(text: str, language: str) -> int:
    words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    stops = _STOPWORDS.get(language) or set()
    # Stopwords may be single unicode words already covered by \w.
    extra = sum(text.count(token) for token in stops if not re.fullmatch(r"\w+", token))
    return sum(1 for word in words if word in stops) + extra


def detect_language(text: str) -> str | None:
    """Best-effort language id. None means the detector is not sure."""
    source = text or ""
    for language, pattern in _UNIQUE_SCRIPTS.items():
        if len(pattern.findall(source)) >= 3:
            return language
    for group, pattern in _SHARED_SCRIPTS.items():
        if len(pattern.findall(source)) < 3:
            continue
        members = [lang for lang, name in _SCRIPT_GROUP.items() if name == group]
        scored = [(lang, _stopword_hits(source, lang)) for lang in members]
        scored.sort(key=lambda item: item[1], reverse=True)
        if scored and scored[0][1] > 0 and (len(scored) == 1 or scored[0][1] > scored[1][1]):
            return scored[0][0]
        return None
    scored = [(lang, _stopword_hits(source, lang)) for lang in ("en", "de", "pt", "it", "vi", "fi", "sw")]
    scored.sort(key=lambda item: item[1], reverse=True)
    if scored[0][1] >= 2 and scored[0][1] > scored[1][1]:
        return scored[0][0]
    return None


def _english_enough(text: str) -> bool:
    return detect_language(text) == "en" or _stopword_hits(text, "en") >= 2


def _search(pattern: str, text: str) -> bool:
    try:
        return re.search(pattern, text, flags=re.IGNORECASE) is not None
    except re.error:
        return False


def _findall(pattern: str, text: str) -> int:
    try:
        return len(re.findall(pattern, text, flags=re.IGNORECASE))
    except re.error:
        return 0


def check_instruction(instruction_id: str, response: str, kwargs: dict | None) -> bool:
    """One instruction. Missing required arguments fail closed. Nothing is random."""
    args = {key: value for key, value in (kwargs or {}).items() if value is not None}
    text = response or ""
    if instruction_id not in INSTRUCTION_IDS:
        return False
    if instruction_id == "keywords:existence":
        keywords = args.get("keywords") or []
        if not keywords:
            return False
        return all(_search(str(keyword), text) for keyword in keywords)
    if instruction_id == "keywords:frequency":
        if not _require(args, "keyword", "frequency", "relation"):
            return False
        if args["relation"] not in _RELATIONS:
            return False
        return _relation(_findall(str(args["keyword"]), text), str(args["relation"]), int(args["frequency"]))
    if instruction_id == "keywords:forbidden_words":
        words = args.get("forbidden_words") or []
        if not words:
            return False
        for word in words:
            if re.search(r"\b" + re.escape(str(word)) + r"\b", text, flags=re.IGNORECASE):
                return False
        return True
    if instruction_id == "keywords:letter_frequency":
        if not _require(args, "letter", "let_frequency", "let_relation"):
            return False
        if args["let_relation"] not in _RELATIONS:
            return False
        letter = str(args["letter"]).lower()[:1]
        count = Counter(text.lower())[letter]
        return _relation(count, str(args["let_relation"]), int(args["let_frequency"]))
    if instruction_id == "language:response_language":
        language = args.get("language")
        if not language:
            return False
        return detect_language(text) == language
    if instruction_id == "length_constraints:number_sentences":
        if not _require(args, "num_sentences", "relation"):
            return False
        if args["relation"] not in _RELATIONS:
            return False
        return _relation(len(split_into_sentences(text)), str(args["relation"]), int(args["num_sentences"]))
    if instruction_id == "length_constraints:number_paragraphs":
        if args.get("num_paragraphs") is None:
            return False
        paragraphs = re.split(r"\s?\*\*\*\s?", text)
        count = len(paragraphs)
        for index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                if index == 0 or index == len(paragraphs) - 1:
                    count -= 1
                else:
                    return False
        return count == int(args["num_paragraphs"])
    if instruction_id == "length_constraints:number_words":
        if not _require(args, "num_words", "relation"):
            return False
        if args["relation"] not in _RELATIONS:
            return False
        return _relation(count_words(text), str(args["relation"]), int(args["num_words"]))
    if instruction_id == "length_constraints:nth_paragraph_first_word":
        if not _require(args, "num_paragraphs", "nth_paragraph", "first_word"):
            return False
        paragraphs = re.split(r"\n\n", text)
        count = len(paragraphs)
        for paragraph in paragraphs:
            if not paragraph.strip():
                count -= 1
        nth = int(args["nth_paragraph"])
        if nth > count or nth < 1:
            return False
        paragraph = paragraphs[nth - 1].strip()
        if not paragraph:
            return False
        word = paragraph.split()[0].strip().lstrip("'").lstrip('"')
        first = ""
        for letter in word:
            if letter in {".", ",", "?", "!", "'", '"'}:
                break
            first += letter.lower()
        return count == int(args["num_paragraphs"]) and first == str(args["first_word"]).lower()
    if instruction_id == "detectable_content:number_placeholders":
        if args.get("num_placeholders") is None:
            return False
        return len(re.findall(r"\[.*?\]", text)) >= int(args["num_placeholders"])
    if instruction_id == "detectable_content:postscript":
        marker = str(args.get("postscript_marker") or "")
        if not marker:
            return False
        lowered = text.lower()
        if marker == "P.P.S":
            pattern = r"\s*p\.\s?p\.\s?s.*$"
        elif marker == "P.S.":
            pattern = r"\s*p\.\s?s\..*$"
        else:
            pattern = r"\s*" + re.escape(marker.lower()) + r".*$"
        return bool(re.findall(pattern, lowered, flags=re.MULTILINE))
    if instruction_id == "detectable_format:number_bullet_lists":
        if args.get("num_bullets") is None:
            return False
        stars = re.findall(r"^\s*\*[^\*].*$", text, flags=re.MULTILINE)
        dashes = re.findall(r"^\s*-.*$", text, flags=re.MULTILINE)
        return len(stars) + len(dashes) == int(args["num_bullets"])
    if instruction_id == "detectable_format:constrained_response":
        stripped = text.strip()
        return any(option in stripped for option in _CONSTRAINED)
    if instruction_id == "detectable_format:number_highlighted_sections":
        if args.get("num_highlights") is None:
            return False
        count = 0
        for highlight in re.findall(r"\*[^\n\*]*\*", text):
            if highlight.strip("*").strip():
                count += 1
        for highlight in re.findall(r"\*\*[^\n\*]*\*\*", text):
            if highlight.removeprefix("**").removesuffix("**").strip():
                count += 1
        return count >= int(args["num_highlights"])
    if instruction_id == "detectable_format:multiple_sections":
        if not _require(args, "section_spliter", "num_sections"):
            return False
        pattern = r"\s?" + re.escape(str(args["section_spliter"])) + r"\s?\d+\s?"
        sections = re.split(pattern, text)
        return (len(sections) - 1) >= int(args["num_sections"])
    if instruction_id == "detectable_format:json_format":
        payload = (
            text.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            json.loads(payload)
        except ValueError:
            return False
        return True
    if instruction_id == "detectable_format:title":
        titles = re.findall(r"<<[^\n]+>>", text)
        return any(title.lstrip("<").rstrip(">").strip() for title in titles)
    if instruction_id == "combination:two_responses":
        valid = []
        parts = text.split("******")
        for index, part in enumerate(parts):
            if not part.strip():
                if index != 0 and index != len(parts) - 1:
                    return False
            else:
                valid.append(part)
        return len(valid) == 2 and valid[0].strip() != valid[1].strip()
    if instruction_id == "combination:repeat_prompt":
        prompt = args.get("prompt_to_repeat")
        if not prompt:
            return False
        return text.strip().lower().startswith(str(prompt).strip().lower())
    if instruction_id == "startend:end_checker":
        phrase = args.get("end_phrase")
        if not phrase:
            return False
        cleaned = text.strip().strip('"').lower()
        return cleaned.endswith(str(phrase).strip().lower())
    if instruction_id == "startend:quotation":
        stripped = text.strip()
        return len(stripped) > 1 and stripped[0] == '"' and stripped[-1] == '"'
    if instruction_id == "change_case:capital_word_frequency":
        if not _require(args, "capital_frequency", "capital_relation"):
            return False
        if args["capital_relation"] not in _RELATIONS:
            return False
        words = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", text)
        count = sum(1 for word in words if word.isupper())
        return _relation(count, str(args["capital_relation"]), int(args["capital_frequency"]))
    if instruction_id == "change_case:english_capital":
        return bool(text) and text.isupper() and _english_enough(text.lower())
    if instruction_id == "change_case:english_lowercase":
        return bool(text) and text.islower() and _english_enough(text)
    if instruction_id == "punctuation:no_comma":
        return re.search(r"\,", text) is None
    return False


def loose_variants(response: str) -> list[str]:
    """The eight strings from evaluation_lib.test_instruction_following_loose."""
    lines = (response or "").split("\n")
    removed_first = "\n".join(lines[1:]).strip()
    removed_last = "\n".join(lines[:-1]).strip()
    removed_both = "\n".join(lines[1:-1]).strip()
    revised = (response or "").replace("*", "")
    return [
        response or "",
        revised,
        removed_first,
        removed_last,
        removed_both,
        removed_first.replace("*", ""),
        removed_last.replace("*", ""),
        removed_both.replace("*", ""),
    ]


def instruction_flags(instruction_ids: list[str], response: str, kwargs: list[dict] | None) -> dict:
    args_list = list(kwargs or [])
    strict_flags = []
    loose_flags = []
    variants = loose_variants(response)
    for index, instruction_id in enumerate(instruction_ids):
        args = args_list[index] if index < len(args_list) else {}
        if not str(response or "").strip():
            strict_flags.append(False)
        else:
            strict_flags.append(check_instruction(instruction_id, response, args))
        loose = False
        for variant in variants:
            if str(variant or "").strip() and check_instruction(instruction_id, variant, args):
                loose = True
                break
        loose_flags.append(loose)
    return {
        "strict": strict_flags,
        "loose": loose_flags,
        "prompt_strict": bool(strict_flags) and all(strict_flags),
        "prompt_loose": bool(loose_flags) and all(loose_flags),
    }


def score_ifeval_item(row: dict, response: str) -> dict:
    flags = instruction_flags(
        list(row.get("instruction_id_list") or []),
        response,
        list(row.get("kwargs") or []),
    )
    passed = flags["prompt_strict"]
    return {
        "predicted": "",
        "expected": "",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "metric": "prompt_strict",
        "prompt_strict": flags["prompt_strict"],
        "prompt_loose": flags["prompt_loose"],
        "instruction_strict": flags["strict"],
        "instruction_loose": flags["loose"],
    }


def aggregate_ifeval(items: list[dict]) -> dict:
    prompt_n = len(items)
    if not prompt_n:
        return {
            "prompt_strict": None,
            "prompt_loose": None,
            "instruction_strict": None,
            "instruction_loose": None,
            "n_items": 0,
            "n_instructions": 0,
        }
    inst_strict = []
    inst_loose = []
    for item in items:
        inst_strict.extend(item.get("instruction_strict") or [])
        inst_loose.extend(item.get("instruction_loose") or [])
    inst_n = len(inst_strict)
    return {
        "prompt_strict": sum(1 for item in items if item.get("prompt_strict")) / prompt_n,
        "prompt_loose": sum(1 for item in items if item.get("prompt_loose")) / prompt_n,
        "instruction_strict": (sum(1 for flag in inst_strict if flag) / inst_n) if inst_n else None,
        "instruction_loose": (sum(1 for flag in inst_loose if flag) / inst_n) if inst_n else None,
        "n_items": prompt_n,
        "n_instructions": inst_n,
    }
