"""Deterministic stand-in model for the standard-bench smoke."""

from __future__ import annotations

from llm_eval.models.base import BaseModel, ModelResponse


class StubStandardModel(BaseModel):
    def __init__(self, name: str = "OLMo 2 1B Instruct"):
        super().__init__(name, {})
        self.device_name = "stub"
        self.precision_used = "fp16"
        self.calls = 0
        self.max_context = 8192

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        self.calls += 1
        return ModelResponse(
            text=stub_reply(prompt),
            latency_ms=0.1,
            tokens_used=8,
            metadata={
                "model": self.name,
                "device": "stub",
                "raw_text": stub_reply(prompt),
                "hit_token_cap": False,
                "think_truncated": False,
                "prompt_over_budget": False,
                "completion_tokens": 8,
            },
        )


def stub_reply(prompt: str) -> str:
    text = prompt or ""
    if "STUBKEY mmlu-a" in text:
        return "The answer is (A)."
    if "STUBKEY gsm-4" in text:
        return "Working.\nThe answer is 4."
    if "STUBKEY math-half" in text:
        return "Half of one.\n\\boxed{1/2}"
    if "STUBKEY ifeval-comma" in text:
        return "This response has no comma at all."
    if "STUBKEY add" in text or "def add(" in text:
        return "    return a + b\n"
    if "STUBKEY shared" in text:
        return "    return tuple(sorted(set(left) & set(right)))\n"
    if "polar_rect(" in text:
        return (
            "import cmath\n"
            "def polar_rect(x, y):\n"
            "    cn = cmath.polar(complex(x, y))\n"
            "    cn1 = cmath.rect(2, cmath.pi)\n"
            "    return (cn, cn1)\n"
        )
    if "check_str(" in text:
        return (
            "import re\n"
            "def check_str(string):\n"
            "    regex = '^[aeiouAEIOU][A-Za-z0-9_]*'\n"
            "    return re.search(regex, string)\n"
        )
    return "I do not know."
