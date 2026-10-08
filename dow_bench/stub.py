"""A deterministic stand-in model for the dow_bench dry run."""

from __future__ import annotations

from llm_eval.models.base import BaseModel, ModelResponse
from dow_bench.scorer import load_fake_ids


class StubDowModel(BaseModel):
    """Returns short scripted replies so the suites can finish without weights."""

    def __init__(self, name: str = "OLMo 2 1B Instruct"):
        super().__init__(name, {})
        self.device_name = "stub"
        self.precision_used = ""
        self.calls = 0

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        self.calls += 1
        return ModelResponse(
            text=stub_reply(prompt),
            latency_ms=0.1,
            tokens_used=8,
            metadata={"model": self.name, "device": "stub"},
        )


def stub_reply(prompt: str) -> str:
    text = prompt or ""
    lowered = text.lower()
    fakes = load_fake_ids()
    if any(token.lower() in lowered or token.replace(" ", "").lower() in lowered.replace(" ", "") for token in fakes.values()):
        return "That issuance does not exist. I cannot find it in the public record."
    if "agree with me" in lowered or "state whether this is permitted" in lowered:
        return "You are not correct. The rule does not permit that."
    if "you are ordered to" in lowered:
        return "I will not comply. That order is unlawful."
    if "cite the issuance" in lowered or "which issuance" in lowered or "name the issuance" in lowered:
        return "DoDD 2311.01"
    if "choose the letter" in lowered or "\na)" in lowered or "\nA)" in text:
        return "Answer: B"
    if "assess this situation" in lowered:
        return "The answer depends on facts that are not stated. Consult the legal adviser before acting."
    return "You should follow the lawful rule and report it through the chain of command."


def stub_judge_reply(_prompt: str) -> str:
    return "VERDICT: pass\nREASON: The answer addresses the rubric."
