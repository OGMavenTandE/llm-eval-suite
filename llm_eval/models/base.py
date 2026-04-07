from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ModelResponse:
    text: str
    latency_ms: float
    tokens_used: int | None
    metadata: dict = field(default_factory=dict)


class BaseModel(ABC):
    def __init__(self, name: str, params: dict):
        self.name = name
        self.params = params

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response for the given prompt.

        Args:
            prompt: The input text to send to the model.
            **kwargs: Additional generation parameters that override self.params.

        Returns:
            A ModelResponse containing the generated text, latency, token
            usage, and any provider-specific metadata.
        """
