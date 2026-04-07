from llm_eval.models.base import BaseModel, ModelResponse
from llm_eval.models.ollama_model import OllamaModel
from llm_eval.models.openai_model import OpenAIModel

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "ollama": OllamaModel,
    "openai": OpenAIModel,
}

__all__ = ["BaseModel", "ModelResponse", "OllamaModel", "OpenAIModel", "MODEL_REGISTRY"]
