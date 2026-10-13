from llm_eval.models.base import BaseModel, ModelResponse
from llm_eval.models.ollama_model import OllamaModel
from llm_eval.models.openai_model import OpenAIModel
from llm_eval.models.served import LlamaCppModel, VllmModel

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "ollama": OllamaModel,
    "openai": OpenAIModel,
    "vllm": VllmModel,
    "llama_cpp": LlamaCppModel,
}

__all__ = [
    "BaseModel",
    "ModelResponse",
    "OllamaModel",
    "OpenAIModel",
    "VllmModel",
    "LlamaCppModel",
    "MODEL_REGISTRY",
]
