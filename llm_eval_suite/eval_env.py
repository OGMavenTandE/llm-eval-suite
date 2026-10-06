"""Pinned eval environment. Training venvs are left alone.

``run_app.bat`` creates ``.venv-eval`` and installs these pins. Torch comes
from the PyTorch CUDA index when ``nvidia-smi`` is on PATH, otherwise from
the CPU index. Install torch before garak so pip does not pull a CPU wheel
from PyPI. The strings here and in ``run_app.bat`` are the same pins.
"""

from __future__ import annotations

TORCH_VERSION = "2.13.0"
GARAK_VERSION = "0.17.0"
TRANSFORMERS_VERSION = "5.18.0"
CUDA_INDEX = "https://download.pytorch.org/whl/cu126"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
VENV_DIR = ".venv-eval"
CPU_WARNING = (
    "WARNING: No NVIDIA GPU was detected (nvidia-smi not found). "
    "Installing the CPU build of torch. Garak and local Hugging Face models will be slow."
)


def torch_pip_args(*, nvidia_gpu: bool) -> list[str]:
    """Arguments after ``python -m pip``. A list, never a shell string."""
    index = CUDA_INDEX if nvidia_gpu else CPU_INDEX
    return ["install", f"torch=={TORCH_VERSION}", "--index-url", index]


def garak_pip_args(*, nvidia_gpu: bool) -> list[str]:
    """Install garak and transformers without replacing the torch build."""
    index = CUDA_INDEX if nvidia_gpu else CPU_INDEX
    return [
        "install",
        f"garak=={GARAK_VERSION}",
        f"transformers=={TRANSFORMERS_VERSION}",
        f"torch=={TORCH_VERSION}",
        "--extra-index-url",
        index,
    ]
