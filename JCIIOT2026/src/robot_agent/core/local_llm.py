"""Local LLM client — loads a GGUF model directly, no server needed.

Usage::

    from robot_agent.core.local_llm import LocalLLM
    llm = LocalLLM("/path/to/model.gguf")
    text = llm.generate("Hello")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LocalLLM:
    """Thin wrapper around llama-cpp-python for direct GGUF inference.

    Drop-in replacement for ``OllamaClient`` — same ``generate()`` signature.
    """

    model_path: str | Path
    n_ctx: int = 4096
    n_threads: int = 4
    verbose: bool = False

    _model: object | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is required for local LLM inference. "
                "Install: pip install llama-cpp-python"
            )
        path = str(self.model_path)
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        logger.info("Loading model: %s", path)
        self._model = Llama(
            model_path=path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=self.verbose,
        )
        logger.info("Model loaded")

    def generate(
        self,
        prompt: str,
        *,
        num_predict: int = 4096,
        temperature: float = 0.1,
        json_mode: bool = False,
    ) -> str:
        """Send *prompt* and return the model's text response.

        Same signature as ``OllamaClient.generate()``.
        """
        self._load()
        result = self._model(
            prompt,
            max_tokens=num_predict,
            temperature=temperature,
            top_p=0.9,
            echo=False,
            stream=False,
        )
        text = str(result["choices"][0]["text"]).strip()
        if not text:
            logger.warning("LocalLLM returned empty response")
        return text

    def healthcheck(self) -> dict[str, str]:
        """Check model availability."""
        path = str(self.model_path)
        if Path(path).exists():
            return {"ok": "true", "message": f"Model file found: {path}", "models": path}
        return {"ok": "false", "message": f"Model file not found: {path}", "models": ""}
