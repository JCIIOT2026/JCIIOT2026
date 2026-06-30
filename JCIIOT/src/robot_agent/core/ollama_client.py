"""Minimal Ollama HTTP client — sends prompt, returns raw text.

This module does ONE thing: talk to the Ollama /api/generate endpoint.
It does NOT handle JSON parsing, repair, retry, or structured output.
Those concerns belong to the caller (planner).

Intentionally simple — changes elsewhere should never break connectivity.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from urllib import error, request

logger = logging.getLogger(__name__)

DEFAULT_NUM_PREDICT = 4096
DEFAULT_TEMPERATURE = 0.1


@dataclass(slots=True)
class OllamaClient:
    """Thin wrapper around Ollama's HTTP API.

    Usage::

        client = OllamaClient("http://localhost:11434", "qwen3")
        text = client.generate("Hello")
        print(text)
    """

    base_url: str
    model: str
    timeout: float = 120.0

    # ── public API ────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        *,
        num_predict: int = DEFAULT_NUM_PREDICT,
        temperature: float = DEFAULT_TEMPERATURE,
        json_mode: bool = False,
    ) -> str:
        """Send *prompt* and return the model's text response.

        If *json_mode* is True, adds ``format: "json"`` to constrain the
        model to valid JSON output.  Some models / servers don't support
        this — the caller should fall back to json_mode=False on empty result.
        """
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }
        if json_mode:
            payload["format"] = "json"

        data = self._post("/api/generate", payload)

        if "error" in data:
            logger.error("Ollama error: %s", data["error"])
            return ""

        text = _extract_response_text(data)
        if not text:
            # Some models occasionally return empty on first attempt (e.g. still
            # loading into VRAM).  Retry once after a short wait.
            logger.warning(
                "Ollama returned empty response (keys: %s), retrying…",
                sorted(data.keys()),
            )
            time.sleep(2.0)
            data = self._post("/api/generate", payload)
            if "error" in data:
                logger.error("Ollama retry error: %s", data["error"])
                return ""
            text = _extract_response_text(data)
            if not text:
                logger.error("Ollama retry also empty — model may not be ready")

        return text

    def healthcheck(self) -> dict[str, str]:
        """Check connectivity and model availability."""
        try:
            root = self._get("/api/tags")
        except RuntimeError as exc:
            return {"ok": "false", "message": str(exc), "models": ""}

        models = root.get("models", []) if isinstance(root, dict) else []
        model_names = {
            str(item.get("name", "")) for item in models if isinstance(item, dict)
        }
        if self.model in model_names:
            return {
                "ok": "true",
                "message": f"Connected, model {self.model} found",
                "models": ", ".join(sorted(model_names)),
            }
        return {
            "ok": "false",
            "message": f"Connected, but model {self.model} is not in the model list",
            "models": ", ".join(sorted(model_names)),
        }

    # ── internal HTTP ─────────────────────────────────────

    def _post(self, path: str, payload: dict) -> dict:
        return self._request("POST", path, payload)

    def _get(self, path: str) -> dict:
        return self._request("GET", path)

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        url = f"{self.base_url.rstrip('/')}{path}"
        body_bytes: bytes | None = None
        if payload is not None:
            body_bytes = json.dumps(payload).encode("utf-8")

        req = request.Request(
            url,
            data=body_bytes,
            headers={"Content-Type": "application/json"} if body_bytes else {},
            method=method,
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                raw_body = resp.read().decode("utf-8")
        except error.URLError as exc:
            raise RuntimeError(f"Ollama unreachable at {url}: {exc}") from exc

        return _parse_ollama_body(raw_body)


def _parse_ollama_body(body: str) -> dict:
    """Parse Ollama HTTP response body.

    Handles both single JSON objects and ndjson (streaming) responses.
    """
    # 1) single JSON object
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass

    # 2) ndjson — accumulate "response" fragments across lines
    lines = [ln for ln in body.strip().splitlines() if ln.strip()]
    accumulated: list[str] = []
    meta: dict = {}
    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "response" in obj and obj["response"]:
            accumulated.append(obj["response"])
        meta = obj

    if accumulated:
        return {**meta, "response": "".join(accumulated)}

    raise RuntimeError(f"Unparseable Ollama response (first 500 chars): {body[:500]}")


def _extract_response_text(data: dict) -> str:
    """Extract the model's text output from an Ollama /api/generate response.

    Most models put output in ``response``.  Reasoning / thinking models
    (e.g. qwen with MTP) may put the final answer in ``thinking`` and leave
    ``response`` empty — we fall back accordingly.
    """
    text = str(data.get("response", "")).strip()
    if text:
        return text

    # Thinking-model fallback: the output may be in the ``thinking`` field
    thinking = str(data.get("thinking", "")).strip()
    if thinking:
        logger.info("Using 'thinking' field as response (thinking model detected)")
        return thinking

    return ""
