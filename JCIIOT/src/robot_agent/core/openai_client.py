"""Generic OpenAI-compatible API client — supports DeepSeek, Zhipu GLM, OpenAI, etc.

Implements the same ``generate()`` interface as ``OllamaClient`` / ``LocalLLM``
so it can be used as a drop-in replacement in ``TaskPlanner``.

Usage::

    # DeepSeek
    client = OpenAIClient(
        api_key="sk-...",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
    )

    # Zhipu GLM via OpenAI-compatible endpoint
    client = OpenAIClient(
        api_key="...",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model="glm-4.6v-flash",
    )

    # Any self-hosted vLLM / llama.cpp server
    client = OpenAIClient(
        api_key="not-needed",
        base_url="http://localhost:8000/v1",
        model="qwen2.5-7b",
    )
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_TIMEOUT = 120.0


@dataclass(slots=True)
class OpenAIClient:
    """Generic OpenAI-compatible chat completions client.

    Drop-in replacement for ``OllamaClient`` — same ``generate()`` and
    ``healthcheck()`` signatures.
    """

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT

    # ── public API ────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        *,
        num_predict: int = 4096,
        temperature: float = 0.1,
        json_mode: bool = False,
    ) -> str:
        """Send *prompt* and return the model's text response.

        If *json_mode* is True, requests ``response_format: {"type": "json_object"}``
        which constrains the model to valid JSON (supported by most OpenAI-compatible
        servers).  Falls back gracefully if the server doesn't support it.
        """
        messages = [{"role": "user", "content": prompt}]

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": num_predict,
            "stream": False,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        # Some providers (Zhipu GLM) support disabling thinking mode
        # This is harmless for providers that don't recognize it
        payload["thinking"] = {"type": "disabled"}

        return _openai_chat(
            payload,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def healthcheck(self) -> dict[str, str]:
        """Check connectivity and model availability.

        Tries to list models via ``/v1/models`` (standard OpenAI endpoint).
        Falls back to a simple chat completion ping if model listing fails.
        """
        base = self.base_url.rstrip("/")

        # Try /v1/models first
        try:
            models = _list_models(self.api_key, base, self.timeout)
            model_ids = {m.get("id", "") for m in models if isinstance(m, dict)}
            if self.model in model_ids:
                return {
                    "ok": "true",
                    "message": f"Connected, model {self.model} found",
                    "models": ", ".join(sorted(model_ids)),
                }
            return {
                "ok": "false",
                "message": (
                    f"Connected, but model {self.model} is not in the model list. "
                    f"Available: {', '.join(sorted(model_ids))}"
                ),
                "models": ", ".join(sorted(model_ids)),
            }
        except Exception:
            pass

        # Fallback: send a minimal chat completion to verify the key + URL work
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 4,
                "stream": False,
            }
            _openai_chat(payload, api_key=self.api_key, base_url=base, timeout=min(self.timeout, 30.0))
            return {
                "ok": "true",
                "message": f"Chat completion ping OK (model: {self.model})",
                "models": self.model,
            }
        except Exception as exc:
            return {"ok": "false", "message": str(exc), "models": ""}


# ── internal helpers ─────────────────────────────────────


def _resolve_chat_url(base_url: str) -> str:
    """Resolve the /v1/chat/completions endpoint from a base URL.

    Handles several common patterns::

        https://api.deepseek.com          → .../v1/chat/completions
        https://api.deepseek.com/v1       → .../v1/chat/completions
        https://open.bigmodel.cn/api/paas/v4  → .../chat/completions
        http://localhost:8000/v1/chat/completions → no change
    """
    url = base_url.rstrip("/")

    # Already a full chat/completions URL
    if url.endswith("/chat/completions"):
        return url

    # If the URL already contains /v1, append /chat/completions
    if "/v1" in url or "/v4" in url:
        return f"{url}/chat/completions"

    # Otherwise, append /v1/chat/completions
    return f"{url}/v1/chat/completions"


def _list_models(api_key: str, base_url: str, timeout: float) -> list[dict]:
    """List models via the /v1/models endpoint."""
    url = base_url.rstrip("/")
    if not url.endswith("/models"):
        # Extract the base: strip /chat/completions if present, then append /models
        if "/chat/completions" in url:
            url = url.rsplit("/chat/completions", 1)[0]
        if not url.endswith("/models"):
            # If URL is like .../v1, append /models
            if url.endswith("/v1") or url.endswith("/v4"):
                url = f"{url}/models"
            else:
                url = f"{url}/v1/models"

    req = urllib.request.Request(
        url,
        method="GET",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"OpenAI models list HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:500]}"
        ) from exc

    if isinstance(body, dict):
        return body.get("data", [])
    return []


def _openai_chat(
    payload: dict,
    *,
    api_key: str,
    base_url: str,
    timeout: float,
) -> str:
    """Send a chat completion request and return the text response.

    Raises ``RuntimeError`` on HTTP errors or empty responses.
    """
    chat_url = _resolve_chat_url(base_url)
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        chat_url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(
            f"OpenAI API HTTP {exc.code} at {chat_url}: {err_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"OpenAI API unreachable at {chat_url}: {exc}"
        ) from exc

    # Extract the message content from the first choice
    choices = body.get("choices", [])
    if not choices:
        logger.error("OpenAI API returned empty choices: %s", str(body)[:300])
        return ""

    message = choices[0].get("message", {})
    content = str(message.get("content", "")).strip()

    # Reasoning / thinking model fallback (e.g. DeepSeek-R1)
    if not content:
        reasoning = str(message.get("reasoning_content", "")).strip()
        if reasoning:
            logger.info("Using 'reasoning_content' field as response (reasoning model detected)")
            return reasoning
        thinking = str(message.get("thinking", "")).strip()
        if thinking:
            logger.info("Using 'thinking' field as response (thinking model detected)")
            return thinking

    if not content:
        logger.warning(
            "OpenAI API returned empty content (finish_reason=%s)",
            choices[0].get("finish_reason", "unknown"),
        )
        # One retry after a short wait
        logger.warning("Retrying OpenAI request after empty response...")
        time.sleep(2.0)
        try:
            req2 = urllib.request.Request(
                chat_url,
                data=data,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
            with urllib.request.urlopen(req2, timeout=timeout) as resp2:
                body2 = json.loads(resp2.read().decode("utf-8"))
            choices2 = body2.get("choices", [])
            if choices2:
                msg2 = choices2[0].get("message", {})
                content = str(msg2.get("content", "")).strip()
        except Exception:
            pass

    return content
