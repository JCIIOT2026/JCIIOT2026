"""Zhipu GLM API client — alternative LLM backend (from llm_task_navigator.py).

More reliable than self-hosted Ollama for production use.  Requires a valid
API key in ``GLM_API_KEY`` env var or the constant below.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

GLM_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_MODEL = "glm-4.6v-flash"


class GlmClient:
    """Minimal GLM API client — same interface as OllamaClient / LocalLLM."""

    def __init__(self, *, api_key: str | None = None, model: str = DEFAULT_MODEL, api_url: str = GLM_API_URL, timeout: float = 60.0):
        import os
        self._api_key = api_key or os.getenv("GLM_API_KEY", "")
        self._model = model
        self._api_url = api_url
        self._timeout = timeout

    def generate(self, prompt: str, *, num_predict: int = 4096, temperature: float = 0.1, json_mode: bool = False) -> str:
        return _glm_generate(prompt, model=self._model, api_key=self._api_key, api_url=self._api_url, timeout=self._timeout, num_predict=num_predict, temperature=temperature)


def _glm_generate(prompt: str, *, model: str, api_key: str, api_url: str, timeout: float, num_predict: int, temperature: float) -> str:
    """Send a single prompt to GLM and return text."""
    import json, urllib.request, urllib.error
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": num_predict,
        "stream": False,
        "thinking": {"type": "disabled"},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        api_url, data=data, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"GLM HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}") from exc
    return body["choices"][0]["message"]["content"]


def call_glm_plan(
    command: str,
    scene: dict,
    *,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    api_url: str = GLM_API_URL,
    timeout: float = 60.0,
) -> dict:
    """Send a planning request to GLM and return the parsed JSON plan."""
    import os
    key = api_key or os.getenv("GLM_API_KEY")
    if not key:
        raise RuntimeError("Missing GLM API key. Set GLM_API_KEY env var or pass api_key= explicitly.")

    map_summary = _summarize_map_for_llm(scene)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a robot task planner. Convert the user's Chinese instruction "
                "into strict JSON. Do not plan low-level velocities or grid paths. "
                "Use only station names that appear in the provided map. "
                "Return exactly one JSON object with keys: task_type, source, target, steps. "
                "For pick-and-place, steps must be navigate→pick→navigate→place."
            ),
        },
        {
            "role": "user",
            "content": json.dumps({
                "user_command": command,
                "available_map_nodes": map_summary,
                "valid_task_type": "pick_and_place",
                "required_output_example": {
                    "task_type": "pick_and_place",
                    "source": "input_1",
                    "target": "output_3",
                    "steps": [
                        {"action": "navigate", "goal": "input_1"},
                        {"action": "pick", "object_from": "input_1"},
                        {"action": "navigate", "goal": "output_3"},
                        {"action": "place", "object_to": "output_3"},
                    ],
                },
            }, ensure_ascii=False),
        },
    ]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "stream": False,
        "thinking": {"type": "disabled"},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GLM API HTTP {exc.code}: {err_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GLM API request failed: {exc}") from exc

    content = body["choices"][0]["message"]["content"]
    return _parse_llm_json(content)


def _summarize_map_for_llm(scene: dict) -> dict:
    """Extract compact station summary for the GLM prompt."""
    nodes: dict[str, dict] = {}
    for group in ("input_ports", "output_ports"):
        for name, obj in scene.get(group, {}).items():
            nodes[name] = {
                "role": obj.get("role"),
                "kind": obj.get("kind"),
                "center": obj.get("center"),
                "approach": obj.get("approach"),
                "display_name": obj.get("display_name", name),
            }
    return nodes


def _parse_llm_json(content: str) -> dict:
    """Extract JSON from GLM response (handles markdown fences)."""
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start:end + 1])
        raise
