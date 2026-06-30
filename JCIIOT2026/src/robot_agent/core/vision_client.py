"""Ollama vision client — read images from Word/docs and send to multimodal LLM."""

from __future__ import annotations

import base64
import json
import urllib.request
from pathlib import Path


def extract_images_from_docx(path: str | Path) -> list[tuple[str, bytes]]:
    """Extract all images from a .docx file. Returns [(filename, bytes), ...]."""
    from docx import Document

    images = []
    doc = Document(str(path))
    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            name = rel.target_ref.split("/")[-1] if rel.target_ref else "image.png"
            images.append((name, rel.target_part.blob))
    return images


def ask_vision(
    prompt: str,
    images: list[bytes] | bytes,
    *,
    base_url: str = "http://localhost:11434",
    model: str = "qwen3-vl:8b",
    timeout: float = 60.0,
) -> str:
    """Send image(s) + text prompt to an Ollama vision model, return response text."""
    img_list = images if isinstance(images, list) else [images]
    b64_list = [base64.b64encode(img).decode() for img in img_list]

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": b64_list,
            }
        ],
        "stream": False,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8"))
    return resp["message"]["content"]


def read_docx_with_vision(
    path: str | Path,
    *,
    prompt: str = "Describe every object, station, and coordinate visible in this image.",
    base_url: str = "http://localhost:11434",
    model: str = "qwen3-vl:8b",
) -> str:
    """Extract images from a .docx file and describe them all with a vision model.

    Returns concatenated descriptions.
    """
    images = extract_images_from_docx(path)
    if not images:
        return "No images found in the document."

    results = []
    for name, img_data in images:
        desc = ask_vision(f"{prompt}\n[image: {name}]", img_data,
                          base_url=base_url, model=model)
        results.append(f"## {name}\n{desc}")
    return "\n\n".join(results)
