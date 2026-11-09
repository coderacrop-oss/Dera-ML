"""Gemini image generation connector.

Takes image_prompt from ContentOutput and returns a PIL Image.
Model: gemini-3.1-flash-image (v1beta API)

Usage:
    from image_generator import generate_for_content
    image = generate_for_content(content, api_key="...")
"""
from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PILImage
    from content_generator import ContentOutput


def generate_image(
    prompt: str,
    api_key: str | None = None,
    model: str = "gemini-3.1-flash-image",
) -> "PILImage.Image":
    """Generate an image from a text prompt via Gemini.

    Args:
        prompt:   English image generation prompt
        api_key:  Gemini API key (falls back to GEMINI_API_KEY env var)
        model:    Gemini image model name

    Returns:
        PIL Image (RGB)
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError("google-genai not installed. Run: pip install google-genai")

    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow not installed. Run: pip install Pillow")

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY not set — pass api_key or set env var")

    client = genai.Client(api_key=key, http_options={"api_version": "v1beta"})

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")

    raise ValueError("Gemini returned no image in response")


def generate_for_content(
    content: "ContentOutput",
    api_key: str | None = None,
    model: str = "gemini-3.1-flash-image",
) -> "PILImage.Image":
    """Generate image for a ContentOutput using its image_prompt.

    Falls back to a generic food photography prompt if image_prompt is empty.
    """
    prompt = content.image_prompt or (
        "Professional food photography, commercial ad quality, "
        "warm natural light, shallow depth of field, no text, no watermark"
    )
    return generate_image(prompt, api_key=api_key, model=model)
