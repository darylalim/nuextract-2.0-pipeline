"""NuExtract3 MLX runtime wrapper.

Loads numind/NuExtract3-mlx-8bits via mlx-vlm and exposes streaming generation
across the three NuExtract3 modes (structured / markdown / template-generation).

Includes a fix-up shim for a conversion bug in the MLX repo: its
processor_config.json declares Qwen3VLImageProcessor, while the model author's
own numind/NuExtract3 declares Qwen2VLImageProcessor. Upstream is authoritative,
so the rewrite stays correct regardless of what transformers ships.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from mlx_vlm import load as mlx_vlm_load
from mlx_vlm import stream_generate as mlx_vlm_stream_generate

DEFAULT_MODEL_ID = "numind/NuExtract3-mlx-8bits"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0

MODE_STRUCTURED = "structured"
MODE_CONTENT = "content"
MODE_MARKDOWN = "markdown"
MODE_TEMPLATE_GENERATION = "template-generation"


def patch_processor_config(local_dir: str | Path) -> bool:
    """Replace Qwen3VLImageProcessor with Qwen2VLImageProcessor in the local copy.

    numind/NuExtract3-mlx-8bits declares "Qwen3VLImageProcessor" in its
    processor_config.json; the model author's own numind/NuExtract3 declares
    "Qwen2VLImageProcessor", with otherwise identical geometry (patch_size 16,
    merge_size 2, temporal_patch_size 2). Upstream is authoritative, so this is
    a conversion bug in the MLX repo — not a workaround for a missing class.

    Do NOT gate this on `hasattr(transformers, "Qwen3VLImageProcessor")`. That
    the class is currently absent from transformers is why the bug is *visible*
    (load fails loudly today), not why the rewrite is *correct*; adding such a
    gate would silently stop patching the day transformers ships the class and
    load preprocessing the model author never specified.

    Idempotent — returns True if it changed anything.
    """
    config_path = Path(local_dir) / "processor_config.json"
    if not config_path.exists():
        return False
    content = config_path.read_text()
    if '"Qwen3VLImageProcessor"' not in content:
        return False
    config_path.write_text(
        content.replace('"Qwen3VLImageProcessor"', '"Qwen2VLImageProcessor"')
    )
    return True


def load_model(model_id: str = DEFAULT_MODEL_ID) -> tuple[Any, Any]:
    """Download, patch, and load NuExtract3-MLX. Returns (model, processor)."""
    local_dir = snapshot_download(repo_id=model_id)
    patch_processor_config(local_dir)
    return mlx_vlm_load(local_dir)


def build_messages(
    text: str = "",
    image_path: str | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Build a chat message list with optional system + user image/text parts.

    The Jinja template inserts the vision placeholder for any user-content item
    that has an 'image' or 'image_url' key or type == 'image'; actual pixel
    data flows separately through stream_generate(image=...). System messages
    must be string-only — the Jinja raises if they contain images.
    """
    messages: list[dict[str, Any]] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    parts: list[dict[str, Any]] = []
    if image_path:
        parts.append({"type": "image", "image": image_path})
    text = (text or "").strip()
    if text:
        parts.append({"type": "text", "text": text})
    if not parts:
        parts.append({"type": "text", "text": ""})
    messages.append({"role": "user", "content": parts})
    return messages


def render_prompt(
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    template: str | None = None,
    instructions: str | None = None,
    mode: str | None = None,
    enable_thinking: bool = False,
) -> str:
    """Render NuExtract3's chat template with task kwargs.

    Kwargs pass inline (HF transformers convention), not nested under
    chat_template_kwargs (which is a vLLM-specific convention).
    """
    kwargs: dict[str, Any] = {"enable_thinking": enable_thinking}
    if template is not None:
        kwargs["template"] = template
    if instructions:
        kwargs["instructions"] = instructions
    if mode is not None:
        kwargs["mode"] = mode
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def stream_extract(
    model: Any,
    processor: Any,
    *,
    text: str = "",
    image_path: str | None = None,
    system_prompt: str | None = None,
    template: str | None = None,
    instructions: str | None = None,
    mode: str | None = None,
    enable_thinking: bool = False,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Iterator[str]:
    """Stream generation. Yields cumulative output text on each chunk."""
    messages = build_messages(
        text=text, image_path=image_path, system_prompt=system_prompt
    )
    prompt = render_prompt(
        processor,
        messages,
        template=template,
        instructions=instructions,
        mode=mode,
        enable_thinking=enable_thinking,
    )
    kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        # mlx-vlm defaults this to False, which leaves the EOS token (<|im_end|>)
        # in the decoded text. Structured/template modes hide it because
        # extract_answer_block re-parses the JSON, but markdown mode renders and
        # downloads the raw string, so the saved .md ends with a literal token.
        "skip_special_tokens": True,
    }
    if image_path:
        kwargs["image"] = [image_path]
    accumulated = ""
    for chunk in mlx_vlm_stream_generate(model, processor, prompt, **kwargs):
        # Read .text directly rather than falling back to str(chunk): if a future
        # mlx-vlm renames the field, an AttributeError here is the loud failure we
        # want. The fallback would instead splice the whole GenerationResult repr
        # into the output — and no test would catch it, since they all feed stubs.
        accumulated += chunk.text
        yield accumulated


def split_reasoning_and_output(text: str, reasoning_enabled: bool) -> tuple[str, str]:
    """Split <think>...</think>... into (reasoning, output).

    When reasoning is disabled, all text is treated as output. When enabled but
    </think> hasn't arrived yet, everything is reasoning and output is empty.
    """
    if not text:
        return "", ""
    if not reasoning_enabled:
        return "", text.strip()
    lower = text.lower()
    end_tag = "</think>"
    if end_tag in lower:
        idx = lower.find(end_tag)
        return text[:idx].strip(), text[idx + len(end_tag) :].strip()
    return text.strip(), ""


_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer_block(text: str) -> str:
    """Pull <answer>...</answer> contents, or the longest valid JSON object.

    Tries `json.JSONDecoder.raw_decode` at every `{` position and returns the
    longest successfully-parsed span. Falls back to the stripped text if no
    valid JSON is found. Handles the common "reasoning text + JSON" case
    correctly, unlike a greedy regex.
    """
    if not text:
        return ""
    match = _ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()
    decoder = json.JSONDecoder()
    best: str | None = None
    start = 0
    while True:
        i = text.find("{", start)
        if i == -1:
            break
        try:
            _obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            start = i + 1
            continue
        span = text[i:end]
        if best is None or len(span) > len(best):
            best = span
        start = end
    return best if best is not None else text.strip()


def pretty_json_or_text(text: str) -> str:
    """Try to pretty-print as JSON; fall back to the original string."""
    stripped = (text or "").strip()
    if not stripped:
        return ""
    try:
        return json.dumps(json.loads(stripped), indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return stripped
