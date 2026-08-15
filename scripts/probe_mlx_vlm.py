"""Probe whether mlx-vlm + HF transformers plumb template kwargs through to NuExtract3's Jinja template.

Run: uv run python scripts/probe_mlx_vlm.py

What this verifies (in order):
  1. mlx_vlm.load() succeeds on numind/NuExtract3-mlx-8bits
  2. processor.apply_chat_template(messages, template=..., enable_thinking=False)
     produces a rendered prompt containing the template JSON and 【task】structured
  3. Passing mode='markdown' (no template) produces 【task】content (markdown maps to content per the template)
  4. Passing mode='template-generation' produces 【task】template generation
  5. End-to-end generate() returns parseable JSON on a trivial extraction
  6. End-to-end generate() with an image reaches the vision path and returns
     parseable JSON (the only check that exercises torchvision / the image
     processor — every test in tests/ mocks mlx_vlm.stream_generate)

NOTE: The HF Space uses vLLM, which accepts kwargs nested under `chat_template_kwargs`.
HF transformers' apply_chat_template expects them as direct keyword arguments instead.

Prints PASS/FAIL per check and a final verdict. Exits non-zero on any failure.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from mlx_vlm import generate, load

MODEL_ID = "numind/NuExtract3-mlx-8bits"


def _print_header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _print_check(name: str, ok: bool, detail: str = "") -> bool:
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def _render_probe_image(dest: Path) -> Path:
    """Render a small document-like PNG so check 6 has real pixels to process.

    Generated rather than committed so the probe stays self-contained and the
    repo carries no binary fixture.
    """
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (640, 220), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=34)
    draw.text((40, 40), "INVOICE", fill="black", font=font)
    draw.text((40, 100), "Name: John Doe", fill="black", font=font)
    draw.text((40, 150), "Age: 30", fill="black", font=font)
    image.save(dest)
    return dest


def _render(processor: Any, messages: list[dict], **kwargs: Any) -> str:
    """Render the chat template via the HF transformers convention."""
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def main() -> int:
    _print_header(f"Loading {MODEL_ID} via mlx_vlm.load()")
    try:
        # mlx-vlm returns dynamically-typed (model, processor) objects; treat them as
        # Any (as nuextract.load_model does). mlx-vlm annotates generate()'s processor
        # param as PreTrainedTokenizer but accepts a full processor at runtime.
        loaded: tuple[Any, Any] = load(MODEL_ID)
        model, processor = loaded
    except Exception as e:
        print(f"  [FAIL] load() raised: {type(e).__name__}: {e}")
        return 2
    print("  [PASS] model + processor loaded")

    template = {"name": "verbatim-string", "age": "integer"}
    template_str = json.dumps(template)
    messages = [{"role": "user", "content": "John Doe is 30 years old."}]

    passes: list[bool] = []

    # --- Check 2: template kwarg reaches the Jinja template ---
    _print_header("Check 2: apply_chat_template(template=..., enable_thinking=False)")
    try:
        rendered_with_template = _render(
            processor,
            messages,
            template=template_str,
            enable_thinking=False,
        )
        print("  Rendered prompt (truncated to 800 chars):")
        print("  " + repr(rendered_with_template[:800]))
        ok = (
            '"name"' in rendered_with_template
            and '"verbatim-string"' in rendered_with_template
            and "【task】structured" in rendered_with_template
            and "【template_start】" in rendered_with_template
        )
        passes.append(
            _print_check("template JSON + structured task section appear in prompt", ok)
        )
    except Exception as e:
        print(f"  [FAIL] apply_chat_template raised: {type(e).__name__}: {e}")
        passes.append(False)
        rendered_with_template = ""

    # --- Check 3: mode='markdown' (no template) maps to 'content' task ---
    _print_header("Check 3: apply_chat_template(mode='markdown')")
    try:
        rendered_markdown = _render(
            processor,
            messages,
            mode="markdown",
            enable_thinking=False,
        )
        print("  Rendered prompt (truncated to 800 chars):")
        print("  " + repr(rendered_markdown[:800]))
        # Per the Jinja: markdown mode reassigns mode='content' (line 12 of chat_template.jinja)
        ok = (
            "【task】content" in rendered_markdown
            and "【template_start】" not in rendered_markdown
        )
        passes.append(
            _print_check(
                "mode='markdown' produces 【task】content with no template", ok
            )
        )
    except Exception as e:
        print(f"  [FAIL] apply_chat_template raised: {type(e).__name__}: {e}")
        passes.append(False)

    # --- Check 4: mode='template-generation' produces its own task type ---
    _print_header("Check 4: apply_chat_template(mode='template-generation')")
    try:
        rendered_tplgen = _render(
            processor,
            messages,
            mode="template-generation",
        )
        print("  Rendered prompt (truncated to 800 chars):")
        print("  " + repr(rendered_tplgen[:800]))
        ok = "【task】template generation" in rendered_tplgen
        passes.append(
            _print_check(
                "template-generation mode produces 【task】template generation", ok
            )
        )
    except Exception as e:
        print(f"  [FAIL] apply_chat_template raised: {type(e).__name__}: {e}")
        passes.append(False)

    # --- Check 5: end-to-end generation with the template ---
    _print_header("Check 5: end-to-end generate() with typed template")
    try:
        print("  Generating (this may take ~30s on first run)...")
        output = generate(
            model,
            processor,
            rendered_with_template,
            max_tokens=256,
            verbose=False,
        )
        output_text = output.text if hasattr(output, "text") else str(output)
        print(f"  Raw output: {output_text!r}")
        stripped = output_text.strip()
        # NuExtract3 may emit `<answer>...</answer>` wrappers or raw JSON
        answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", stripped, re.DOTALL)
        if answer_match:
            stripped = answer_match.group(1).strip()
        # Or it may emit thinking trace + JSON
        json_match = re.search(r"\{[\s\S]*\}", stripped)
        if json_match:
            stripped = json_match.group(0)
        try:
            parsed = json.loads(stripped)
            ok = isinstance(parsed, dict) and "name" in parsed and "age" in parsed
            passes.append(
                _print_check(
                    "output parses as JSON with expected keys",
                    ok,
                    f"parsed={parsed}",
                )
            )
        except json.JSONDecodeError:
            passes.append(
                _print_check(
                    "output parses as JSON",
                    False,
                    "not parseable; raw output above",
                )
            )
    except Exception as e:
        print(f"  [FAIL] generate() raised: {type(e).__name__}: {e}")
        passes.append(False)

    # --- Check 6: end-to-end generation from an image (vision path) ---
    _print_header("Check 6: end-to-end generate() with an image")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = _render_probe_image(Path(tmpdir) / "probe.png")
            image_messages = [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": str(image_path)}],
                }
            ]
            rendered_image = _render(
                processor,
                image_messages,
                template=template_str,
                enable_thinking=False,
            )
            print("  Generating from image (exercises the image processor)...")
            output = generate(
                model,
                processor,
                rendered_image,
                image=[str(image_path)],
                max_tokens=256,
                verbose=False,
            )
        output_text = output.text if hasattr(output, "text") else str(output)
        print(f"  Raw output: {output_text!r}")
        json_match = re.search(r"\{[\s\S]*\}", output_text)
        parsed = json.loads(json_match.group(0)) if json_match else None
        # The point is that the vision path runs and produces our schema; OCR
        # accuracy on a synthetic image is not what this check is gating.
        ok = isinstance(parsed, dict) and "name" in parsed and "age" in parsed
        passes.append(
            _print_check(
                "image input reaches the vision path and returns typed JSON",
                ok,
                f"parsed={parsed}",
            )
        )
    except Exception as e:
        print(f"  [FAIL] image generate() raised: {type(e).__name__}: {e}")
        passes.append(False)

    _print_header("VERDICT")
    if all(passes):
        print("  ALL CHECKS PASSED — safe to proceed with Phase 1.")
        return 0
    failed = sum(1 for p in passes if not p)
    print(f"  {failed} CHECK(S) FAILED — review output above before Phase 1.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
