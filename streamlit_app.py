"""NuExtract3 Streamlit app — mirrors the official HF Space, runs locally on MLX.

Single image + optional text input, JSON template editor, three modes:
  - Extract JSON (structured extraction)
  - Convert to Markdown (document-to-markdown)
  - Generate template (NL description → JSON template)

Streams output with optional <think>...</think> reasoning parsing.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from nuextract import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    MODE_MARKDOWN,
    MODE_TEMPLATE_GENERATION,
    extract_answer_block,
    load_model,
    pretty_json_or_text,
    split_reasoning_and_output,
    stream_extract,
)

DEFAULT_TEMPLATE = json.dumps(
    {
        "title": "string",
        "entities": ["string"],
        "dates": ["YYYY-MM-DD"],
        "amounts": [{"value": "number", "currency": "string"}],
    },
    indent=2,
)

TEMPLATE_GEN_GUIDANCE = (
    "Generate a concise JSON extraction template for this document. "
    "Use descriptive field names and simple type hints like string, number, "
    "verbatim-string, date, boolean, or arrays of objects. Return only the JSON template."
)


@st.cache_resource
def get_model() -> tuple[Any, Any]:
    return load_model()


_IMG_PATH_KEY = "_uploaded_image_path"
_IMG_ID_KEY = "_uploaded_image_id"


def _save_uploaded_image(uploaded_file: Any) -> str | None:
    """Persist the uploaded image to a temp file once per upload.

    Cached in session_state keyed on uploaded_file.file_id so reruns don't
    re-write the image; the previous temp file is cleaned up when a new upload
    arrives.
    """
    if uploaded_file is None:
        # Upload was removed: delete the orphaned temp file and clear stale
        # session_state so nothing leaks on disk or lingers in state.
        cached_path = st.session_state.pop(_IMG_PATH_KEY, None)
        st.session_state.pop(_IMG_ID_KEY, None)
        if cached_path:
            Path(cached_path).unlink(missing_ok=True)
        return None
    file_id = uploaded_file.file_id
    cached_path = st.session_state.get(_IMG_PATH_KEY)
    if (
        st.session_state.get(_IMG_ID_KEY) == file_id
        and cached_path
        and Path(cached_path).exists()
    ):
        return cached_path
    if cached_path:
        Path(cached_path).unlink(missing_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    st.session_state[_IMG_ID_KEY] = file_id
    st.session_state[_IMG_PATH_KEY] = tmp.name
    return tmp.name


def _validate_template(template_str: str) -> tuple[dict | None, str | None]:
    """Validate template is non-empty JSON dict. Returns (parsed, error)."""
    stripped = (template_str or "").strip()
    if not stripped:
        return None, "Template is empty."
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"
    if not isinstance(parsed, dict):
        return None, "Template must be a JSON object."
    if not parsed:
        return None, "Template must not be empty."
    return parsed, None


def _render_output_pane(
    output_placeholder: Any,
    reasoning_placeholder: Any,
    accumulated: str,
    *,
    reasoning_enabled: bool,
    is_structured: bool,
) -> None:
    """Update the reasoning + output panes from a single accumulated stream chunk."""
    think, output = split_reasoning_and_output(accumulated, reasoning_enabled)

    if reasoning_enabled:
        if think:
            reasoning_placeholder.markdown(f"```text\n{think}\n```")
        else:
            reasoning_placeholder.caption("_(no reasoning yet)_")
    else:
        reasoning_placeholder.caption("_(reasoning disabled)_")

    if not output:
        if reasoning_enabled:
            output_placeholder.caption("_(waiting for output after `</think>`)_")
        else:
            output_placeholder.caption("_(generating...)_")
        return

    if is_structured:
        answer = extract_answer_block(output)
        pretty = pretty_json_or_text(answer)
        if pretty.startswith("{") or pretty.startswith("["):
            output_placeholder.code(pretty, language="json")
        else:
            output_placeholder.markdown(pretty)
    else:
        output_placeholder.markdown(output)


_DOWNLOAD_CONFIGS = {
    "extract": ("Download JSON", "extraction.json", "application/json", True),
    "template": ("Download template", "template.json", "application/json", True),
    "markdown": ("Download Markdown", "document.md", "text/markdown", False),
}


def _render_download_button(
    placeholder: Any, accumulated: str, *, download_kind: str, reasoning: bool
) -> None:
    """Render a download button with content cleaned of reasoning trace + wrappers."""
    _, output = split_reasoning_and_output(accumulated, reasoning)
    label, file_name, mime, is_json = _DOWNLOAD_CONFIGS[download_kind]
    data = extract_answer_block(output) if is_json else output
    with placeholder.container():
        # on_click="ignore" keeps the download client-side so clicking it does
        # not rerun the _output_section fragment — a rerun would repaint the idle
        # hint over the result and drop this button (no generate button is active).
        st.download_button(
            label,
            data=data,
            file_name=file_name,
            mime=mime,
            on_click="ignore",
            icon=":material/download:",
            key=f"download_{download_kind}",
        )


def _run_mode(
    *,
    mode_label: str,
    model: Any,
    processor: Any,
    image_path: str | None,
    text: str,
    system_prompt: str | None = None,
    template: str | None,
    instructions: str | None,
    mode: str | None,
    reasoning: bool,
    temperature: float,
    max_tokens: int,
    download_kind: str,
    reasoning_placeholder: Any,
    output_placeholder: Any,
    download_placeholder: Any,
) -> None:
    """Drive a streamed generation for one mode and update the UI panes live."""
    is_structured = template is not None and mode is None
    with st.spinner(f"{mode_label}..."):
        accumulated = ""
        try:
            for chunk in stream_extract(
                model,
                processor,
                text=text,
                image_path=image_path,
                system_prompt=system_prompt,
                template=template,
                instructions=instructions,
                mode=mode,
                enable_thinking=reasoning,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                accumulated = chunk
                _render_output_pane(
                    output_placeholder,
                    reasoning_placeholder,
                    accumulated,
                    reasoning_enabled=reasoning,
                    is_structured=is_structured or mode == MODE_TEMPLATE_GENERATION,
                )
        except Exception as e:
            output_placeholder.error(f"{type(e).__name__}: {e}")
            return

    if not accumulated.strip():
        output_placeholder.warning("Empty output from model.")
        return

    _render_download_button(
        download_placeholder,
        accumulated,
        download_kind=download_kind,
        reasoning=reasoning,
    )


@st.fragment
def _output_section(model: Any, processor: Any) -> None:
    """Fragment: action buttons + the streamed reasoning/result/download panes.

    Isolated in a fragment so clicking a generate button reruns only this
    region — the input widgets in the left column keep their state and are not
    re-rendered while a generation runs. The buttons must live inside the
    fragment for that isolation to apply (a fragment only reruns on its own
    when the triggering widget is inside it). Input values are read from
    session_state, which the keyed left-column widgets populate on the full
    rerun that precedes this fragment.
    """
    with st.container(horizontal=True):
        btn_extract = st.button(
            "Extract JSON",
            type="primary",
            icon=":material/data_object:",
            width="stretch",
            key="extract_button",
        )
        btn_markdown = st.button(
            "Convert to Markdown",
            icon=":material/article:",
            width="stretch",
            key="markdown_button",
        )
        btn_template = st.button(
            "Generate template",
            icon=":material/auto_awesome:",
            width="stretch",
            key="template_button",
        )

    st.markdown("**Reasoning**")
    reasoning_placeholder = st.empty()
    st.markdown("**Result**")
    output_placeholder = st.empty()
    download_placeholder = st.empty()

    # Idle hint, shown only when no generate button fired this run: keeps the
    # Result pane from being blank on first load, without lingering over the
    # spinner during generation or repainting after an output-less rerun.
    if not (btn_extract or btn_markdown or btn_template):
        output_placeholder.caption("Choose an action above to generate output.")

    # Inputs live in the left column (outside this fragment); read their current
    # values from session_state via their widget keys.
    image_path = _save_uploaded_image(st.session_state.get("image_input"))
    text = st.session_state.get("text_input", "")
    template_str = st.session_state.get("template_input", "")
    instructions = st.session_state.get("instructions_input", "")
    temperature = st.session_state.get("temperature_slider", DEFAULT_TEMPERATURE)
    reasoning = st.session_state.get("reasoning_checkbox", False)
    max_tokens = st.session_state.get("max_tokens_slider", DEFAULT_MAX_TOKENS)

    if btn_extract:
        _, error = _validate_template(template_str)
        if error:
            output_placeholder.error(f"Template error: {error}")
        elif not image_path and not text.strip():
            output_placeholder.warning("Provide an image, text, or both.")
        else:
            _run_mode(
                mode_label="Extracting",
                model=model,
                processor=processor,
                image_path=image_path,
                text=text,
                template=template_str,
                instructions=instructions or None,
                mode=None,
                reasoning=reasoning,
                temperature=temperature,
                max_tokens=max_tokens,
                download_kind="extract",
                reasoning_placeholder=reasoning_placeholder,
                output_placeholder=output_placeholder,
                download_placeholder=download_placeholder,
            )
    elif btn_markdown:
        if not image_path:
            output_placeholder.warning(
                "Markdown conversion requires an image of the document."
            )
        else:
            _run_mode(
                mode_label="Converting to Markdown",
                model=model,
                processor=processor,
                image_path=image_path,
                text="",
                template=None,
                instructions=None,
                mode=MODE_MARKDOWN,
                reasoning=reasoning,
                temperature=temperature,
                max_tokens=max_tokens,
                download_kind="markdown",
                reasoning_placeholder=reasoning_placeholder,
                output_placeholder=output_placeholder,
                download_placeholder=download_placeholder,
            )
    elif btn_template:
        if not image_path and not text.strip():
            output_placeholder.warning(
                "Template generation needs an image or text to describe the document."
            )
        else:
            _run_mode(
                mode_label="Generating template",
                model=model,
                processor=processor,
                image_path=image_path,
                text=text,
                system_prompt=TEMPLATE_GEN_GUIDANCE,
                template=None,
                instructions=None,
                mode=MODE_TEMPLATE_GENERATION,
                reasoning=False,
                temperature=temperature,
                max_tokens=max_tokens,
                download_kind="template",
                reasoning_placeholder=reasoning_placeholder,
                output_placeholder=output_placeholder,
                download_placeholder=download_placeholder,
            )


# --- Streamlit UI ---

st.set_page_config(
    page_title="NuExtract Studio",
    page_icon=":material/document_scanner:",
    layout="wide",
)
st.title("NuExtract Studio")

col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.subheader("Input")
    uploaded_image = st.file_uploader(
        "Image",
        type=["jpg", "jpeg", "png", "webp"],
        help="JPG, PNG, or WEBP image of the document.",
        key="image_input",
    )
    if uploaded_image is not None:
        st.image(uploaded_image, width="stretch")

    # Keyed inputs feed session_state; the _output_section fragment reads their
    # values by key rather than capturing the return values here.
    st.text_area(
        "Text (optional)",
        height=100,
        placeholder="Paste document text here, or use the image above.",
        key="text_input",
    )

    st.space("medium")
    st.markdown("**Template (JSON)**")
    st.caption(
        "Describe each field with a type hint, e.g. string, number, or YYYY-MM-DD."
    )
    st.text_area(
        "Template",
        value=DEFAULT_TEMPLATE,
        height=320,
        label_visibility="collapsed",
        key="template_input",
    )

    st.text_area(
        "Instructions (optional)",
        height=80,
        placeholder="Extra guidance for the model, e.g. 'use British date format'.",
        key="instructions_input",
    )

    col_temp, col_tokens = st.columns(2)
    with col_temp:
        st.slider(
            "Temperature",
            0.0,
            1.0,
            DEFAULT_TEMPERATURE,
            0.05,
            help="0 is deterministic; raise for more varied output.",
            key="temperature_slider",
        )
    with col_tokens:
        st.slider(
            "Max tokens",
            256,
            8192,
            DEFAULT_MAX_TOKENS,
            256,
            help="Upper bound on generated tokens.",
            key="max_tokens_slider",
        )
    # Own full-width line so the "Reasoning" label never wraps (it did when
    # squeezed into a narrow middle column alongside the two sliders).
    st.checkbox(
        "Reasoning",
        value=False,
        help="Show the model's `<think>` trace in the Reasoning pane.",
        key="reasoning_checkbox",
    )

with col_right:
    st.subheader("Output")
    # The model load is the app's only slow step (~5 GB on a cold start) and it
    # sits here, below the left column, rather than above the columns: Streamlit
    # emits a UI delta per st.* call, so anything after a blocking call cannot
    # paint until it returns. Loading here lets every input widget render first,
    # so a first-time user can upload an image and edit the template while the
    # download runs. Only the output pane, which genuinely needs the model,
    # waits on it. st.spinner rather than st.skeleton because the message
    # ("~5 GB") is the part that stops the wait from reading as a hang.
    with st.spinner("Loading model (first run downloads ~5 GB)..."):
        model, processor = get_model()
    _output_section(model, processor)
