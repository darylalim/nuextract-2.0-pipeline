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
def get_model() -> tuple[Any, Any] | Exception:
    """Load the model + processor pair once per server process.

    Returns a failure rather than raising it, so the failure is cached too:
    st.cache_resource writes an entry only when the function *returns*, so a
    raising load is re-attempted on every rerun. The input widgets now render
    before this runs and stay live through a failed load, which means every
    slider drag or upload would otherwise queue another download attempt.
    """
    try:
        return load_model()
    except Exception as exc:
        return exc


# Megabytes; see the file_uploader call that passes it.
_MAX_IMAGE_UPLOAD_MB = 25

# The last completed run, replayed when a full rerun re-creates the output
# placeholders with no generate button pressed.
_LAST_RUN_KEY = "_last_run"

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
    final: bool = False,
) -> None:
    """Update the reasoning + output panes from a single accumulated stream chunk.

    `final` marks the one call made after the stream completes. Only then does
    the text satisfy extract_answer_block's whole-document contract — see the
    structured branch below.
    """
    think, output = split_reasoning_and_output(accumulated, reasoning_enabled)

    if reasoning_enabled:
        if think:
            # st.code, not a hand-built ```text fence: `think` is untrusted model
            # output, and a fence inside it would close ours and hand the rest to
            # the Markdown renderer. Markdown-mode reasoning quotes fences often.
            # Fixed height so the trace scrolls in place instead of growing the
            # page: the Result pane and its download button sit *below* this, and
            # an uncapped trace pushes them off-screen for the whole run while the
            # scroll target keeps moving. An int height is what makes st.code
            # scroll; "content" (the default) is what lets it grow.
            reasoning_placeholder.code(
                think, language=None, wrap_lines=True, height=300
            )
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

    if is_structured and not final:
        # Mid-stream: show the raw partial, and do not inspect it to decide how.
        # extract_answer_block and pretty_json_or_text have whole-document
        # contracts every partial violates — the outer object is still truncated,
        # so the longest span that *parses* is whichever nested object closed
        # first, and the pane would shrink to it and freeze until the last token.
        # Sniffing the prefix instead is no better: a preamble, an <answer>
        # wrapper in any case, or a stray <think> would each route the JSON
        # through the Markdown renderer for the whole run. Structured mode is
        # asking for JSON, so render it as JSON and let the final pass judge.
        output_placeholder.code(output, language="json", wrap_lines=True)
    elif is_structured:
        body = pretty_json_or_text(extract_answer_block(output))
        if body.startswith("{") or body.startswith("["):
            output_placeholder.code(body, language="json", wrap_lines=True)
        else:
            output_placeholder.markdown(body)
    else:
        output_placeholder.markdown(output)


_DOWNLOAD_CONFIGS = {
    "extract": ("Download JSON", "extraction.json", "application/json", True),
    "template": ("Download template", "template.json", "application/json", True),
    "markdown": ("Download Markdown", "document.md", "text/markdown", False),
}


def _download_payload(accumulated: str, *, download_kind: str, reasoning: bool) -> str:
    """The exact text a download would write: reasoning trace and wrappers gone.

    This is also the only honest test for "did the run produce anything?".
    `accumulated` still holds the trace and any `<answer>` wrapper, so it is
    non-empty for runs that produced no answer at all — see the guard in
    `_run_mode`.
    """
    _, output = split_reasoning_and_output(accumulated, reasoning)
    is_json = _DOWNLOAD_CONFIGS[download_kind][3]
    return extract_answer_block(output) if is_json else output


def _render_download_button(
    placeholder: Any, payload: str, *, download_kind: str
) -> None:
    """Render a download button for an already-cleaned `_download_payload`."""
    label, file_name, mime, _ = _DOWNLOAD_CONFIGS[download_kind]
    with placeholder.container():
        # on_click="ignore" keeps the download client-side so clicking it does
        # not rerun the _output_section fragment — a rerun would repaint the idle
        # hint over the result and drop this button (no generate button is active).
        st.download_button(
            label,
            data=payload,
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
    render_as_json = is_structured or mode == MODE_TEMPLATE_GENERATION
    with st.spinner(f"{mode_label}...", show_time=True):
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
                    is_structured=render_as_json,
                )
        except Exception as e:
            output_placeholder.error(f"{type(e).__name__}: {e}")
            return

    # Guard on the payload a download would actually write, never on `accumulated`:
    # the raw text still carries the reasoning trace and any <answer> wrapper, so
    # two runs that produced no answer at all are non-empty there — a budget spent
    # entirely inside <think> (</think> never arrives), and a closed but empty
    # <answer></answer>. Both used to slip past this guard and end with a stale
    # mid-stream caption as the terminal state, or a blanked pane, and an
    # empty-payload download button beside it.
    payload = _download_payload(
        accumulated, download_kind=download_kind, reasoning=reasoning
    )
    if not payload.strip():
        think, output = split_reasoning_and_output(accumulated, reasoning)
        if think and not output:
            # Actionable: the run did work, it just never got to an answer.
            output_placeholder.warning(
                "The model spent its whole token budget reasoning and never "
                "reached an answer. Raise Max tokens, or turn Reasoning off."
            )
        else:
            output_placeholder.warning("Empty output from model.")
        return

    # The stream is complete, so the text finally satisfies extract_answer_block's
    # whole-document contract: render once more to strip any <answer> wrapper and
    # pretty-print. Deliberately below the empty-output guard — above it, an
    # empty run would repaint "(generating...)" over its own warning. Skipped for
    # markdown mode, where this render is byte-identical to the last streaming one
    # and would re-send a whole document — the largest one the app produces.
    if render_as_json:
        _render_output_pane(
            output_placeholder,
            reasoning_placeholder,
            accumulated,
            reasoning_enabled=reasoning,
            is_structured=render_as_json,
            final=True,
        )

    # Remember the finished run so a *full* rerun can repaint it. Any left-column
    # widget triggers one — those widgets sit outside this fragment — which
    # re-creates all three placeholders empty with none of the generate buttons
    # pressed, and would otherwise discard a result that cost a whole local
    # generation. Stored below the empty-payload guard, so a run that never
    # produced an answer can never come back as a replayable "result", and the
    # payload is carried rather than re-derived: deriving it twice is how the
    # guard and the button drift apart.
    st.session_state[_LAST_RUN_KEY] = {
        "accumulated": accumulated,
        "payload": payload,
        "download_kind": download_kind,
        "reasoning": reasoning,
        "render_as_json": render_as_json,
    }

    _render_download_button(download_placeholder, payload, download_kind=download_kind)


@st.fragment
def _output_section() -> None:
    """Fragment: action buttons + the streamed reasoning/result/download panes.

    Isolated in a fragment so clicking a generate button reruns only this
    region — the input widgets in the left column keep their state and are not
    re-rendered while a generation runs. The buttons must live inside the
    fragment for that isolation to apply (a fragment only reruns on its own
    when the triggering widget is inside it). Input values are read from
    session_state, which the keyed left-column widgets populate on the full
    rerun that precedes this fragment.

    Loads the model itself rather than taking it as an argument, so the load
    can sit below this section's own chrome instead of above it — see the
    comment on the load below.
    """
    with st.container(horizontal=True):
        btn_extract = st.button(
            "Extract JSON",
            help="Needs a valid JSON template, plus an image or text.",
            type="primary",
            icon=":material/data_object:",
            width="stretch",
            key="extract_button",
        )
        btn_markdown = st.button(
            "Convert to Markdown",
            help="Needs an image of the document.",
            icon=":material/article:",
            width="stretch",
            key="markdown_button",
        )
        btn_template = st.button(
            "Generate template",
            help="Needs an image or text to describe the document.",
            icon=":material/auto_awesome:",
            width="stretch",
            key="template_button",
        )

    st.markdown("**Reasoning**")
    reasoning_placeholder = st.empty()
    st.markdown("**Result**")
    output_placeholder = st.empty()
    download_placeholder = st.empty()

    # The load runs here, after the buttons and both pane headers have claimed
    # their positions, so the whole page is painted before it blocks: Streamlit
    # emits a UI delta per st.* call, so only what follows this line waits on
    # it. The wait shows inside the Result slot, where the output will land.
    with output_placeholder.container():
        with st.spinner("Loading model (first run downloads ~5 GB)...", show_time=True):
            loaded = get_model()
    if isinstance(loaded, Exception):
        with output_placeholder.container():
            st.error(f"Model failed to load — {type(loaded).__name__}")
            # Message through st.code, not st.error: st.error renders its body as
            # GitHub-flavored Markdown with soft breaks disabled, so the
            # multi-line messages this path actually produces (an offline hub
            # download, an unpatched processor_config.json) collapse onto one
            # line. st.code also keeps untrusted text out of the Markdown
            # renderer, the same reason the reasoning trace uses it.
            st.code(str(loaded) or repr(loaded), language=None, wrap_lines=True)
            # get_model returns the exception instead of raising it, so Streamlit
            # never reports it and nothing logs it — this expander is the only
            # place the frame that names the cause survives. Collapsed by
            # default: the headline is enough unless you are debugging.
            with st.expander("Traceback", icon=":material/bug_report:"):
                st.exception(loaded)
            # Cached failure (see get_model): retrying is an explicit click,
            # not something every widget interaction re-triggers.
            if st.button(
                "Retry model load",
                icon=":material/refresh:",
                key="retry_model_load_button",
            ):
                get_model.clear()
                st.rerun()
        return
    model, processor = loaded

    # Idle hint, shown only when no generate button fired this run: keeps the
    # Result pane from being blank on first load, without lingering over the
    # spinner during generation or repainting after an output-less rerun.
    if not (btn_extract or btn_markdown or btn_template):
        last_run = st.session_state.get(_LAST_RUN_KEY)
        if last_run is None:
            # Both panes get a hint, not just Result: reasoning_placeholder is
            # only ever written from _render_output_pane, so before the first run
            # the bold "Reasoning" header renders over an unwritten st.empty() of
            # zero height — the same void the mid-run "(reasoning disabled)"
            # caption prevents.
            reasoning_placeholder.caption("_(no run yet)_")
            output_placeholder.caption("Choose an action above to generate output.")
        else:
            # Replay the last completed run rather than painting the idle hint
            # over it. `final=True` reproduces the terminal state exactly: this
            # text is a whole document, so it satisfies extract_answer_block's
            # contract — unlike a mid-stream chunk. Markdown mode gets the final
            # pass here too; _run_mode skips it only because there the render is
            # byte-identical to its last streaming one, which is not the case for
            # a placeholder that has just been re-created empty.
            _render_output_pane(
                output_placeholder,
                reasoning_placeholder,
                last_run["accumulated"],
                reasoning_enabled=last_run["reasoning"],
                is_structured=last_run["render_as_json"],
                final=True,
            )
            _render_download_button(
                download_placeholder,
                last_run["payload"],
                download_kind=last_run["download_kind"],
            )

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
        # One document page, in megabytes. Without this the widget inherits
        # server.maxUploadSize (200 MB), and _save_uploaded_image reads whatever
        # arrives whole into memory via getvalue() before the image processor
        # decodes it to raw pixels — on the same unified memory already holding a
        # ~5 GB model. Generous for a 300dpi page scan, cheap insurance otherwise.
        max_upload_size=_MAX_IMAGE_UPLOAD_MB,
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
        # Collapsed, but still the widget's accessible name — so it has to match
        # the "Template (JSON)" heading a sighted user reads above it.
        "Template (JSON)",
        value=DEFAULT_TEMPLATE,
        height=320,
        label_visibility="collapsed",
        key="template_input",
    )

    st.text_area(
        # 98 is the floor Streamlit enforces for a visible label, not a chosen
        # size: the 80 that used to sit here was silently clamped up to it, so the
        # number read as intent while doing nothing. Stating the floor keeps the
        # rendering identical and makes the constraint visible. Dropping the
        # parameter instead would take the default — three lines, i.e. *taller*
        # than the Text box above — inverting the intent this field was written with.
        "Instructions (optional)",
        height=98,
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
        help=(
            "Show the model's `<think>` trace in the Reasoning pane. Ignored by "
            "**Generate template** — the model's template only permits reasoning "
            "for extraction and Markdown."
        ),
        key="reasoning_checkbox",
    )

with col_right:
    st.subheader("Output")
    # No model load here: _output_section loads it itself, below its own buttons
    # and pane headers, so nothing on the page waits on the ~5 GB download
    # except the Result slot the output lands in.
    _output_section()
