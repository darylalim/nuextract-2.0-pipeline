"""AppTest end-to-end tests for streamlit_app.py.

Covers script-body wiring (button validation, default state, streaming flow)
that the helper-function tests in test_streamlit_app.py don't exercise. Mocks
nuextract.load_model and the underlying snapshot/mlx loaders so no 5 GB model
download happens.

File-upload paths use the at_with_image fixture, which drives a fake image
into st.file_uploader via AppTest's native set_value API (added in Streamlit
1.56). AppTest does expose at.download_button, but its DownloadButton carries
only label/help/value, so file_name/mime/payload assertions live in
test_streamlit_app.py.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


def _stub_model_loading(monkeypatch, *, on_load=None):
    """Stub every path that could reach a real 5 GB download.

    Patches load_model + snapshot_download + mlx_vlm_load as belt-and-suspenders
    insurance: even if a code path bypasses load_model, the lower-level functions
    are no-op'd so no real download can happen. Shared by the `at` fixture and
    the ordering guard so the set of stubbed boundaries is defined once — a
    second copy would drift, and a missed boundary fails as a 4.8 GB download
    rather than an obvious error. `on_load` fires when load_model is called, for
    tests that need to observe the load itself.
    """

    def fake_load_model(*_, **__):
        if on_load is not None:
            on_load()
        return MagicMock(), MagicMock()

    def fake_model_pair(*_, **__):
        return MagicMock(), MagicMock()

    monkeypatch.setattr("nuextract.load_model", fake_load_model)
    monkeypatch.setattr("nuextract.snapshot_download", lambda *_, **__: "/fake/dir")
    monkeypatch.setattr("nuextract.mlx_vlm_load", fake_model_pair)


@pytest.fixture
def cold_model_cache():
    """Clear the process-global st.cache_resource entry around a test.

    get_model is cached for the life of the process, so a test that needs to
    observe the load itself must clear first — otherwise an earlier test's entry
    means the stub never runs and the assertion silently observes nothing.
    Clearing on the way out too keeps the mutation from leaking: without it the
    suite's cache would hold a pair built by this test's stub after monkeypatch
    has already torn that stub down.
    """
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.fixture
def at(monkeypatch):
    """Build a fresh AppTest with model loading stubbed, already run once."""
    _stub_model_loading(monkeypatch)
    instance = AppTest.from_file(APP_PATH)
    instance.run()
    return instance


@pytest.fixture
def at_with_image(at, monkeypatch):
    """AppTest with a fake image driven into st.file_uploader.

    Streamlit 1.56+ lets AppTest drive st.file_uploader natively, so we register
    a fake image via file_uploader(...).set_value(...) — exercising the real
    widget instead of patching it out. The script's natural code path then calls
    _save_uploaded_image(...), which writes a real temp file and produces a valid
    image_path. st.image is no-op'd because the fake bytes aren't a decodable
    image. Cleans up the resulting temp file on teardown.
    """
    monkeypatch.setattr("streamlit.image", lambda *_, **__: None)
    at.file_uploader(key="image_input").set_value(
        ("test.png", b"\x89PNG\r\n\x1a\n", "image/png")
    )
    at.run()
    yield at
    # Lazy import is safe here: AppTest already loaded streamlit_app, so this
    # is a sys.modules lookup, not a re-execution of the script body.
    from streamlit_app import _IMG_PATH_KEY

    if _IMG_PATH_KEY in at.session_state:
        Path(at.session_state[_IMG_PATH_KEY]).unlink(missing_ok=True)


@pytest.fixture
def stream_captor(monkeypatch):
    """Patch nuextract.stream_extract with a recording fake.

    Returns `(captured, set_chunks)` where `captured` is a dict that fills with
    the kwargs stream_extract was called with, and `set_chunks(*chunks)` queues
    output to yield. Tests that want an empty stream just don't call set_chunks.
    Tests that need the stream to raise should use an inline fake_stream instead.
    """
    captured: dict = {}
    chunks: list[str] = []

    def fake_stream(*_, **kwargs):
        captured.update(kwargs)
        yield from chunks

    def set_chunks(*new_chunks: str) -> None:
        chunks.extend(new_chunks)

    monkeypatch.setattr("nuextract.stream_extract", fake_stream)
    return captured, set_chunks


# --- Initial render ---


def test_no_exception_on_initial_load(at):
    assert not at.exception


def test_title_renders(at):
    assert at.title[0].value == "NuExtract Studio"


def test_default_template_loads_in_editor(at):
    template = at.text_area(key="template_input").value
    parsed = json.loads(template)
    assert isinstance(parsed, dict)
    assert len(parsed) > 0


def test_three_buttons_present(at):
    labels = [b.label for b in at.button]
    assert "Extract JSON" in labels
    assert "Convert to Markdown" in labels
    assert "Generate template" in labels


def test_no_warnings_or_errors_on_initial_load(at):
    assert len(at.error) == 0
    assert len(at.warning) == 0


def test_model_loads_after_the_page_chrome_renders(monkeypatch, cold_model_cache):
    """The ~5 GB load must run below *everything* that doesn't depend on it.

    Streamlit emits a UI delta per st.* call, so a blocking load stops every
    element after it from painting until it returns. Nothing but an actual
    generation needs the model, so both the left column's inputs and the right
    column's own chrome (action buttons, pane headers) must render first.

    Asserted structurally rather than by timing, since render order is not
    observable from AppTest: keyed widgets register themselves in session_state
    as they render, so an anchor key is present when load_model is called if
    and only if that widget already ran.

    Both anchors are the *last* keyed widget of their group, and that is the
    whole point — an anchor further up still passes with the load sitting in
    the middle of the group it is supposed to be guarding. reasoning_checkbox
    is the last of col_left's 7 inputs (template_input, the 3rd, would let the
    load sit mid-column); template_button is the last of the three action
    buttons.
    """
    seen: dict = {}

    def record() -> None:
        seen["inputs"] = "reasoning_checkbox" in st.session_state
        seen["chrome"] = "template_button" in st.session_state

    _stub_model_loading(monkeypatch, on_load=record)

    at = AppTest.from_file(APP_PATH)
    at.run()

    # Without this, a crash before either anchor leaves `seen` empty and the
    # assertions below fail with a message blaming the wrong thing.
    assert not at.exception
    assert seen.get("inputs") is True, (
        "get_model() ran before the left column finished — move it below col_left"
    )
    assert seen.get("chrome") is True, (
        "get_model() ran before the action buttons rendered — move it below them "
        "inside _output_section"
    )


def test_result_pane_shows_idle_hint_on_initial_load(at):
    """Before any run, the Result pane shows an empty-state hint (rendered into
    its placeholder by the _output_section fragment) instead of a blank pane."""
    assert any("Choose an action" in c.value for c in at.caption)


def test_idle_hint_cleared_after_run(at, stream_captor):
    """Once a generation runs, the idle hint is gone — the output replaces it in
    the same placeholder rather than lingering beside the result."""
    _, set_chunks = stream_captor
    set_chunks('{"k": 1}')

    at.text_area(key="text_input").set_value("doc text")
    at.button(key="extract_button").click()
    at.run()

    assert not any("Choose an action" in c.value for c in at.caption)
    assert any('"k": 1' in c.value for c in at.code)


# --- Extract button validation ---


def test_extract_with_no_input_shows_warning(at):
    at.button(key="extract_button").click().run()
    assert any("Provide an image, text, or both" in w.value for w in at.warning)


def test_extract_with_invalid_template_shows_error(at):
    at.text_area(key="template_input").set_value("not json {{{").run()
    at.button(key="extract_button").click().run()
    assert any("Template error" in e.value for e in at.error)


def test_extract_with_empty_template_shows_error(at):
    at.text_area(key="template_input").set_value("").run()
    at.button(key="extract_button").click().run()
    assert any("Template error" in e.value for e in at.error)


def test_extract_with_non_dict_template_shows_error(at):
    at.text_area(key="template_input").set_value("[1, 2, 3]").run()
    at.button(key="extract_button").click().run()
    assert any("Template error" in e.value for e in at.error)


# --- Markdown button validation ---


def test_markdown_without_image_shows_warning(at):
    at.button(key="markdown_button").click().run()
    assert any("requires an image" in w.value.lower() for w in at.warning)


# --- Template-gen button validation ---


def test_template_gen_with_no_input_shows_warning(at):
    at.button(key="template_button").click().run()
    assert any("needs an image or text" in w.value.lower() for w in at.warning)


# --- Streaming flow (mocked stream_extract) ---


def test_extract_with_text_streams_json_output(at, stream_captor):
    """Happy path: text input + default template → stream_extract called →
    JSON renders as code block → download button appears."""
    captured, set_chunks = stream_captor
    set_chunks('{"name": "Alice"}')

    at.text_area(key="text_input").set_value("doc text")
    at.button(key="extract_button").click()
    at.run()

    assert len(at.error) == 0
    assert len(at.warning) == 0
    # JSON appearing as a code block proves the streaming flow completed and
    # _render_output_pane ran in structured mode. The download button's
    # file_name/mime/payload are asserted in test_streamlit_app.py, since
    # AppTest's DownloadButton exposes only label/help/value.
    assert any('"name": "Alice"' in c.value for c in at.code)
    assert captured["text"] == "doc text"
    assert captured["mode"] is None
    assert captured.get("system_prompt") is None  # Extract sends no system prompt


def test_template_gen_with_text_streams_json_output(at, stream_captor):
    """Template-gen mode passes mode='template-generation' and renders the
    generated template as JSON."""
    captured, set_chunks = stream_captor
    set_chunks('{"field_a": "string", "field_b": "number"}')

    at.text_area(key="text_input").set_value("describe a document")
    at.button(key="template_button").click()
    at.run()

    assert len(at.error) == 0
    assert len(at.warning) == 0
    assert any('"field_a"' in c.value for c in at.code)
    assert captured["mode"] == "template-generation"


def test_extract_empty_stream_shows_warning(at, stream_captor):
    """When stream_extract yields nothing, the user gets an explicit warning
    rather than a silent empty pane."""
    # Don't call set_chunks — fixture's default is to yield nothing.
    at.text_area(key="text_input").set_value("doc text")
    at.button(key="extract_button").click()
    at.run()

    assert any("Empty output" in w.value for w in at.warning)


def test_extract_exception_during_stream_shows_error(at, monkeypatch):
    """Model errors mid-stream surface in the output pane, not as a crash.

    Uses an inline fake_stream because stream_captor doesn't support raising
    mid-stream — the special case is rare enough not to complicate the fixture.
    """

    def fake_stream(*_, **__):
        yield "partial"
        raise RuntimeError("model crashed mid-stream")

    monkeypatch.setattr("nuextract.stream_extract", fake_stream)

    at.text_area(key="text_input").set_value("doc text")
    at.button(key="extract_button").click()
    at.run()

    assert not at.exception
    assert any(
        "RuntimeError" in e.value and "model crashed" in e.value for e in at.error
    )


def test_extract_passes_slider_values_to_stream_extract(at, stream_captor):
    """Temperature and max_tokens sliders flow through to the streaming call."""
    captured, set_chunks = stream_captor
    set_chunks('{"k": 1}')

    at.text_area(key="text_input").set_value("doc text")
    # Values must align to each slider's step (temp step=0.05, max_tokens step=256)
    at.slider(key="temperature_slider").set_value(0.7)
    at.slider(key="max_tokens_slider").set_value(512)
    at.button(key="extract_button").click()
    at.run()

    assert captured["temperature"] == 0.7
    assert captured["max_tokens"] == 512


def test_reasoning_enabled_splits_reasoning_and_output_panes(at, stream_captor):
    """With reasoning on, <think>...</think> goes to the reasoning pane and
    the JSON answer goes to the result pane."""
    _, set_chunks = stream_captor
    set_chunks('thinking step by step</think>{"k": 1}')

    at.text_area(key="text_input").set_value("doc text")
    at.checkbox(key="reasoning_checkbox").check()
    at.button(key="extract_button").click()
    at.run()

    # Both panes render as code elements now, so assert by *position*, not by
    # content: the fragment creates the reasoning placeholder before the output
    # one, so at.code[0] is the Reasoning pane and at.code[1] the Result pane.
    # Matching on content alone would still pass with the two panes swapped.
    assert len(at.code) == 2
    assert "thinking step by step" in at.code[0].value
    assert '"k": 1' not in at.code[0].value
    assert '"k": 1' in at.code[1].value
    assert "thinking step by step" not in at.code[1].value


def test_completed_run_strips_answer_wrapper_and_pretty_prints(at, stream_captor):
    """End-to-end cover for the post-stream `final=True` render in _run_mode.
    Asserting the *exact* indented body is what makes this fail if that call is
    ever dropped: the streaming render shows the raw wrapped text, so a test
    that only checked for a substring would pass either way."""
    _, set_chunks = stream_captor
    set_chunks('<answer>{"k": 1}</answer>')

    at.text_area(key="text_input").set_value("doc text")
    at.button(key="extract_button").click()
    at.run()

    assert [c.value for c in at.code] == ['{\n  "k": 1\n}']


def test_reasoning_trace_containing_a_code_fence_stays_in_one_element(
    at, stream_captor
):
    """A fence inside the trace used to close the hand-built ```text wrapper and
    hand everything after it to the Markdown renderer. st.code takes the body as
    a value, so the whole trace — inner fence and all — stays in one element."""
    _, set_chunks = stream_captor
    trace = 'converting to markdown:\n```json\n{"a": 1}\n```\ndone'
    set_chunks(f'{trace}</think>{{"k": 2}}')

    at.text_area(key="text_input").set_value("doc text")
    at.checkbox(key="reasoning_checkbox").check()
    at.button(key="extract_button").click()
    at.run()

    holding = [c.value for c in at.code if "converting to markdown" in c.value]
    assert len(holding) == 1
    # Nothing after the inner fence escaped into a separate rendered element.
    assert "```json" in holding[0]
    assert holding[0].endswith("done")


def test_extract_passes_instructions_to_stream_extract(at, stream_captor):
    """The optional instructions field flows through to the streaming call."""
    captured, set_chunks = stream_captor
    set_chunks('{"k": 1}')

    at.text_area(key="text_input").set_value("doc text")
    at.text_area(key="instructions_input").set_value("use British date format")
    at.button(key="extract_button").click()
    at.run()

    assert captured["instructions"] == "use British date format"


def test_template_gen_passes_system_prompt(at, stream_captor):
    """Template-gen sends the TEMPLATE_GEN_GUIDANCE system prompt (the Extract
    and Markdown paths pass no system prompt)."""
    # sys.modules lookup — AppTest already loaded streamlit_app, so this does
    # not re-execute the script body without mocks.
    from streamlit_app import TEMPLATE_GEN_GUIDANCE

    captured, set_chunks = stream_captor
    set_chunks('{"field_a": "string"}')

    at.text_area(key="text_input").set_value("describe a document")
    at.button(key="template_button").click()
    at.run()

    assert captured["system_prompt"] == TEMPLATE_GEN_GUIDANCE


def test_template_gen_forces_reasoning_off(at, stream_captor):
    """Template-gen overrides the reasoning checkbox: enable_thinking is always
    False even when the user has reasoning on (the Jinja only allows thinking
    for structured/content modes)."""
    captured, set_chunks = stream_captor
    set_chunks('{"field_a": "string"}')

    at.text_area(key="text_input").set_value("describe a document")
    at.checkbox(key="reasoning_checkbox").check()
    at.button(key="template_button").click()
    at.run()

    assert captured["enable_thinking"] is False


# --- Streaming flow with image (file_uploader patched via at_with_image) ---


def test_markdown_happy_path_renders_markdown(at_with_image, stream_captor):
    """Image + Markdown button → stream yields markdown → result renders as
    markdown (not JSON code block); mode='markdown' and image_path flow
    through to stream_extract."""
    captured, set_chunks = stream_captor
    set_chunks("# Document Title\n\nThis is the body.")

    at_with_image.button(key="markdown_button").click()
    at_with_image.run()

    assert len(at_with_image.error) == 0
    assert len(at_with_image.warning) == 0
    # Markdown mode renders via st.markdown, not st.code
    assert any("Document Title" in m.value for m in at_with_image.markdown)
    assert captured["mode"] == "markdown"
    assert captured["image_path"]  # real temp file path written by _save_uploaded_image
    assert captured.get("system_prompt") is None  # Markdown sends no system prompt


def test_extract_with_image_only_streams_json(at_with_image, stream_captor):
    """Image-only (no text) + Extract → stream_extract receives image_path
    and empty text; output renders as JSON code block."""
    captured, set_chunks = stream_captor
    set_chunks('{"extracted": "from image"}')

    at_with_image.button(key="extract_button").click()
    at_with_image.run()

    assert len(at_with_image.error) == 0
    assert len(at_with_image.warning) == 0
    assert any('"extracted"' in c.value for c in at_with_image.code)
    assert captured["image_path"]
    assert captured["text"] == ""
    assert captured["mode"] is None


def test_template_gen_with_image_only_streams_json(at_with_image, stream_captor):
    """Image-only (no text) + Template-gen → stream_extract receives image_path
    and mode='template-generation'."""
    captured, set_chunks = stream_captor
    set_chunks('{"title": "string", "date": "YYYY-MM-DD"}')

    at_with_image.button(key="template_button").click()
    at_with_image.run()

    assert len(at_with_image.error) == 0
    assert len(at_with_image.warning) == 0
    assert any('"title"' in c.value for c in at_with_image.code)
    assert captured["image_path"]
    assert captured["mode"] == "template-generation"
