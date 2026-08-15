import json
from unittest.mock import MagicMock, patch

import pytest

from nuextract import (
    MODE_CONTENT,
    MODE_MARKDOWN,
    MODE_STRUCTURED,
    MODE_TEMPLATE_GENERATION,
    build_messages,
    extract_answer_block,
    load_model,
    patch_processor_config,
    pretty_json_or_text,
    render_prompt,
    split_reasoning_and_output,
    stream_extract,
)

# --- patch_processor_config ---


def test_patch_processor_config_replaces_string(tmp_path):
    cfg = tmp_path / "processor_config.json"
    cfg.write_text(json.dumps({"image_processor_type": "Qwen3VLImageProcessor"}))
    changed = patch_processor_config(tmp_path)
    assert changed is True
    new_content = cfg.read_text()
    assert "Qwen3VLImageProcessor" not in new_content
    assert "Qwen2VLImageProcessor" in new_content


def test_patch_processor_config_idempotent(tmp_path):
    cfg = tmp_path / "processor_config.json"
    cfg.write_text(json.dumps({"image_processor_type": "Qwen2VLImageProcessor"}))
    changed = patch_processor_config(tmp_path)
    assert changed is False
    assert "Qwen2VLImageProcessor" in cfg.read_text()


def test_patch_processor_config_missing_file(tmp_path):
    assert patch_processor_config(tmp_path) is False


# --- transformers compatibility invariants ---
#
# The transformers pin is load-bearing and these two facts are what make the
# stack work. Both are model-free one-liners, so CI gates them instead of a
# checklist item in CLAUDE.md that a future bump can quietly skip.


def test_qwen3_5_is_in_the_auto_config_resolver():
    """The model's architecture must be resolvable by AutoConfig.

    NuExtract3 is Qwen3.5-family; if a transformers bump drops 'qwen3_5' from
    the mapping, load_model() fails with an unrecognized-model-type error.
    """
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    assert "qwen3_5" in CONFIG_MAPPING_NAMES


def test_qwen2_vl_image_processor_is_still_exported():
    """patch_processor_config rewrites the config to name this class.

    If a transformers bump removes or renames it, the shim would point the
    processor at a class that cannot be resolved.
    """
    import transformers

    assert hasattr(transformers, "Qwen2VLImageProcessor")


# --- build_messages ---


def test_build_messages_text_only():
    msgs = build_messages(text="Hello world")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == [{"type": "text", "text": "Hello world"}]


def test_build_messages_image_only():
    msgs = build_messages(image_path="/tmp/x.png")
    assert msgs[0]["content"] == [{"type": "image", "image": "/tmp/x.png"}]


def test_build_messages_image_and_text():
    msgs = build_messages(text="describe", image_path="/tmp/x.png")
    content = msgs[0]["content"]
    assert content[0] == {"type": "image", "image": "/tmp/x.png"}
    assert content[1] == {"type": "text", "text": "describe"}


def test_build_messages_empty_yields_empty_text_part():
    msgs = build_messages()
    assert msgs[0]["content"] == [{"type": "text", "text": ""}]


def test_build_messages_strips_whitespace_only_text():
    msgs = build_messages(text="   \n  ", image_path="/tmp/x.png")
    # Whitespace-only text is dropped, only image remains
    assert msgs[0]["content"] == [{"type": "image", "image": "/tmp/x.png"}]


def test_build_messages_with_system_prompt():
    """System prompt prepends a system message; user message stays second."""
    msgs = build_messages(text="user text", system_prompt="be a JSON expert")
    assert len(msgs) == 2
    assert msgs[0] == {"role": "system", "content": "be a JSON expert"}
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == [{"type": "text", "text": "user text"}]


def test_build_messages_whitespace_system_prompt_dropped():
    """Empty/whitespace system_prompt is ignored — no system message added."""
    msgs = build_messages(text="hi", system_prompt="   \n  ")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"


# --- render_prompt ---


def test_render_prompt_passes_template_kwarg_inline():
    processor = MagicMock()
    processor.apply_chat_template.return_value = "rendered"
    messages = [{"role": "user", "content": []}]

    out = render_prompt(
        processor, messages, template='{"name":"string"}', enable_thinking=False
    )
    assert out == "rendered"
    call = processor.apply_chat_template.call_args
    assert call.args[0] == messages
    assert call.kwargs["template"] == '{"name":"string"}'
    assert call.kwargs["enable_thinking"] is False
    assert call.kwargs["tokenize"] is False
    assert call.kwargs["add_generation_prompt"] is True
    # Critical: should NOT nest under chat_template_kwargs (vLLM convention)
    assert "chat_template_kwargs" not in call.kwargs


def test_render_prompt_omits_none_kwargs():
    processor = MagicMock()
    processor.apply_chat_template.return_value = "rendered"

    render_prompt(processor, [], template=None, mode=None, instructions=None)
    call = processor.apply_chat_template.call_args
    assert "template" not in call.kwargs
    assert "mode" not in call.kwargs
    assert "instructions" not in call.kwargs
    assert call.kwargs["enable_thinking"] is False


def test_render_prompt_includes_mode():
    processor = MagicMock()
    processor.apply_chat_template.return_value = ""

    render_prompt(processor, [], mode="markdown")
    assert processor.apply_chat_template.call_args.kwargs["mode"] == "markdown"


def test_render_prompt_includes_instructions_when_set():
    processor = MagicMock()
    processor.apply_chat_template.return_value = ""

    render_prompt(processor, [], instructions="use ISO dates")
    assert (
        processor.apply_chat_template.call_args.kwargs["instructions"]
        == "use ISO dates"
    )


def test_render_prompt_drops_empty_instructions():
    processor = MagicMock()
    processor.apply_chat_template.return_value = ""

    render_prompt(processor, [], instructions="")
    # Empty string is falsy → dropped, model sees no instructions
    assert "instructions" not in processor.apply_chat_template.call_args.kwargs


def test_render_prompt_passes_image_message_through():
    """Image-bearing messages reach apply_chat_template unchanged so the Jinja
    template can insert the vision placeholder."""
    processor = MagicMock()
    processor.apply_chat_template.return_value = ""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/tmp/x.png"},
                {"type": "text", "text": "describe"},
            ],
        }
    ]
    render_prompt(processor, messages, mode="markdown")
    assert processor.apply_chat_template.call_args.args[0] == messages


# --- mode constants ---


def test_mode_constants_have_expected_values():
    """Sanity check the mode strings the Jinja template branches on."""
    assert MODE_STRUCTURED == "structured"
    assert MODE_CONTENT == "content"
    assert MODE_MARKDOWN == "markdown"
    assert MODE_TEMPLATE_GENERATION == "template-generation"


# --- split_reasoning_and_output ---


def test_split_no_reasoning_returns_full_as_output():
    think, output = split_reasoning_and_output("hello", reasoning_enabled=False)
    assert think == ""
    assert output == "hello"


def test_split_with_complete_think_block():
    text = 'thinking about it</think>{"name":"x"}'
    think, output = split_reasoning_and_output(text, reasoning_enabled=True)
    assert think == "thinking about it"
    assert output == '{"name":"x"}'


def test_split_incomplete_think_returns_only_reasoning():
    think, output = split_reasoning_and_output(
        "still thinking...", reasoning_enabled=True
    )
    assert think == "still thinking..."
    assert output == ""


def test_split_case_insensitive_end_tag():
    think, output = split_reasoning_and_output("R</THINK>O", reasoning_enabled=True)
    assert think == "R"
    assert output == "O"


def test_split_empty_text():
    assert split_reasoning_and_output("", reasoning_enabled=True) == ("", "")
    assert split_reasoning_and_output("", reasoning_enabled=False) == ("", "")


# --- extract_answer_block ---


@pytest.mark.parametrize(
    "text,expected",
    [
        pytest.param('<answer>{"k":1}</answer>', '{"k":1}', id="answer_wrapped"),
        pytest.param("<ANSWER>x</ANSWER>", "x", id="answer_case_insensitive"),
        pytest.param(
            'prefix {"a":1} middle {"b":1,"c":2} suffix',
            '{"b":1,"c":2}',
            id="picks_longest_valid_json",
        ),
        pytest.param(
            'Looking at the doc... {"name": "John", "age": 30}',
            '{"name": "John", "age": 30}',
            id="reasoning_prefix_plus_json",
        ),
        pytest.param(
            '{"outer": 2, "inner": {"k": 1}}',
            '{"outer": 2, "inner": {"k": 1}}',
            id="nested_json_as_single_span",
        ),
        pytest.param(
            "{not json {still bad",
            "{not json {still bad",
            id="unparseable_brace_runs",
        ),
        pytest.param("  just text  ", "just text", id="no_match_returns_stripped"),
        pytest.param("", "", id="empty"),
    ],
)
def test_extract_answer_block(text, expected):
    assert extract_answer_block(text) == expected


# --- pretty_json_or_text ---


@pytest.mark.parametrize(
    "text,expected",
    [
        pytest.param(
            '{"a":1,"b":2}', '{\n  "a": 1,\n  "b": 2\n}', id="valid_json_indented"
        ),
        pytest.param(
            '{"name":"élise"}', '{\n  "name": "élise"\n}', id="preserves_unicode"
        ),
        pytest.param("not json {{{", "not json {{{", id="invalid_returns_original"),
        pytest.param("", "", id="empty"),
        pytest.param("   ", "", id="whitespace_only"),
    ],
)
def test_pretty_json_or_text(text, expected):
    assert pretty_json_or_text(text) == expected


# --- load_model integration boundary ---


def test_load_model_invokes_snapshot_and_patch_and_load():
    """load_model orchestrates: snapshot_download → patch → mlx_vlm.load."""
    with (
        patch("nuextract.patch_processor_config") as mock_patch,
        patch("nuextract.snapshot_download", return_value="/fake/dir") as mock_dl,
        patch("nuextract.mlx_vlm_load", return_value=("M", "P")) as mock_load,
    ):
        model, processor = load_model("test/repo")

    mock_dl.assert_called_once_with(repo_id="test/repo")
    mock_patch.assert_called_once_with("/fake/dir")
    mock_load.assert_called_once_with("/fake/dir")
    assert (model, processor) == ("M", "P")


# --- stream_extract integration boundary ---


def test_stream_extract_yields_accumulated_text():
    """stream_extract should accumulate per-chunk .text deltas."""
    chunks = [MagicMock(text="Hel"), MagicMock(text="lo"), MagicMock(text=" world")]
    processor = MagicMock()
    processor.apply_chat_template.return_value = "PROMPT"

    with patch(
        "nuextract.mlx_vlm_stream_generate", return_value=iter(chunks)
    ) as mock_sg:
        outputs = list(stream_extract(MagicMock(), processor, text="hi"))

    assert outputs == ["Hel", "Hello", "Hello world"]
    # image kwarg is omitted entirely when no image_path is given
    assert "image" not in mock_sg.call_args.kwargs


def test_stream_extract_passes_image_path_as_list():
    chunks = [MagicMock(text="ok")]
    processor = MagicMock()
    processor.apply_chat_template.return_value = "PROMPT"

    with patch(
        "nuextract.mlx_vlm_stream_generate", return_value=iter(chunks)
    ) as mock_sg:
        list(stream_extract(MagicMock(), processor, image_path="/tmp/x.png"))

    assert mock_sg.call_args.kwargs["image"] == ["/tmp/x.png"]


def test_stream_extract_forwards_template_and_generation_kwargs():
    """The UI sliders (temperature, max_tokens) and template editor must reach
    the underlying calls — otherwise the controls become inert."""
    chunks = [MagicMock(text="ok")]
    processor = MagicMock()
    processor.apply_chat_template.return_value = "PROMPT"

    with patch(
        "nuextract.mlx_vlm_stream_generate", return_value=iter(chunks)
    ) as mock_sg:
        list(
            stream_extract(
                MagicMock(),
                processor,
                text="hi",
                template='{"x": "string"}',
                instructions="be brief",
                mode=None,
                enable_thinking=True,
                temperature=0.7,
                max_tokens=512,
            )
        )

    # Template + instructions + thinking flag reach apply_chat_template inline
    tpl_kwargs = processor.apply_chat_template.call_args.kwargs
    assert tpl_kwargs["template"] == '{"x": "string"}'
    assert tpl_kwargs["instructions"] == "be brief"
    assert tpl_kwargs["enable_thinking"] is True

    # Generation kwargs reach mlx_vlm_stream_generate
    gen_kwargs = mock_sg.call_args.kwargs
    assert gen_kwargs["temperature"] == 0.7
    assert gen_kwargs["max_tokens"] == 512


def test_stream_extract_skips_special_tokens():
    """mlx-vlm defaults skip_special_tokens=False, which leaves <|im_end|> in
    the decoded text — visible in markdown output and the downloaded .md."""
    chunks = [MagicMock(text="ok")]
    processor = MagicMock()
    processor.apply_chat_template.return_value = "PROMPT"

    with patch(
        "nuextract.mlx_vlm_stream_generate", return_value=iter(chunks)
    ) as mock_sg:
        list(stream_extract(MagicMock(), processor, text="hi"))

    assert mock_sg.call_args.kwargs["skip_special_tokens"] is True


def test_stream_extract_forwards_system_prompt():
    """system_prompt is forwarded to build_messages and becomes a system message."""
    chunks = [MagicMock(text="ok")]
    processor = MagicMock()
    processor.apply_chat_template.return_value = "PROMPT"

    with patch("nuextract.mlx_vlm_stream_generate", return_value=iter(chunks)):
        list(
            stream_extract(
                MagicMock(),
                processor,
                text="user content",
                system_prompt="be a JSON expert",
            )
        )

    sent_messages = processor.apply_chat_template.call_args.args[0]
    assert sent_messages[0] == {"role": "system", "content": "be a JSON expert"}
    assert sent_messages[1]["role"] == "user"
