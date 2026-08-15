# CLAUDE.md

## Project Overview

Streamlit application for structured extraction, document understanding, and template generation with NuMind NuExtract on Apple Silicon with MLX. Uses `numind/NuExtract3-mlx-8bits` (a 4B-parameter Qwen3-family vision-language model) via [mlx-vlm](https://github.com/Blaizzy/mlx-vlm). Mirrors the official [NuExtract3 HF Space](https://huggingface.co/spaces/numind/NuExtract3) architecture but uses local MLX inference instead of remote vLLM.

Three modes: structured JSON extraction (typed template + image/text → JSON), document-to-markdown (image → Markdown), and template generation (NL description → JSON template). Streaming output with optional `<think>...</think>` reasoning trace.

## Commands

```bash
uv sync                                    # Setup environment
uv run streamlit run streamlit_app.py      # Run the app
uv run ruff check .                        # Lint
uv run ruff format .                       # Format
uv run ty check                            # Type check
uv run pytest                              # Run tests
uv run pytest tests/test_file.py::test_name  # Run a single test
uv run python scripts/probe_mlx_vlm.py     # End-to-end model probe
```

When working with Python, invoke the relevant `/astral:<skill>` for uv, ty, and ruff to ensure best practices are followed.

CI via GitHub Actions (`.github/workflows/ci.yml`): lint, format check, type check, and `pytest` on every push and PR to `main`. Uses `macos-14` (Apple Silicon) runners for MLX compatibility. All four steps gate the build.

**Everything CI depends on is pinned**, so a green build can't turn red without a repo change: Python deps via `==` + `uv sync --frozen`, the interpreter via `.python-version` (3.13), uv itself via setup-uv's `version:` input, and the actions by tag. Note `astral-sh/setup-uv` stopped publishing moving major tags after `v7`, so it is referenced by exact tag (`@v10.0.1`) while `actions/checkout` uses the conventional moving major (`@v7`).

Lint/type config lives in `pyproject.toml`: Ruff selects `E`/`F`/`I` (import sort)/`UP` (pyupgrade), with line length deferred to the formatter (`ignore = ["E501"]`); ty targets the `requires-python` floor (3.12) with `error-on-warning`. **The runtime/type-check version split is deliberate** — code runs on 3.13 (`.python-version`) while ty analyzes against the 3.12 floor, so `requires-python = ">=3.12"` stays honest without holding the runtime back.

Team-shared Claude Code hooks in `.claude/hooks/` (wired via `.claude/settings.json`; both tracked in git) enforce these same gates at edit-time: `ruff format` + `ruff check --fix` + `ty check` after each `.py` edit (PostToolUse), a block on editing `.env*`/`.streamlit/secrets.toml` (PreToolUse), and `pytest` when a turn changed `.py` files (Stop, loop-guarded). They need `jq` + `uv` (missing `jq` degrades to a silent no-op) and are snapshotted at session start, so restart after pulling changes. Details in `.claude/hooks/README.md`; personal overrides go in the gitignored `.claude/settings.local.json`.

## Architecture

App in `streamlit_app.py`, runtime wrapper in `nuextract.py`, probe in `scripts/probe_mlx_vlm.py`.

### `nuextract.py`

Thin wrapper around mlx-vlm. Module-level imports of `huggingface_hub.snapshot_download`, `mlx_vlm.load`, `mlx_vlm.stream_generate` so tests can patch via the `nuextract.*` namespace (standard "patch where used" pattern).

- **Constants** — `DEFAULT_MODEL_ID = "numind/NuExtract3-mlx-8bits"`, `DEFAULT_MAX_TOKENS = 4096`, `DEFAULT_TEMPERATURE = 0.0`, `MODE_STRUCTURED`, `MODE_CONTENT`, `MODE_MARKDOWN`, `MODE_TEMPLATE_GENERATION`
- **`patch_processor_config(local_dir)`** — Patches the MLX repo's `processor_config.json` packaging bug (see Key Details). Idempotent, returns `True` if changed.
- **`load_model(model_id)`** — `snapshot_download` → `patch_processor_config` → `mlx_vlm.load`. Returns `(model, processor)`.
- **`build_messages(text, image_path, system_prompt)`** — Builds a chat message list: optional `{"role": "system", "content": ...}` first (string-only — the Jinja raises if it contains images), then a `{"role": "user", "content": [...]}` with `{"type": "image", "image": path}` and/or `{"type": "text", "text": ...}` parts. The Jinja template inserts the vision placeholder; actual pixel data flows separately via `stream_generate(image=...)`.
- **`render_prompt(processor, messages, *, template, instructions, mode, enable_thinking)`** — Calls `processor.apply_chat_template(...)` with task kwargs **inline** (HF transformers convention), NOT nested under `chat_template_kwargs` (that's a vLLM-specific convention). Omits `None`/empty values so the template's defaults apply.
- **`stream_extract(model, processor, *, text, image_path, system_prompt, template, instructions, mode, enable_thinking, temperature, max_tokens)`** — Builds messages → renders prompt → calls `mlx_vlm.stream_generate`. Yields **cumulative** text on each chunk (not per-chunk deltas) for direct rendering into a Streamlit placeholder.
- **`split_reasoning_and_output(text, reasoning_enabled)`** — Splits on `</think>` (case-insensitive). Returns `("", text)` when reasoning is off; returns `(text, "")` when reasoning is on but `</think>` hasn't arrived yet.
- **`extract_answer_block(text)`** — Pulls `<answer>...</answer>` contents if present, else the longest span at any `{` position that parses via `json.JSONDecoder.raw_decode`, else the stripped text. Correctly handles the "reasoning text + JSON" case without merging unrelated spans.
- **`pretty_json_or_text(text)`** — Tries `json.loads` + indented re-dump; falls back to the original on parse failure.

### `streamlit_app.py`

Two-column layout. Left: inputs. Right: action buttons + outputs (wrapped in an `st.fragment`).

- **`get_model()`** — `@st.cache_resource` wrapper around `nuextract.load_model`.
- **`_save_uploaded_image(uploaded_file)`** — Persists `st.file_uploader` bytes to a temp file once per upload, keyed on `uploaded_file.file_id` in `st.session_state`. Reruns return the cached path; a new upload deletes the previous temp file before writing the new one, and removing the upload deletes the orphaned temp file and clears the cached `session_state` keys. Required because mlx-vlm wants file paths/URLs.
- **`_validate_template(template_str)`** — Validates JSON template is a non-empty dict; returns `(parsed, error)`.
- **`_render_output_pane(output_placeholder, reasoning_placeholder, accumulated, *, reasoning_enabled, is_structured)`** — Updates both panes from a single stream chunk: routes reasoning trace to its placeholder (or shows `_(reasoning disabled)_` when off), then renders output as JSON code-block (structured/template-gen modes) or markdown (markdown mode). Shows `_(waiting for output after </think>)_` placeholder while reasoning is in progress.
- **`_render_download_button(placeholder, accumulated, *, download_kind, reasoning)`** — After a successful run, renders a `st.download_button` inside the download placeholder, with `on_click="ignore"` so downloading stays client-side and doesn't rerun the fragment (a rerun would repaint the idle hint over the result and drop the button). `download_kind` is one of `"extract"`, `"template"`, `"markdown"` and controls the label, filename, and whether `extract_answer_block` is applied to strip wrappers.
- **`_run_mode(...)`** — Drives one streamed generation: calls `_render_output_pane` on every chunk yielded by `stream_extract`, then fills the download placeholder on completion.
- **`_output_section(model, processor)`** — `@st.fragment`-decorated. Renders the three action buttons (Extract / Markdown / Template-gen) in a horizontal container, plus the always-rendered Reasoning + Result placeholders and the download placeholder (the Result pane shows an idle empty-state hint until a run produces output), then runs the button handlers. Lives in a fragment so clicking a generate button reruns **only** this region — the left-column input widgets keep their state and aren't re-rendered during generation. The buttons must be inside the fragment for that isolation to apply (a fragment only reruns independently when the triggering widget is inside it). Reads input values from `st.session_state` by widget key (the left-column widgets populate them on the full rerun that precedes the fragment). Handlers validate inputs (markdown requires image; extract requires template + either image or text) and call `_run_mode`.
- **Top-level UI** — `st.set_page_config(page_icon=":material/document_scanner:", layout="wide")`, title, model load spinner, then `st.columns([1, 1])` for the two panes. Left pane: image uploader (with preview), optional text area, JSON template editor (`DEFAULT_TEMPLATE`), optional instructions, temperature + max-tokens sliders and a reasoning checkbox on its own row — all keyed, return values unused (read via `session_state` in the fragment). Right pane: `Output` subheader then `_output_section(model, processor)`.

### `scripts/probe_mlx_vlm.py`

Standalone probe to verify the migration still works on a fresh machine. Six checks:
1. `mlx_vlm.load()` succeeds on `numind/NuExtract3-mlx-8bits`
2. `processor.apply_chat_template(template=..., enable_thinking=False)` produces a rendered prompt with the typed template embedded under `【template_start】`
3. `mode='markdown'` reassigns to `【task】content` (per the Jinja's mode-collapsing logic)
4. `mode='template-generation'` produces `【task】template generation`
5. End-to-end `mlx_vlm.generate()` returns parseable JSON for a trivial extraction
6. End-to-end `mlx_vlm.generate(image=[...])` on a PIL-rendered PNG returns typed JSON — the **only** coverage of the vision path (torchvision + the image processor), since every test in `tests/` mocks `mlx_vlm.stream_generate`. The image is generated at run time, so no binary fixture is committed.

Run: `uv run python scripts/probe_mlx_vlm.py`. Exits non-zero on any failure.

## Key Details

- **Model**: `numind/NuExtract3-mlx-8bits` — 8-bit affine quant of NuExtract3 (Qwen3.5 4B base), ~5 GB on disk
- **Context**: 131K tokens supported by the model, but practical limit is bounded by unified memory + KV cache size
- **Pinned transformers**: `==5.15.0`. The floor is **not freely chosen** — `mlx-vlm==0.6.13` declares `transformers>=5.14.0`, so transformers cannot be rolled back without also downgrading mlx-vlm. (The model's own declared `transformers_version` is `5.5.4`, but `qwen3_5` only enters the auto-resolver mapping in much later versions.) Two invariants make the stack work — `qwen3_5` in `CONFIG_MAPPING_NAMES` and `Qwen2VLImageProcessor` still exported — and both are asserted by model-free tests in `tests/test_nuextract.py`, so CI gates them rather than a checklist.
- **Required torchvision**: as of transformers 5.15 the image-processor mapping is backend-keyed — `qwen3_5 → {"torchvision": "Qwen2VLImageProcessor", "pil": "Qwen2VLImageProcessorPil"}` — so the image processor alone no longer forces torchvision. The hard requirement now comes from `Qwen3VLVideoProcessor`, which the processor constructs eagerly on load and which requires PyTorch. `torchvision` stays a direct dependency (it is what pulls `torch` into the tree); dropping it is not a simple deletion.
- **Packaging-bug shim** in `nuextract.patch_processor_config()`: `numind/NuExtract3-mlx-8bits` declares `Qwen3VLImageProcessor`; the model author's own `numind/NuExtract3` declares `Qwen2VLImageProcessor` with otherwise identical geometry. **Upstream is authoritative, so this is a conversion bug in the MLX repo, not a workaround for a class missing from transformers.** Do not gate the shim on `hasattr(transformers, "Qwen3VLImageProcessor")` — the class being absent today is why the bug is *visible* (load fails loudly), not why the rewrite is *correct*; such a gate would silently stop patching the day transformers ships the class. The shim patches the locally cached copy on every `load_model()` call (idempotent).
- **Template kwarg API**: HF transformers' `apply_chat_template` accepts template variables as **direct keyword arguments** (e.g. `template="..."`, `mode="..."`, `enable_thinking=False`). The HF Space uses vLLM which expects them nested under `chat_template_kwargs={...}` — that convention does **not** apply to direct HF/mlx-vlm usage. As of transformers 5.15 this is conditional on the model's template, not universal: `apply_chat_template` introspects the Jinja for declared variables and routes anything undeclared into `processor_kwargs`, warning as it does. NuExtract3's template declares all four kwargs `render_prompt` sends, so nothing warns today — but a template change (or a stricter transformers) could turn this into a silent no-op, and the warning is the signal to watch for.
- **Unexercised transitive paths**: the dependency set carries `pandas` 3.x (Streamlit's dataframe/Altair rendering) and `opencv-python` 5.x (mlx-vlm's video path). Nothing in this app or its tests touches either, so both crossed a major version **unverified rather than verified** — a regression there would surface to a user, not to CI.
- **Modes (from `chat_template.jinja`)**: setting `template=...` forces `mode='structured'`; `mode='markdown'` is reassigned to `'content'`; `mode='template-generation'` and `'document-detection'` are also valid. `enable_thinking=True` is only allowed for `structured` and `content` modes.
- **Output parsing**: NuExtract3 may emit `<answer>...</answer>` wrappers around structured output or raw JSON. `extract_answer_block` handles both.
- **Reasoning trace**: model emits `<think>...</think>` before the answer when `enable_thinking=True`. `split_reasoning_and_output` separates them for the two-pane UI.
- **Image input**: `st.file_uploader` (jpg/jpeg/png/webp), persisted to a temp file before passing to mlx-vlm. No clipboard support (Streamlit limitation vs the Gradio-based HF Space).
- **Theme**: `.streamlit/config.toml` sets a GitHub-inspired light/dark palette (keys validated against the Streamlit 1.61 schema). Tracked in git — only `.streamlit/secrets.toml` is ignored.
- **No PDF support** in v1 — matches the HF Space; users convert PDFs to images externally.
- **License**: project code is **MIT** (`LICENSE` at repo root — verbatim SPDX text so GitHub's detector recognizes it; also declared in `pyproject.toml` via PEP 639 `license = "MIT"` + `license-files = ["LICENSE"]`). The NuExtract3 model weights are separately **Apache-2.0** and downloaded at runtime (not redistributed by this repo), so the model license governs model usage, not this project's code. README has Acknowledgments + License sections crediting the upstream MIT HF Space.

## Tests

Total: 93 tests across three files, no real model loaded. These are `pytest`-collected counts — parametrization expands `test_nuextract.py`'s 32 `def test_` functions into 43 cases, so a raw `grep -c 'def test_'` undercounts.

- **`tests/test_nuextract.py`** (43) — Pure function tests for the runtime wrapper. `extract_answer_block` and `pretty_json_or_text` cases are parametrized; integration boundaries (`load_model`, `stream_extract`) are tested by patching the `nuextract.*` namespace. Two tests import `transformers` directly to assert the compatibility invariants (see Key Details) — the only tests that touch a real dependency, and they load no model.
- **`tests/test_streamlit_app.py`** (25) — Helper function tests. The module-scoped `app` fixture mocks all Streamlit primitives + `nuextract.load_model` so `streamlit_app` imports cleanly without a real model (`st.fragment` is patched to an identity decorator so the `_output_section` fragment body runs under the mocks). Includes a regression test that removing an upload cleans up the orphaned temp file and session state.
- **`tests/test_streamlit_app_apptest.py`** (25) — End-to-end UI wiring via Streamlit's `AppTest`. `nuextract.load_model` is stubbed out. The `at_with_image` fixture drives a fake image into `st.file_uploader` via AppTest's native `set_value` API (added in Streamlit 1.56), exercising the real widget. Covers initial render (including the Result-pane empty-state hint rendered by the `_output_section` fragment), button validation, and streaming flow for text- and image-input paths (including the instructions field, template-gen system prompt, and template-gen reasoning override). `st.download_button` isn't exposed by AppTest — its rendering is tested in `test_streamlit_app.py`.
