import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
_PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)


def _forbid(target: str):
    """Build a stand-in that fails loudly instead of fetching or loading a model."""

    def _raise(*_args, **_kwargs):
        raise AssertionError(
            f"{target} was called for real during a test — this would trigger a "
            "real model download/load (~4.8 GB). Mock the nuextract.* namespace "
            "before importing the module under test; see the app fixture in "
            "tests/test_streamlit_app.py."
        )

    return _raise


@pytest.fixture(autouse=True, scope="session")
def _no_real_model_loads():
    """Make a forgotten mock fail in milliseconds rather than download 4.8 GB.

    This exists because the invariant "no test loads a real model" was asserted
    in CLAUDE.md for months while being false: the app fixture patched
    `streamlit_app.load_model`, which mock.patch resolves *by importing*
    streamlit_app, whose module body calls get_model(). Every CI run downloaded
    the model, and two runs hung for 6h inside it before GitHub's 360-minute
    default killed them.

    A hand-run check (`HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run pytest`) is
    the wrong enforcement mechanism on its own: with a warm cache, offline mode
    *succeeds*, so it passes locally and only fails on a cold CI runner — the
    exact blind spot that hid the bug. This guard has no such gap.

    Deliberately narrow. Blocking sockets outright would also catch it, but
    Streamlit gathers usage stats and AppTest runs the real script in-process,
    so a broad network ban risks failures unrelated to the bug being prevented.
    Tests that legitimately exercise these boundaries patch over the guard and
    restore it on exit, which is what makes it safe to leave always-on.
    """
    import huggingface_hub

    import nuextract

    with (
        patch.object(
            nuextract,
            "snapshot_download",
            side_effect=_forbid("nuextract.snapshot_download"),
        ),
        patch.object(
            nuextract, "mlx_vlm_load", side_effect=_forbid("nuextract.mlx_vlm_load")
        ),
        # nuextract binds snapshot_download at import time, so patching the
        # source as well only matters for a future module that imports it
        # directly — cheap insurance against the same mistake in a new file.
        patch.object(
            huggingface_hub,
            "snapshot_download",
            side_effect=_forbid("huggingface_hub.snapshot_download"),
        ),
    ):
        yield
