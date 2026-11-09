"""Skip, rather than error, the tests that need model weights when the model isn't in the local Hugging Face cache.

On a machine with the weights cached nothing changes: every test runs. Elsewhere `pytest -q` runs
the 8 weight-free tests and reports the rest as skipped with the reason. Set NL_DOWNLOAD=1 to let the tests download
the model instead.
"""
import os

import pytest

REPO = os.environ.get("NL_MODEL", "Qwen/Qwen3.5-0.8B")
NEEDS_MODEL = {"nm", "ids"}  # fixtures that load the model
LOADS_MODEL_ITSELF = {"test_steer_grad_predicts_small_doses"}


def _cached(repo: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(repo, "config.json"), str)
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if os.environ.get("NL_DOWNLOAD") or _cached(REPO):
        return
    skip = pytest.mark.skip(reason=f"{REPO} is not in the local Hugging Face cache (set NL_DOWNLOAD=1 to download it)")
    for item in items:
        if NEEDS_MODEL & set(getattr(item, "fixturenames", ())) or item.name in LOADS_MODEL_ITSELF:
            item.add_marker(skip)
