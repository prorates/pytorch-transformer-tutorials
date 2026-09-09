"""Every module imports cleanly.

Cheap, but it is the check that actually breaks: these modules are edited one
tutorial at a time and share config.py / utils.py, so a signature change in the
shared pair surfaces here rather than partway through a training run. Every
entry point is `if __name__ == "__main__"`-guarded, so importing runs no work.
"""

import importlib

import pytest

MODULES = [
    "config",
    "utils",
    "train",
    "translate",
    "test",
    *[f"model{n}" for n in range(1, 9)],
    *[f"tutorial{n}" for n in range(1, 9)],
    *[f"dataset{n}" for n in (1, 2, 3, 6, 7, 8)],
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    assert importlib.import_module(name) is not None


@pytest.mark.parametrize("n", range(1, 9))
def test_each_model_exposes_its_builder(n: int) -> None:
    """model<n>.py exports build_transformer<n>."""
    module = importlib.import_module(f"model{n}")
    assert callable(getattr(module, f"build_transformer{n}"))
