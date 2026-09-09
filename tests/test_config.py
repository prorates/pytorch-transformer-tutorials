"""config.py's helpers — the shared surface every tutorial calls.

get_config() with no arguments creates a model folder on disk; these tests always
pass an explicit path so the suite stays side-effect free.
"""

from pathlib import Path

import pytest

import config

REQUIRED_KEYS = {
    "batch_size",
    "num_epochs",
    "lr",
    "seq_len",
    "d_model",
    "N",
    "h",
    "dropout",
    "d_ff",
    "datasource",
    "lang_src",
    "lang_tgt",
    "model_basename",
    "tokenizer_file",
    "experiment_name",
}

# Every model folder's config.yaml — one whose config drifted from what
# get_config() can load is a training run that dies on startup. get_model_folder()
# names them "<datasource>_<src>_<tgt>[_<altmodel>]", so the underscore is what
# separates them from unrelated dirs that also hold a config.yaml (openspec/).
TRACKED_CONFIGS = sorted(str(p) for p in Path(__file__).resolve().parent.parent.glob("*/config.yaml") if "_" in p.parent.name)


def test_default_config_has_every_required_key() -> None:
    assert set(config.get_default_config()) >= REQUIRED_KEYS


def test_default_config_dropout_is_a_probability() -> None:
    assert 0.0 <= config.get_default_config()["dropout"] < 1.0


def test_d_model_divides_evenly_by_head_count() -> None:
    """Multi-head attention splits d_model across h; a remainder is a silent bug."""
    default = config.get_default_config()
    assert default["d_model"] % default["h"] == 0


def test_tracked_configs_are_discovered() -> None:
    assert TRACKED_CONFIGS, "expected at least one <model folder>/config.yaml"


@pytest.mark.parametrize("path", TRACKED_CONFIGS)
def test_tracked_config_loads_and_carries_required_keys(path: str) -> None:
    loaded = config.get_config(filename=path)
    assert set(loaded) >= REQUIRED_KEYS


def test_missing_config_file_falls_back_to_defaults() -> None:
    assert config.get_config(filename="does/not/exist.yaml") == config.get_default_config()


def test_model_folder_uses_alt_model_when_set() -> None:
    cfg = config.get_default_config()
    cfg["datasource"], cfg["lang_src"], cfg["lang_tgt"], cfg["alt_model"] = "ds", "en", "fr", "model3"
    assert config.get_model_folder(cfg) == "ds_en_fr_model3"


def test_model_folder_omits_alt_model_when_unset() -> None:
    cfg = config.get_default_config()
    cfg["datasource"], cfg["lang_src"], cfg["lang_tgt"], cfg["alt_model"] = "ds", "en", "fr", None
    assert config.get_model_folder(cfg) == "ds_en_fr"


def test_weights_path_sits_under_the_model_folder() -> None:
    cfg = config.get_default_config()
    path = Path(config.get_weights_file_path(cfg, "07"))
    assert path.parent.name == config.get_model_folder(cfg)
    assert path.name.endswith("07.pt")


def test_latest_weights_returns_none_when_folder_is_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = config.get_default_config()
    Path(config.get_model_folder(cfg)).mkdir(parents=True)
    assert config.latest_weights_file_path(cfg) is None


def test_get_device_returns_a_known_backend() -> None:
    assert config.get_device() in {"cuda", "mps", "cpu"}
