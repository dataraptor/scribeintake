"""Deterministic tests for config: pinned IDs, limits, paths, no-key import."""

from scribeintake import config


def test_model_ids_are_pinned():
    assert config.MODEL_INTAKE == "claude-sonnet-4-6"
    assert config.MODEL_SUMMARY == "claude-opus-4-8"
    assert config.MODEL_JUDGE == "claude-opus-4-8"


def test_limits_and_cache_floors_present():
    assert config.MAX_AGENT_STEPS == 4
    assert config.MAX_INTAKE_TURNS == 20
    assert config.CACHE_FLOOR_OPUS == 4096
    assert config.CACHE_FLOOR_SONNET == 2048


def test_locale_and_crisis_numbers():
    assert config.LOCALE == "en-US"
    assert config.CRISIS_NUMBERS == {"lifeline": "988", "emergency": "911"}


def test_reproducibility_versions():
    assert config.RULES_VERSION == "v1"
    assert config.PROMPT_VERSION == "v1"


def test_data_dir_is_created_on_access():
    data_dir = config.settings.DATA_DIR
    assert data_dir.exists()
    assert data_dir.is_dir()
    # Derived paths hang off DATA_DIR.
    assert config.settings.DB_PATH.parent == data_dir
    assert config.settings.CHROMA_DIR.parent == data_dir


def test_kb_dir_points_into_package():
    assert config.settings.KB_DIR.name == "kb"
    assert config.settings.KB_DIR.parent.name == "scribeintake"


def test_import_does_not_require_api_key():
    # settings imported fine without ANTHROPIC_API_KEY set; the field is optional.
    assert config.settings is not None
    key = config.settings.anthropic_api_key
    assert key is None or isinstance(key, str)
