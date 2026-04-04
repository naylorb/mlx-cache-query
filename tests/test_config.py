from unittest.mock import patch
from mcq.core.config import McqConfig, _parse_toml


def test_default_config():
    config = McqConfig()
    assert config.model == "mlx-community/Qwen2.5-3B-Instruct-4bit"
    assert config.max_tokens == 512
    assert config.query_budget == 2048


def test_parse_toml_basic():
    text = 'model = "my-model"\nmax_tokens = 1024\n'
    data = _parse_toml(text)
    assert data["model"] == "my-model"
    assert data["max_tokens"] == 1024


def test_parse_toml_comments():
    text = "# comment\nmodel = \"test\"\n# another comment\n"
    data = _parse_toml(text)
    assert data["model"] == "test"


def test_config_load_from_file(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text('model = "custom-model"\nmax_tokens = 256\n')
    with patch("mcq.core.config._CONFIG_PATH", config_file):
        config = McqConfig.load()
    assert config.model == "custom-model"
    assert config.max_tokens == 256
    assert config.query_budget == 2048  # default


def test_config_load_missing_file(tmp_path):
    with patch("mcq.core.config._CONFIG_PATH", tmp_path / "nonexistent.toml"):
        config = McqConfig.load()
    assert config.model == "mlx-community/Qwen2.5-3B-Instruct-4bit"
