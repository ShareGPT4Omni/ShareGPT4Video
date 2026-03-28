"""
Unit tests for llm_provider module.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from llava.eval.llm_provider import (
    PROVIDER_PRESETS,
    SUPPORTED_PROVIDERS,
    detect_provider,
    create_llm_client,
    get_model_name,
    add_provider_args,
    clamp_temperature,
    strip_think_tags,
)


class TestProviderPresets(unittest.TestCase):
    """Test provider preset configurations."""

    def test_openai_preset_exists(self):
        self.assertIn("openai", PROVIDER_PRESETS)

    def test_minimax_preset_exists(self):
        self.assertIn("minimax", PROVIDER_PRESETS)

    def test_minimax_base_url(self):
        self.assertEqual(
            PROVIDER_PRESETS["minimax"]["base_url"],
            "https://api.minimax.io/v1"
        )

    def test_minimax_default_model(self):
        self.assertEqual(
            PROVIDER_PRESETS["minimax"]["default_model"],
            "MiniMax-M2.7"
        )

    def test_openai_default_model(self):
        self.assertEqual(
            PROVIDER_PRESETS["openai"]["default_model"],
            "gpt-3.5-turbo-0125"
        )

    def test_minimax_api_key_env(self):
        self.assertEqual(
            PROVIDER_PRESETS["minimax"]["api_key_env"],
            "MINIMAX_API_KEY"
        )

    def test_supported_providers_list(self):
        self.assertIn("openai", SUPPORTED_PROVIDERS)
        self.assertIn("minimax", SUPPORTED_PROVIDERS)

    def test_all_presets_have_required_keys(self):
        required_keys = {"base_url", "api_key_env", "default_model"}
        for provider, preset in PROVIDER_PRESETS.items():
            for key in required_keys:
                self.assertIn(key, preset, f"{provider} missing {key}")


class TestDetectProvider(unittest.TestCase):
    """Test auto-detection of LLM provider from environment variables."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}, clear=False)
    def test_detect_minimax_from_env(self):
        self.assertEqual(detect_provider(), "minimax")

    @patch.dict(os.environ, {"OPENAIKEY": "test-key"}, clear=False)
    def test_detect_openai_from_env(self):
        # Remove MINIMAX_API_KEY if present
        env = os.environ.copy()
        env.pop("MINIMAX_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            os.environ["OPENAIKEY"] = "test-key"
            self.assertEqual(detect_provider(), "openai")

    @patch.dict(os.environ, {}, clear=True)
    def test_detect_defaults_to_openai(self):
        self.assertEqual(detect_provider(), "openai")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "mk", "OPENAIKEY": "ok"}, clear=True)
    def test_minimax_takes_priority(self):
        self.assertEqual(detect_provider(), "minimax")


class TestCreateLLMClient(unittest.TestCase):
    """Test LLM client creation for different providers."""

    @patch("llava.eval.llm_provider.OpenAI")
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-minimax-key"}, clear=True)
    def test_create_minimax_client(self, mock_openai):
        create_llm_client("minimax")
        mock_openai.assert_called_once_with(
            api_key="test-minimax-key",
            base_url="https://api.minimax.io/v1"
        )

    @patch("llava.eval.llm_provider.OpenAI")
    @patch.dict(os.environ, {"OPENAIKEY": "test-openai-key"}, clear=True)
    def test_create_openai_client(self, mock_openai):
        create_llm_client("openai")
        mock_openai.assert_called_once_with(api_key="test-openai-key")

    @patch("llava.eval.llm_provider.OpenAI")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-alt-key"}, clear=True)
    def test_create_openai_client_alt_env(self, mock_openai):
        create_llm_client("openai")
        mock_openai.assert_called_once_with(api_key="test-alt-key")

    def test_create_unsupported_provider_raises(self):
        with self.assertRaises(ValueError) as ctx:
            create_llm_client("unsupported_provider")
        self.assertIn("Unsupported provider", str(ctx.exception))

    @patch("llava.eval.llm_provider.OpenAI")
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "mk"}, clear=True)
    def test_auto_detect_creates_minimax(self, mock_openai):
        create_llm_client(None)
        mock_openai.assert_called_once_with(
            api_key="mk",
            base_url="https://api.minimax.io/v1"
        )

    @patch("llava.eval.llm_provider.OpenAI")
    @patch.dict(os.environ, {"OPENAIKEY": "ok", "OPENAIBASE": "https://custom.api/v1"}, clear=True)
    def test_openaibase_env_override(self, mock_openai):
        create_llm_client("openai")
        mock_openai.assert_called_once_with(
            api_key="ok",
            base_url="https://custom.api/v1"
        )

    @patch("llava.eval.llm_provider.OpenAI")
    def test_case_insensitive_provider(self, mock_openai):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "mk"}, clear=True):
            create_llm_client("MiniMax")
            mock_openai.assert_called_once()


class TestGetModelName(unittest.TestCase):
    """Test model name resolution."""

    def test_minimax_default_model(self):
        args = argparse.Namespace(
            provider="minimax", gpt_version="gpt-3.5-turbo-0125"
        )
        self.assertEqual(get_model_name(args), "MiniMax-M2.7")

    def test_openai_default_model(self):
        args = argparse.Namespace(
            provider="openai", gpt_version="gpt-3.5-turbo-0125"
        )
        self.assertEqual(get_model_name(args), "gpt-3.5-turbo-0125")

    def test_explicit_model_overrides_default(self):
        args = argparse.Namespace(
            provider="minimax", gpt_version="MiniMax-M2.7-highspeed"
        )
        self.assertEqual(get_model_name(args), "MiniMax-M2.7-highspeed")

    def test_explicit_custom_model(self):
        args = argparse.Namespace(
            provider="openai", gpt_version="gpt-4o"
        )
        self.assertEqual(get_model_name(args), "gpt-4o")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "mk"}, clear=True)
    def test_auto_detect_model(self):
        args = argparse.Namespace(
            provider=None, gpt_version="gpt-3.5-turbo-0125"
        )
        self.assertEqual(get_model_name(args), "MiniMax-M2.7")


class TestAddProviderArgs(unittest.TestCase):
    """Test CLI argument injection."""

    def test_adds_provider_argument(self):
        parser = argparse.ArgumentParser()
        add_provider_args(parser)
        args = parser.parse_args(["--provider", "minimax"])
        self.assertEqual(args.provider, "minimax")

    def test_provider_default_is_none(self):
        parser = argparse.ArgumentParser()
        add_provider_args(parser)
        args = parser.parse_args([])
        self.assertIsNone(args.provider)

    def test_invalid_provider_rejected(self):
        parser = argparse.ArgumentParser()
        add_provider_args(parser)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--provider", "invalid"])

    def test_openai_provider_accepted(self):
        parser = argparse.ArgumentParser()
        add_provider_args(parser)
        args = parser.parse_args(["--provider", "openai"])
        self.assertEqual(args.provider, "openai")


class TestClampTemperature(unittest.TestCase):
    """Test temperature clamping for different providers."""

    def test_minimax_clamp_zero(self):
        self.assertEqual(clamp_temperature(0.0, "minimax"), 0.01)

    def test_minimax_clamp_negative(self):
        self.assertEqual(clamp_temperature(-1.0, "minimax"), 0.01)

    def test_minimax_clamp_above_one(self):
        self.assertEqual(clamp_temperature(1.5, "minimax"), 1.0)

    def test_minimax_valid_temperature(self):
        self.assertEqual(clamp_temperature(0.7, "minimax"), 0.7)

    def test_minimax_boundary_one(self):
        self.assertEqual(clamp_temperature(1.0, "minimax"), 1.0)

    def test_openai_no_clamp(self):
        self.assertEqual(clamp_temperature(0.0, "openai"), 0.0)
        self.assertEqual(clamp_temperature(2.0, "openai"), 2.0)

    @patch.dict(os.environ, {}, clear=True)
    def test_auto_detect_openai_no_clamp(self):
        self.assertEqual(clamp_temperature(0.0), 0.0)


class TestStripThinkTags(unittest.TestCase):
    """Test think tag stripping."""

    def test_strip_simple_think_block(self):
        text = "<think>reasoning here</think>{'score': 5}"
        self.assertEqual(strip_think_tags(text), "{'score': 5}")

    def test_strip_multiline_think_block(self):
        text = "<think>\nlong\nreasoning\n</think>\n{'score': 3}"
        self.assertEqual(strip_think_tags(text), "{'score': 3}")

    def test_no_think_tags_unchanged(self):
        text = "{'score': 4}"
        self.assertEqual(strip_think_tags(text), "{'score': 4}")

    def test_none_returns_none(self):
        self.assertIsNone(strip_think_tags(None))

    def test_empty_string(self):
        self.assertEqual(strip_think_tags(""), "")

    def test_think_only_returns_empty(self):
        text = "<think>just thinking</think>"
        self.assertEqual(strip_think_tags(text), "")

    def test_preserves_surrounding_text(self):
        text = "before <think>middle</think> after"
        self.assertEqual(strip_think_tags(text), "before  after")


if __name__ == "__main__":
    unittest.main()
