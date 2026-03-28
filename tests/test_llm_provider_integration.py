"""
Integration tests for MiniMax LLM provider in evaluation benchmarks.

These tests verify that the MiniMax provider can be correctly configured
and used with the evaluation benchmark scripts. They require a valid
MINIMAX_API_KEY environment variable to run.

Run with: MINIMAX_API_KEY=your-key python -m pytest tests/test_llm_provider_integration.py -v
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llava.eval.llm_provider import (
    create_llm_client,
    get_model_name,
    strip_think_tags,
    PROVIDER_PRESETS,
)


@unittest.skipUnless(
    os.environ.get("MINIMAX_API_KEY"),
    "MINIMAX_API_KEY not set, skipping integration tests"
)
class TestMiniMaxIntegration(unittest.TestCase):
    """Integration tests that call the real MiniMax API."""

    def test_minimax_client_chat_completion(self):
        """Test a real chat completion call to MiniMax API."""
        client = create_llm_client("minimax")
        completion = client.chat.completions.create(
            model="MiniMax-M2.7",
            messages=[
                {"role": "user", "content": "Reply with exactly: OK"}
            ],
            max_tokens=10,
        )
        self.assertIsNotNone(completion.choices)
        self.assertGreater(len(completion.choices), 0)
        self.assertIsNotNone(completion.choices[0].message.content)

    def test_minimax_eval_style_prompt(self):
        """Test an evaluation-style scoring prompt with MiniMax."""
        client = create_llm_client("minimax")
        completion = client.chat.completions.create(
            model="MiniMax-M2.7",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an evaluator. Output ONLY a Python dictionary "
                        "with a 'score' key. No explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Question: What color is the sky?\n"
                        "Correct Answer: Blue\n"
                        "Predicted Answer: Blue\n\n"
                        "Output only: {'score': <integer 0-5>}"
                    ),
                },
            ],
            max_tokens=100,
        )
        raw_response = completion.choices[0].message.content
        self.assertIsNotNone(raw_response)
        # Verify the model returns a parseable response (with or without think tags)
        response = strip_think_tags(raw_response)
        # The response should contain score-related content
        self.assertTrue(
            "score" in response or "score" in raw_response,
            f"Expected 'score' in response, got: {raw_response!r}"
        )

    def test_minimax_highspeed_model(self):
        """Test MiniMax-M2.7-highspeed model works."""
        client = create_llm_client("minimax")
        completion = client.chat.completions.create(
            model="MiniMax-M2.7-highspeed",
            messages=[
                {"role": "user", "content": "Reply with exactly: OK"}
            ],
            max_tokens=10,
        )
        self.assertIsNotNone(completion.choices[0].message.content)


class TestMiniMaxClientConfig(unittest.TestCase):
    """Test MiniMax client configuration without API calls."""

    @patch("llava.eval.llm_provider.OpenAI")
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"}, clear=True)
    def test_minimax_client_uses_correct_base_url(self, mock_openai):
        create_llm_client("minimax")
        call_kwargs = mock_openai.call_args[1]
        self.assertEqual(call_kwargs["base_url"], "https://api.minimax.io/v1")

    @patch("llava.eval.llm_provider.OpenAI")
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"}, clear=True)
    def test_minimax_client_uses_api_key(self, mock_openai):
        create_llm_client("minimax")
        call_kwargs = mock_openai.call_args[1]
        self.assertEqual(call_kwargs["api_key"], "test-key-123")

    @patch("llava.eval.llm_provider.OpenAI")
    @patch.dict(
        os.environ,
        {"MINIMAX_API_KEY": "mk", "OPENAIBASE": "https://override.api/v1"},
        clear=True,
    )
    def test_openaibase_overrides_minimax_base_url(self, mock_openai):
        create_llm_client("minimax")
        call_kwargs = mock_openai.call_args[1]
        self.assertEqual(call_kwargs["base_url"], "https://override.api/v1")


if __name__ == "__main__":
    unittest.main()
