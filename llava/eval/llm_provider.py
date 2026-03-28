"""
LLM provider utility for evaluation benchmarks.

Supports multiple LLM providers (OpenAI, MiniMax) via OpenAI-compatible API.
Provider can be selected via --provider CLI argument or auto-detected from
environment variables.

Usage:
    from llm_provider import create_llm_client, add_provider_args, get_model_name

    # In parse_args():
    add_provider_args(parser)

    # In main():
    args = parse_args()
    client = create_llm_client(args.provider)
    model = get_model_name(args)
"""

import os
import re
from openai import OpenAI


# Provider presets: (base_url, api_key_env, default_model)
PROVIDER_PRESETS = {
    "openai": {
        "base_url": None,  # Uses OpenAI SDK default
        "api_key_env": "OPENAIKEY",
        "api_key_env_alt": "OPENAI_API_KEY",
        "default_model": "gpt-3.5-turbo-0125",
    },
    "minimax": {
        "base_url": "https://api.minimax.io/v1",
        "api_key_env": "MINIMAX_API_KEY",
        "api_key_env_alt": None,
        "default_model": "MiniMax-M2.7",
    },
}

SUPPORTED_PROVIDERS = list(PROVIDER_PRESETS.keys())


def detect_provider():
    """Auto-detect provider from available environment variables."""
    if os.environ.get("MINIMAX_API_KEY"):
        return "minimax"
    # Default to openai if OPENAIKEY or OPENAI_API_KEY is set, or as fallback
    return "openai"


def create_llm_client(provider=None):
    """
    Create an OpenAI-compatible client for the given provider.

    Args:
        provider: Provider name ('openai', 'minimax') or None for auto-detect.

    Returns:
        OpenAI client configured for the specified provider.
    """
    if provider is None:
        provider = detect_provider()

    provider = provider.lower()
    if provider not in PROVIDER_PRESETS:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    preset = PROVIDER_PRESETS[provider]

    # Resolve API key from environment
    api_key = os.environ.get(preset["api_key_env"])
    if not api_key and preset.get("api_key_env_alt"):
        api_key = os.environ.get(preset["api_key_env_alt"])

    # Resolve base URL: env override takes priority, then preset
    base_url = os.environ.get("OPENAIBASE") or preset["base_url"]

    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)


def get_model_name(args):
    """
    Get the model name from CLI args, falling back to provider default.

    Args:
        args: Parsed argparse namespace with 'gpt_version' and 'provider' attrs.

    Returns:
        Model name string.
    """
    provider = getattr(args, "provider", None) or detect_provider()
    provider = provider.lower()
    preset = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["openai"])

    # If user explicitly set --gpt_version (non-default), use it
    gpt_version = getattr(args, "gpt_version", None)
    if gpt_version and gpt_version != "gpt-3.5-turbo-0125":
        return gpt_version

    return preset["default_model"]


def add_provider_args(parser):
    """
    Add --provider argument to an argparse parser.

    Args:
        parser: argparse.ArgumentParser instance.
    """
    parser.add_argument(
        "--provider",
        default=None,
        type=str,
        choices=SUPPORTED_PROVIDERS,
        help=(
            "LLM provider to use for evaluation "
            f"({', '.join(SUPPORTED_PROVIDERS)}). "
            "Auto-detected from environment variables if not specified."
        ),
    )


def clamp_temperature(temperature, provider=None):
    """
    Clamp temperature to valid range for the given provider.

    MiniMax requires temperature in (0.0, 1.0].

    Args:
        temperature: Requested temperature value.
        provider: Provider name or None for auto-detect.

    Returns:
        Clamped temperature value.
    """
    if provider is None:
        provider = detect_provider()

    if provider == "minimax":
        # MiniMax requires temperature in (0.0, 1.0]
        if temperature <= 0.0:
            return 0.01
        if temperature > 1.0:
            return 1.0
    return temperature


def strip_think_tags(text):
    """
    Strip <think>...</think> tags from model responses.

    Some models (e.g., MiniMax-M2.7) may include reasoning traces
    wrapped in <think> tags. This function removes them to extract
    only the final answer.

    Args:
        text: Raw response text from the model.

    Returns:
        Text with <think>...</think> blocks removed, stripped of
        leading/trailing whitespace.
    """
    if text is None:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
