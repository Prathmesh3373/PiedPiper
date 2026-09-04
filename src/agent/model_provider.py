"""Gemini model factory. The application has no OpenAI fallback."""

from __future__ import annotations

import os


def _timeout_seconds() -> float:
    try:
        return max(5.0, float(os.getenv("TOURISM_LLM_TIMEOUT_SECONDS", "30")))
    except ValueError:
        return 30.0


def provider_name() -> str:
    return (os.getenv("TOURISM_LLM_PROVIDER") or "gemini").strip().lower()


def llm_enabled() -> bool:
    if os.environ.get("TOURISM_LLM_BRAIN", "1") != "1":
        return False
    return provider_name() == "gemini" and bool((os.getenv("GEMINI_API_KEY") or "").strip())


def _base_model_kwargs(*, temperature: float, max_tokens: int) -> dict:
    """Common kwargs shared by brain and writer models."""
    return {
        "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "api_key": os.getenv("GEMINI_API_KEY"),
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "timeout": _timeout_seconds(),
        "max_retries": 2,
    }


def make_chat_model(*, temperature: float, max_tokens: int):
    """Writer / slot model — plain text generation, no tool binding.
    Used by compose.py (answer writer) and slots.py (slot extraction).
    """
    if provider_name() != "gemini":
        raise RuntimeError("TOURISM_LLM_PROVIDER must be 'gemini'; no other LLM is supported.")
    if not llm_enabled():
        raise RuntimeError("GEMINI_API_KEY is required to run the tourism agent.")
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(**_base_model_kwargs(temperature=temperature, max_tokens=max_tokens))


def make_brain_model(*, max_tokens: int = 1200):
    """Brain / tool-calling model — temperature=0, thinking disabled for fast tool dispatch.

    thinking_budget=0 tells Gemini 2.5 flash NOT to do extended reasoning before
    emitting tool calls. Without this, the model thinks through the whole problem
    internally and returns plain prose instead of tool calls — which breaks our
    manual tool-dispatch loop entirely.
    """
    if provider_name() != "gemini":
        raise RuntimeError("TOURISM_LLM_PROVIDER must be 'gemini'; no other LLM is supported.")
    if not llm_enabled():
        raise RuntimeError("GEMINI_API_KEY is required to run the tourism agent.")
    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs = _base_model_kwargs(temperature=0, max_tokens=max_tokens)
    # Only add thinking_budget if the model supports it (2.5-flash full does, lite may not).
    model_name = (os.getenv("GEMINI_MODEL", "gemini-2.5-flash") or "").lower()
    if "2.5-flash" in model_name and "lite" not in model_name:
        kwargs["thinking_budget"] = 0
    return ChatGoogleGenerativeAI(**kwargs)
