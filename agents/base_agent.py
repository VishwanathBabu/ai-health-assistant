"""
agents/base_agent.py
====================
Abstract base class that every agent in the system must inherit.

Enforces:
  - Standard run() interface
  - Timeout handling
  - Structured error output (never a crash)
  - Type-annotated input/output
"""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any

from core.config import settings, LLMProvider
from core.logger import AgentLogger, get_request_id


class AgentError(Exception):
    """Raised when an agent fails in a non-recoverable way."""
    pass


class BaseAgent(ABC):
    """
    Abstract base for all health assistant agents.

    Subclasses MUST implement:
      - name: str  (class attribute)
      - _build_prompt(input_data) -> str
      - _parse_response(raw: str) -> dict
      - _fallback_response() -> dict

    Subclasses MUST NOT:
      - Access the LLM directly — use self._call_llm()
      - Raise exceptions from run() — errors must be caught and returned safely
    """

    name: str = "base_agent"

    def __init__(self) -> None:
        self._validate_config()

    def _validate_config(self) -> None:
        """Fail fast at startup if configuration is missing."""
        if not settings.active_api_key:
            raise AgentError(
                f"[{self.name}] No API key configured for provider "
                f"'{settings.llm_provider}'. Set it in .env."
            )
        # Ollama-specific: warn if the base URL looks wrong
        if settings.llm_provider == LLMProvider.OLLAMA:
            url = settings.ollama_base_url
            if not url.startswith("http"):
                raise AgentError(
                    f"[{self.name}] OLLAMA_BASE_URL must start with http/https. "
                    f"Got: '{url}'"
                )

    # ── Public interface ─────────────────────────────────────────────────────

    async def run(
        self,
        input_data: dict[str, Any],
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Main entry point. Always returns a dict — never raises.

        On any failure, returns self._fallback_response() so the pipeline
        can continue safely.
        """
        rid = request_id or get_request_id()

        with AgentLogger(self.name, request_id=rid) as agent_log:
            try:
                prompt = self._build_prompt(input_data)
                raw = await asyncio.wait_for(
                    self._call_llm(prompt),
                    timeout=settings.agent_timeout_seconds,
                )
                result = self._parse_response(raw)
                agent_log.set_result(result)
                return result

            except asyncio.TimeoutError:
                agent_log.set_error(
                    TimeoutError(
                        f"Agent '{self.name}' timed out after "
                        f"{settings.agent_timeout_seconds}s"
                    )
                )
                return self._fallback_response()

            except Exception as exc:  # noqa: BLE001
                agent_log.set_error(exc)
                return self._fallback_response()

    # ── Abstract interface ───────────────────────────────────────────────────

    @abstractmethod
    def _build_prompt(self, input_data: dict[str, Any]) -> str:
        """
        Construct the full prompt string to send to the LLM.
        Include the system prompt + formatted user content.
        """

    @abstractmethod
    def _parse_response(self, raw: str) -> dict[str, Any]:
        """
        Parse the LLM's raw string response into a structured dict.
        Must never raise — return _fallback_response() on parse failure.
        """

    @abstractmethod
    def _fallback_response(self) -> dict[str, Any]:
        """
        Safe default output used when the agent fails or times out.
        Must be conservative (e.g. assume worst case for emergency agent).
        """

    # ── LLM call (provider-abstracted) ──────────────────────────────────────

    async def _call_llm(self, prompt: str) -> str:
        """
        Dispatch to the correct LLM backend based on settings.llm_provider.
        Returns the raw text response string.
        """
        if settings.llm_provider == LLMProvider.OPENAI:
            return await self._call_openai(prompt)
        if settings.llm_provider == LLMProvider.ANTHROPIC:
            return await self._call_anthropic(prompt)
        # LLMProvider.OLLAMA
        return await self._call_ollama(prompt)

    # ── Provider implementations ─────────────────────────────────────────────

    async def _call_openai(self, prompt: str) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=settings.active_api_key)
        response = await client.chat.completions.create(
            model=settings.active_model,
            temperature=settings.agent_temperature,
            max_tokens=settings.agent_max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    async def _call_anthropic(self, prompt: str) -> str:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=settings.active_api_key)
        message = await client.messages.create(
            model=settings.active_model,
            temperature=settings.agent_temperature,
            max_tokens=settings.agent_max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text if message.content else ""

    async def _call_ollama(self, prompt: str) -> str:
        """
        Call the local Ollama server via its HTTP API.

        Uses httpx (already in requirements.txt) for async HTTP.
        Ollama's /api/generate endpoint with stream=false returns a single
        JSON object whose 'response' field contains the model's reply.

        Ollama API reference: https://github.com/ollama/ollama/blob/main/docs/api.md
        """
        import httpx

        url = f"{settings.ollama_base_url.rstrip('/')}/api/generate"
        payload = {
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False,  # get one complete JSON response
            "options": {
                "temperature": settings.agent_temperature,
                "num_predict": settings.agent_max_tokens,  # Ollama's name for max_tokens
            },
        }

        async with httpx.AsyncClient(timeout=settings.agent_timeout_seconds) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()  # surface HTTP errors (4xx/5xx) as exceptions
            data = resp.json()

        # Ollama returns {"response": "...", "done": true, ...}
        text = data.get("response", "").strip()
        if not text:
            raise ValueError(
                f"Ollama returned an empty response for model '{settings.ollama_model}'. "
                f"Full payload: {data}"
            )
        return text

    # ── JSON parsing helper ──────────────────────────────────────────────────

    def _extract_json(self, raw: str) -> dict[str, Any]:
        """
        Robustly extract a JSON object from an LLM response.
        Handles markdown fences (```json ... ```) and leading/trailing text.
        """
        cleaned = raw.strip()
        if "```" in cleaned:
            parts = cleaned.split("```")
            for part in parts:
                candidate = part.strip().lstrip("json").strip()
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue

        # Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Last attempt: find first { ... } block
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not extract JSON from response: {raw[:200]}")
