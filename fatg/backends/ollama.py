"""
Ollama backend.

Talks to a local Ollama server via its REST API.
Ollama handles hardware selection automatically:
  - Apple Silicon → Metal (GPU)
  - NVIDIA        → CUDA
  - CPU fallback  → llama.cpp CPU

Install Ollama: https://ollama.com
"""

import json
import logging
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)


class OllamaBackend:
    """Async client for the Ollama REST API."""

    def __init__(self, host: str = "http://localhost:11434", timeout: float = 30.0):
        self.host = host.rstrip("/")
        self.timeout = timeout

    async def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.host}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """Return list of locally available model names."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.host}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]

    async def is_model_available(self, model: str) -> bool:
        """Check if a specific model is pulled locally."""
        available = await self.list_models()
        # Match by prefix (e.g. "qwen2.5:1.5b" matches "qwen2.5:1.5b-instruct-q4_K_M")
        model_base = model.split(":")[0]
        for m in available:
            if m == model or m.startswith(model_base):
                return True
        return False

    async def pull_model(self, model: str) -> None:
        """Pull a model from Ollama registry (blocking until done)."""
        logger.info(f"Pulling model {model} from Ollama registry...")
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"{self.host}/api/pull",
                json={"name": model},
            ) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        data = json.loads(line)
                        status = data.get("status", "")
                        if "pulling" in status or "downloading" in status:
                            logger.debug(f"  {status}")

    async def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> str:
        """
        Generate a completion using Ollama.

        If json_mode=True, Ollama will constrain output to valid JSON.
        This is the key feature for structured generation.
        """
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system:
            payload["system"] = system

        if json_mode:
            payload["format"] = "json"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.host}/api/generate",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()

    async def generate_structured(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> dict:
        """
        Generate and return parsed JSON.
        Raises ValueError if output is not valid JSON.
        """
        raw = await self.generate(
            model=model,
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )

        # Strip markdown fences if model wraps output
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1])

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Model returned invalid JSON: {e}\nRaw output: {raw[:200]}") from e
