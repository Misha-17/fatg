"""
FATG configuration.

You can override any setting by passing a FATGConfig to FATGEngine.create().
"""

from dataclasses import dataclass, field


@dataclass
class FATGConfig:
    # Ollama server
    ollama_host: str = "http://localhost:11434"
    ollama_timeout: float = 30.0

    # Model selection
    # If None, auto-selected based on hardware detection
    draft_model: str | None = None   # tier 2 — tiny model
    verify_model: str | None = None  # tier 3 — medium model

    # Generation settings
    temperature: float = 0.7
    max_tokens: int = 512

    # Retry settings
    max_retries: int = 3

    # Finnish validator
    enable_finnish_validation: bool = True

    # Cache settings
    enable_cache: bool = True
    cache_max_size: int = 1000

    # Logging
    verbose: bool = False
