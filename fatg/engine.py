"""
FATGEngine — Finnish Adaptive Tiered Generation Engine.

Main entry point. Auto-detects hardware, selects models,
and provides a simple async API for generating Finnish
language learning content.

Usage:
    from fatg import FATGEngine

    engine = await FATGEngine.create()
    print(engine.hardware)  # see what was detected

    question = await engine.generate_quest_question(
        target_word="kahvia",
        scenario="cafe_order",
        difficulty=0.1,
    )
    print(question.question_fi)
    print(question.options)
"""

import logging
from typing import TYPE_CHECKING

from fatg.backends.ollama import OllamaBackend
from fatg.config import FATGConfig
from fatg.hardware import HardwareProfile, ModelTier, TIER_MODELS, detect
from fatg.tiers.llm import LLMTier, QuestQuestion

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class FATGEngine:
    """
    Hardware-adaptive tiered generation engine.

    Always use FATGEngine.create() — not FATGEngine() directly —
    because setup involves async checks (Ollama availability, model pulling).
    """

    def __init__(
        self,
        hardware: HardwareProfile,
        backend: OllamaBackend,
        tier: LLMTier,
        config: FATGConfig,
    ):
        self.hardware = hardware
        self._backend = backend
        self._tier = tier
        self._config = config

    @classmethod
    async def create(
        cls,
        config: FATGConfig | None = None,
        auto_pull: bool = True,
    ) -> "FATGEngine":
        """
        Create and initialise a FATGEngine.

        Args:
            config: Optional config overrides. If None, uses smart defaults.
            auto_pull: If True, automatically pulls the recommended model
                       from Ollama if it isn't already downloaded.

        Raises:
            RuntimeError: If Ollama is not running or reachable.
        """
        if config is None:
            config = FATGConfig()

        # Detect hardware
        hardware = detect()
        logger.info(f"Hardware detected: {hardware}")

        # Set up Ollama backend
        backend = OllamaBackend(
            host=config.ollama_host,
            timeout=config.ollama_timeout,
        )

        # Check Ollama is running
        if not await backend.is_available():
            raise RuntimeError(
                f"Ollama is not running at {config.ollama_host}.\n"
                "Start it with: ollama serve\n"
                "Install from: https://ollama.com"
            )

        # Pick model based on hardware or config override
        model = config.verify_model or TIER_MODELS[hardware.recommended_tier]

        # Auto-pull if needed
        if auto_pull:
            if not await backend.is_model_available(model):
                logger.info(f"Model {model} not found locally. Pulling...")
                print(f"[FATG] Pulling {model} — this may take a few minutes...")
                await backend.pull_model(model)
                print(f"[FATG] {model} ready.")

        # Build LLM tier
        tier = LLMTier(
            backend=backend,
            model=model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            max_retries=config.max_retries,
            enable_validation=config.enable_finnish_validation,
        )

        engine = cls(
            hardware=hardware,
            backend=backend,
            tier=tier,
            config=config,
        )

        logger.info(
            f"FATGEngine ready — model: {model} | "
            f"backend: {hardware.backend.value}"
        )

        return engine

    async def generate_quest_question(
        self,
        target_word: str,
        scenario: str = "general",
        difficulty: float = 0.3,
    ) -> QuestQuestion:
        """
        Generate a fill-in-the-blank Finnish quest question.

        Args:
            target_word: The Finnish word/phrase to build the question around.
            scenario: One of "cafe_order", "job_interview", "asking_directions",
                      "kela_boss", or "general".
            difficulty: Float 0.0-1.0 controlling sentence complexity.

        Returns:
            QuestQuestion with sentence, blanked version, options, and validation result.
        """
        return await self._tier.generate_quest_question(
            target_word=target_word,
            scenario=scenario,
            difficulty=difficulty,
        )

    async def generate_kela_questions(
        self,
        deck: list[dict],
    ) -> list[QuestQuestion]:
        """
        Generate KELA boss fight questions for a player's deck.

        Args:
            deck: List of card dicts, each with at least a "word_fi" key.

        Returns:
            List of QuestQuestion objects, one per card.
        """
        questions = []
        for card in deck:
            word = card.get("word_fi") or card.get("target_fi", "")
            if not word:
                continue
            try:
                q = await self._tier.generate_quest_question(
                    target_word=word,
                    scenario="kela_boss",
                    difficulty=0.9,
                )
                questions.append(q)
            except RuntimeError as e:
                logger.warning(f"Skipping card '{word}': {e}")

        return questions

    @property
    def model(self) -> str:
        """The model currently in use."""
        return self._tier.model

    def info(self) -> dict:
        """Return engine configuration summary."""
        return {
            "hardware": {
                "backend": self.hardware.backend.value,
                "ram_gb": self.hardware.ram_gb,
                "recommended_tier": self.hardware.recommended_tier.value,
                "notes": self.hardware.notes,
            },
            "model": self.model,
            "config": {
                "temperature": self._config.temperature,
                "max_tokens": self._config.max_tokens,
                "max_retries": self._config.max_retries,
                "finnish_validation": self._config.enable_finnish_validation,
            },
        }
