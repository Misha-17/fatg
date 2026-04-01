"""
FATG — Finnish Adaptive Tiered Generation

Hardware-adaptive LLM inference framework for Finnish language
learning content generation. Runs on Apple Silicon, NVIDIA GPUs,
and CPU-only machines via Ollama.

Quick start:
    pip install fatg
    ollama serve  # in a separate terminal

    import asyncio
    from fatg import FATGEngine

    async def main():
        engine = await FATGEngine.create()
        print(engine.hardware)

        q = await engine.generate_quest_question(
            target_word="kahvia",
            scenario="cafe_order",
            difficulty=0.1,
        )
        print(q.question_fi)
        print(q.options)

    asyncio.run(main())
"""

from fatg.engine import FATGEngine
from fatg.config import FATGConfig
from fatg.hardware import detect as detect_hardware, HardwareProfile, Backend, ModelTier
from fatg.tiers.llm import QuestQuestion

__all__ = [
    "FATGEngine",
    "FATGConfig",
    "detect_hardware",
    "HardwareProfile",
    "Backend",
    "ModelTier",
    "QuestQuestion",
]

__version__ = "0.1.1"
