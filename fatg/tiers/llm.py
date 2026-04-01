"""
LLM generation tier.

Handles structured content generation using the Ollama backend.
For Lingo Deck, generates fill-in-the-blank Finnish language questions
in the correct JSON schema, with retry logic and validation.
"""

import json
import logging
from dataclasses import dataclass

from fatg.backends.ollama import OllamaBackend
from fatg.validators.finnish import ValidationResult, validate_quest_question

logger = logging.getLogger(__name__)


@dataclass
class QuestQuestion:
    """A generated fill-in-the-blank Finnish question."""
    sentence_fi: str        # full Finnish sentence
    sentence_en: str        # full English translation
    target_fi: str          # word being tested (blanked out)
    target_en: str          # English translation of target
    question_fi: str        # sentence_fi with target replaced by ....
    question_en: str        # sentence_en with target replaced by ....
    distractor_1_fi: str
    distractor_2_fi: str
    distractor_3_fi: str
    difficulty: float       # 0.0-1.0
    scenario: str
    validation: ValidationResult | None = None

    @property
    def options(self) -> list[str]:
        """All 4 options shuffled — caller should shuffle before returning to user."""
        return [self.target_fi, self.distractor_1_fi, self.distractor_2_fi, self.distractor_3_fi]


QUEST_SYSTEM_PROMPT = """You are a Finnish language learning content generator.
You generate fill-in-the-blank sentences for language learners.
You always respond with valid JSON only. No explanation, no markdown, just JSON."""

QUEST_PROMPT_TEMPLATE = """Create a Finnish language exercise. Use the word "{target_word}" in a sentence.

Context: {scenario}
Difficulty: {difficulty} (0.0=easy, 1.0=hard)

IMPORTANT: The value of "target_fi" MUST appear word-for-word inside "sentence_fi".

Return ONLY this JSON:
{{
  "sentence_fi": "a Finnish sentence containing {target_word}",
  "sentence_en": "English translation of the sentence",
  "target_fi": "{target_word}",
  "target_en": "English meaning of {target_word}",
  "distractor_1_fi": "wrong Finnish word 1",
  "distractor_2_fi": "wrong Finnish word 2",
  "distractor_3_fi": "wrong Finnish word 3"
}}"""

KELA_SYSTEM_PROMPT = """You are a Finnish language learning content generator specialised
in bureaucratic Finnish as used in KELA (the Finnish Social Insurance Institution).
KELA is known for formal register, passive voice, partitive case, and compound nouns.
You always respond with valid JSON only. No explanation, no markdown, just JSON."""

KELA_PROMPT_TEMPLATE = """Generate a KELA-themed fill-in-the-blank Finnish question.

The player's card word: {card_word}
This word should appear (possibly inflected) in a KELA office context.

Scenario context: KELA office — filling forms, asking about benefits, bureaucratic Finnish

Rules:
1. Write a realistic KELA-office Finnish sentence containing the word
2. Use the grammatically correct inflected form for the sentence
3. target_fi must appear EXACTLY in sentence_fi
4. 3 distractors must be real Finnish words, plausible but wrong
5. Make it feel authentically bureaucratic (passive voice welcome)

Respond ONLY with this JSON:
{{
  "sentence_fi": "Finnish KELA sentence",
  "sentence_en": "English translation",
  "target_fi": "exact word/form to blank",
  "target_en": "English of target",
  "distractor_1_fi": "wrong option 1",
  "distractor_2_fi": "wrong option 2",
  "distractor_3_fi": "wrong option 3"
}}"""


class LLMTier:
    """Handles LLM-based question generation with validation and retry."""

    def __init__(
        self,
        backend: OllamaBackend,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        max_retries: int = 3,
        enable_validation: bool = True,
    ):
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.enable_validation = enable_validation

    async def generate_quest_question(
        self,
        target_word: str,
        scenario: str = "general",
        difficulty: float = 0.3,
    ) -> QuestQuestion:
        """
        Generate a fill-in-the-blank quest question for a given Finnish word.

        Retries up to max_retries times if validation fails.
        """
        is_kela = scenario == "kela_boss"

        system = KELA_SYSTEM_PROMPT if is_kela else QUEST_SYSTEM_PROMPT
        prompt = (
            KELA_PROMPT_TEMPLATE.format(card_word=target_word)
            if is_kela
            else QUEST_PROMPT_TEMPLATE.format(
                target_word=target_word,
                scenario=scenario,
                difficulty=difficulty,
            )
        )

        last_error: str = ""

        for attempt in range(self.max_retries):
            try:
                raw = await self.backend.generate_structured(
                    model=self.model,
                    prompt=prompt,
                    system=system,
                    temperature=self.temperature + (attempt * 0.1),  # increase temp on retry
                    max_tokens=self.max_tokens,
                )

                question = self._parse_raw(raw, scenario, difficulty)

                if self.enable_validation:
                    validation = validate_quest_question(
                        sentence_fi=question.sentence_fi,
                        target_fi=question.target_fi,
                        distractors=[
                            question.distractor_1_fi,
                            question.distractor_2_fi,
                            question.distractor_3_fi,
                        ],
                    )
                    question.validation = validation

                    if not validation.valid:
                        last_error = f"Validation failed: {validation.issues}"
                        logger.warning(
                            f"Attempt {attempt + 1}/{self.max_retries} — {last_error}"
                        )
                        continue

                logger.debug(
                    f"Generated question for '{target_word}' "
                    f"in {attempt + 1} attempt(s)"
                )
                return question

            except (ValueError, KeyError) as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} — {e}")

        raise RuntimeError(
            f"Failed to generate valid question for '{target_word}' "
            f"after {self.max_retries} attempts. Last error: {last_error}"
        )

    def _parse_raw(self, raw: dict, scenario: str, difficulty: float) -> QuestQuestion:
        """Parse raw LLM JSON into a QuestQuestion."""
        required = [
            "sentence_fi", "sentence_en", "target_fi", "target_en",
            "distractor_1_fi", "distractor_2_fi", "distractor_3_fi",
        ]
        for key in required:
            if key not in raw:
                raise KeyError(f"Missing required field: {key}")

        sentence_fi = raw["sentence_fi"]
        sentence_en = raw["sentence_en"]
        target_fi = raw["target_fi"]
        target_en = raw["target_en"]

        question_fi = sentence_fi.replace(target_fi, "....", 1)
        question_en = sentence_en.replace(target_en, "....", 1)

        return QuestQuestion(
            sentence_fi=sentence_fi,
            sentence_en=sentence_en,
            target_fi=target_fi,
            target_en=target_en,
            question_fi=question_fi,
            question_en=question_en,
            distractor_1_fi=raw["distractor_1_fi"],
            distractor_2_fi=raw["distractor_2_fi"],
            distractor_3_fi=raw["distractor_3_fi"],
            difficulty=difficulty,
            scenario=scenario,
        )
