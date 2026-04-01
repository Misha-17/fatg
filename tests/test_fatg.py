"""
Tests for FATG.

Run with: pytest tests/ -v
Note: LLM tests require Ollama running locally.
"""

import pytest
from fatg.hardware import detect, Backend
from fatg.validators.finnish import validate_quest_question
from fatg.config import FATGConfig


# ── Hardware detection ────────────────────────────────────────────────────────

def test_hardware_detection_returns_profile():
    profile = detect()
    assert profile.backend in Backend
    assert profile.ram_gb > 0
    assert profile.recommended_model


def test_hardware_profile_str():
    profile = detect()
    s = str(profile)
    assert "Backend:" in s
    assert "RAM:" in s


# ── Finnish validator ─────────────────────────────────────────────────────────

def test_validator_passes_correct_input():
    result = validate_quest_question(
        sentence_fi="Haluaisin kupillisen kahvia, kiitos.",
        target_fi="kahvia",
        distractors=["teetä", "vettä", "maitoa"],
    )
    assert result.valid
    assert result.score >= 0.5
    assert len(result.issues) == 0


def test_validator_fails_when_target_missing():
    result = validate_quest_question(
        sentence_fi="Haluaisin kupillisen teetä, kiitos.",
        target_fi="kahvia",  # not in sentence
        distractors=["teetä", "vettä", "maitoa"],
    )
    assert not result.valid
    assert result.score == 0.0
    assert any("not found" in issue for issue in result.issues)


def test_validator_fails_duplicate_options():
    result = validate_quest_question(
        sentence_fi="Haluaisin kupillisen kahvia, kiitos.",
        target_fi="kahvia",
        distractors=["kahvia", "vettä", "maitoa"],  # duplicate
    )
    assert not result.valid or result.score < 0.8


def test_validator_catches_vowel_harmony_violation():
    result = validate_quest_question(
        sentence_fi="Menen kauppaan ostamaan maitoa.",
        target_fi="maitoa",
        distractors=["kahvia", "vettä", "teetä"],
    )
    # maitoa is valid Finnish, should pass
    assert result.valid


def test_validator_flags_numeric_target():
    # Numbers clearly not Finnish
    result = validate_quest_question(
        sentence_fi="Haluaisin kupillisen 123, kiitos.",
        target_fi="123",
        distractors=["teetae", "vetta", "maitoa"],
    )
    assert not result.valid or result.score < 1.0

def test_validator_flags_empty_target():
    result = validate_quest_question(
        sentence_fi="Haluaisin kupillisen kahvia.",
        target_fi="",
        distractors=["teetae", "vetta", "maitoa"],
    )
    assert not result.valid


# ── Config ────────────────────────────────────────────────────────────────────

def test_default_config():
    config = FATGConfig()
    assert config.ollama_host == "http://localhost:11434"
    assert config.max_retries == 3
    assert config.enable_finnish_validation is True


def test_config_override():
    config = FATGConfig(
        verify_model="phi3.5:3.8b",
        temperature=0.5,
        max_retries=1,
    )
    assert config.verify_model == "phi3.5:3.8b"
    assert config.temperature == 0.5


# ── Engine (requires Ollama) ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_engine_fails_without_ollama():
    """Engine should raise RuntimeError if Ollama is not running."""
    from fatg import FATGEngine, FATGConfig

    config = FATGConfig(
        ollama_host="http://localhost:19999",  # wrong port
    )
    with pytest.raises(RuntimeError, match="Ollama is not running"):
        await FATGEngine.create(config=config, auto_pull=False)


@pytest.mark.asyncio
@pytest.mark.skipif(
    True,  # set to False if Ollama is running locally
    reason="Requires Ollama running locally with a model pulled"
)
async def test_generate_quest_question():
    from fatg import FATGEngine

    engine = await FATGEngine.create(auto_pull=False)
    q = await engine.generate_quest_question(
        target_word="kahvia",
        scenario="cafe_order",
        difficulty=0.1,
    )

    assert "kahvia" in q.sentence_fi or q.target_fi
    assert len(q.options) == 4
    assert q.question_fi != q.sentence_fi  # blanking happened
    assert q.validation is not None
    assert q.validation.valid
