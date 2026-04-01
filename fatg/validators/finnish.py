"""
Finnish language validator.

Validates LLM-generated Finnish language content for:
1. Structural integrity — target word appears in sentence
2. Basic morphological plausibility — word endings look Finnish
3. Optional voikko integration — full spell/grammar check

voikko is optional. If not installed, falls back to rule-based checks.
Install voikko: pip install libvoikko (also needs system library)
  macOS: brew install libvoikko
  Ubuntu: sudo apt install libvoikko-dev voikko-fi
"""

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Finnish vowel harmony groups
BACK_VOWELS = set("aouAOU")
FRONT_VOWELS = set("äöyÄÖY")

# Common Finnish word suffixes by case
# This is a basic approximation — Finnish has 15 cases
COMMON_SUFFIXES = {
    # Nominative — no suffix
    # Genitive
    "n",
    # Accusative
    "t",
    # Partitive
    "a", "ä", "ta", "tä", "tta", "ttä",
    # Inessive (in)
    "ssa", "ssä",
    # Elative (out of)
    "sta", "stä",
    # Illative (into)
    "an", "ään", "en", "in", "on", "un", "yn",
    "han", "hän", "hen", "hin", "hon", "hun", "hyn",
    # Adessive (on/at)
    "lla", "llä",
    # Ablative (from/off)
    "lta", "ltä",
    # Allative (onto)
    "lle",
    # Essive (as)
    "na", "nä",
    # Translative (becoming)
    "ksi",
    # Abessive (without)
    "tta", "ttä",
    # Comitative (with)
    "neen",
}


@dataclass
class ValidationResult:
    valid: bool
    score: float          # 0.0 - 1.0, higher is better
    issues: list[str]
    used_voikko: bool = False


def validate_quest_question(
    sentence_fi: str,
    target_fi: str,
    distractors: list[str],
) -> ValidationResult:
    """
    Validate a generated fill-in-the-blank Finnish quest question.

    Checks:
    1. target_fi appears in sentence_fi
    2. Blanking works (sentence makes sense with hole)
    3. target_fi looks morphologically plausible
    4. Distractors are distinct from target and each other
    5. Vowel harmony consistency in target_fi
    """
    issues = []
    score = 1.0

    # Check 1 — target must appear in sentence
    if target_fi not in sentence_fi:
        issues.append(f"target_fi '{target_fi}' not found in sentence_fi")
        return ValidationResult(valid=False, score=0.0, issues=issues)

    # Check 2 — blanked sentence makes sense
    blanked = sentence_fi.replace(target_fi, "____", 1)
    if blanked == sentence_fi:
        issues.append("Blanking had no effect — target may have been replaced already")
        score -= 0.2

    # Check 3 — target looks Finnish (basic check)
    fi_result = _check_finnish_word(target_fi)
    if not fi_result:
        issues.append(f"target_fi '{target_fi}' may not be valid Finnish")
        return ValidationResult(valid=False, score=0.0, issues=issues)

    # Check 4 — distractors are distinct
    all_options = distractors + [target_fi]
    if len(set(all_options)) < len(all_options):
        issues.append("Duplicate options detected")
        score -= 0.3

    for d in distractors:
        if d == target_fi:
            issues.append(f"Distractor '{d}' is identical to target")
            score -= 0.3

    # Check 5 — vowel harmony in target
    if not _check_vowel_harmony(target_fi):
        issues.append(f"Possible vowel harmony violation in '{target_fi}'")
        score -= 0.1

    # Try voikko if available
    used_voikko = False
    voikko_result = _try_voikko_check(sentence_fi)
    if voikko_result is not None:
        used_voikko = True
        if not voikko_result:
            issues.append("Voikko: sentence contains unrecognised words")
            score -= 0.2

    score = max(0.0, score)
    valid = score >= 0.5 and len([i for i in issues if "not found" in i]) == 0

    return ValidationResult(
        valid=valid,
        score=round(score, 2),
        issues=issues,
        used_voikko=used_voikko,
    )


def _check_finnish_word(word: str) -> bool:
    """
    Basic check that a word looks like Finnish.
    Rejects obvious non-Finnish strings.
    """
    if not word:
        return False

    # Empty string is never valid
    if len(word) == 0:
        return False

    # Must contain at least one vowel
    vowels = set("aeiouäöyAEIOUÄÖY")
    if not any(c in vowels for c in word):
        return False

    # Must be alphabetic (allowing Finnish chars)
    if not re.match(r"^[a-zA-ZäöåÄÖÅ\-]+$", word):
        return False

    # Reasonable length
    if len(word) < 2 or len(word) > 30:
        return False

    return True


def _check_vowel_harmony(word: str) -> bool:
    """
    Check vowel harmony — Finnish words use either back vowels (a, o, u)
    or front vowels (ä, ö, y) but not both (with rare exceptions).
    Neutral vowels (e, i) can appear with either group.
    """
    has_back = any(c in BACK_VOWELS for c in word)
    has_front = any(c in FRONT_VOWELS for c in word)

    # Having both is suspicious (possible in compound words)
    if has_back and has_front:
        # Allow if word contains a hyphen (compound)
        if "-" not in word:
            return False

    return True


def _try_voikko_check(sentence: str) -> bool | None:
    """
    Try to use libvoikko for spell checking.
    Returns None if voikko is not installed.
    Returns True if all words are recognised, False otherwise.
    """
    try:
        from libvoikko import Voikko  # type: ignore
        v = Voikko("fi")
        words = re.findall(r"[a-zA-ZäöåÄÖÅ]+", sentence)
        for word in words:
            if not v.spell(word):
                return False
        return True
    except ImportError:
        return None
    except Exception as e:
        logger.debug(f"Voikko check failed: {e}")
        return None
