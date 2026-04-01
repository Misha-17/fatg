# FATG — Finnish Adaptive Tiered Generation

Hardware-adaptive LLM inference framework for Finnish language learning content.
Runs on **Apple Silicon** (M1/M2/M3), **NVIDIA GPUs**, and **CPU-only** machines
via [Ollama](https://ollama.com).

Built for [Lingo Deck](https://github.com/Wirlaa/LingoDeck) — a Finnish language
learning card game.

## Features

- **Auto hardware detection** — detects M1/NVIDIA/CPU and picks the right model
- **Tiered generation** — tiny model for simple content, medium model for complex
- **Finnish morphological validation** — catches hallucinated inflections before they reach users
- **Structured JSON output** — Ollama's JSON mode ensures parseable responses every time
- **Optional voikko integration** — full spell/grammar checking if libvoikko is installed
- **Zero cloud dependency** — everything runs locally

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally (`ollama serve`)

## Install

```bash
pip install fatg
```

With voikko support (recommended for production):
```bash
pip install fatg[voikko]
# macOS: brew install libvoikko
# Ubuntu: sudo apt install libvoikko-dev voikko-fi
```

## Quick Start

```python
import asyncio
from fatg import FATGEngine

async def main():
    # Auto-detects hardware, pulls model if needed
    engine = await FATGEngine.create()

    # See what hardware was detected
    print(engine.hardware)
    # Backend: apple_silicon | RAM: 16.0GB | Recommended: qwen2.5:7b (medium)

    # Generate a quest question
    q = await engine.generate_quest_question(
        target_word="kahvia",
        scenario="cafe_order",
        difficulty=0.1,
    )

    print(q.question_fi)
    # "Haluaisin kupillisen ...., kiitos."

    print(q.question_en)
    # "I would like a cup of ...., please."

    print(q.options)
    # ["kahvia", "teetä", "vettä", "maitoa"]  (shuffled)

    print(q.validation.score)
    # 0.9

asyncio.run(main())
```

## KELA Boss Fight (LLM-generated questions from deck)

```python
deck = [
    {"word_fi": "tukea"},
    {"word_fi": "hakemusta"},
    {"word_fi": "liitteitä"},
]

questions = await engine.generate_kela_questions(deck)
for q in questions:
    print(q.question_fi)
    print(q.options)
```

## Hardware Performance

| Hardware | Model | Speed | 100-token response |
|---|---|---|---|
| M1 Air 8GB | phi3.5:3.8b | ~25 tok/s | ~4s |
| M1 Air 16GB | qwen2.5:7b | ~14 tok/s | ~7s |
| RTX 3060 6GB | phi3.5:3.8b | ~35 tok/s | ~3s |
| RTX 3060 8GB | qwen2.5:7b | ~45 tok/s | ~2s |
| CPU only | qwen2.5:1.5b | ~8 tok/s | ~12s |

## Config

```python
from fatg import FATGEngine, FATGConfig

config = FATGConfig(
    ollama_host="http://localhost:11434",
    verify_model="phi3.5:3.8b",   # override auto-selection
    temperature=0.7,
    max_retries=3,
    enable_finnish_validation=True,
)

engine = await FATGEngine.create(config=config)
```

## Scenarios

| Scenario | Description |
|---|---|
| `cafe_order` | Ordering at a Finnish café |
| `job_interview` | Job interview in Finnish |
| `asking_directions` | Asking for directions |
| `kela_boss` | KELA bureaucratic Finnish (hardest) |
| `general` | No specific context |

## License

MIT
