"""
Hardware detection for FATG.

Detects the best available backend and recommends model sizes
based on available memory. Supports Apple Silicon (M1/M2/M3/M4),
NVIDIA GPUs, and CPU-only fallback.
"""

import platform
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum


class Backend(str, Enum):
    APPLE_SILICON = "apple_silicon"
    NVIDIA = "nvidia"
    CPU = "cpu"


class ModelTier(str, Enum):
    TINY = "tiny"      # 1.5B  — fastest, least accurate
    SMALL = "small"    # 3.8B  — good balance
    MEDIUM = "medium"  # 7B    — best quality, needs more RAM


# Recommended Ollama model tags per tier
TIER_MODELS = {
    ModelTier.TINY: "qwen2.5:1.5b",
    ModelTier.SMALL: "phi3.5:3.8b",
    ModelTier.MEDIUM: "qwen2.5:7b",
}


@dataclass
class HardwareProfile:
    backend: Backend
    ram_gb: float
    recommended_tier: ModelTier
    recommended_model: str
    can_run_medium: bool
    notes: str

    def __str__(self) -> str:
        return (
            f"Backend: {self.backend.value} | "
            f"RAM: {self.ram_gb:.1f}GB | "
            f"Recommended: {self.recommended_model} ({self.recommended_tier.value}) | "
            f"{self.notes}"
        )


def detect() -> HardwareProfile:
    """
    Auto-detect hardware and return a HardwareProfile with
    model recommendations.
    """
    system = platform.system()
    machine = platform.machine()

    # Apple Silicon detection
    if system == "Darwin" and machine == "arm64":
        return _profile_apple_silicon()

    # NVIDIA detection
    nvidia = _detect_nvidia()
    if nvidia:
        vram_gb, notes = nvidia
        return _profile_nvidia(vram_gb, notes)

    # CPU fallback
    ram_gb = _get_system_ram_gb()
    return HardwareProfile(
        backend=Backend.CPU,
        ram_gb=ram_gb,
        recommended_tier=ModelTier.TINY,
        recommended_model=TIER_MODELS[ModelTier.TINY],
        can_run_medium=False,
        notes="CPU only — use tiny model for acceptable speed (~5-8 tok/s)",
    )


def _profile_apple_silicon() -> HardwareProfile:
    ram_gb = _get_system_ram_gb()

    # M1/M2 Air 8GB → tiny or small
    # M1/M2 Air 16GB → small or medium
    # M1 Pro/Max 16GB+ → medium
    if ram_gb >= 16:
        tier = ModelTier.MEDIUM
        can_run_medium = True
        notes = f"Apple Silicon {ram_gb:.0f}GB — use MLX for +30-50% speed (pip install mlx-lm)"
    elif ram_gb >= 8:
        tier = ModelTier.SMALL
        can_run_medium = False
        notes = f"Apple Silicon {ram_gb:.0f}GB — phi3.5 3.8B fits comfortably (~25 tok/s)"
    else:
        tier = ModelTier.TINY
        can_run_medium = False
        notes = f"Apple Silicon {ram_gb:.0f}GB — use tiny model to avoid swap"

    return HardwareProfile(
        backend=Backend.APPLE_SILICON,
        ram_gb=ram_gb,
        recommended_tier=tier,
        recommended_model=TIER_MODELS[tier],
        can_run_medium=can_run_medium,
        notes=notes,
    )


def _profile_nvidia(vram_gb: float, gpu_notes: str) -> HardwareProfile:
    ram_gb = _get_system_ram_gb()

    if vram_gb >= 8:
        tier = ModelTier.MEDIUM
        can_run_medium = True
        notes = f"{gpu_notes} — 7B Q4 fits in VRAM (~40-70 tok/s)"
    elif vram_gb >= 6:
        tier = ModelTier.SMALL
        can_run_medium = False
        notes = f"{gpu_notes} — phi3.5 3.8B Q4 recommended (~30-40 tok/s)"
    else:
        tier = ModelTier.TINY
        can_run_medium = False
        notes = f"{gpu_notes} — low VRAM, tiny model only"

    return HardwareProfile(
        backend=Backend.NVIDIA,
        ram_gb=ram_gb,
        recommended_tier=tier,
        recommended_model=TIER_MODELS[tier],
        can_run_medium=can_run_medium,
        notes=notes,
    )


def _detect_nvidia() -> tuple[float, str] | None:
    """Try to detect NVIDIA GPU and return (vram_gb, label)."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        lines = result.stdout.strip().split("\n")
        if not lines:
            return None

        # Take the first GPU
        parts = lines[0].split(",")
        if len(parts) < 2:
            return None

        name = parts[0].strip()
        vram_mb = float(parts[1].strip())
        return vram_mb / 1024, name

    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


def _get_system_ram_gb() -> float:
    """Get total system RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        # Fallback without psutil
        system = platform.system()
        if system == "Darwin":
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=3
                )
                return int(result.stdout.strip()) / (1024 ** 3)
            except Exception:
                pass
        return 8.0  # safe default assumption
