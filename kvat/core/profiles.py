"""
Built-in workload profiles for KVCache Auto-Tuner.

Provides realistic, production-tested profiles for common use cases:
- Chat/Agent: Low latency, short responses
- RAG: Large context, medium responses
- Longform: Extended generation
"""

from __future__ import annotations

import json
from pathlib import Path

from kvat.core.schema import ProfileType, WorkloadProfile

# =============================================================================
# Built-in Profiles
# =============================================================================

CHAT_AGENT_PROFILE = WorkloadProfile(
    name="chat-agent",
    profile_type=ProfileType.CHAT_AGENT,
    context_lengths=[2048, 4096, 8192],
    output_lengths=[64, 128, 256],
    system_prompt_tokens=2000,
    warmup_runs=2,
    measurement_runs=5,
    weight_ttft=0.50,       # TTFT is critical for chat
    weight_throughput=0.35,
    weight_memory=0.15,
)

RAG_PROFILE = WorkloadProfile(
    name="rag",
    profile_type=ProfileType.RAG,
    context_lengths=[8192, 16384, 32768],
    output_lengths=[256, 512],
    system_prompt_tokens=500,
    warmup_runs=2,
    measurement_runs=5,
    weight_ttft=0.35,
    weight_throughput=0.35,
    weight_memory=0.30,     # Memory matters for large contexts
)

LONGFORM_PROFILE = WorkloadProfile(
    name="longform",
    profile_type=ProfileType.LONGFORM,
    context_lengths=[4096, 8192],
    output_lengths=[1024, 2048],
    system_prompt_tokens=500,
    warmup_runs=2,
    measurement_runs=3,     # Fewer runs due to longer generation
    weight_ttft=0.25,
    weight_throughput=0.50, # Throughput critical for long generation
    weight_memory=0.25,
)

# Quick profile for fast CI testing
CI_MICRO_PROFILE = WorkloadProfile(
    name="ci-micro",
    profile_type=ProfileType.CUSTOM,
    context_lengths=[512],
    output_lengths=[32],
    system_prompt_tokens=100,
    warmup_runs=1,
    measurement_runs=2,
    weight_ttft=0.50,
    weight_throughput=0.35,
    weight_memory=0.15,
)


BUILTIN_PROFILES: dict[str, WorkloadProfile] = {
    "chat-agent": CHAT_AGENT_PROFILE,
    "rag": RAG_PROFILE,
    "longform": LONGFORM_PROFILE,
    "ci-micro": CI_MICRO_PROFILE,
}


def list_profiles() -> list[str]:
    """List all available profile names."""
    return list(BUILTIN_PROFILES.keys())


def get_profile(name: str) -> WorkloadProfile | None:
    """Get a built-in profile by name."""
    return BUILTIN_PROFILES.get(name)


def load_profile_from_json(path: str | Path) -> WorkloadProfile:
    """
    Load a custom profile from a JSON file.

    Args:
        path: Path to JSON file containing profile definition

    Returns:
        WorkloadProfile instance

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Ensure custom type if not specified
    if "profile_type" not in data:
        data["profile_type"] = ProfileType.CUSTOM.value

    return WorkloadProfile(**data)


def save_profile_to_json(profile: WorkloadProfile, path: str | Path) -> None:
    """
    Save a profile to a JSON file.

    Args:
        profile: Profile to save
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile.model_dump(mode="json"), f, indent=2)


def create_custom_profile(
    name: str,
    context_lengths: list[int],
    output_lengths: list[int],
    *,
    system_prompt_tokens: int = 500,
    warmup_runs: int = 2,
    measurement_runs: int = 5,
    weight_ttft: float = 0.5,
    weight_throughput: float = 0.35,
    weight_memory: float = 0.15,
) -> WorkloadProfile:
    """
    Create a custom workload profile.

    Args:
        name: Profile name
        context_lengths: List of context lengths to test
        output_lengths: List of output lengths to test
        system_prompt_tokens: Approximate system prompt size
        warmup_runs: Number of warmup runs
        measurement_runs: Number of measurement runs
        weight_ttft: Weight for TTFT in scoring (0-1)
        weight_throughput: Weight for throughput in scoring (0-1)
        weight_memory: Weight for memory efficiency in scoring (0-1)

    Returns:
        Configured WorkloadProfile
    """
    return WorkloadProfile(
        name=name,
        profile_type=ProfileType.CUSTOM,
        context_lengths=context_lengths,
        output_lengths=output_lengths,
        system_prompt_tokens=system_prompt_tokens,
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
        weight_ttft=weight_ttft,
        weight_throughput=weight_throughput,
        weight_memory=weight_memory,
    )


# =============================================================================
# Profile Recommendations
# =============================================================================

def recommend_profile(
    typical_context_length: int,
    typical_output_length: int,
    latency_sensitive: bool = True,
) -> str:
    """
    Recommend a profile based on workload characteristics.

    Args:
        typical_context_length: Typical input context length
        typical_output_length: Typical output length
        latency_sensitive: Whether low latency is critical

    Returns:
        Recommended profile name
    """
    if typical_context_length >= 16384:
        return "rag"
    elif typical_output_length >= 512:
        return "longform"
    elif latency_sensitive:
        return "chat-agent"
    else:
        return "chat-agent"  # Default


def get_profile_description(name: str) -> str:
    """Get a human-readable description of a profile."""
    descriptions = {
        "chat-agent": (
            "Chat/Agent workload: Large system prompts (2-4k tokens), "
            "short responses (64-256 tokens). Optimized for minimal TTFT."
        ),
        "rag": (
            "RAG workload: Large context windows (8k-32k tokens), "
            "medium responses (256-512 tokens). Balanced TTFT/throughput/memory."
        ),
        "longform": (
            "Longform generation: Medium context (4k-8k tokens), "
            "long responses (1k-2k tokens). Optimized for throughput."
        ),
        "ci-micro": (
            "CI/Testing profile: Minimal context (512 tokens), "
            "short responses (32 tokens). For quick validation only."
        ),
    }
    return descriptions.get(name, "Custom profile")
