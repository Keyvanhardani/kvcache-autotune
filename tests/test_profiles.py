"""Tests for profiles module."""

import pytest
import json
import tempfile
from pathlib import Path

from kvat.core.profiles import (
    get_profile,
    list_profiles,
    load_profile_from_json,
    save_profile_to_json,
    create_custom_profile,
    recommend_profile,
    get_profile_description,
    BUILTIN_PROFILES,
)
from kvat.core.schema import ProfileType


class TestBuiltinProfiles:
    """Tests for built-in profiles."""

    def test_list_profiles(self):
        """Test listing available profiles."""
        profiles = list_profiles()

        assert "chat-agent" in profiles
        assert "rag" in profiles
        assert "longform" in profiles

    def test_get_chat_agent_profile(self):
        """Test getting chat-agent profile."""
        profile = get_profile("chat-agent")

        assert profile is not None
        assert profile.name == "chat-agent"
        assert profile.profile_type == ProfileType.CHAT_AGENT
        assert profile.weight_ttft == 0.5  # TTFT is critical for chat

    def test_get_rag_profile(self):
        """Test getting RAG profile."""
        profile = get_profile("rag")

        assert profile is not None
        assert profile.name == "rag"
        assert 8192 in profile.context_lengths  # Large context
        assert profile.weight_memory == 0.30  # Memory matters for large contexts

    def test_get_longform_profile(self):
        """Test getting longform profile."""
        profile = get_profile("longform")

        assert profile is not None
        assert profile.name == "longform"
        assert profile.weight_throughput == 0.50  # Throughput critical

    def test_get_nonexistent_profile(self):
        """Test getting a profile that doesn't exist."""
        profile = get_profile("nonexistent")
        assert profile is None

    def test_all_profiles_have_valid_weights(self):
        """Test that all profiles have weights summing to 1.0."""
        for name, profile in BUILTIN_PROFILES.items():
            total = (
                profile.weight_ttft
                + profile.weight_throughput
                + profile.weight_memory
            )
            assert abs(total - 1.0) < 0.01, f"Profile {name} weights sum to {total}"


class TestCustomProfiles:
    """Tests for custom profile functionality."""

    def test_create_custom_profile(self):
        """Test creating a custom profile."""
        profile = create_custom_profile(
            name="my-profile",
            context_lengths=[1024, 2048],
            output_lengths=[64, 128],
            weight_ttft=0.4,
            weight_throughput=0.4,
            weight_memory=0.2,
        )

        assert profile.name == "my-profile"
        assert profile.profile_type == ProfileType.CUSTOM
        assert profile.context_lengths == [1024, 2048]

    def test_save_and_load_profile(self):
        """Test saving and loading a profile from JSON."""
        profile = create_custom_profile(
            name="test-save",
            context_lengths=[512, 1024],
            output_lengths=[32, 64],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profile.json"

            save_profile_to_json(profile, path)
            loaded = load_profile_from_json(path)

            assert loaded.name == profile.name
            assert loaded.context_lengths == profile.context_lengths
            assert loaded.output_lengths == profile.output_lengths

    def test_load_invalid_profile(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_profile_from_json("/nonexistent/path.json")


class TestProfileRecommendation:
    """Tests for profile recommendation."""

    def test_recommend_rag_for_large_context(self):
        """Test RAG recommendation for large contexts."""
        recommended = recommend_profile(
            typical_context_length=20000,
            typical_output_length=256,
        )

        assert recommended == "rag"

    def test_recommend_longform_for_long_output(self):
        """Test longform recommendation for long outputs."""
        recommended = recommend_profile(
            typical_context_length=4000,
            typical_output_length=1000,
        )

        assert recommended == "longform"

    def test_recommend_chat_agent_for_latency(self):
        """Test chat-agent recommendation for latency-sensitive."""
        recommended = recommend_profile(
            typical_context_length=2000,
            typical_output_length=100,
            latency_sensitive=True,
        )

        assert recommended == "chat-agent"


class TestProfileDescriptions:
    """Tests for profile descriptions."""

    def test_get_chat_agent_description(self):
        """Test getting chat-agent description."""
        desc = get_profile_description("chat-agent")

        assert "Chat" in desc or "chat" in desc
        assert "TTFT" in desc

    def test_get_rag_description(self):
        """Test getting RAG description."""
        desc = get_profile_description("rag")

        assert "RAG" in desc
        assert "context" in desc.lower()

    def test_unknown_profile_description(self):
        """Test description for unknown profile."""
        desc = get_profile_description("unknown")
        assert desc == "Custom profile"
