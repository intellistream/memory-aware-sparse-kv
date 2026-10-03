"""Load and validate model-specific M0-A runtime and trace contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_PROFILE = "deepseek_v4"
PROFILE_FILE = Path(__file__).with_name("model_profiles.json")


def _validate_profile(name: str, profile: dict[str, Any]) -> None:
    required = {
        "model_id", "model_revision", "model_path", "served_model_name",
        "image", "container", "launcher", "tensor_parallel_size", "devices",
        "gpu_memory_utilization", "max_model_len", "trace_schema_version",
        "attention_backend", "native_operators", "raw_index_unit",
        "compression_ratio", "logical_block_size", "selected_width",
        "expected_layers", "computed_layers", "reused_layers",
        "default_pairs", "baseline_file", "pilot_kind",
        "requires_equal_pair_tokens",
    }
    missing = sorted(required - profile.keys())
    if missing:
        raise ValueError(f"Model profile {name} missing fields: {missing}")
    devices = profile["devices"]
    if not devices or len(set(devices)) != len(devices):
        raise ValueError(f"Model profile {name} has invalid devices")
    if profile["tensor_parallel_size"] != len(devices):
        raise ValueError(f"Model profile {name} TP size must match device count")
    if profile["expected_layers"] != (
        profile["computed_layers"] + profile["reused_layers"]
    ):
        raise ValueError(f"Model profile {name} layer counts do not add up")
    for field in (
        "compression_ratio", "logical_block_size", "selected_width",
        "expected_layers",
    ):
        if not isinstance(profile[field], int) or profile[field] < 1:
            raise ValueError(f"Model profile {name} has invalid {field}")
    if profile["trace_schema_version"] not in {1, 2}:
        raise ValueError(f"Model profile {name} has unknown trace schema")
    if not isinstance(profile["requires_equal_pair_tokens"], bool):
        raise ValueError(f"Model profile {name} has invalid pair-length policy")


def load_profiles(path: Path = PROFILE_FILE) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported model profile schema")
    profiles = data.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Model profile file contains no profiles")
    for name, profile in profiles.items():
        _validate_profile(name, profile)
    return profiles


def load_profile(name: str, path: Path = PROFILE_FILE) -> dict[str, Any]:
    profiles = load_profiles(path)
    if name not in profiles:
        raise ValueError(
            f"Unknown model profile {name!r}; choose one of {sorted(profiles)}"
        )
    profile = dict(profiles[name])
    profile["name"] = name
    return profile


def resolve_profile_path(root: Path, profile: dict[str, Any], field: str) -> Path:
    value = Path(profile[field])
    return value if value.is_absolute() else root / value


def validate_pair_policy(pair_set: dict[str, Any], profile: dict[str, Any]) -> None:
    if not profile["requires_equal_pair_tokens"]:
        return
    unequal = [
        pair["pair_id"] for pair in pair_set["pairs"]
        if pair["event"]["prompt_tokens_expected"]
        != pair["control"]["prompt_tokens_expected"]
    ]
    if unequal:
        raise ValueError(
            f"Model profile {profile['name']} requires equal pair lengths: {unequal}"
        )
