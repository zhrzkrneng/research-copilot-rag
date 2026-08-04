"""Configuration file loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config.config import AppConfig


def load_config(path: str | Path) -> AppConfig:
    """Load an application configuration from JSON or YAML."""
    config_path = _validate_path(path)
    text = config_path.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError("configuration file cannot be empty.")

    data = _parse_mapping(
        text,
        suffix=config_path.suffix.lower(),
    )

    return AppConfig.from_mapping(data)


def _validate_path(path: str | Path) -> Path:
    if not isinstance(path, (str, Path)):
        raise TypeError(
            "path must be a string or pathlib.Path."
        )

    config_path = Path(path).expanduser()

    if not config_path.exists():
        raise FileNotFoundError(config_path)

    if not config_path.is_file():
        raise ValueError("path must point to a file.")

    if config_path.suffix.lower() not in {
        ".json",
        ".yaml",
        ".yml",
    }:
        raise ValueError(
            "configuration file must use .json, .yaml, or .yml."
        )

    return config_path


def _parse_mapping(
    text: str,
    *,
    suffix: str,
) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        if suffix == ".json":
            raise ValueError(
                "invalid JSON configuration."
            ) from None

        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required for non-JSON YAML syntax."
            ) from exc

        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(
                "invalid YAML configuration."
            ) from exc

    if not isinstance(data, dict):
        raise TypeError(
            "configuration root must be a mapping."
        )

    return data
