"""Load project defaults from the shared YAML configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Return parsed project configuration, reporting invalid config clearly."""
    try:
        with CONFIG_PATH.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except OSError as exc:
        raise RuntimeError(f"Unable to read configuration file: {CONFIG_PATH}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration in {CONFIG_PATH}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {CONFIG_PATH}")
    return config
