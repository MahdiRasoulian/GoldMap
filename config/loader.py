"""Configuration loader for Goldmap platform."""

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. Defaults to config/settings.yaml.
        
    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        config_path = str(
            Path(__file__).parent / "settings.yaml"
        )
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Override with environment variables if present
    if os.environ.get("MT5_LOGIN"):
        config["mt5"]["login"] = int(os.environ["MT5_LOGIN"])
    if os.environ.get("MT5_PASSWORD"):
        config["mt5"]["password"] = os.environ["MT5_PASSWORD"]
    if os.environ.get("MT5_SERVER"):
        config["mt5"]["server"] = os.environ["MT5_SERVER"]
    
    return config


# Global config instance
CONFIG = load_config()