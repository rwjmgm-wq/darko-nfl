"""
Configuration loader utility.

Loads configuration from YAML files and environment variables.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
import os


class ConfigLoader:
    """Loads and manages configuration settings."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._load_env_vars()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def _load_env_vars(self):
        """Load environment variables from .env file."""
        # Try to load .env file
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)

        # Override config with environment variables if present
        cfb_api_key = os.getenv("CFB_API_KEY")
        if cfb_api_key:
            self.config['cfb_data']['api_key'] = cfb_api_key

        pff_data_path = os.getenv("PFF_DATA_PATH")
        if pff_data_path:
            self.config['pff_data']['data_path'] = pff_data_path
            self.config['pff_data']['enabled'] = True

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path to config value (e.g., 'decay.epa_beta')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration dictionary."""
        return self.config


# Global config instance
_config = None


def get_config(config_path: str = "config/config.yaml") -> ConfigLoader:
    """
    Get global configuration instance.

    Args:
        config_path: Path to configuration file

    Returns:
        ConfigLoader instance
    """
    global _config
    if _config is None:
        _config = ConfigLoader(config_path)
    return _config
