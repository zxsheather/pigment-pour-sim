"""Shared paths for source assets and local simulation outputs."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENE_CONFIG = Path(__file__).with_name("scene_config.json")
