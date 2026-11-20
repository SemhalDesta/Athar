"""Paths and configuration."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent
DATA_DIR = Path(os.getenv("ATHAR_DATA_DIR", PROJECT_DIR / "data"))
OUTPUT_DIR = DATA_DIR / "outputs"
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = DATA_DIR / "model"
RULES_PATH = BACKEND_DIR / "config" / "rules.yaml"

if load_dotenv:
    load_dotenv(BACKEND_DIR / ".env")
    load_dotenv(PROJECT_DIR / ".env")


@lru_cache(maxsize=1)
def rules() -> dict:
    with open(RULES_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def llm_model() -> str:
    return os.getenv("ATHAR_LLM_MODEL") or rules()["llm"]["model"]


def openrouter_key() -> str | None:
    return os.getenv("OPENROUTER_API_KEY")


for _d in (DATA_DIR, OUTPUT_DIR, CACHE_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)
