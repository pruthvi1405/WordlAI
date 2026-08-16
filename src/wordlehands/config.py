from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_model: str
    target_base_url: str
    allowlist_path: Path
    artifacts_dir: Path
    evidence_dir: Path
    headless: bool

    @classmethod
    def load(cls) -> Settings:
        return cls(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            openai_model=os.environ.get("WORDLEHANDS_MODEL", "gpt-4.1"),
            target_base_url=os.environ.get(
                "WORDLEHANDS_TARGET_URL", "https://hellowordl.net/"
            ),
            allowlist_path=ROOT_DIR / "config" / "allowlist.yaml",
            artifacts_dir=ROOT_DIR / "artifacts",
            evidence_dir=ROOT_DIR / "evidence",
            headless=os.environ.get("WORDLEHANDS_HEADLESS", "0") == "1",
        )


settings = Settings.load()
