"""Load and validate the YAML config."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any] = field(repr=False)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def tickers(self) -> list[str]:
        return list(self.raw["universe"]["tickers"])

    @property
    def benchmark(self) -> str:
        return self.raw["universe"]["benchmark"]

    def path(self, key: str) -> Path:
        """Resolve one of the run.*_dir settings to an absolute path."""
        p = ROOT / self.raw["run"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p


def load(path: str | Path | None = None) -> Config:
    path = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(path) as fh:
        return Config(yaml.safe_load(fh))
