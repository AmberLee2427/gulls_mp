"""Small parser for Gulls ``.prm`` KEY=VALUE files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class PrmEntry:
    """One parsed KEY=VALUE line from a parameter file."""

    key: str
    value: str
    line_number: int


@dataclass(frozen=True)
class PrmDocument:
    """Parsed ``.prm`` content with order and duplicate-key metadata."""

    entries: tuple[PrmEntry, ...]
    key_order: tuple[str, ...]
    settings: dict[str, str]
    duplicates: dict[str, tuple[PrmEntry, ...]] = field(default_factory=dict)

    def to_text(self, overrides: Mapping[str, object] | None = None) -> str:
        """Serialize settings to ``.prm`` text, applying optional overrides."""

        overrides = overrides or {}
        lines: list[str] = []

        for key in self.key_order:
            value = overrides.get(key, self.settings[key])
            lines.append(f"{key}={value}")

        for key, value in overrides.items():
            if key not in self.settings:
                lines.append(f"{key}={value}")

        return "\n".join(lines) + ("\n" if lines else "")


def parse_prm_text(text: str) -> PrmDocument:
    """Parse ``.prm`` text into settings, preserving first-seen key order."""

    entries: list[PrmEntry] = []
    key_order: list[str] = []
    settings: dict[str, str] = {}
    duplicate_entries: dict[str, list[PrmEntry]] = {}

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        content = raw_line.split("#", 1)[0].strip()
        if not content or "=" not in content:
            continue

        key, value = content.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        entry = PrmEntry(key=key, value=value, line_number=line_number)
        entries.append(entry)

        if key in settings:
            duplicate_entries.setdefault(key, []).append(entry)
        else:
            key_order.append(key)

        settings[key] = value

    duplicates = {
        key: tuple(values)
        for key, values in duplicate_entries.items()
    }

    return PrmDocument(
        entries=tuple(entries),
        key_order=tuple(key_order),
        settings=settings,
        duplicates=duplicates,
    )


def parse_prm_file(path: str | Path) -> PrmDocument:
    """Parse a ``.prm`` file from disk."""

    return parse_prm_text(Path(path).read_text(encoding="utf-8"))
