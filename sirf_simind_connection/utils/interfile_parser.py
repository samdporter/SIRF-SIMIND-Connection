"""
Shared interfile parsing utilities.

This module provides consistent parsing for STIR interfile header files,
eliminating duplicate parsing logic across the codebase.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import re

Number = Union[tuple, list, float, int, str]


def parse_interfile_line(line: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse a single interfile line and return (key, value) tuple.

    Args:
        line: A line from an interfile header file

    Returns:
        Tuple of (key, value) or (None, None) if line is not parseable

    Examples:
        >>> parse_interfile_line("matrix size [1] := 128")
        ('matrix size [1]', '128')
        >>> parse_interfile_line("; This is a comment")
        (None, None)
        >>> parse_interfile_line("!INTERFILE :=")
        (None, None)
    """
    line = line.strip()

    # Skip comments, empty lines, and section headers
    if (
        not line
        or line.startswith(";")
        or line.startswith("#")
        or line.endswith(":=")
    ):
        return None, None

    # Handle := separator (preferred)
    if ":=" in line:
        key, _, value = line.partition(":=")
        return key.strip(), value.strip()

    return None, None


def parse_interfile_header(filename: str) -> dict[str, str]:
    """Parse an entire interfile header file and return key-value dict.

    Args:
        filename: Path to interfile header file (.hs, .hv, .hct, etc.)

    Returns:
        Dictionary of key-value pairs from the header

    Examples:
        >>> attrs = parse_interfile_header("template.hs")
        >>> attrs["number of projections"]
        '64'
    """
    values = {}
    with open(filename, "r") as file:
        for line in file:
            key, value = parse_interfile_line(line)
            if key is not None:
                values[key] = value
    return values


def parse_interfile_with_regex(filename: str) -> dict[str, str]:
    """Parse interfile using regex pattern (legacy compatibility).

    This is the original parsing method from stir_utils.py.
    Use parse_interfile_header() for new code.

    Args:
        filename: Path to interfile header file

    Returns:
        Dictionary of key-value pairs
    """
    values = {}
    with open(filename, "r") as file:
        for line in file:
            if match := re.search(r"([^;].*?)\s*:=\s*(.*)", line):
                key = match[1].strip()
                value = match[2].strip()
                values[key] = value
    return values


@dataclass
class InterfileEntry:
    """Represents a single line inside an interfile header."""

    text: str
    key: Optional[str]
    value: Optional[str]

    @classmethod
    def from_line(cls, line: str) -> "InterfileEntry":
        key, value = parse_interfile_line(line)
        # Preserve newline from source to avoid formatting churn
        if not line.endswith("\n"):
            line = line + "\n"
        return cls(text=line, key=key, value=value)

    @classmethod
    def from_key_value(cls, key: str, value: Number) -> "InterfileEntry":
        value_str = cls._coerce_value(value)
        return cls(text=f"{key} := {value_str}\n", key=key, value=value_str)

    @staticmethod
    def _coerce_value(value: Number) -> str:
        if isinstance(value, (tuple, list)):
            return "{" + ", ".join(str(v) for v in value) + "}"
        return str(value)

    def set_value(self, value: Number) -> None:
        value_str = self._coerce_value(value)
        self.value = value_str
        if self.key is None:
            raise ValueError("Cannot set value on entry without a key")
        self.text = f"{self.key} := {value_str}\n"

    def copy(self) -> "InterfileEntry":
        return InterfileEntry(text=self.text, key=self.key, value=self.value)


class InterfileHeader:
    """Editable representation of a parsed interfile header."""

    def __init__(self, entries: List[InterfileEntry]):
        self._entries = entries

    @classmethod
    def from_file(cls, filename: str) -> "InterfileHeader":
        with open(filename, "r") as file:
            entries = [InterfileEntry.from_line(line) for line in file]
        return cls(entries)

    def copy(self) -> "InterfileHeader":
        return InterfileHeader([entry.copy() for entry in self._entries])

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        for entry in reversed(self._entries):
            if entry.key == key:
                return entry.value
        return default

    def set(self, key: str, value: Number) -> None:
        for entry in self._entries:
            if entry.key == key:
                entry.set_value(value)
                return
        self._entries.append(InterfileEntry.from_key_value(key, value))

    def insert(self, index: int, key: str, value: Number) -> None:
        entry = InterfileEntry.from_key_value(key, value)
        index = max(0, min(index, len(self._entries)))
        self._entries.insert(index, entry)

    def write(self, filename: str) -> None:
        with open(filename, "w") as file:
            for entry in self._entries:
                file.write(entry.text)


__all__ = [
    "InterfileHeader",
    "InterfileEntry",
    "parse_interfile_line",
    "parse_interfile_header",
    "parse_interfile_with_regex",
]
