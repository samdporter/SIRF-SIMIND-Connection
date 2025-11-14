"""
Shared interfile parsing utilities.

This module provides consistent parsing for STIR interfile header files,
eliminating duplicate parsing logic across the codebase.
"""

import re
from typing import Dict, Optional, Tuple


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


def parse_interfile_header(filename: str) -> Dict[str, str]:
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


def parse_interfile_with_regex(filename: str) -> Dict[str, str]:
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


__all__ = [
    "parse_interfile_line",
    "parse_interfile_header",
    "parse_interfile_with_regex",
]
