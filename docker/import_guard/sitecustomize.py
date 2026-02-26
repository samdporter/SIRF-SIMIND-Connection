"""
Import guard used by backend-isolated Docker environments.

Set SIRF_SIMIND_BLOCK_IMPORTS to a comma-separated list of top-level module
names. Any import matching one of those module roots will raise ImportError.
"""

from __future__ import annotations

import builtins
import importlib.abc
import os
import sys


BLOCKED_IMPORTS = {
    entry.strip()
    for entry in os.environ.get("SIRF_SIMIND_BLOCK_IMPORTS", "").split(",")
    if entry.strip()
}

if BLOCKED_IMPORTS:
    _ORIGINAL_IMPORT = builtins.__import__

    class _BlockedModuleFinder(importlib.abc.MetaPathFinder):
        """Prevent loading blocked module roots through all import paths."""

        def find_spec(self, fullname, path=None, target=None):
            root = fullname.split(".", 1)[0]
            if root in BLOCKED_IMPORTS:
                raise ImportError(
                    f"Module '{root}' is blocked in this backend-isolated container."
                )
            return None

    for module_name in list(sys.modules):
        if module_name.split(".", 1)[0] in BLOCKED_IMPORTS:
            del sys.modules[module_name]

    sys.meta_path.insert(0, _BlockedModuleFinder())

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".", 1)[0]
        if root in BLOCKED_IMPORTS:
            raise ImportError(
                f"Module '{root}' is blocked in this backend-isolated container."
            )
        return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)

    builtins.__import__ = _guarded_import
