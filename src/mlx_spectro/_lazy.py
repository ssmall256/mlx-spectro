"""Deferred imports for optional heavy dependencies."""

from __future__ import annotations

import importlib
from typing import Any


class LazyModule:
    """Stand-in for a module that is imported on first attribute access.

    On first use it imports the module and rebinds ``namespace[name]`` to it, so
    every later access in that module is a plain global lookup with no proxy
    overhead. mlx-spectro uses NumPy only for filterbank design and host-side
    feature helpers; STFT/iSTFT never touch it, so code that only transforms
    audio (Demucs, for one) never imports NumPy at all.
    """

    __slots__ = ("_module_name", "_namespace", "_name")

    def __init__(self, module_name: str, namespace: dict, name: str):
        self._module_name = module_name
        self._namespace = namespace
        self._name = name

    def _load(self) -> Any:
        module = importlib.import_module(self._module_name)
        self._namespace[self._name] = module
        return module

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._load(), attr)

    def __repr__(self) -> str:
        return f"<lazy module {self._module_name!r}>"
