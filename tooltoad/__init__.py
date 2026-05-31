"""Public package exports for tooltoad.

Calculator backends are resolved lazily so importing Tooltoad's chemistry and
visualization helpers does not require ORCA, xTB, g-xTB, or OpenMPI to be
configured.
"""

from __future__ import annotations

import sys
from importlib import import_module
from typing import Any

_MODULES: dict[str, str] = {
    "chemutils": "tooltoad.chemutils",
    "scene3d": "tooltoad.scene3d",
    "thermochemistry": "tooltoad.thermochemistry",
    "vis": "tooltoad.vis",
}

_PUBLIC_API: dict[str, tuple[str, str]] = {
    "orca_calculate": ("tooltoad.orca", "orca_calculate"),
    "xtb_calculate": ("tooltoad.xtb", "xtb_calculate"),
    "gxtb_calculate": ("tooltoad.gxtb", "gxtb_calculate"),
    "orca": ("tooltoad.orca", "orca_calculate"),
    "xtb": ("tooltoad.xtb", "xtb_calculate"),
    "gxtb": ("tooltoad.gxtb", "gxtb_calculate"),
    "PeriodicTable": ("tooltoad.periodictable", "PeriodicTable"),
    "run_crest": ("tooltoad.crest", "run_crest"),
}

_FALLBACK_MODULES = (
    "tooltoad.chemutils",
    "tooltoad.scene3d",
    "tooltoad.vis",
    "tooltoad.thermochemistry",
)

_CALCULATOR_ALIASES = {
    "tooltoad.orca": ("orca_calculate", "orca"),
    "tooltoad.xtb": ("xtb_calculate", "xtb"),
    "tooltoad.gxtb": ("gxtb_calculate", "gxtb"),
}

__all__ = sorted({*_MODULES, *_PUBLIC_API})


def _repair_calculator_aliases() -> None:
    """Keep historical short calculator aliases ahead of submodule attrs."""
    for module_name, (attr_name, alias_name) in _CALCULATOR_ALIASES.items():
        module = sys.modules.get(module_name)
        if module is None or not hasattr(module, attr_name):
            continue
        value = getattr(module, attr_name)
        globals()[attr_name] = value
        globals()[alias_name] = value


def __getattr__(name: str) -> Any:
    """Lazily resolve public Tooltoad API symbols."""
    if name in _MODULES:
        module = import_module(_MODULES[name])
        globals()[name] = module
        return module

    if name in _PUBLIC_API:
        module_name, attr_name = _PUBLIC_API[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        _repair_calculator_aliases()
        return value

    for module_name in _FALLBACK_MODULES:
        module = import_module(module_name)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            _repair_calculator_aliases()
            return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return module globals plus lazy public API names."""
    return sorted(set(globals()) | set(__all__))
