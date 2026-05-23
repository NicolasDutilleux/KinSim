"""Labeler registry (modular methylation labeling sources).

Adding a new labeler:
    1. Create ``kinsim_NN/labelers/<my_name>.py`` with a
       ``@register``ed subclass of :class:`BaseLabeler`.
    2. Import the module here so it gets registered on package import.
    3. Reference it from the YAML ``labelers:`` list with
       ``type: <my_name>``.

Pure-stdlib labelers are imported eagerly so ``kinsim_NN.labelers``
remains importable on CPU-only / torch-less / pysam-less nodes that
just want to read or inspect the config. Pysam-dependent labelers are
imported lazily on first ``create_labeler`` call.
"""
from .base import BaseLabeler
from .registry import create_labeler as _create_labeler_raw
from .registry import list_labelers, register

# Eagerly register stdlib-only labelers.
from . import gff           # noqa: F401  registers GFFLabeler


def create_labeler(name: str, **kwargs):
    """Wrapper that lazy-imports pysam-dependent labelers on first request."""
    if name == "jasmine_mm_ml":
        from . import jasmine_mm_ml  # noqa: F401 registers JasmineMMMLLabeler
    return _create_labeler_raw(name, **kwargs)


__all__ = [
    "BaseLabeler",
    "register",
    "create_labeler",
    "list_labelers",
]
