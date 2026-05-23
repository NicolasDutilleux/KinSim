"""Labeler registry (modular methylation labeling sources).

Adding a new labeler:
    1. Create ``kinsim_NN/labelers/<my_name>.py`` with a
       ``@register``ed subclass of :class:`BaseLabeler`.
    2. Import the module here so it gets registered on package import.
    3. Reference it from the YAML ``labelers:`` list with
       ``type: <my_name>``.
"""
from .base import BaseLabeler
from .registry import register, create_labeler, list_labelers

# Import submodules to trigger registration
from . import gff           # noqa: F401  registers GFFLabeler
from . import jasmine_mm_ml # noqa: F401  registers JasmineMMMLLabeler


__all__ = [
    "BaseLabeler",
    "register",
    "create_labeler",
    "list_labelers",
]
