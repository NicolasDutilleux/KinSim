"""Labeler registry: @register decorator and factory.

Pattern modelled after ``kinsim/utils/parsers/registry.py`` — keeps
labeler discovery decoupled from import order. Implementations
register themselves on import; users construct them via
:func:`create_labeler` keyed by the YAML config's ``type`` field.
"""
from __future__ import annotations

import logging
from typing import Type

from .base import BaseLabeler


log = logging.getLogger(__name__)


_REGISTRY: dict[str, Type[BaseLabeler]] = {}


def register(cls: Type[BaseLabeler]) -> Type[BaseLabeler]:
    """Decorator: register a labeler class by its :attr:`name`."""
    name = getattr(cls, "name", "")
    if not name:
        raise ValueError(f"{cls.__name__} must set the 'name' class attribute")
    if name in _REGISTRY:
        existing = _REGISTRY[name].__name__
        if existing != cls.__name__:
            log.warning("Labeler name %r already registered to %s — overwriting with %s",
                        name, existing, cls.__name__)
    _REGISTRY[name] = cls
    return cls


def create_labeler(name: str, **kwargs) -> BaseLabeler:
    """Instantiate a registered labeler by name.

    Raises:
        KeyError: if no labeler matches ``name``.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown labeler {name!r}. Registered: {sorted(_REGISTRY.keys())}"
        )
    cls = _REGISTRY[name]
    return cls(**kwargs)


def list_labelers() -> list[str]:
    return sorted(_REGISTRY.keys())


__all__ = ["register", "create_labeler", "list_labelers"]
