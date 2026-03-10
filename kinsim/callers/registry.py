"""Parser registry with @register decorator and factory functions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseOutputParser

log = logging.getLogger(__name__)

_REGISTRY: dict[str, type[BaseOutputParser]] = {}


def register(cls: type[BaseOutputParser]) -> type[BaseOutputParser]:
    """Class decorator that registers a parser in the global registry."""
    _REGISTRY[cls.name] = cls
    return cls


def create_parser(name: str, **kwargs) -> BaseOutputParser:
    """Instantiate a registered parser by name.

    Args:
        name:    Parser name (e.g. "pacbio", "modkit", "ipd_summary").
        **kwargs: Passed to the parser constructor.

    Raises:
        KeyError: If no parser is registered with that name.
    """
    if name not in _REGISTRY:
        available = ', '.join(sorted(_REGISTRY))
        raise KeyError(
            f"Unknown parser '{name}'. Available: {available}"
        )
    return _REGISTRY[name](**kwargs)


def list_parsers() -> list[str]:
    """Return sorted list of registered parser names."""
    return sorted(_REGISTRY)


def auto_detect_parser(filepath: str) -> BaseOutputParser | None:
    """Try each registered parser's heuristic to find a match.

    Returns the first parser whose is_file_for_this_parser() returns True,
    or None if no parser matches.
    """
    for cls in _REGISTRY.values():
        instance = cls()
        if instance.is_file_for_this_parser(filepath):
            log.info("Auto-detected parser '%s' for %s", cls.name, filepath)
            return instance
    return None
