"""Backward-compatibility shim — callers moved to kinsim.prep.callers.

All imports from kinsim.callers are re-exported from kinsim.prep.callers.
Update your imports to use kinsim.prep.callers directly.
"""

from ..prep.callers import (  # noqa: F401
    BaseOutputParser,
    auto_detect_parser,
    create_parser,
    list_parsers,
)

__all__ = [
    "BaseOutputParser",
    "auto_detect_parser",
    "create_parser",
    "list_parsers",
]
