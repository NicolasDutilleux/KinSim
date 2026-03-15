"""Backward-compatibility shim — prepare moved to kinsim.prep.prepare.

All public names are re-exported. Update your imports to use
kinsim.prep.prepare directly.
"""

from .prep.prepare import main, prepare_config  # noqa: F401
