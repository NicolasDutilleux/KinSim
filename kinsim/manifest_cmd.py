"""Backward-compatibility shim — manifest_cmd moved to kinsim.prep.manifest.

All public names are re-exported. Update your imports to use
kinsim.prep.manifest directly.
"""

from .prep.manifest import main  # noqa: F401
