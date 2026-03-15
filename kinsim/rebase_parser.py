"""Backward-compatibility shim — rebase_parser moved to kinsim.prep.rebase.

All public names are re-exported. Update your imports to use
kinsim.prep.rebase directly.
"""

from .prep.rebase import (  # noqa: F401
    decode_fuzznuc_pattern_name,
    main,
    parse_rebase_annotation,
    parse_rebase_file,
    parse_rebase_isoschizomers,
    parse_rebase_simple,
    parse_rebase_withrefm,
    write_fuzznuc_pattern_file,
)
