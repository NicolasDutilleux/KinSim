"""Abstract base class for methylation caller output parsers."""

from abc import ABC, abstractmethod
from typing import ClassVar


class BaseOutputParser(ABC):
    """Base class for parsing output files from methylation calling tools.

    Each parser reads an output file (CSV, TSV, GFF, etc.) from a specific
    methylation caller and converts it into a KinSim motif string.

    Subclasses must define:
        name:            Short identifier used for registry lookup.
        supported_mods:  List of mod types this format can carry.
        parse():         File → KinSim motif string conversion.
    """

    name: ClassVar[str]
    supported_mods: ClassVar[list[str]]

    @abstractmethod
    def parse(
        self,
        filepath: str,
        min_fraction: float = 0.40,
        min_detected: int = 20,
    ) -> str:
        """Parse a caller output file into a KinSim motif string.

        Returns a semicolon-delimited string, e.g.:
            "m6A,GATC,1,925,0.87;m4C,CCWGG,2,310,0.92"

        Args:
            filepath:      Path to the caller output file.
            min_fraction:  Minimum fraction threshold for filtering.
            min_detected:  Minimum nDetected threshold for filtering.

        Returns:
            KinSim motif string (semicolon-delimited entries).
        """

    def is_file_for_this_parser(self, filepath: str) -> bool:
        """Heuristic: return True if filepath looks like this format.

        Used by auto_detect_parser() to guess the right parser.
        Default returns False — subclasses should override.
        """
        return False
