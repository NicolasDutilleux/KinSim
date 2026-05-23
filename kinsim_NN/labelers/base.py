"""Abstract base class for methylation labelers.

A labeler takes a reference contig (id + sequence) plus per-source
input files (GFF, jasmine BAM, custom CSV, etc.) and yields per-position
methylation labels as ``(ref_id, pos_0based, meth_id, strand)`` tuples.

Multiple labelers are chained in order; earlier labelers win on
conflicts (see ``extract.merge_labels``).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable


class BaseLabeler(ABC):
    """Methylation labeler base class.

    Subclasses must:
      * set the class attribute :attr:`name` to a unique registry key.
      * implement :meth:`label` to yield ``(ref_id, pos, meth_id,
        strand)`` tuples.

    Subclass constructors typically take a path + kwargs from the YAML
    ``labelers`` entry. The kwargs match the YAML keys (e.g.,
    ``qv_threshold``, ``ml_threshold``).
    """

    #: Unique registry key; class attribute.
    name: str = ""

    def __init__(self, **kwargs):
        self._kwargs = dict(kwargs)

    @abstractmethod
    def label(
        self,
        ref_id: str,
        ref_seq: str,
        strain_dir: Path,
        **kwargs,
    ) -> Iterable[tuple[str, int, int, str]]:
        """Yield label tuples for one reference contig.

        Args:
            ref_id: contig name as it appears in the BAM / GFF.
            ref_seq: full reference sequence for that contig.
            strain_dir: per-strain pipeline directory (used to resolve
                ``file_pattern`` from the YAML config).
            kwargs: free-form passthrough (currently: ``meth_id_by_name``,
                ``treat_modified_base_as``).

        Yields:
            ``(ref_id, pos_0based, meth_id, strand)`` where ``strand`` is
            ``"+"`` or ``"-"``.
        """
        raise NotImplementedError


__all__ = ["BaseLabeler"]
