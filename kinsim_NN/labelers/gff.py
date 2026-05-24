"""GFF labeler — parse pbmotifmaker reprocess GFF for methylation labels.

The reprocessed GFF (output of ``pbmotifmaker reprocess``) has one line
per modified position with columns:

    seqid  source  type  start  end  score  strand  frame  attributes

We map ``type`` ∈ {``m6A``, ``m4C``, ``m5C``, ``modified_base``} to
:attr:`meth_id_by_name`, filter by ``score (modificationQv) ≥
qv_threshold``, and emit ``(seqid, pos_0based, meth_id, strand)``.

The ``modified_base`` type is unclassified by ipdSummary; jasmine
+ modkit downstream typically rebrand these as 5mC at CpG sites. By
default (``treat_modified_base_as: m5C`` in YAML), we treat them as
m5C. Set ``treat_modified_base_as: null`` in the config to ignore them.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .base import BaseLabeler
from .registry import register


log = logging.getLogger(__name__)


@register
class GFFLabeler(BaseLabeler):
    """Per-position GFF (motifs.gff) parser."""

    name = "gff"

    def __init__(
        self,
        qv_threshold: float = 10.0,
        file_pattern: str = "{strain_dir}/motifs.gff",
        require_motif: bool = False,
        **kwargs,
    ):
        super().__init__(
            qv_threshold=qv_threshold,
            file_pattern=file_pattern,
            require_motif=require_motif,
            **kwargs,
        )
        self.qv_threshold = float(qv_threshold)
        self.file_pattern = str(file_pattern)
        # When True, only keep lines whose col-9 attributes contain "motif="
        # (i.e. pbmotifmaker reprocess matched this position to a discovered
        # enriched motif). Filters out high-QV ipdSummary calls without motif
        # support — essential on noisy chemistries (Vega P1-C1) where 96%+
        # of GFF lines are random kinetic blips.
        self.require_motif = bool(require_motif)
        # Per-instance cache of parsed files. Each entry maps
        # (path, frozenset(meth_id_by_name.items()), treat_modified_base_as,
        #  require_motif) → dict[ref_id, list[(pos_0based, meth_id, strand)]].
        self._cache: dict[tuple, dict[str, list]] = {}

    def _resolve_path(self, strain_dir: Path) -> Path:
        p = self.file_pattern.format(strain_dir=str(strain_dir))
        return Path(p)

    def _parse_file(
        self,
        path: Path,
        meth_id_by_name: dict[str, int],
        treat_modified_base_as: str | None,
    ) -> dict[str, list[tuple[int, int, str]]]:
        """Parse the GFF once, return labels grouped by ref_id."""
        key = (
            str(path),
            frozenset(meth_id_by_name.items()),
            treat_modified_base_as,
            self.require_motif,
        )
        if key in self._cache:
            return self._cache[key]
        groups: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
        n_kept = 0
        n_skipped_qv = 0
        n_skipped_type = 0
        n_skipped_no_motif = 0
        with open(path) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9:
                    continue
                gff_seqid = parts[0]
                gff_type = parts[2]
                try:
                    qv = float(parts[5])
                    start_1based = int(parts[3])
                except ValueError:
                    continue
                if qv < self.qv_threshold:
                    n_skipped_qv += 1
                    continue
                # Optional: skip ipdSummary calls without motif support.
                # col 9 (attributes) carries "motif=GATC;id=GATC;..." only on
                # positions that pbmotifmaker reprocess matched to a discovered
                # enriched motif.
                if self.require_motif and "motif=" not in parts[8]:
                    n_skipped_no_motif += 1
                    continue
                strand = parts[6]

                meth_name: str | None
                if gff_type in meth_id_by_name and gff_type != "none":
                    meth_name = gff_type
                elif gff_type == "modified_base" and treat_modified_base_as:
                    meth_name = treat_modified_base_as
                else:
                    meth_name = None

                if meth_name is None or meth_name not in meth_id_by_name:
                    n_skipped_type += 1
                    continue

                meth_id = meth_id_by_name[meth_name]
                groups[gff_seqid].append((start_1based - 1, meth_id, strand))
                n_kept += 1
        log.info(
            "[GFFLabeler] %s parsed once  kept=%d  skipped(qv<%g)=%d  "
            "skipped(type)=%d  skipped(no_motif)=%d  require_motif=%s",
            path.name, n_kept, self.qv_threshold, n_skipped_qv,
            n_skipped_type, n_skipped_no_motif, self.require_motif,
        )
        self._cache[key] = dict(groups)
        return self._cache[key]

    def label(
        self,
        ref_id: str,
        ref_seq: str,
        strain_dir: Path,
        *,
        meth_id_by_name: dict[str, int],
        treat_modified_base_as: str | None = None,
        **kwargs,
    ) -> Iterable[tuple[str, int, int, str]]:
        path = self._resolve_path(strain_dir)
        if not path.is_file():
            log.warning("[GFFLabeler] missing: %s", path)
            return
        groups = self._parse_file(path, meth_id_by_name, treat_modified_base_as)
        for pos, mid, strand in groups.get(ref_id, []):
            yield (ref_id, pos, mid, strand)


__all__ = ["GFFLabeler"]
