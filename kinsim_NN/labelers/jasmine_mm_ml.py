"""Jasmine MM/ML BAM labeler — extract 5mC positions from MM/ML tags.

The jasmine + modkit pipeline emits a BAM where each read carries the
MM tag (modification base list) and ML tag (per-modification likelihood
0-255). We parse these tags to find per-position 5mC calls with
likelihood ≥ ``ml_threshold``.

This complements :class:`GFFLabeler` when 5mC isn't called by
ipdSummary (which only reliably handles m6A/m4C). Optional in the YAML
chain.

MM tag format (SAM specification):
    "C+m,n1,n2,...;"  — "m" mod on C bases, n_i = number of C bases
    to skip between modifications (delta encoding).

ML tag:
    Array of uint8, one entry per mod position (in the order they
    appear across all MM segments). Value = round(prob * 256).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pysam

from .base import BaseLabeler
from .registry import register


log = logging.getLogger(__name__)


@register
class JasmineMMMLLabeler(BaseLabeler):
    """Parse MM/ML tags from a jasmine BAM to label 5mC positions."""

    name = "jasmine_mm_ml"

    def __init__(
        self,
        ml_threshold: int = 200,
        file_pattern: str = "{strain_dir}/jasmine_5mC.bam",
        meth_name: str = "m5C",
        **kwargs,
    ):
        super().__init__(
            ml_threshold=ml_threshold,
            file_pattern=file_pattern,
            meth_name=meth_name,
            **kwargs,
        )
        self.ml_threshold = int(ml_threshold)
        self.file_pattern = str(file_pattern)
        self.meth_name = str(meth_name)

    def _resolve_path(self, strain_dir: Path) -> Path:
        return Path(self.file_pattern.format(strain_dir=str(strain_dir)))

    def _decode_mm_ml(self, read: pysam.AlignedSegment, c_positions_query: list[int]
                     ) -> list[tuple[int, int]]:
        """Decode MM/ML for one read.

        Returns list of (query_pos, ml_score) for 5mC sites only.
        Assumes the canonical PacBio MM segment ``C+m`` (5mC on C).
        """
        if not (read.has_tag("MM") and read.has_tag("ML")):
            return []
        mm = read.get_tag("MM")
        ml = list(read.get_tag("ML"))
        out: list[tuple[int, int]] = []
        ml_idx = 0
        # MM is semicolon-separated segments per modification
        for segment in str(mm).rstrip(";").split(";"):
            if not segment:
                continue
            head, _, deltas_str = segment.partition(",")
            # head is like 'C+m' or 'C+m?'
            if not head.startswith("C+m"):
                # Skip non-5mC segments (m6A '+A+a', etc.)
                # but still consume the deltas in ML
                if deltas_str:
                    ml_idx += len(deltas_str.split(","))
                continue
            deltas = [int(d) for d in deltas_str.split(",") if d != ""]
            c_idx = 0
            for d in deltas:
                # advance c_idx by d+1 to land on the next called C
                c_idx += d
                if c_idx >= len(c_positions_query):
                    break
                q_pos = c_positions_query[c_idx]
                score = int(ml[ml_idx]) if ml_idx < len(ml) else 0
                ml_idx += 1
                out.append((q_pos, score))
                c_idx += 1
        return out

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
            log.warning("[JasmineMMMLLabeler] missing: %s", path)
            return
        if self.meth_name not in meth_id_by_name:
            log.warning(
                "[JasmineMMMLLabeler] meth_name=%r not in methylation_types, skipping",
                self.meth_name,
            )
            return
        meth_id = meth_id_by_name[self.meth_name]

        # Aggregate per (ref_pos, strand) → list of ML scores from reads
        pos_scores: dict[tuple[int, str], list[int]] = defaultdict(list)
        with pysam.AlignmentFile(str(path), "rb") as bam:
            for read in bam.fetch(ref_id):
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    continue
                qseq = read.query_sequence
                if qseq is None:
                    continue
                # Find C positions in the (synthesized strand) query sequence
                c_positions = [i for i, b in enumerate(qseq) if b in "Cc"]
                if not c_positions:
                    continue
                calls = self._decode_mm_ml(read, c_positions)
                if not calls:
                    continue
                # Map query → ref via aligned pairs
                pairs = read.get_aligned_pairs(matches_only=True)
                q_to_r = {q: r for q, r in pairs}
                strand = "-" if read.is_reverse else "+"
                for q_pos, score in calls:
                    rp = q_to_r.get(q_pos)
                    if rp is None:
                        continue
                    pos_scores[(rp, strand)].append(score)

        n_kept = 0
        n_below = 0
        for (rp, strand), scores in pos_scores.items():
            # Use the MEDIAN score across covering reads
            scores_sorted = sorted(scores)
            median = scores_sorted[len(scores_sorted) // 2]
            if median < self.ml_threshold:
                n_below += 1
                continue
            yield (ref_id, rp, meth_id, strand)
            n_kept += 1
        log.info(
            "[JasmineMMMLLabeler] %s  ref=%s  kept=%d  skipped(ml<%d)=%d",
            path.name, ref_id, n_kept, self.ml_threshold, n_below,
        )


__all__ = ["JasmineMMMLLabeler"]
