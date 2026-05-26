# Bugs found during bc2034 validate run (2026-05-26)

Context: re-running `kinsim_NN generate` on bc2034 with v2 changes
(per_read_z + emit_unaligned by default) then chaining bystrandify →
align → ipdSummary → motifmaker. Found via comparison with the prior
v12_run3 manual run.

---

## Bug 4 — `template_length=0` triggered bystrandify discard (THE *actual* root cause)

**Symptom:** After fixing Bug 1 (flag=4), Bug 2 (SO:unknown), AND Bug 3
(strip ip/pw), bystrandify STILL discarded 99.8 % of reads. Output BAM
still 6.3 MB.

**Diagnosis:** Side-by-side `samtools view` of one specific read
(261166274/ccs) between the OLD working v12_run3 SIM.bam (passed
bystrandify) and the NEW broken validate_bc2034_perreadz SIM.bam
(rejected) — the only differing field across all 11 SAM columns + all
tags was **TLEN (col 9)**:

* OLD:  ``flag=4 ... TLEN=3960`` (= read length, left over from aligned input)
* NEW:  ``flag=4 ... TLEN=0``    (my _unalign_read cleared it)

Per BAM spec, TLEN is undefined for unmapped single-end reads — 0 is
the "spec-correct" value. But ccs-kinetics-bystrandify apparently uses
``TLEN > 0`` as a heuristic to detect "valid HiFi record" and silently
discards records with TLEN=0 — with the same misleading "has 0
PulseWidths" warning.

**Fix:** stop clearing template_length in ``_unalign_read``. Match
legacy ``kinsim/generate.py`` which never touches the field — TLEN
stays at whatever the aligned input had (typically = seq length).

```python
# DON'T do this:
#   read.template_length = 0   # ← triggers bystrandify rejection
```

Also: switched ``cigarstring=None`` to ``cigartuples=None`` to match
legacy exactly. Both should work, but the legacy form is the proven one.

**Lesson:** Four separate things contributed to bystrandify rejection,
all surfacing the same misleading "0 PulseWidths" warning. Only a
column-by-column ``samtools view`` diff between a known-working record
and a broken one localised the actual root cause. The other three
fixes were also necessary (without them, the output would have failed
other validators downstream), but only the TLEN fix unblocked
bystrandify.

The methodology: when a downstream tool gives misleading errors,
**diff a single record at the SAM/BAM level against a known-working
sample**. Don't trust the warning text.

---

## Bug 3 — pbmm2's `ip` / `pw` tags survived from aligned input

**Symptom:** After fixing Bug 1 (flag=4) AND Bug 2 (SO:unknown),
bystrandify STILL discarded 99.8 % of reads with the same misleading
"has 0 PulseWidths" warning. Bystrandified BAM = 6.3 MB (vs 4 GB for the
legacy `kinsim generate` chain).

**Root cause:** The @RG DS field on every PacBio HiFi BAM (preserved
from the original sequencer output via prep + align) declares:

```
DS:READTYPE=CCS;Ipd:CodecV1=ip;PulseWidth:CodecV1=pw;...
```

This tells downstream PacBio tools "kinetics live in ``ip`` and ``pw``
tags", **not** in ``fi`` / ``fp`` / ``ri`` / ``rp``. ``fi`` / ``fp`` /
``ri`` / ``rp`` are the legacy per-read tag names used on raw HiFi BAMs.
pbmm2 align can convert those into per-strand ``ip`` / ``pw`` on the
aligned output. When our input is the aligned BAM, the read may already
carry stale ``ip`` / ``pw`` from pbmm2.

kinsim_NN generate wrote fresh ``fi`` / ``fp`` / ``ri`` / ``rp`` over
the existing read but **did not strip the stale ``ip`` / ``pw``**.
ccs-kinetics-bystrandify obeys the @RG and reads ``pw`` for
PulseWidths — finds it empty or invalid → discards the read with the
"has 0 PulseWidths" warning.

**Confirmation:** Legacy ``kinsim/generate.py`` explicitly strips ``ip``
/ ``pw`` before writing (see comment at line ~1475: *"pbmm2 may have
converted fi/fp/ri/rp → ip/pw on the aligned input. Strip ip/pw so
downstream tools read our fresh fi/fp/ri/rp."*). Legacy chain produced
4 GB bystrandified output. kinsim_NN didn't strip → 6 MB.

**Fix:** `_unalign_read` now also strips ``ip`` / ``pw`` tags. Three
clean lines:

```python
for stale in ("ip", "pw"):
    if read.has_tag(stale):
        read.set_tag(stale, None)
```

**Lesson:** When converting an aligned PacBio BAM into "raw HiFi
shape" for downstream PacBio tooling, you cannot simply clear alignment
fields — you must also strip alignment-induced tag rewrites. The @RG
DS field decides which tag bystrandify obeys, not the tag-presence in
the read.

The misleading "0 PulseWidths" warning surfaced THREE separate bugs in
sequence, each masked by the others. The actionable diagnostic is to
diff the BAM record (header + flag + full tag set) against a
known-working legacy ``kinsim generate`` output, NOT to trust the
warning text.

---

## Bug 2 — emit_unaligned kept `@HD SO:coordinate` on unaligned output (CRITICAL)

**Symptom:** After fixing Bug 1 (flag=4), bystrandify STILL discarded
99.8 % of reads with the same "has 0 PulseWidths" warning. Output BAM
shrunk from 3.2 GB → 6.3 MB just like before.

**Root cause:** The input was a coordinate-sorted aligned BAM
(`@HD SO:coordinate`). `_sanitize_header_for_unaligned` stripped @SQ
but left SO:coordinate intact. After unaligning every record, every
read has ref_id=-1 / no CIGAR — there is no coordinate to sort by.
bystrandify reads SO:coordinate, dispatches to its aligned-processing
path, then rejects every record (which now lacks alignment fields)
with the misleading "0 PulseWidths" warning.

**Confirmation:** OLD validate run (v12_run3) used the legacy
`kinsim generate` which natively writes unaligned output with
`@HD SO:unknown`. Its bystrandified BAM = 4.0 GB (kept everything).
NEW kinsim_NN bystrandified BAM (with SO:coordinate left over) =
6.3 MB. Identical reads, identical tags, only SO differs.

**Fix:** `_sanitize_header_for_unaligned` now also forces `SO:unknown`
on the @HD entry, matching the canonical raw HiFi BAM shape.

**Lesson:** Both Bug 1 ("0 PulseWidths" → wrong flag bit) and Bug 2
("0 PulseWidths" → wrong SO header) hide behind the same misleading
warning text. bystrandify's discard message is generic — debug by
comparing the whole header + flag against a known-working BAM.

---

## Bug 1 — `_unalign_read` wrote flag=12 instead of flag=4 (CRITICAL)

**Symptom:** ccs-kinetics-bystrandify discarded 99.8 % of reads with the
misleading log warning:

```
WARN | New read 'm84151_.../ccs/fwd' has '0' PulseWidths, discarding
```

Out of 124,446 SIM.bam reads, only ~250 made it into the bystrandified
output (6.3 MB vs 3.2 GB input). The downstream chain (pbmm2 align →
ipdSummary → pbmotifmaker) then ran on essentially nothing and produced
0 motifs.

**Root cause:** `kinsim_NN/generate.py::_unalign_read` set:

```python
read.is_unmapped = True
read.is_reverse = False
read.is_secondary = False
read.is_supplementary = False
read.mate_is_unmapped = True   # ← BUG
read.is_proper_pair = False
```

`mate_is_unmapped = True` sets bit 8, giving flag = 4 + 8 = **12**.
PacBio CCS reads are not paired — raw HiFi BAMs from the sequencer have
flag = **4** exactly. ccs-kinetics-bystrandify uses the flag byte to
validate the input is a real raw HiFi record; flag=12 fails the check
and the read is discarded with the (misleading) PulseWidths warning.

**Confirmed by diagnostic:** a discarded read (`261166274/ccs`) had
qlen=3960, all four kinetic tags size=3960, ~all bytes non-zero, mean ~24,
diverse value distribution. Data was perfect; only the flag was wrong.

**Fix (commit pending):** set `read.flag = 4` directly. Clears all flag
bits in one shot, including the mate_unmapped and is_reverse bits.
Leaves the paired-end bits untouched because CCS reads aren't paired
anyway.

```python
def _unalign_read(read):
    read.flag = 4
    read.reference_id = -1
    read.reference_start = -1
    read.mapping_quality = 0
    read.cigarstring = None
    read.next_reference_id = -1
    read.next_reference_start = -1
    read.template_length = 0
```

**Lesson:** "0 PulseWidths" was a red herring. Always confirm the flag
byte matches the canonical raw HiFi shape (`samtools view -c -f 4`) when
emitting unaligned BAMs for downstream PacBio tooling.

---

## Observation — bystrandify message is misleading

The warning text suggests a tag-content problem ("0 PulseWidths") when
the real cause is flag validation. Worth keeping in mind when debugging
future cases: don't trust the warning text, inspect the actual tags AND
the flag.

---

## Non-bugs verified during the same run

* **Model output is good.** First sampled read fp values: min=0,
  max=112, mean=23.74, 105 unique bytes — real model signal, not
  default placeholder.
* **Precompute path works.** 12.5 M model inferences across 32 z samples
  filled the 1050 MB kin_map for ctg.s1; multiprocess workers correctly
  looked up the cached values.
* **REF path discrepancy** between `validate.sh` and reality —
  validate.sh expects `pipeline/<sample>/<sample>_assembly.fasta`,
  but for bc2034 the actual file is at `<sample>/final_assembly.fasta`.
  Already handled by the `[ -f "$REF" ] || REF=...` fallback in
  validate.sh; just noting it.

---

## After-the-fact follow-ups (post-Friday)

* The misleading bystrandify warning argues for a per-read flag
  pre-flight check in the validate chain: assert all reads in SIM.bam
  have flag=4 before submitting bystrandify, fail loudly otherwise.
* The `mate_is_unmapped=True` mistake came from copy-pasting a standard
  "make read unmapped" snippet that assumes paired-end. KinSim/kinsim_NN
  generate is single-end CCS only — worth documenting the canonical
  unalign recipe somewhere central (e.g. in `utils/bam_io.py`).
