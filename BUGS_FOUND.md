# Bugs found during bc2034 validate run (2026-05-26)

Context: re-running `kinsim_NN generate` on bc2034 with v2 changes
(per_read_z + emit_unaligned by default) then chaining bystrandify →
align → ipdSummary → motifmaker. Found via comparison with the prior
v12_run3 manual run.

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
