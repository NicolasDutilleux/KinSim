# Bugs found during validation

Each entry: symptom, root cause, fix, lesson. All five share a single
misleading downstream warning (`ccs-kinetics-bystrandify`: `has '0'
PulseWidths, discarding`), surfaced sequentially during the bc2034 chain
validation in 2026-05.

The methodology that ultimately localised each bug: byte-level
`samtools view` diff between a known-working legacy record and a
rejected new one, then targeted controlled substitutions.

---

## Bug 1 — `flag = 4` violation on the unaligned output

**Symptom.** Bystrandify discarded 99.8 % of records. Output ~6 MB
where the legacy chain produced ~4 GB.

**Root cause.** The emission code set `mate_is_unmapped = True` on the
unaligned record, which silently added bit 8, raising the SAM flag from
4 to 12. PacBio HiFi convention requires exactly `flag = 4` (unmapped,
single-end) on raw HiFi BAMs.

**Fix.** Set `read.flag = 4` directly in `_unalign_read`.

**Lesson.** Strictly equal-to checks on flag values in downstream
PacBio tools — bit semantics are not always honoured.

---

## Bug 2 — `@HD SO:coordinate` left on unaligned output

**Symptom.** With `flag = 4` correct, bystrandify still discarded
99.8 % of records.

**Root cause.** `_sanitize_header_for_unaligned` stripped `@SQ` lines
but inherited `@HD SO:coordinate` from the aligned input. That value
routed bystrandify into its aligned-input code path, which then
rejected every record now lacking alignment fields.

**Fix.** Force `SO:unknown` on the `@HD` line when emitting an
unaligned output.

**Lesson.** Header-level convention can override per-record content in
downstream tools' control flow.

---

## Bug 3 — stale `ip` / `pw` tags inherited from `pbmm2 align`

**Symptom.** With Bugs 1 and 2 fixed, the chain still rejected 99.8 %.

**Root cause.** The `@RG DS` field declares
`Ipd:CodecV1=ip;PulseWidth:CodecV1=pw`, telling downstream tools to
read kinetics from `ip` / `pw`. When the aligned input carried
per-strand `ip` / `pw` tags from a prior `pbmm2 align`, those survived
into the generated BAM and were preferred by bystrandify over the
freshly written `fi` / `fp` / `ri` / `rp`.

**Fix.** Strip `ip` and `pw` in `_unalign_read` before writing.

**Lesson.** When converting an aligned PacBio BAM into raw-HiFi shape,
alignment-induced tag rewrites must be stripped, not only the
alignment fields themselves.

---

## Bug 4 — `template_length = 0` triggers bystrandify discard

**Symptom.** With Bugs 1–3 fixed, bystrandify still rejected 99.8 %.

**Root cause.** Bystrandify uses `TLEN > 0` as a heuristic to detect a
"valid HiFi record". The `_unalign_read` code had set
`template_length = 0` on the unaligned output, which is spec-correct
for an unmapped single-end record but triggered the discard.

**Fix.** Leave `template_length` untouched in `_unalign_read`. The
inherited value (typically the read length) satisfies bystrandify's
heuristic.

**Lesson.** Diff one specific record column-by-column against a
known-working sample; do not trust the warning text.

---

## Bug 5 — zero values in `fi` / `fp` / `ri` / `rp` are codec-invalid

**Symptom.** With Bugs 1–4 fixed, bystrandify still discarded a small
but non-zero fraction of records under the same "0 PulseWidths"
warning.

**Root cause.** The PacBio uint8 frame-count codec uses values in
`[1, 255]`; byte value `0` denotes "missing / invalid".
`ccs-kinetics-bystrandify` discards any record containing a `0` in its
kinetic arrays. The generator's log-space output occasionally rounds
down to `0` (~0.1 – 0.3 % of positions per read).

**Fix.** Clamp the four kinetic arrays to `≥ 1` immediately before
`set_tag`, in both the sequential and the multiprocess paths of
`kinsim_NN/generate.py`:

```python
np.maximum(fi, 1, out=fi)
np.maximum(fp, 1, out=fp)
np.maximum(ri, 1, out=ri)
np.maximum(rp, 1, out=rp)
```

**Lesson.** The decisive diagnostic was a controlled substitution
(replace generated kinetics with a constant) rather than further
structural diffing.

---

## Aggregate effect

End-to-end yield on the bc2034 test strain, after all five corrections,
went from ~250 retained per-position kinetic calls to **16.7 million**.

---

# Bugs found during code review (post-pipeline-validation)

A subsequent code-review pass surfaced four correctness defects in the
extraction, evaluation, and generation paths that were independent of
the BAM-emission boundary bugs above. All four are now corrected.

## Bug 6 — Jasmine MM/ML labeler mis-indexed C positions on reverse-mapped reads

**Symptom.** Approximately half of the 5mC calls produced by
`JasmineMMMLLabeler` were placed at incorrect reference positions on
reverse-mapped reads.

**Root cause.** Per the SAM specification, the `MM` tag enumerates
modified-base positions in the **original (forward) read orientation**,
regardless of the alignment direction. The labeler enumerated C
positions in `query_sequence`, which `pysam` reverse-complements when
`read.is_reverse` is set. For those reads, the C indices used to decode
MM deltas no longer corresponded to the bases MM was describing.

**Fix.** When `read.is_reverse`, enumerate C positions in
`read.get_forward_sequence()` (the original CCS orientation) and convert
each forward-frame index to its BAM-SEQ index (`n − 1 − i`) for the
alignment lookup. Forward-mapped reads are unchanged.

**Lesson.** Tags that the SAM specification defines in the
original-read frame must not be indexed via `query_sequence` for
reverse-mapped records.

## Bug 7 — Baseline samples double-counted in the held-out W1 evaluation

**Symptom.** The `w1_baseline` and `w1_overall` metrics were
systematically inflated relative to the per-meth W1 buckets, making
cross-bucket comparison meaningless.

**Root cause.** Both the in-training evaluator
(`kinsim_NN.train._evaluate_on_shards`) and the standalone
`kinsim_NN.evaluate` appended *two* values (channels 0 and 2) to the
baseline pool while methylation buckets received only one. Baselines
were therefore weighted twice in the pool.

**Fix.** Append exactly one value per sample to every bucket. The
baseline branch now uses channel 0 only, picked deterministically so
the bucketing is reproducible across runs.

## Bug 8 — Palindromic methylation sites dropped from the held-out W1 evaluation

**Symptom.** Roughly half of the methylated-position signal on
palindromic motifs (e.g. m6A on both strands of `GATC`) was silently
omitted from the eval pool.

**Root cause.** The same evaluator used an `if mf > 0 / elif mr > 0`
chain that picked only one strand when both were methylated. On
palindromic motifs this discarded the contribution of the other
strand. In Strepto, where palindromic sites dominate, the loss was
substantial.

**Fix.** Replace the `elif` with an independent `if`: when both
strands carry a methylation, append to both per-meth buckets. The
per-category bucket still receives only one contribution per sample.

## Bug 9 — Generator RNG seeded after the precompute path consumed randomness

**Symptom.** `kinsim_nn generate --seed N` did not produce
byte-reproducible BAM output across runs that used the precompute fast
path.

**Root cause.** `torch.manual_seed`, `np.random.seed`, and
`random.Random(seed)` were called after `_load_generator` and after
ancillary helpers that may consume PRNG state. Although in the current
code path no consumer actually drew before the seed call, the ordering
was load-bearing and brittle.

**Fix.** Move the seeding to the top of `kinsim_NN.generate.generate`,
immediately after `setup_logging` and before any other call. Also
seed `torch.cuda.manual_seed_all` to cover CUDA streams.

## Aggregate effect (bugs 6–9)

The held-out W1 estimates produced after bugs 7 and 8 are corrected
are directly comparable across buckets, the 5mC subset of labelled
positions now reflects the jasmine catalogue without orientation bias,
and `kinsim_nn generate --seed N` is reproducible end-to-end.

---

## Bug 10 — Missing `fn` / `rn` per-strand subread-count tags drop every record

**Symptom.** When feeding the v6 validation chain a kinsim BAM (124 k
unaligned records with valid `fi`/`fp`/`ri`/`rp`, all per-record
checks green — flag = 4, SO = unknown, TLEN ≠ 0, no zero in kinetic
arrays, no stale `ip`/`pw`), `ccs-kinetics-bystrandify` produced an
output BAM with the header only and zero records. Silent on stderr.
Downstream `ipdSummary` then errored with `OSError: No mapped reads
found`.

**Diagnostic methodology.** Comparing the tag profile of the broken
kinsim BAM against a Sequel raw HiFi BAM that bystrandifies cleanly
(`samtools view tag.bam | head -1 | tr '\t' '\n' | grep ':'`)
revealed two scalar tags present on the working input and absent on
the kinsim BAM: `fn:i:N` (forward subread count) and `rn:i:N` (reverse
subread count).

A controlled ablation on the working raw HiFi BAM (target output 392
records on a 196-read tiny subset) confirmed the role of each tag:

| ablation | output records |
|---|---|
| baseline | 392 |
| minus `ec` | 392 (not required) |
| minus `fn` | 196 (only `/rev` emitted) |
| minus `rn` | 196 (only `/fwd` emitted) |
| minus both `fn` AND `rn` | **0** |

**Root cause.** Bystrandify uses the presence of `fn` to emit the
`/fwd` record and the presence of `rn` to emit the `/rev` record.
Neither has anything to do with the actual subread counts — only
presence is checked. `kinsim_NN.generate` did not write either tag,
so every record was double-rejected.

**Fix.** Write `fn:i:1` and `rn:i:1` alongside the kinetic-array
`set_tag` calls, in both the multiprocess and single-threaded branches
of `kinsim_NN/generate.py`. Value `1` is arbitrary; bystrandify checks
presence only.

**Lesson.** When a PacBio tool is silent on rejection, the tag-profile
diff against a known-working input — not the documented field
semantics — is the only reliable diagnostic. Cf. the SAME methodology
that resolved bugs 1–5.
