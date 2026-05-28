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
