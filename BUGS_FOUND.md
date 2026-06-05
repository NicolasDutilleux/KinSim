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

---

## Bug 11 — Stripping `@RG DS` codec declarations crashes `ipdSummary` with `KeyError: 'Ipd'`

**Symptom.** With the bug 10 fix in place, the v6 validation chain
sailed past bystrandify (248 382 records — exactly 2× the 124 191 ZMW
input) and pbmm2 align (248 586 mapped reads), then `ipdSummary`
crashed in ~5 minutes with `KeyError: 'Ipd'` raised inside every
`KineticWorkerProcess`:

```
File ".../kineticsTools/KineticWorker.py", line 484, in _loadRawIpds
    rawIpd = aln.IPD() * factor
File ".../pbcore/io/align/BamAlignment.py", line 50, in f
    return self.baseFeature(featureName, aligned, orientation)
File ".../pbcore/io/align/BamAlignment.py", line 532, in baseFeature
    concreteFeatureName = self.bam._baseFeatureNameMappings[self.qId][featureName]
KeyError: 'Ipd'
```

**Root cause.** Earlier in the v6 debugging cycle (commit `0c2bd2f`),
`scripts/strip_bystrandified_to_hifi.py` was modified to remove the
`Ipd:CodecV1=ip;` and `PulseWidth:CodecV1=pw;` substrings from the
`@RG DS` field on the suspicion that those declarations were causing
bystrandify to silently drop records. They were not — bug 10 (missing
`fn`/`rn`) was the actual cause, and the wrong-track DS cleanup
remained in the script after that fix.

`pbcore` (the BAM I/O library used internally by `kineticsTools`)
builds a per-read-group lookup `_baseFeatureNameMappings[qId]` from
exactly those `Ipd:CodecV1=...` and `PulseWidth:CodecV1=...`
substrings. When `ipdSummary` calls `aln.IPD()`, the lookup queries
this mapping with the feature name `"Ipd"`. With the codec
declarations stripped from `@RG DS`, the mapping is empty and the
query raises `KeyError: 'Ipd'`.

**Fix.** Remove the `_clean_rg_ds` / `_clean_header` helpers from
`scripts/strip_bystrandified_to_hifi.py` and pass `bam_in.header`
through to the output verbatim. The codec declarations are a static
PacBio convention present on every PacBio BAM (raw HiFi and
bystrandified alike) and stripping them is never a correct move.

**Lesson.** A "wrong-track" fix that does not break the immediate
symptom can survive several diagnostic cycles. After the real cause
(bug 10) was identified, the earlier band-aid still needed to be
reverted — its consequences only surfaced once the chain ran further
than the bystrandify step. Bisect-style attribution: if a partial fix
made it through review without being reverted, audit it before
claiming the chain is unblocked.

---

## Bug 12 — PBSIM3 read names are not parseable by `ccs-kinetics-bystrandify` / `ipdSummary`

**Symptom.** Feeding a PBSIM3-simulated HiFi BAM directly into the
downstream PacBio chain (with `kinsim_nn generate` having injected
synthetic `fi/fp/ri/rp`) caused tool-specific failures: `pbindex`
either failed outright or produced an empty `.pbi`,
`ccs-kinetics-bystrandify` silently dropped every record (the same
silent-failure pattern as bugs 1–5, 10), and `ipdSummary` crashed
inside `pbcore` while extracting the ZMW number.

**Root cause.** PacBio HiFi BAM convention requires the read name to
match `m<MovieName>/<HoleNumber>/ccs`, where `<HoleNumber>` is a
non-negative decimal integer. `pbcore` parses the second
slash-separated token as the ZMW hole number and indexes records by
it. PBSIM3 emits simulator-native names (e.g. `S1_1`, `S1_2`, …,
optionally prefixed by an organism tag) that do not contain the
required `m...`/`<int>`/`ccs` structure. The PacBio readers either
fail the integer parse or fall through to a code path that assumes a
sentinel ZMW value and discards the record.

**Fix (workflow level).** Before piping PBSIM3 output through KinSim's
generation step, rewrite read names to a PacBio-conformant template
with a synthetic movie tag and a monotonically increasing integer
hole number. Reserved for a future preprocessing helper
(`scripts/pbsim3_to_pacbio_names.py`); the strip-and-rename pattern
mirrors the bystrandified-to-HiFi script we already maintain.

**Lesson.** PacBio tools do not just consume a BAM — they consume a
BAM under a structured read-name convention. Format conformance at
the byte level (flags, tags, codec) is necessary but not sufficient;
the read name itself is a load-bearing field. Validate the name
pattern before submitting upstream of any PacBio binary.

---

## Bug 13 — Test-strain shards leak into the training set (`list_shards` exclusion logic)

**Symptom.** The training log reported `Training shards: 65
test_strains=('bc2034', 'bc2045', 'bc2048', 'bc2082')` for the v6
bilateral run on a corpus of exactly 65 bilateral shards (58 training
+ 7 test shards). The "test_strains exclusion" produced *the entire
corpus*. The held-out W1 evaluator separately re-found the test
shards via `glob(f"*{sid}_shard.pkl")` (line 387) and dutifully
reported W1 numbers — but those numbers were measured on data the
model had already seen during training. The champion W1 = 2.017 at
step 45 000 is therefore not a held-out generalisation metric on the
v6 bilateral run; it is a training-set fidelity metric.

**Root cause.** `kinsim_NN/data/dataset.py` `list_shards()` used
`if sid in exclude_strains: continue`, where `sid` is the full
sample_id like `"strepto_bc2034"` and `exclude_strains` is the set
from `cfg.split.test_strains` like `{"bc2034", "bc2045", …}`. The
`in` check is exact-membership — `"strepto_bc2034"` is not in
`{"bc2034", …}`, so the exclusion never fires for lineage-prefixed
shards. The eval logic used a glob with a wildcard prefix
(`*{sid}_shard.pkl`) which happens to be permissive; the training-side
exclusion was not.

**Fix.** Match the test_strain against both the full sample_id and
the trailing underscore-separated component:

```python
sid_tail = sid.rsplit("_", 1)[-1]
if sid in exclude_strains or sid_tail in exclude_strains:
    continue
```

With this, `"strepto_bc2034".rsplit("_", 1)[-1] == "bc2034"` is in
the exclusion set and the shard is correctly held out.

**Impact on v6 results.** Every W1 number reported from the
2026-05-31 v6 bilateral training run (`train_v6_bilateral_17274833`)
was computed on data the model had been trained on. The 28 %
improvement vs the legacy ConvPredictor remains *directionally*
meaningful (the metric definition and corpus did change in ways
documented in the thesis), but the absolute number cannot be cited
as a held-out generalisation result without re-evaluating against
shards that were actually excluded from training. Easiest path
post-fix: take the snapshotted `best_G_step45k_snapshot.pt`, run
`kinsim_nn evaluate` on the test_strains shards with the corrected
`list_shards`, and report whatever number falls out. Document the
gap from the corrected number to the published one as a thesis
limitation.

**Lesson.** Two pieces of code that look like they should agree on
"is this shard in the test split?" can silently disagree when one
uses set membership and the other uses fuzzy glob matching. When a
sample-identifier appears in multiple places (YAML, filename glob,
set membership), normalise it in exactly one place and pass the
canonical form everywhere. The split logic needs a single source of
truth — either the YAML carries full sample_ids (`strepto_bc2034`,
`vega_bc2034`) explicitly, or both training and eval use the same
helper that does the suffix-match. Either, not both.
