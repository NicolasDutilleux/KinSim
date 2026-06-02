# PacBio tool compatibility

Hard-won rules for chaining `pbmm2`, `ccs-kinetics-bystrandify`,
`ipdSummary`, `pbmotifmaker`, `jasmine`, and `modkit` without silent
data loss. Every rule here is the consequence of a bug we hit and
recovered from — see [`BUGS_FOUND.md`](BUGS_FOUND.md) for the full
case history of each.

The single fact that makes this document necessary:
**`ccs-kinetics-bystrandify` rejects records silently.** No stderr
warning, no log line, just a smaller (or empty) output BAM. Every
"silent drop" rule below was diagnosed by tag-profile diff against a
known-working raw HiFi BAM (`samtools view tag.bam | head -1`), not by
reading the tool's output.

---

## 1. Tool source matrix

Apptainer SIF first for everything PacBio. The cluster's conda
`kinsim_env` has *newer* `pbmm2` (26.x) that **cannot read** BAMs
produced by the SIF's `pbmm2` 1.18.0 — never mix sources inside a chain.

| Tool | Source | Version | Why this source |
|---|---|---|---|
| `pbmm2` | SIF | 1.18.0 | conda 26.x rejects SIF-produced BAMs |
| `pbindex` | SIF | 3.5.0 | matches `pbmm2` in the SIF |
| `ccs-kinetics-bystrandify` | SIF | 3.5.0 | only place it ships |
| `pbmotifmaker` | **Falquet host install** (SMRT-Link 25.1) | 1.1.0 | SIF's 1.2.0 is more stringent at the same `--min-score` and recovers fewer motifs |
| `ipdSummary` | **Falquet host install** (SMRT-Link 25.1) | 3.0 (SP3-C3 model) | not shipped in the SIF; the cluster module `SMRT-Link/12.0.0` produces lower detection rates than Falquet's reference |
| `jasmine` | conda (`kinsim_env`) | 2.4.0 | **not in the SIF**; matches the Revio production version |
| `modkit` | conda (`kinsim_env`) | 0.6.1 | rust binary, never shipped in the SIF |
| `samtools` | conda or SIF | 1.20 / 1.21 | plain htslib, stable across versions |

**Invocation form** (always `--bind /data` so paths under `/data/` resolve inside the container):

```bash
SMRT_SIF=/containers/apptainer/pacbio-smrt-tools-25.3.sif
apptainer exec --bind /data "$SMRT_SIF" <tool> <args...>
```

**For `ipdSummary` and `pbmotifmaker`** (Falquet 25.1 host install):

```bash
FALQUET=/data/users/lfalquet/SOFTS/SMRTlink/smrtlink251R
"$FALQUET/smrtcmds/bin/ipdSummary"   <args...>
"$FALQUET/smrtcmds/bin/pbmotifmaker" <args...>
```

The SP3-C3 model lives at:
`$FALQUET/install/.../kineticsTools/resources/SP3-C3.npz.gz`

---

## 2. Apptainer poisoning by `module load SMRT-Link`

`module load SMRT-Link/12.0.0.177059-cli-tools-only` mutates `PATH`,
`LD_LIBRARY_PATH`, and `PYTHONPATH`. Apptainer inherits the host
environment by default, so a module loaded *before* `apptainer exec`
breaks SIF tools — observed: `pbmm2 1.18` in the SIF emits *"Could
not determine read input type(s)"* when the module was loaded earlier
in the same shell.

**Rule.** If you also need `ipdSummary` from the SMRT-Link module
(i.e. the *cluster* module, not the Falquet host install), load it
**lazily** right before the `ipdSummary` invocation, never in the
script header. Equivalent escape hatch:
`apptainer exec --cleanenv --bind /data "$SIF" ...` on every PacBio
invocation.

Since the project pins `ipdSummary` to Falquet's 25.1 (not the module),
this issue does not arise on the v6 chain.

---

## 3. `ccs-kinetics-bystrandify` silent-drop rules

Bystrandify accepts a HiFi-shaped BAM (one record per ZMW, unaligned,
with raw kinetics `fi`/`fp`/`ri`/`rp`) and emits two records per ZMW
(`<zmw>/fwd` and `<zmw>/rev`) carrying per-strand `ip`/`pw`. It
discards records silently — every rule below was diagnosed by tag-diff,
not by stderr.

A single record is dropped when **any** of the following hold:

1. **SAM flag is not exactly `4`.** PacBio HiFi convention is `flag = 4`
   (unmapped, single-end). Anything else — even `flag = 12`
   (`mate_is_unmapped` added) or `flag = 20` (`is_reverse` set) —
   silently discards the record. See bug 1 in [`BUGS_FOUND.md`](BUGS_FOUND.md).

2. **`@HD SO` is `coordinate`.** Bystrandify routes records into the
   aligned-input code path on `SO:coordinate`, then rejects each one
   for lacking alignment fields. The unaligned output must carry
   `@HD SO:unknown` (or `unsorted`). See bug 2.

3. **Stale `ip` or `pw` tags are present.** The `@RG DS` field
   `Ipd:CodecV1=ip;PulseWidth:CodecV1=pw` declares the codec for the
   bystrandified output, but on a raw HiFi BAM the kinetics live in
   `fi`/`fp`/`ri`/`rp`. If a prior `pbmm2 align` left stale `ip`/`pw`
   on the records, bystrandify reads those (per the @RG) and discards
   the record on the empty/junk data. Strip `ip` and `pw` before
   writing. See bug 3.

4. **`TLEN = 0` (column 9).** Bystrandify uses non-zero TLEN as a
   heuristic to recognise a HiFi record. The spec-correct
   `template_length = 0` for an unmapped single-end record triggers
   the silent discard. Inherit the value from the aligned input
   (typically equal to the read length). See bug 4.

5. **Any `0` byte in `fi`, `fp`, `ri`, or `rp`.** The PacBio uint8
   frame-count codec uses `[1, 255]`; `0` denotes "missing/invalid".
   Even one zero in the kinetic arrays drops the record. Clamp the
   four arrays to `≥ 1` immediately before `set_tag`. See bug 5.

6. **Missing `fn` tag.** Per-strand subread count for the forward
   strand (`fn:i:N`, scalar int). Without it, bystrandify emits only
   the `/rev` record; the `/fwd` is silently dropped.

7. **Missing `rn` tag.** Same for the reverse strand. Without it, only
   `/fwd` is emitted.

8. **Missing both `fn` AND `rn`.** Both records are dropped — every
   ZMW is gone. Output BAM has the header only, zero records.
   Diagnosed 2026-06-02 by tag ablation on a known-good Sequel raw
   HiFi BAM (200 reads, target 392 bystrandified records, missing
   both → 0).

`kinsim_NN/generate.py` writes `fn:i:1` and `rn:i:1` (the value `1` is
arbitrary — bystrandify only checks presence, not magnitude). The
arrays-vs-scalar typing matters: `fn` and `rn` are scalar ints, not
per-base arrays.

**`ec` (effective coverage, scalar float)** is *not* required —
bystrandify accepts records without it.

---

## 4. The `@RG DS` field is information, not configuration

A typical PacBio HiFi `@RG` looks like:

```
@RG ID:... PL:PACBIO PM:REVIO PU:m... LB:... SM:...
    DS:READTYPE=CCS;Ipd:CodecV1=ip;PulseWidth:CodecV1=pw;BINDINGKIT=...;...
```

The `Ipd:CodecV1=ip` and `PulseWidth:CodecV1=pw` declarations are
**normal** on raw HiFi BAMs even when the actual kinetics live in
`fi`/`fp`/`ri`/`rp`. They tell *bystrandified* tooling where to put
the post-bystrandify kinetics, not where to read them from on the raw
input. Do **not** strip them — verified 2026-06-02 by comparing the
@RG of a known-good Sequel raw HiFi (which has these declarations and
bystrandifies fine) against a kinsim BAM (where stripping them changed
nothing).

The DS field order is not load-bearing for any of the chain tools we
have tested.

---

## 5. Bystrandified BAM conventions

After `bystrandify`, each ZMW becomes two records:

- `<readname>/fwd` — `is_reverse = 0` after `pbmm2 align`, carries
  `ip = fi` and `pw = fp` (forward-strand kinetics).
- `<readname>/rev` — `is_reverse = 1` after `pbmm2 align`, carries
  `ip = reverse-complement(ri)` and `pw = reverse-complement(rp)`.

**Invariant.** After `pbmm2 align`, the `/fwd` and `/rev` records of
the *same* ZMW always have **opposite** `is_reverse` flags. Same
`is_reverse` on both is the pathological case that caused v3
`extract samples=0` — fixed in `7ae9d66`. If you re-implement an
extract pass, assert this invariant on at least one ZMW per shard
during a sanity run.

---

## 6. PacBio uint8 kinetic codec

`fi`, `fp`, `ri`, `rp`, `ip`, `pw` are stored as `B:C` arrays
(unsigned uint8). The codec range is `[1, 255]`; `0` is reserved for
"invalid/missing":

| codec value | meaning |
|---|---|
| `0` | invalid / missing / discard the record |
| `1` | shortest valid frame count |
| `2..255` | log-spaced frame counts (see `kinsim_NN/utils/pacbio_codec.py`) |

Always clamp generator output with `np.maximum(arr, 1, out=arr)`
before writing. The clamp lives in `kinsim_NN/generate.py` immediately
above `read.set_tag("fi", …)` in both the multiprocess and
single-threaded branches.

---

## 7. `pbmm2` cannot read across versions

The cluster has two `pbmm2` installations:

- **SIF 1.18.0** — required by `ccs-kinetics-bystrandify` 3.5.0 output.
- **conda 26.x** — newer, but rejects SIF-produced BAMs with
  *"Could not determine read input type(s)"*.

Never mix sources inside one chain. The validation chain uses the
SIF's `pbmm2` for both the bystrandified→aligned alignment after the
generator and any prior alignments — keep the version line consistent.

---

## 8. `jasmine` 5mC model chemistry mismatch (Vega)

`jasmine 2.4.0` ships a 5mC CNN trained on the Revio P2-C2 chemistry.
Vega (P1-C1) data triggers a *WARN* on every read — the call is
quantitatively biased but qualitatively usable. This cannot be fixed
without retraining the jasmine model. We acknowledge the warning and
proceed.

---

## 9. The downstream chain end-to-end (what shapes look like)

```
kinsim_nn generate → output.bam
    1 record per ZMW, unaligned (flag=4, SO=unknown, @SQ stripped)
    tags: RG, np, rq, zm, fi, fp, ri, rp, fn=1, rn=1 (+ all preserved aux)

ccs-kinetics-bystrandify → output_bys.bam
    2 records per ZMW (/fwd, /rev), still unaligned
    tags: RG, np, rq, zm, ip, pw

pbmm2 align (SIF 1.18) → output_aln.bam
    sorted aligned BAM (SO:coordinate, @SQ from ref)
    is_reverse on /fwd and /rev is OPPOSITE per ZMW (invariant)

ipdSummary (Falquet 25.1, SP3-C3) → basemods.gff + basemods.csv
    per-position m6A and m4C calls

pbmotifmaker find (Falquet 25.1) → motifs.csv
    motif-confirmed positions with fraction and meanScore
```

Comparing this output to the real-data ipdSummary GFF at the same
positions is the validation test — `scripts/plot_perread_ipd_at_gff_sites.py`.

---

## 10. Quick "is my BAM HiFi-shaped?" check

Before feeding a BAM to `ccs-kinetics-bystrandify`, sanity-check with:

```bash
samtools view -H BAM | grep '^@HD'                     # SO:unknown ?
samtools view BAM | head -1 | awk -F'\t' '{print "flag="$2, "tlen="$9}'  # flag=4, tlen != 0 ?
samtools view BAM | head -1 | tr '\t' '\n' | grep -E '^(fi|fp|ri|rp|fn|rn|RG):' | head
# expect: fi:B:C,..., fp:B:C,..., ri:B:C,..., rp:B:C,..., fn:i:N, rn:i:N, RG:Z:...
samtools view BAM | head -1 | tr '\t' '\n' | grep -E '^(ip|pw):'
# expect: nothing (ip/pw must be absent on raw HiFi input)
```

If anything from `fi, fp, ri, rp, fn, rn` is missing, bystrandify will
silently drop records.

---

## Sources / version evidence

Versions in section 1 verified empirically 2026-04-22 by `--version`
queries across SIF and conda. Re-verify with the probe block in
[`.claude/skills/kinsim/SKILL.md`](.claude/skills/kinsim/SKILL.md) if
you suspect drift.
