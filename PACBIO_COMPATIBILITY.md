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

Default to the Apptainer SIF (SMRT-Tools 25.3) for PacBio binaries
that ship in it — and crucially **never mix sources inside one chain**.
The cluster's conda `kinsim_env` has a newer `pbmm2` (26.x) that
**cannot read** BAMs produced by the SIF's `pbmm2` 1.18.0; a chain that
bystrandifies through the SIF and then aligns through conda breaks
silently with *"Could not determine read input type(s)"*. Two
exceptions to the SIF default: `ipdSummary` and `pbmotifmaker` are
pinned to L. Falquet's host install of SMRT-Link 25.1 because the
SIF's 25.3 versions of these are more stringent (`pbmotifmaker` 1.2.0)
or absent (`ipdSummary` not shipped in the SIF) and produce lower
detection rates against the reference catalogue.

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

The SP3-C3 kinetics model (used by `ipdSummary --ipdModel`) lives at:

```
$FALQUET/install/smrtlink-release_25.1.0.257715/bundles/smrttools/install/smrttools-release_25.1.0.257715/private/pacbio/python3pkgs/kineticstools-py3/lib/python3.9/site-packages/kineticsTools/resources/SP3-C3.npz.gz
```

This is NOT the same SP3-C3 file as the one bundled with the cluster's
`SMRT-Link/12.0.0.177059-cli-tools-only` module (which sits under
`/mnt/ss/sib/ibu/rocky8/...`). Use Falquet's path with Falquet's
`ipdSummary` for chain consistency; mixing the 25.1 binary with the
12.0 model has not been validated.

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

3. **Stale `ip` or `pw` tags are present on the records.** The
   `@RG DS` field declares `Ipd:CodecV1=ip;PulseWidth:CodecV1=pw` as a
   PacBio convention regardless of whether actual `ip`/`pw` tags exist
   on the reads. When `pbmm2 align` runs on a bystrandified input it
   leaves per-strand `ip`/`pw` on the aligned records; if those
   survive a downstream HiFi-shape conversion, bystrandify prefers
   them over `fi`/`fp`/`ri`/`rp` and discards the record on the
   empty / wrong-shape data. Strip `ip` and `pw` from every record
   before writing — but leave the `@RG DS` declaration alone (see
   section 4). See bug 3.

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

## 4. The `@RG DS` field is a static convention, not a read-from pointer

A typical PacBio HiFi `@RG` looks like:

```
@RG ID:... PL:PACBIO PM:REVIO PU:m... LB:... SM:...
    DS:READTYPE=CCS;Ipd:CodecV1=ip;PulseWidth:CodecV1=pw;BINDINGKIT=...;...
```

`Ipd:CodecV1=ip` and `PulseWidth:CodecV1=pw` are codec-name
declarations: "if a tag named `ip` is present, its codec is `V1`;
same for `pw`". They are **always** present on PacBio BAMs (raw HiFi
and bystrandified alike) by tool convention — verified 2026-06-02 by
comparing the @RG of a known-good Sequel raw HiFi (which has these
declarations and bystrandifies cleanly) against a kinsim BAM (where
stripping them changes nothing).

Practical consequence: do **not** strip these declarations. The
silent-drop trigger of bug 3 was the *presence of stale `ip`/`pw`
tags on the records*, not the DS field — see section 3 item 3.

The DS-field key order is not load-bearing for any chain tool we have
tested (rebuilding the @RG via `pysam.AlignmentHeader.from_dict`
reorders fields and bystrandify still accepts the result).

---

## 5. Bystrandified BAM conventions

After `bystrandify`, each ZMW becomes two records distinguished by a
suffix on the read name:

- `<readname>/fwd` — carries forward-strand kinetics:
  `SEQ = original HiFi consensus`, `ip = fi`, `pw = fp`.
- `<readname>/rev` — carries reverse-strand kinetics:
  `SEQ = reverse-complement of HiFi consensus`,
  `ip = reverse-complement(ri)`, `pw = reverse-complement(rp)`.

The `/fwd` / `/rev` suffix refers to which strand's kinetics the
record carries, **not** to its alignment direction. Both records of
the same ZMW go through `pbmm2 align` independently and either may
end up forward- or reverse-aligned in the BAM depending on which
strand of the reference the original HiFi consensus came from.

**Invariant.** After `pbmm2 align`, the `/fwd` and `/rev` records of
the *same* ZMW always have **opposite** `is_reverse` flags — because
their SEQs are reverse-complements of each other, they cannot align
in the same orientation. Same `is_reverse` on both is the pathological
case that caused the v3 `extract samples=0` regression (root-cause fix
in commit `7ae9d66`). If you re-implement an extract or downstream
pass that pairs `/fwd` with `/rev`, assert this invariant on at least
one ZMW per shard during a sanity run.

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

Two `pbmm2` versions are reachable on the cluster:

- **SIF 1.18.0** — ships with SMRT-Tools 25.3, paired with the SIF's
  `ccs-kinetics-bystrandify` 3.5.0. Used by every alignment in the
  validation chain.
- **conda 26.x** — newer, lives in `kinsim_env`. **Rejects** BAMs
  produced by the SIF's `pbmm2` 1.18.0 with the message
  *"Could not determine read input type(s)"*. Verified empirically on
  the bc2071 production chain.

Rule: pin every `pbmm2` invocation in a chain to the same installation.
Practically, this means the validation chain uses the SIF's `pbmm2`
for the post-bystrandify alignment, and the training-data BAMs we
strip from were also aligned with the SIF's `pbmm2` — the version line
is internally consistent.

If you ever need to feed a BAM produced by one `pbmm2` into the other,
re-emit the alignment with the receiving version first.

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
    1 record per ZMW, unaligned
    @HD SO:unknown, @SQ stripped, flag=4 on every record
    kinetics: fi, fp, ri, rp (uint8 B:C arrays, clamped ≥1)
    scalar gating tags: fn=1, rn=1
    PacBio aux: RG, np, rq, zm, sn, qs, qe, cx, ec, ws, etc.

ccs-kinetics-bystrandify (SIF 3.5.0) → output_bys.bam
    2 records per ZMW with /fwd and /rev read-name suffixes,
    still unaligned (flag=4)
    kinetics consumed: fi, fp, ri, rp → per-strand ip, pw
    (the fn/rn scalars are also consumed at this step)

pbmm2 align --preset CCS --sort (SIF 1.18) → output_aln.bam
    sorted aligned BAM (@HD SO:coordinate, @SQ from ref)
    each ZMW's /fwd and /rev have OPPOSITE is_reverse (section 5)
    samtools view -F 4 to drop the rare unmappable reads before
    feeding ipdSummary — kineticsTools dies on unmapped records
    with "No mapped reads found"

ipdSummary (Falquet 25.1, --ipdModel SP3-C3.npz.gz) → basemods.gff + basemods.csv
    per-position m6A and m4C calls with methylation fraction

pbmotifmaker find --min-score 40 (Falquet 25.1) → motifs.csv
    motif-confirmed positions with motifString, centerPos, fraction,
    nDetected, nGenome, meanScore
```

Comparing this output to the real-data ipdSummary GFF at the same
positions is the validation test
([`scripts/plot_perread_ipd_at_gff_sites.py`](scripts/plot_perread_ipd_at_gff_sites.py)).

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
queries across SIF and conda. To re-verify on the cluster (one-shot
srun, ~5 min):

```bash
srun --partition=pibu_el8 --account=p774 --time=00:05:00 --mem=2G --cpus-per-task=1 bash -c '
SIF=/containers/apptainer/pacbio-smrt-tools-25.3.sif
echo "=== SIF tools ==="
for t in pbmm2 pbindex ccs-kinetics-bystrandify pbmotifmaker; do
  echo -n "  $t: "
  apptainer exec --bind /data "$SIF" "$t" --version 2>&1 | head -1
done
echo "=== Falquet 25.1 host install ==="
F=/data/users/lfalquet/SOFTS/SMRTlink/smrtlink251R/smrtcmds/bin
for t in ipdSummary pbmotifmaker; do
  echo -n "  $t: "
  "$F/$t" --version 2>&1 | head -1
done
echo "=== conda kinsim_env ==="
source ~/.bashrc; conda activate kinsim_env
for t in jasmine modkit samtools; do
  echo -n "  $t: "; "$t" --version 2>&1 | head -1
done
'
```

If anything drifts, update the table in section 1 and the references
in `slurm/callers/*.slurm` together.
