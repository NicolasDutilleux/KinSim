# Design decisions

Architectural and engineering choices behind `kinsim_NN`, with rationale.
Earlier decisions related to the legacy `kinsim` / `kinsim2` /
`kinsim_baseline` packages are retired with those packages and not
listed here. The relevant history can be recovered from Git.

---

## 1. Conditional WGAN-GP with a transformer generator and critic

**Decision.** Train an adversarial sample-level model on the joint
21 × 4 kinetic window, rather than a per-position likelihood model
predicting `(μ, σ)` for IPD and PW.

**Rationale.** The legacy ConvPredictor formulation, trained under
Gaussian or Beta-NLL on per-position marginals, converged on the
target dataset to a degenerate solution in which the predicted `σ`
absorbed the per-read variance and `μ` stayed close to baseline. The
adversarial objective targets the distribution of samples directly and
removes the parametric Gaussian assumption.

**References.** Gulrajani et al., *Improved Training of Wasserstein
GANs*, NeurIPS 2017, [doi:10.48550/arXiv.1704.00028](https://doi.org/10.48550/arXiv.1704.00028).
Mirza & Osindero, *Conditional Generative Adversarial Nets*,
arXiv:1411.1784, 2014. Seitzer et al., *On the Pitfalls of
Heteroscedastic Uncertainty Estimation with Probabilistic Neural
Networks*, ICLR 2022, [doi:10.48550/arXiv.2203.09168](https://doi.org/10.48550/arXiv.2203.09168).

---

## 2. Transformer backbone and AdaLN-Zero conditioning

**Decision.** Use a transformer (8 layers, 8 heads, `d_model = 256`)
for the generator and a smaller transformer (6 layers, 6 heads,
`d_model = 192`, spectral-norm on every Linear) for the critic. Inject
the conditioning through AdaLN-Zero modulation rather than through
FiLM on a CNN backbone.

**Rationale.** On a 21 bp window the receptive-field argument does not
favour CNN over transformer (attention costs `K² = 441` operations per
token, negligible at this scale). The decisive criterion is the
conditioning mechanism: AdaLN-Zero produces per-sample `(shift, scale,
gate)` per block with the gate zero-initialised so each block starts
as the identity transform. This identity-init property smooths the
early training phase and has no clean analogue under FiLM on a CNN.

**Reference.** Peebles & Xie, *Scalable Diffusion Models with
Transformers*, ICCV 2023, [doi:10.1109/ICCV51070.2023.00387](https://doi.org/10.1109/ICCV51070.2023.00387).

---

## 3. Manual self-attention (no fused kernel)

**Decision.** Implement the scaled dot-product self-attention by hand
in `MultiHeadSelfAttention.forward`, not via
`torch.nn.functional.scaled_dot_product_attention`.

**Rationale.** WGAN-GP's gradient penalty requires a double backward
(`create_graph=True`) on the critic. The fused / flash-attention
kernels shipped with PyTorch do not currently implement this
double-backward path. The manual implementation exposes each matmul
and the softmax as standard autograd operations with a defined
second-order gradient. The throughput cost is negligible at `K = 21`.

---

## 4. Bystrandified BAM as the input format

**Decision.** `kinsim_NN extract` reads bystrandified BAMs (2 records
per ZMW with `/fwd` and `/rev` suffixes, kinetic tags `ip` and `pw`),
not raw HiFi BAMs (1 record per ZMW with `fi` / `fp` / `ri` / `rp`).

**Rationale.** Bystrandified BAMs are the format produced by the
existing PacBio preprocessing chain (`ccs-kinetics-bystrandify` →
`pbmm2 align`) and are what the production strains are stored as.
Supporting raw HiFi was attempted in an earlier iteration; the strand
routing required to recover per-strand kinetics from `fi` / `fp` / `ri`
/ `rp` without alignment-induced ordering loss was implemented but
required maintaining two code paths in `iter_window_samples` and
`iter_chunk_samples`. Removing one of the two simplified the
extraction surface and removed a class of orientation bugs.

---

## 5. Three-category labelling with positive-offset emission

**Decision.** For each labelled methylation position `p` of type `T`,
emit candidate rows at every offset `k ∈ [0, near_meth_max_dist]`,
labelled `SLOWED` if `k ∈ signal_offsets[T]` else `NEAR_METH`. Baseline
candidates are sampled separately at positions `≥ baseline_min_dist`
from any labelled methylation.

**Rationale.** The kinetic signature of a methylation extends over
several bases downstream of the modified site (m6A: offsets 0 and 5;
m4C: offset 0; m5C: offsets 2 and 6). Training only at the modified
base itself would discard the off-site footprint that downstream
methylation callers actually rely on. The NEAR_METH category provides
a tight negative control: it shares "is close to a methylation" with
SLOWED but the polymerase is not expected to deviate from baseline at
those positions.

**Limitation.** Upstream offsets are not currently sampled. Recent
literature (`Jasmine`, `KinMethyl`) reports detectable signal at
upstream positions; extending `signal_offsets` to negative values and
`near_meth_max_dist` symmetrically is a candidate extension.

---

## 6. Methylation context on both strands across the full 21 bp window

**Decision.** The shard stores `meth_fwd[K]` and `meth_rev[K]` at all
21 positions (not just `[-1, 0, +1]` on the reverse strand as in the
legacy single-strand layout).

**Rationale.** Methylation in non-palindromic restriction-modification
systems is asymmetric across strands, and the kinetic effect of a
modification on the antisense strand reaches into the focal window. A
restricted reverse-strand context was a simplification valid only
under the palindromic assumption.

---

## 7. Frozen `model_config.json` at training start

**Decision.** Write the model architecture and the `meth_id_by_name`
mapping into `model_config.json` at training start, and refuse to
overwrite it on `--resume`.

**Rationale.** The methylation IDs are persisted into the trained
weights through the input embedding. Editing
`kinsim_nn_config.yaml`'s `methylation_types` between training and
inference would silently produce a model that disagrees with the
generator at inference time. Pinning the mapping at training time
makes the contract explicit and the failure mode auditable.

---

## 8. `best_G.pt` selection on held-out Wasserstein-1

**Decision.** During training, evaluate every `eval_every` steps on
the test-strain shards and write `best_G.pt` whenever the global W1
metric improves. Keep `G.pt` and `D.pt` as the latest checkpoint pair
required for `--resume` (resuming `G` alone would violate the
critic's Lipschitz constraint).

**Rationale.** The training-time W1 estimator is the only available
held-out metric inside the training loop; preserving the best snapshot
on this metric allows downstream evaluation to compare a "current"
generator against a "best held-out" generator without re-running.

---

## 9. Vendored motif handling inside `kinsim_NN`

**Decision.** `kinsim_NN/utils/motifs.py` and
`kinsim_NN/utils/parsers/` are now a self-contained motif-parsing
toolchain inside the package, with no dependency on any sibling
package.

**Rationale.** The earlier `from kinsim.utils.motifs import …` chain
made the package non-installable without the legacy `kinsim/`. With
`kinsim/` retired, the motif utilities are owned by `kinsim_NN` and
the configuration lookups (`get_modified_base`, `get_modified_base_map`,
`get_meth_ids`) are wired to `kinsim_nn_config.yaml`.

---

## 10. Future work — eukaryote / prokaryote mode preset

**Status.** Not yet implemented; documented here so the design space
is captured before it is forgotten.

**The problem.** KinSim's labelling chain is configuration-driven via
`kinsim_nn_config.yaml`. A user simulating prokaryotic data wants
`gff` (ipdSummary + pbmotifmaker for m6A / m4C / m5C at motif sites);
a user simulating eukaryotic data wants `jasmine_mm_ml` (5mC at CpG
dinucleotides on a per-read basis). The two labellers are mutually
exclusive in practice: jasmine's CNN is restricted to CpG contexts,
while ipdSummary's m5C calls require motif-level evidence that
bacterial m5C systems provide but eukaryotic CpG islands typically do
not. Asking the user to assemble the right `labelers:` list by hand
is a UX cliff.

**Proposed design.** Add a top-level YAML key:

```yaml
mode: prokaryote   # one of: prokaryote, eukaryote
```

with a presets layer that, at config load time, expands this to:

* `mode: prokaryote` → `labelers: [gff]`, `methylation_types`
  keeps m6A / m4C / m5C as currently defined.
* `mode: eukaryote` → `labelers: [jasmine_mm_ml]`, `methylation_types`
  reduced to `{none, m5C}` with `m5C.signal_offsets = [...]` retained.

The user-facing CLI gains `--mode {prokaryote,eukaryote}` on every
subcommand that reads the config. The model itself (architecture,
loss, training schedule) is unchanged — only the labeller selection
and the active methylation alphabet differ.

**Trade-off.** Adding the preset hides the `labelers:` list under a
mode toggle; advanced users still need access to the raw list for
hybrid configurations (e.g. a bacterial corpus with jasmine-confirmed
5mC). The proposed design keeps the explicit `labelers:` key as the
authority — `mode:` is a convenience that sets defaults the user can
override.

**Rationale.** The biology cleanly bifurcates by kingdom. Hiding the
labeller composition behind a kingdom-level toggle reduces the
cognitive surface for new users and makes "I'm doing eukaryote 5mC"
a one-line YAML change. The trade-off (a level of indirection) is
small relative to the user-facing simplification.
