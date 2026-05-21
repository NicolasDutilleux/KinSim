"""Per-kmer prediction of (μ, σ) under every methylation scenario in kinsim_config.yaml.

Given a trained checkpoint, this runs the model on **all 4^K = 4,194,304 kmers**
under each scenario:

  - ``none``                 — no methylation anywhere in the kmer window
  - ``<T>@+<k>`` for each ``(T, k)`` in ``kinetic_signatures.<T>.signal_offsets``
    e.g. ``m6A@+0``, ``m6A@+5``, ``m4C@+0``, ``m5C@+2``, ``m5C@+6``

For each scenario we set ``meth_full[pred_idx - k, meth_id[T]] = 1.0`` —
that places the methylation ``k`` positions upstream of the prediction
position (i.e. the model predicts the kinetics ``+k`` downstream of the
modification, matching the signature offset convention in extract).

Outputs (next to the chosen output path):
    <out>.tsv   wide-format human-readable table:
                kmer | kmer_id | <scenario>_mu_ipd | <scenario>_mu_pw |
                <scenario>_sigma_ipd | <scenario>_sigma_pw |
                <scenario>_ratio_ipd_vs_none | <scenario>_ratio_pw_vs_none
                (5 prediction cols × N_scenarios + 2 ratio cols × (N-1) for non-none)
    <out>.npz   compact binary (one array per scenario, plus kmer ids)
    <out>.html  per-scenario distribution of μ_ipd / μ_baseline across all
                kmers — one histogram per non-none scenario, with median +
                IQR annotations. Diagnostic for "did the model learn a
                plausible shift per (meth_type, offset)?".

Predicted means/sigmas are returned in **raw uint8-equivalent space**
(via ``inv_log_transform``), so they are directly comparable to PacBio
``fi`` / ``fp`` tag values and to the empirical means from
``kinsim_baseline``.

Usage::

    kinsim predict-kmers <ckpt_dir> <output_prefix>

``ckpt_dir`` should contain ``model_config.json`` plus at least one
``.pt`` checkpoint. The most recent ``best_*.pt`` (or otherwise the most
recent ``.pt``) is used.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from .data.dataset import inv_log_transform
from .models.predictor import create_from_config, load_state_dict_from_ckpt
from .utils.config import get_modified_base_map, load_kinsim_config, setup_logging
from .utils.encoding import BASE_MAP, KMER_PRED_IDX, K, decode_kmer, get_meth_ids
from .utils.sample_layout import REV_METH_LEN

log = logging.getLogger(__name__)

# Synthesized-strand base required at the meth position, given the meth's
# modified_base on the TEMPLATE strand (matches the model's biology_mask,
# which does `bases ^ 3` to get the template). m4C/m5C (template C) → G;
# m6A (template A) → T.
_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


# ---------------------------------------------------------------------------
# Scenario enumeration from kinsim_config.yaml
# ---------------------------------------------------------------------------


def _scenarios_from_yaml() -> list[tuple[str, int, int, str | None]]:
    """Return ``[(label, meth_id, signal_offset_k, required_synth_base), ...]``.

    ``required_synth_base`` is the base the kmer must have at the meth
    position for the scenario to be biology-valid (complement of the
    YAML's ``modified_base``). ``None`` for the ``none`` scenario.
    """
    cfg = load_kinsim_config()
    sigs = cfg.get("kinetic_signatures", {}) or {}
    meth_ids = get_meth_ids()
    base_map = get_modified_base_map()

    out: list[tuple[str, int, int, str | None]] = [("none", 0, 0, None)]
    for T, info in sigs.items():
        m_id = meth_ids.get(T)
        if m_id is None or m_id == 0:
            log.warning("Meth type '%s' has no id in encoding — skipping", T)
            continue
        tpl_base = str(base_map.get(T, "")).upper()
        req_base = _COMPLEMENT.get(tpl_base)
        if req_base is None:
            log.warning("Meth type '%s' has no modified_base — skipping", T)
            continue
        for k in info.get("signal_offsets", []) or []:
            k = int(k)
            pos = KMER_PRED_IDX - k
            if pos < 0 or pos >= K:
                log.warning(
                    "Scenario %s@+%d places meth at position %d, "
                    "outside meth_context window [0, %d] — skipping",
                    T,
                    k,
                    pos,
                    K - 1,
                )
                continue
            out.append((f"{T}@{k:+d}", m_id, k, req_base))
    return out


def _biology_valid_mask(
    kmer_size: int,
    meth_pos: int,
    required_synth_base: str,
) -> np.ndarray:
    """Return bool[4^kmer_size]: True where kmer's base at ``meth_pos`` is the
    required synthesized base.

    The kmer is encoded with the LEFTMOST base in the top 2 bits, so the
    base at position ``meth_pos`` sits at bit shift ``2 * (kmer_size - 1 - meth_pos)``.
    """
    n_kmers = 4 ** kmer_size
    shift = 2 * (kmer_size - 1 - meth_pos)
    base_id = (np.arange(n_kmers, dtype=np.uint64) >> shift) & 3
    return base_id == BASE_MAP[required_synth_base]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _find_checkpoint(ckpt_dir: Path) -> Path:
    """Pick the best (or most recent) checkpoint in ``ckpt_dir``."""
    candidates = list(ckpt_dir.glob("*.pt")) + list(ckpt_dir.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No .pt/.ckpt files in {ckpt_dir}")
    best = [c for c in candidates if "best" in c.name.lower()]
    if best:
        return max(best, key=lambda p: p.stat().st_mtime)
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_model(ckpt_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Load model + its YAML-compatible config from ``ckpt_dir``."""
    config_path = ckpt_dir / "model_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{config_path} missing — run train first")
    config = json.loads(config_path.read_text())

    ckpt_path = _find_checkpoint(ckpt_dir)
    log.info("Loading checkpoint: %s", ckpt_path)
    model = create_from_config(config).to(device)
    model.load_state_dict(load_state_dict_from_ckpt(ckpt_path))
    model.eval()
    return model, config


# ---------------------------------------------------------------------------
# Inference over all kmers under a single scenario
# ---------------------------------------------------------------------------


def _meth_full_for_scenario(
    batch_size: int,
    meth_id: int,
    k_offset: int,
    num_meth_types: int = 4,
    device: torch.device | str = "cpu",
    kmer_size: int = K,
    active_site_index: int = KMER_PRED_IDX,
    n_rev_meth: int = REV_METH_LEN,
) -> torch.Tensor:
    """Build the ``(B, kmer_size + n_rev_meth, num_meth_types)`` tensor.

    ``meth_id == 0`` → all-zero tensor (no methylation).
    Otherwise: methylation is placed at ``active_site_index - k_offset``.

    All geometry params default to the K=11 legacy constants but can be
    overridden — predict_all passes the model's actual kmer_size /
    active_site_index / n_rev_meth so K=21 checkpoints just work.
    """
    total_pos = kmer_size + n_rev_meth
    meth_full = torch.zeros(batch_size, total_pos, num_meth_types, dtype=torch.float32)
    if meth_id != 0:
        pos = active_site_index - int(k_offset)
        # Use 1.0 to mean "fully methylated" — consistent with the one-hot
        # encoding for non-pred positions. For the pred position the
        # training Dataset uses ``frac`` (stoichiometry); we use 1.0 here
        # too so the scenario represents 100% methylation occupancy,
        # which is what the user wants to see ("max effect").
        meth_full[:, pos, meth_id] = 1.0
    return meth_full.to(device)


@torch.no_grad()
def _run_scenario(
    model: torch.nn.Module,
    meth_id: int,
    k_offset: int,
    n_kmers: int,
    batch_size: int,
    num_meth_types: int,
    device: torch.device,
    kmer_size: int = K,
    active_site_index: int = KMER_PRED_IDX,
    n_rev_meth: int = REV_METH_LEN,
    biology_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Run the model on every kmer for one scenario.

    Returns ``(n_kmers, 4)`` array with ``[mu_ipd_log, mu_pw_log, log_sigma_ipd,
    log_sigma_pw]``. Biology-invalid rows (per ``biology_mask``) are NaN-filled.
    """
    preds = np.empty((n_kmers, 4), dtype=np.float32)
    template = _meth_full_for_scenario(
        batch_size,
        meth_id,
        k_offset,
        num_meth_types,
        device,
        kmer_size=kmer_size,
        active_site_index=active_site_index,
        n_rev_meth=n_rev_meth,
    )
    for start in range(0, n_kmers, batch_size):
        end = min(start + batch_size, n_kmers)
        n_batch = end - start
        kmer_ids = torch.arange(start, end, dtype=torch.long, device=device)
        mf = template if n_batch == batch_size else template[:n_batch]
        params = model(kmer_ids, mf)
        preds[start:end] = params.detach().cpu().numpy()
    if biology_mask is not None:
        preds[~biology_mask] = np.nan
    return preds


# ---------------------------------------------------------------------------
# Postprocess: raw model output → physical units + ratios
# ---------------------------------------------------------------------------


def _to_physical(preds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert raw model output to (mu_ipd, mu_pw, sigma_ipd, sigma_pw) in uint8-like space.

    The model emits (μ, log σ) in **log1p space**. We convert μ via
    ``inv_log_transform`` (expm1 clamped to [0, 255]). For σ we apply the
    delta-method approximation:

        σ_raw ≈ |d/dy expm1(y)| · σ_log  =  (μ_raw + 1) · σ_log

    so the returned σ is in the same units as μ (uint8-like frame counts),
    comparable to the empirical IPD std-dev measured on the BAMs.
    """
    mu_ipd_log = preds[:, 0]
    mu_pw_log = preds[:, 1]
    log_sig_ipd = np.clip(preds[:, 2], -6.0, 3.0)
    log_sig_pw = np.clip(preds[:, 3], -6.0, 3.0)
    sigma_ipd_log = np.exp(log_sig_ipd)
    sigma_pw_log = np.exp(log_sig_pw)
    mu_ipd = inv_log_transform(torch.from_numpy(mu_ipd_log)).numpy()
    mu_pw = inv_log_transform(torch.from_numpy(mu_pw_log)).numpy()
    sigma_ipd = (mu_ipd + 1.0) * sigma_ipd_log
    sigma_pw = (mu_pw + 1.0) * sigma_pw_log
    return mu_ipd, mu_pw, sigma_ipd, sigma_pw


# ---------------------------------------------------------------------------
# TSV + NPZ writers
# ---------------------------------------------------------------------------


def _write_tsv(
    output_tsv: Path,
    scenarios: list[tuple[str, int, int, str | None]],
    raw_preds: dict[str, np.ndarray],
) -> None:
    """Wide-format TSV: one row per kmer, all scenarios side by side."""
    n_kmers = next(iter(raw_preds.values())).shape[0]

    physical: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, *_ in scenarios:
        if label not in raw_preds:
            continue
        physical[label] = _to_physical(raw_preds[label])

    none_mu_ipd, none_mu_pw, _, _ = physical["none"]
    none_mu_ipd_safe = np.maximum(none_mu_ipd, 1e-6)
    none_mu_pw_safe = np.maximum(none_mu_pw, 1e-6)

    cols = ["kmer", "kmer_id"]
    for label, *_ in scenarios:
        if label not in physical:
            continue
        sk = label.replace("@", "_at_").replace("+", "p").replace("-", "m")
        cols += [f"{sk}_mu_ipd", f"{sk}_mu_pw", f"{sk}_sigma_ipd", f"{sk}_sigma_pw"]
        if label != "none":
            cols += [f"{sk}_ratio_ipd_vs_none", f"{sk}_ratio_pw_vs_none"]

    log.info("Writing %s ... (%d cols × %d rows)", output_tsv, len(cols), n_kmers)
    kmer_strings = np.array([decode_kmer(i) for i in range(n_kmers)])

    col_arrays: list[np.ndarray] = [kmer_strings, np.arange(n_kmers).astype(str)]
    for label, *_ in scenarios:
        if label not in physical:
            continue
        mu_ipd, mu_pw, sig_ipd, sig_pw = physical[label]
        col_arrays.append(np.char.mod("%.3f", mu_ipd))
        col_arrays.append(np.char.mod("%.3f", mu_pw))
        col_arrays.append(np.char.mod("%.3f", sig_ipd))
        col_arrays.append(np.char.mod("%.3f", sig_pw))
        if label != "none":
            r_ipd = mu_ipd / none_mu_ipd_safe
            r_pw = mu_pw / none_mu_pw_safe
            col_arrays.append(np.char.mod("%.3f", r_ipd))
            col_arrays.append(np.char.mod("%.3f", r_pw))

    # Stack into N × cols and write
    with open(output_tsv, "w") as f:
        f.write("\t".join(cols) + "\n")
        # Build row strings in chunks to bound memory
        chunk = 100_000
        for start in range(0, n_kmers, chunk):
            end = min(start + chunk, n_kmers)
            rows = ["\t".join(col[start:end][i] for col in col_arrays) for i in range(end - start)]
            f.write("\n".join(rows))
            f.write("\n")


def _write_npz(
    output_npz: Path,
    scenarios: list[tuple[str, int, int, str | None]],
    raw_preds: dict[str, np.ndarray],
) -> None:
    """Compact binary output. Writes both **physical** (uint8-comparable) and
    **log-space** (model native) arrays per scenario.

    Physical arrays are for inspection/plotting and comparison with empirical
    BAM kinetics. Log-space arrays (``mu_*_log``, ``sigma_*_log``) are what
    ``kinsim generate --use-lookup`` consumes — sampling is done in log1p
    space exactly like at training time, then ``inv_log_transform``-ed back
    to uint8. Plus a small ``scenarios`` ledger so generate can build its
    ``(meth_id, offset) → scenario_idx`` lookup table without recomputing
    anything from the YAML.
    """
    n_kmers = next(iter(raw_preds.values())).shape[0]
    bundle: dict[str, np.ndarray] = {"kmer_id": np.arange(n_kmers)}

    # Per-scenario metadata: arrays of (label, meth_id, k_offset) — saved as
    # 1D string/int arrays so the LUT consumer can iterate without parsing.
    labels = []
    m_ids = []
    offsets = []

    for label, m_id, k_off, *_ in scenarios:
        if label not in raw_preds:
            continue
        labels.append(label)
        m_ids.append(m_id)
        offsets.append(k_off)

        preds = raw_preds[label]
        sk = label.replace("@", "_at_").replace("+", "p").replace("-", "m")

        # Log-space (model native — for sampling in generate)
        mu_ipd_log = preds[:, 0].astype(np.float32)
        mu_pw_log = preds[:, 1].astype(np.float32)
        sigma_ipd_log = np.exp(np.clip(preds[:, 2], -6.0, 3.0)).astype(np.float32)
        sigma_pw_log = np.exp(np.clip(preds[:, 3], -6.0, 3.0)).astype(np.float32)
        bundle[f"{sk}__mu_ipd_log"] = mu_ipd_log
        bundle[f"{sk}__mu_pw_log"] = mu_pw_log
        bundle[f"{sk}__sigma_ipd_log"] = sigma_ipd_log
        bundle[f"{sk}__sigma_pw_log"] = sigma_pw_log

        # Physical (uint8-comparable — for inspection & legacy consumers)
        mu_ipd, mu_pw, sig_ipd, sig_pw = _to_physical(preds)
        bundle[f"{sk}__mu_ipd"] = mu_ipd
        bundle[f"{sk}__mu_pw"] = mu_pw
        bundle[f"{sk}__sigma_ipd"] = sig_ipd
        bundle[f"{sk}__sigma_pw"] = sig_pw

    bundle["scenarios_label"] = np.asarray(labels)
    bundle["scenarios_meth_id"] = np.asarray(m_ids, dtype=np.int64)
    bundle["scenarios_offset"] = np.asarray(offsets, dtype=np.int64)

    log.info("Writing %s ... (%d arrays)", output_npz, len(bundle))
    np.savez_compressed(output_npz, **bundle)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def predict_all(
    ckpt_dir: Path,
    output_prefix: Path,
    batch_size: int = 65536,
    device_str: str | None = None,
) -> None:
    """Enumerate scenarios, predict on all kmers, write TSV + NPZ."""
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    log.info("Device: %s", device)

    model, config = _load_model(Path(ckpt_dir), device)
    num_meth_types = int(config.get("num_meth_types", 4))

    # K-aware: enumerate from the checkpoint's actual geometry, not the
    # module-level K=11 constants. A K=21 checkpoint produces 4**21 ≈ 4.4T
    # kmers — that's too many to fully enumerate. Cap to a sample and warn.
    ckpt_k = int(config.get("kmer_size", K))
    ckpt_pred_idx = int(config.get("active_site_index", KMER_PRED_IDX))
    ckpt_n_rev = int(config.get("n_rev_meth", REV_METH_LEN))
    scenarios = _scenarios_from_yaml()
    log.info("Scenarios: %s", [s[0] for s in scenarios])
    log.info(
        "Checkpoint geometry: K=%d  active_site_index=%d  n_rev_meth=%d",
        ckpt_k, ckpt_pred_idx, ckpt_n_rev,
    )
    n_kmers = 4**ckpt_k
    if n_kmers > 100_000_000:
        log.warning(
            "K=%d implies %.2g kmers — exceeds the 1e8 safety cap. "
            "Predict-kmers full enumeration is not viable at this scale; "
            "rewrite to sample-and-extrapolate.",
            ckpt_k, n_kmers,
        )
        sys.exit(1)
    log.info(
        "Enumerating %d kmers (4^%d) for %d scenarios → %d predictions total",
        n_kmers,
        ckpt_k,
        len(scenarios),
        n_kmers * len(scenarios),
    )

    raw_preds: dict[str, np.ndarray] = {}
    for label, m_id, k_off, req_base in scenarios:
        if req_base is None:
            bio_mask = None
            n_valid = n_kmers
        else:
            meth_pos = ckpt_pred_idx - k_off
            bio_mask = _biology_valid_mask(ckpt_k, meth_pos, req_base)
            n_valid = int(bio_mask.sum())
        log.info(
            "  ▸ %s  (meth_id=%d, offset=%+d)  required_base=%s  valid=%d/%d",
            label, m_id, k_off, req_base or "-", n_valid, n_kmers,
        )
        raw_preds[label] = _run_scenario(
            model,
            m_id,
            k_off,
            n_kmers,
            batch_size,
            num_meth_types,
            device,
            kmer_size=ckpt_k,
            active_site_index=ckpt_pred_idx,
            n_rev_meth=ckpt_n_rev,
            biology_mask=bio_mask,
        )

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(output_prefix.with_suffix(".tsv"), scenarios, raw_preds)
    _write_npz(output_prefix.with_suffix(".npz"), scenarios, raw_preds)
    _write_ratio_html(output_prefix.with_suffix(".html"), scenarios, raw_preds)
    log.info("Done.")


def _write_ratio_html(
    output_html: Path,
    scenarios: list[tuple[str, int, int, str | None]],
    raw_preds: dict[str, np.ndarray],
) -> None:
    """HTML dashboard — per-scenario distribution of μ_ipd / μ_baseline across all kmers.

    One subplot per non-``none`` scenario, with the histogram of the per-kmer
    IPD ratio (model μ for this scenario / model μ for ``none``). Annotates
    median and IQR so the reader sees the central tendency at a glance.

    Lets you eyeball whether the model has learned a plausible per-scenario
    shift (median ratio > 1 for methylated scenarios) and how heterogeneous
    the response is across kmers (wide IQR = high kmer-specificity).
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        log.warning("plotly not installed — skipping ratio distribution HTML")
        return

    if "none" not in raw_preds:
        log.warning("predict-kmers: no 'none' scenario, can't compute ratios — skipping HTML")
        return

    none_mu_ipd, none_mu_pw, _, _ = _to_physical(raw_preds["none"])
    eps = 1e-6
    none_mu_ipd_safe = np.maximum(none_mu_ipd, eps)
    none_mu_pw_safe = np.maximum(none_mu_pw, eps)

    meth_scenarios = [
        (label, m_id, k_off, req_base)
        for (label, m_id, k_off, req_base) in scenarios
        if label != "none" and label in raw_preds
    ]
    if not meth_scenarios:
        log.info("predict-kmers: no methylated scenarios — skipping HTML")
        return

    cols = 3
    rows = (len(meth_scenarios) + cols - 1) // cols
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[lbl for lbl, *_ in meth_scenarios],
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )

    # Wong/Okabe-Ito colorblind-safe palette (m6A=orange, m4C=sky blue, m5C=bluish green, other=yellow).
    color_by_mid = {1: "#E69F00", 2: "#56B4E9", 3: "#009E73", 4: "#F0E442"}

    for i, (label, m_id, _k_off, _req) in enumerate(meth_scenarios):
        r, c = i // cols + 1, i % cols + 1
        mu_ipd, _, _, _ = _to_physical(raw_preds[label])
        ratio = mu_ipd / none_mu_ipd_safe
        # Restrict to biology-valid kmers (NaN-filled in _run_scenario).
        valid_ratio = ratio[~np.isnan(ratio)]
        if valid_ratio.size == 0:
            continue

        q05, q25, q50, q75, q95 = np.percentile(valid_ratio, [5, 25, 50, 75, 95])
        fig.add_trace(
            go.Histogram(
                x=valid_ratio,
                xbins=dict(start=0, end=6.0, size=0.05),
                marker_color=color_by_mid.get(m_id, "#888"),
                opacity=0.75,
                showlegend=False,
            ),
            row=r,
            col=c,
        )
        fig.add_vline(x=1.0, line=dict(color="black", dash="dot", width=1), row=r, col=c)
        fig.add_vline(x=q50, line=dict(color="#222", width=2), row=r, col=c)
        fig.add_annotation(
            row=r, col=c, xref="x domain", yref="y domain",
            x=0.98, y=0.98, xanchor="right", yanchor="top",
            text=(
                f"median={q50:.2f}<br>IQR=[{q25:.2f}, {q75:.2f}]<br>"
                f"p05={q05:.2f}  p95={q95:.2f}<br>n_valid={valid_ratio.size:,}/{ratio.size:,}"
            ),
            showarrow=False,
            font=dict(size=10, color="#222"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            borderpad=3,
        )
        fig.update_xaxes(range=[0, 6.0], title_text="μ_ipd / μ_baseline" if r == rows else None,
                         row=r, col=c)
        fig.update_yaxes(title_text="count" if c == 1 else None, row=r, col=c)

    fig.update_layout(
        title=(
            "Per-kmer μ_ipd ratio vs unmethylated baseline — biology-valid kmers only "
            "(kmer base at meth position = complement of YAML modified_base). "
            "Dotted line at 1.0 = no shift; solid line = median."
        ),
        height=320 * rows,
        bargap=0.02,
    )

    log.info("Writing %s ...", output_html)
    fig.write_html(str(output_html), include_plotlyjs="cdn", full_html=True)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="kinsim predict-kmers",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("checkpoint_dir", help="Directory containing model_config.json + .pt files.")
    p.add_argument(
        "output_prefix", help="Output path WITHOUT extension. Writes .tsv (wide) + .npz (binary)."
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=65536,
        help="Inference batch size (default 65 536). Lower on small GPU.",
    )
    p.add_argument(
        "--device", default=None, help="'cuda' / 'cpu' / 'cuda:0' (default: cuda if available)."
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(verbose=args.verbose)

    predict_all(
        Path(args.checkpoint_dir),
        Path(args.output_prefix),
        batch_size=args.batch_size,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
