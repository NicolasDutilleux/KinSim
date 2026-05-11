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
from pathlib import Path

import numpy as np
import torch

from .data.dataset import inv_log_transform
from .models.predictor import create_from_config
from .utils.config import load_kinsim_config, setup_logging
from .utils.encoding import K, KMER_PRED_IDX, decode_kmer, get_meth_ids
from .utils.sample_layout import REV_METH_LEN

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenario enumeration from kinsim_config.yaml
# ---------------------------------------------------------------------------


def _scenarios_from_yaml() -> list[tuple[str, int, int]]:
    """Return ``[(label, meth_id, signal_offset_k), ...]`` from YAML.

    The first scenario is always the ``none`` baseline (meth_id=0,
    offset=0). Following entries are per ``(T, k)`` in
    ``kinetic_signatures.<T>.signal_offsets``.

    Any (T, k) whose ``KMER_PRED_IDX - k`` would fall outside
    ``[0, K-1]`` is skipped with a warning (the methylation would be
    outside the model's meth_context window).
    """
    cfg = load_kinsim_config()
    sigs = cfg.get("kinetic_signatures", {}) or {}
    meth_ids = get_meth_ids()

    out: list[tuple[str, int, int]] = [("none", 0, 0)]
    for T, info in sigs.items():
        m_id = meth_ids.get(T)
        if m_id is None or m_id == 0:
            log.warning("Meth type '%s' has no id in encoding — skipping", T)
            continue
        for k in info.get("signal_offsets", []) or []:
            k = int(k)
            pos = KMER_PRED_IDX - k
            if pos < 0 or pos >= K:
                log.warning(
                    "Scenario %s@+%d places meth at position %d, "
                    "outside meth_context window [0, %d] — skipping",
                    T, k, pos, K - 1,
                )
                continue
            out.append((f"{T}@{k:+d}", m_id, k))
    return out


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
    state = torch.load(str(ckpt_path), map_location=device)

    model = create_from_config(config).to(device)

    # Handle both legacy ('model' key) and Lightning ('state_dict' with 'model.' prefix).
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    elif isinstance(state, dict) and "state_dict" in state:
        sd = {
            k.replace("model.", "", 1): v
            for k, v in state["state_dict"].items()
            if k.startswith("model.")
        }
        model.load_state_dict(sd)
    else:
        model.load_state_dict(state)

    model.eval()
    return model, config


# ---------------------------------------------------------------------------
# Inference over all kmers under a single scenario
# ---------------------------------------------------------------------------


def _meth_full_for_scenario(
    batch_size: int, meth_id: int, k_offset: int,
    num_meth_types: int = 4, device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build the ``(B, kmer_size + REV_METH_LEN, num_meth_types)`` tensor.

    ``meth_id == 0`` → all-zero tensor (no methylation).
    Otherwise: methylation is placed at ``KMER_PRED_IDX - k_offset`` (the
    one-hot encoding the model was trained on for positions != pred_idx).
    """
    total_pos = K + REV_METH_LEN
    meth_full = torch.zeros(batch_size, total_pos, num_meth_types, dtype=torch.float32)
    if meth_id != 0:
        pos = KMER_PRED_IDX - int(k_offset)
        # Use 1.0 to mean "fully methylated" — consistent with the one-hot
        # encoding for non-pred positions. For the pred position the
        # training Dataset uses ``frac`` (stoichiometry); we use 1.0 here
        # too so the scenario represents 100% methylation occupancy,
        # which is what the user wants to see ("max effect").
        meth_full[:, pos, meth_id] = 1.0
    return meth_full.to(device)


@torch.no_grad()
def _run_scenario(
    model: torch.nn.Module, meth_id: int, k_offset: int,
    n_kmers: int, batch_size: int, num_meth_types: int,
    device: torch.device,
) -> np.ndarray:
    """Run the model on every kmer for one scenario.

    Returns ``(n_kmers, 4)`` array with raw model output:
    ``[mu_ipd_log, mu_pw_log, log_sigma_ipd, log_sigma_pw]``.
    """
    preds = np.empty((n_kmers, 4), dtype=np.float32)
    template = _meth_full_for_scenario(
        batch_size, meth_id, k_offset, num_meth_types, device,
    )
    for start in range(0, n_kmers, batch_size):
        end = min(start + batch_size, n_kmers)
        n_batch = end - start
        kmer_ids = torch.arange(start, end, dtype=torch.long, device=device)
        # Slice the template instead of rebuilding when batch is full-size
        mf = template if n_batch == batch_size else template[:n_batch]
        params = model(kmer_ids, mf)
        preds[start:end] = params.detach().cpu().numpy()
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
    mu_pw_log  = preds[:, 1]
    log_sig_ipd = np.clip(preds[:, 2], -6.0, 3.0)
    log_sig_pw  = np.clip(preds[:, 3], -6.0, 3.0)
    sigma_ipd_log = np.exp(log_sig_ipd)
    sigma_pw_log  = np.exp(log_sig_pw)
    mu_ipd = inv_log_transform(torch.from_numpy(mu_ipd_log)).numpy()
    mu_pw  = inv_log_transform(torch.from_numpy(mu_pw_log)).numpy()
    sigma_ipd = (mu_ipd + 1.0) * sigma_ipd_log
    sigma_pw  = (mu_pw  + 1.0) * sigma_pw_log
    return mu_ipd, mu_pw, sigma_ipd, sigma_pw


# ---------------------------------------------------------------------------
# TSV + NPZ writers
# ---------------------------------------------------------------------------


def _write_tsv(
    output_tsv: Path, scenarios: list[tuple[str, int, int]],
    raw_preds: dict[str, np.ndarray],
) -> None:
    """Wide-format TSV: one row per kmer, all scenarios side by side."""
    n_kmers = next(iter(raw_preds.values())).shape[0]

    # Physical units per scenario
    physical: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, _, _ in scenarios:
        if label not in raw_preds:
            continue
        physical[label] = _to_physical(raw_preds[label])

    none_mu_ipd, none_mu_pw, _, _ = physical["none"]
    none_mu_ipd_safe = np.maximum(none_mu_ipd, 1e-6)
    none_mu_pw_safe  = np.maximum(none_mu_pw,  1e-6)

    # Header
    cols = ["kmer", "kmer_id"]
    for label, _, _ in scenarios:
        if label not in physical:
            continue
        sk = label.replace("@", "_at_").replace("+", "p").replace("-", "m")
        cols += [f"{sk}_mu_ipd", f"{sk}_mu_pw", f"{sk}_sigma_ipd", f"{sk}_sigma_pw"]
        if label != "none":
            cols += [f"{sk}_ratio_ipd_vs_none", f"{sk}_ratio_pw_vs_none"]

    log.info("Writing %s ... (%d cols × %d rows)", output_tsv, len(cols), n_kmers)
    # Vectorised row formatting — much faster than per-row Python loops.
    # Decode kmer strings in bulk (still O(N) but cheap).
    kmer_strings = np.array([decode_kmer(i) for i in range(n_kmers)])

    # Build columns array
    col_arrays: list[np.ndarray] = [kmer_strings, np.arange(n_kmers).astype(str)]
    for label, _, _ in scenarios:
        if label not in physical:
            continue
        mu_ipd, mu_pw, sig_ipd, sig_pw = physical[label]
        col_arrays.append(np.char.mod("%.3f", mu_ipd))
        col_arrays.append(np.char.mod("%.3f", mu_pw))
        col_arrays.append(np.char.mod("%.3f", sig_ipd))
        col_arrays.append(np.char.mod("%.3f", sig_pw))
        if label != "none":
            r_ipd = mu_ipd / none_mu_ipd_safe
            r_pw  = mu_pw  / none_mu_pw_safe
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
    output_npz: Path, scenarios: list[tuple[str, int, int]],
    raw_preds: dict[str, np.ndarray],
) -> None:
    """Compact binary output. One array per scenario in physical units."""
    bundle: dict[str, np.ndarray] = {"kmer_id": np.arange(next(iter(raw_preds.values())).shape[0])}
    for label, _, _ in scenarios:
        if label not in raw_preds:
            continue
        mu_ipd, mu_pw, sig_ipd, sig_pw = _to_physical(raw_preds[label])
        sk = label.replace("@", "_at_").replace("+", "p").replace("-", "m")
        bundle[f"{sk}__mu_ipd"]    = mu_ipd
        bundle[f"{sk}__mu_pw"]     = mu_pw
        bundle[f"{sk}__sigma_ipd"] = sig_ipd
        bundle[f"{sk}__sigma_pw"]  = sig_pw
    log.info("Writing %s ... (%d arrays)", output_npz, len(bundle))
    np.savez_compressed(output_npz, **bundle)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def predict_all(
    ckpt_dir: Path, output_prefix: Path,
    batch_size: int = 65536, device_str: str | None = None,
) -> None:
    """Enumerate scenarios, predict on all kmers, write TSV + NPZ."""
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    log.info("Device: %s", device)

    model, config = _load_model(Path(ckpt_dir), device)
    num_meth_types = int(config.get("num_meth_types", 4))

    scenarios = _scenarios_from_yaml()
    log.info("Scenarios: %s", [s[0] for s in scenarios])
    n_kmers = 4 ** K
    log.info("Enumerating %d kmers (4^%d) for %d scenarios → %d predictions total",
             n_kmers, K, len(scenarios), n_kmers * len(scenarios))

    raw_preds: dict[str, np.ndarray] = {}
    for label, m_id, k_off in scenarios:
        log.info("  ▸ %s  (meth_id=%d, offset=%+d)", label, m_id, k_off)
        raw_preds[label] = _run_scenario(
            model, m_id, k_off, n_kmers, batch_size, num_meth_types, device,
        )

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(output_prefix.with_suffix(".tsv"), scenarios, raw_preds)
    _write_npz(output_prefix.with_suffix(".npz"), scenarios, raw_preds)
    log.info("Done.")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="kinsim predict-kmers",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("checkpoint_dir",
                   help="Directory containing model_config.json + .pt files.")
    p.add_argument("output_prefix",
                   help="Output path WITHOUT extension. Writes .tsv (wide) + .npz (binary).")
    p.add_argument("--batch-size", type=int, default=65536,
                   help="Inference batch size (default 65 536). Lower on small GPU.")
    p.add_argument("--device", default=None,
                   help="'cuda' / 'cpu' / 'cuda:0' (default: cuda if available).")
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
