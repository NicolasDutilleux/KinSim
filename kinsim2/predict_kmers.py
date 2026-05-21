"""Per-kmer (mu, sigma) under every methylation scenario, bilateral v2.

Given a trained checkpoint, enumerate all 4^K kmers under each
``(meth_type, signal_offset)`` scenario declared in ``kinsim_config.yaml``.
The methylation is placed on the FWD strand context only; the REV
context is all-zero (no rev methylation). This matches the most common
training-time pattern (one meth at a time).

Outputs::

    <out>.tsv   wide-format table with mu/sigma per scenario
                (4 kinetic channels: ipd_fwd, pw_fwd, ipd_rev, pw_rev)
    <out>.npz   compact binary, consumed by lookup-table generate
    <out>.html  per-scenario distribution of mu_ipd_fwd / mu_baseline_fwd
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
from .utils.encoding import BASE_MAP, decode_kmer, get_meth_ids

log = logging.getLogger(__name__)


_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}
_KMER_ENUMERATION_CAP: int = 100_000_000


def _scenarios_from_yaml() -> list[tuple[str, int, int, str | None]]:
    """Return ``[(label, meth_id, signal_offset_k, required_synth_base), ...]``."""
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
            out.append((f"{T}@{int(k):+d}", m_id, int(k), req_base))
    return out


def _biology_valid_mask(kmer_size: int, meth_pos: int, required_synth_base: str) -> np.ndarray:
    n_kmers = 4 ** kmer_size
    shift = 2 * (kmer_size - 1 - meth_pos)
    base_id = (np.arange(n_kmers, dtype=np.uint64) >> shift) & 3
    return base_id == BASE_MAP[required_synth_base]


def _find_checkpoint(ckpt_dir: Path) -> Path:
    candidates = list(ckpt_dir.glob("*.pt")) + list(ckpt_dir.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No .pt/.ckpt files in {ckpt_dir}")
    best = [c for c in candidates if "best" in c.name.lower()]
    if best:
        return max(best, key=lambda p: p.stat().st_mtime)
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_model(ckpt_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
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


def _build_meth_ctx(
    batch_size: int,
    meth_id: int,
    k_offset: int,
    kmer_size: int,
    active_site_index: int,
    num_meth_types: int,
    device: torch.device,
) -> torch.Tensor:
    """One-hot meth context (B, K, M). Active-site-relative offset placement."""
    mc = torch.zeros(batch_size, kmer_size, num_meth_types, dtype=torch.float32)
    if meth_id != 0:
        pos = active_site_index - int(k_offset)
        if 0 <= pos < kmer_size:
            mc[:, pos, meth_id] = 1.0
    return mc.to(device)


@torch.no_grad()
def _run_scenario(
    model: torch.nn.Module,
    meth_id: int,
    k_offset: int,
    n_kmers: int,
    batch_size: int,
    num_meth_types: int,
    device: torch.device,
    kmer_size: int,
    active_site_index: int,
    biology_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``(n_kmers, 8)``: [4 mu_log, 4 log_sigma] from the bilateral head."""
    preds = np.empty((n_kmers, 8), dtype=np.float32)
    mc_fwd_full = _build_meth_ctx(
        batch_size, meth_id, k_offset, kmer_size, active_site_index, num_meth_types, device,
    )
    mc_rev_full = torch.zeros_like(mc_fwd_full)
    for start in range(0, n_kmers, batch_size):
        end = min(start + batch_size, n_kmers)
        n_batch = end - start
        kmer_ids = torch.arange(start, end, dtype=torch.long, device=device)
        mf = mc_fwd_full if n_batch == batch_size else mc_fwd_full[:n_batch]
        mr = mc_rev_full if n_batch == batch_size else mc_rev_full[:n_batch]
        params = model(kmer_ids, mf, mr)
        preds[start:end] = params.detach().cpu().numpy()
    if biology_mask is not None:
        preds[~biology_mask] = np.nan
    return preds


def _sigma_clamp_from_model(model) -> tuple[float, float]:
    return (
        float(getattr(model, "log_sigma_clamp_min", -6.0)),
        float(getattr(model, "log_sigma_clamp_max", 1.5)),
    )


def _to_physical(
    preds: np.ndarray, sigma_clamp: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu_phys[:,4], sigma_phys[:,4]) in uint8-equivalent units."""
    mu_log = preds[:, :4]
    log_sig = np.clip(preds[:, 4:], *sigma_clamp)
    sigma_log = np.exp(log_sig)
    mu_phys = inv_log_transform(torch.from_numpy(mu_log)).numpy()
    sigma_phys = (mu_phys + 1.0) * sigma_log
    return mu_phys, sigma_phys


_CHANNEL_NAMES = ["ipd_fwd", "pw_fwd", "ipd_rev", "pw_rev"]


def _write_tsv(
    output_tsv: Path,
    scenarios: list[tuple[str, int, int, str | None]],
    raw_preds: dict[str, np.ndarray],
    sigma_clamp: tuple[float, float],
) -> None:
    n_kmers = next(iter(raw_preds.values())).shape[0]
    physical = {
        label: _to_physical(raw_preds[label], sigma_clamp)
        for label, *_ in scenarios if label in raw_preds
    }
    none_mu, _ = physical["none"]
    none_mu_safe = np.maximum(none_mu, 1e-6)

    cols = ["kmer", "kmer_id"]
    for label, *_ in scenarios:
        if label not in physical:
            continue
        sk = label.replace("@", "_at_").replace("+", "p").replace("-", "m")
        for ch in _CHANNEL_NAMES:
            cols.append(f"{sk}_mu_{ch}")
        for ch in _CHANNEL_NAMES:
            cols.append(f"{sk}_sigma_{ch}")
        if label != "none":
            for ch in _CHANNEL_NAMES:
                cols.append(f"{sk}_ratio_{ch}_vs_none")

    log.info("Writing %s ... (%d cols x %d rows)", output_tsv, len(cols), n_kmers)
    kmer_strings = np.array([decode_kmer(i) for i in range(n_kmers)])
    col_arrays: list[np.ndarray] = [kmer_strings, np.arange(n_kmers).astype(str)]
    for label, *_ in scenarios:
        if label not in physical:
            continue
        mu_phys, sig_phys = physical[label]
        for j in range(4):
            col_arrays.append(np.char.mod("%.3f", mu_phys[:, j]))
        for j in range(4):
            col_arrays.append(np.char.mod("%.3f", sig_phys[:, j]))
        if label != "none":
            ratio = mu_phys / none_mu_safe
            for j in range(4):
                col_arrays.append(np.char.mod("%.3f", ratio[:, j]))

    with open(output_tsv, "w") as f:
        f.write("\t".join(cols) + "\n")
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
    sigma_clamp: tuple[float, float],
) -> None:
    n_kmers = next(iter(raw_preds.values())).shape[0]
    bundle: dict[str, np.ndarray] = {"kmer_id": np.arange(n_kmers)}
    labels, m_ids, offsets = [], [], []
    for label, m_id, k_off, *_ in scenarios:
        if label not in raw_preds:
            continue
        labels.append(label)
        m_ids.append(m_id)
        offsets.append(k_off)
        preds = raw_preds[label]
        sk = label.replace("@", "_at_").replace("+", "p").replace("-", "m")
        mu_log = preds[:, :4].astype(np.float32)
        sigma_log = np.exp(np.clip(preds[:, 4:], *sigma_clamp)).astype(np.float32)
        for j, ch in enumerate(_CHANNEL_NAMES):
            bundle[f"{sk}__mu_{ch}_log"] = mu_log[:, j]
            bundle[f"{sk}__sigma_{ch}_log"] = sigma_log[:, j]
        mu_phys, sig_phys = _to_physical(preds, sigma_clamp)
        for j, ch in enumerate(_CHANNEL_NAMES):
            bundle[f"{sk}__mu_{ch}"] = mu_phys[:, j]
            bundle[f"{sk}__sigma_{ch}"] = sig_phys[:, j]
    bundle["scenarios_label"] = np.asarray(labels)
    bundle["scenarios_meth_id"] = np.asarray(m_ids, dtype=np.int64)
    bundle["scenarios_offset"] = np.asarray(offsets, dtype=np.int64)
    log.info("Writing %s ... (%d arrays)", output_npz, len(bundle))
    np.savez_compressed(output_npz, **bundle)


def _write_ratio_html(
    output_html: Path,
    scenarios: list[tuple[str, int, int, str | None]],
    raw_preds: dict[str, np.ndarray],
    sigma_clamp: tuple[float, float],
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        log.warning("plotly not installed — skipping ratio distribution HTML")
        return
    if "none" not in raw_preds:
        return
    none_mu, _ = _to_physical(raw_preds["none"], sigma_clamp)
    none_mu_safe = np.maximum(none_mu, 1e-6)
    meth_scenarios = [s for s in scenarios if s[0] != "none" and s[0] in raw_preds]
    if not meth_scenarios:
        return

    cols = 3
    rows = (len(meth_scenarios) + cols - 1) // cols
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[lbl for lbl, *_ in meth_scenarios],
        vertical_spacing=0.12,
    )
    color_by_mid = {1: "#E69F00", 2: "#56B4E9", 3: "#009E73", 4: "#F0E442"}

    for i, (label, m_id, _k, _req) in enumerate(meth_scenarios):
        r, c = i // cols + 1, i % cols + 1
        mu_phys, _ = _to_physical(raw_preds[label], sigma_clamp)
        ratio = mu_phys[:, 0] / none_mu_safe[:, 0]  # ipd_fwd ratio
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
            row=r, col=c,
        )
        fig.add_vline(x=1.0, line=dict(color="black", dash="dot", width=1), row=r, col=c)
        fig.add_vline(x=q50, line=dict(color="#222", width=2), row=r, col=c)
        fig.add_annotation(
            row=r, col=c, xref="x domain", yref="y domain",
            x=0.98, y=0.98, xanchor="right", yanchor="top",
            text=(
                f"median={q50:.2f}<br>IQR=[{q25:.2f}, {q75:.2f}]<br>"
                f"p05={q05:.2f}  p95={q95:.2f}<br>n_valid={valid_ratio.size:,}"
            ),
            showarrow=False,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.85)",
        )
        fig.update_xaxes(range=[0, 6.0], row=r, col=c)

    fig.update_layout(
        title="Per-kmer mu_ipd_fwd ratio vs unmethylated baseline (fwd channel)",
        height=320 * rows,
        bargap=0.02,
    )
    log.info("Writing %s ...", output_html)
    fig.write_html(str(output_html), include_plotlyjs="cdn", full_html=True)


def predict_all(
    ckpt_dir: Path,
    output_prefix: Path,
    batch_size: int = 65536,
    device_str: str | None = None,
) -> None:
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    log.info("Device: %s", device)
    model, config = _load_model(Path(ckpt_dir), device)

    num_meth_types = int(config.get("num_meth_types", 4))
    ckpt_k = int(config.get("kmer_size", 11))
    ckpt_pred_idx = int(config.get("active_site_index", ckpt_k // 2))
    scenarios = _scenarios_from_yaml()
    log.info("Scenarios: %s", [s[0] for s in scenarios])
    log.info(
        "Checkpoint geometry: K=%d  active_site_index=%d  num_meth_types=%d",
        ckpt_k, ckpt_pred_idx, num_meth_types,
    )
    n_kmers = 4 ** ckpt_k
    if n_kmers > _KMER_ENUMERATION_CAP:
        log.warning(
            "K=%d implies %.2g kmers > cap %.0g — full enumeration not viable.",
            ckpt_k, n_kmers, _KMER_ENUMERATION_CAP,
        )
        sys.exit(1)
    log.info(
        "Enumerating %d kmers (4^%d) for %d scenarios -> %d predictions total",
        n_kmers, ckpt_k, len(scenarios), n_kmers * len(scenarios),
    )

    raw_preds: dict[str, np.ndarray] = {}
    for label, m_id, k_off, req_base in scenarios:
        if req_base is None:
            bio_mask = None
            n_valid = n_kmers
        else:
            meth_pos = ckpt_pred_idx - k_off
            if 0 <= meth_pos < ckpt_k:
                bio_mask = _biology_valid_mask(ckpt_k, meth_pos, req_base)
                n_valid = int(bio_mask.sum())
            else:
                log.warning("Scenario %s: meth_pos %d outside window — skipping", label, meth_pos)
                continue
        log.info("  scenario %s (meth_id=%d offset=%+d valid=%d/%d)",
                 label, m_id, k_off, n_valid, n_kmers)
        raw_preds[label] = _run_scenario(
            model, m_id, k_off, n_kmers, batch_size, num_meth_types, device,
            kmer_size=ckpt_k, active_site_index=ckpt_pred_idx, biology_mask=bio_mask,
        )

    sigma_clamp = _sigma_clamp_from_model(model)
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(output_prefix.with_suffix(".tsv"), scenarios, raw_preds, sigma_clamp)
    _write_npz(output_prefix.with_suffix(".npz"), scenarios, raw_preds, sigma_clamp)
    _write_ratio_html(output_prefix.with_suffix(".html"), scenarios, raw_preds, sigma_clamp)
    log.info("Done.")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="kinsim predict-kmers",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("checkpoint_dir")
    p.add_argument("output_prefix")
    p.add_argument("--batch-size", type=int, default=65536)
    p.add_argument("--device", default=None)
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
