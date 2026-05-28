"""Smoke tests for the kinsim_NN package.

Covers import-time integrity, YAML config loading, encoding helpers, and a
forward pass of a tiny generator. Does not require GPU, BAM files, or a
shard pickle on disk.
"""
from __future__ import annotations

import numpy as np
import pytest

from kinsim_NN.utils.config import load_config
from kinsim_NN.utils.encoding import (
    BASE_MAP,
    BASE_RC,
    METH_IDS,
    encode_seq,
    get_meth_ids,
)
from kinsim_NN.utils.pacbio_codec import (
    FRAMES_TABLE,
    frames_to_uint8,
    uint8_to_frames,
)


def test_imports():
    """All public modules import cleanly."""
    import kinsim_NN  # noqa: F401
    import kinsim_NN.extract  # noqa: F401
    import kinsim_NN.generate  # noqa: F401
    import kinsim_NN.train  # noqa: F401
    import kinsim_NN.evaluate  # noqa: F401
    import kinsim_NN.analyze  # noqa: F401


def test_base_map_consistency():
    assert BASE_MAP == {"A": 0, "C": 1, "G": 2, "T": 3}
    # A↔T, C↔G
    assert BASE_RC[0] == 3
    assert BASE_RC[1] == 2
    assert BASE_RC[2] == 1
    assert BASE_RC[3] == 0


def test_meth_ids_canonical():
    assert METH_IDS["none"] == 0
    assert METH_IDS["m6A"] == 1
    assert METH_IDS["m4C"] == 2
    assert METH_IDS["m5C"] == 3


def test_get_meth_ids_from_yaml():
    ids = get_meth_ids()
    assert ids["none"] == 0
    # The three canonical pinned IDs must survive any YAML config.
    assert ids.get("m6A") == 1
    assert ids.get("m4C") == 2
    assert ids.get("m5C") == 3


def test_encode_seq_roundtrip():
    out = encode_seq("ACGT")
    assert tuple(int(x) for x in out) == (0, 1, 2, 3)
    # Lower-case must work too.
    assert tuple(int(x) for x in encode_seq("acgt")) == (0, 1, 2, 3)
    # Non-ACGT bytes map to 0 (A) silently.
    assert tuple(int(x) for x in encode_seq("NNN")) == (0, 0, 0)


def test_pacbio_codec_roundtrip():
    """uint8 → frames → uint8 must be a fixed point on every byte."""
    bytes_in = np.arange(256, dtype=np.uint8)
    frames = uint8_to_frames(bytes_in)
    assert frames.shape == (256,)
    bytes_out = frames_to_uint8(frames)
    np.testing.assert_array_equal(bytes_in, bytes_out)
    # FRAMES_TABLE entries are monotonic.
    assert np.all(np.diff(FRAMES_TABLE) >= 0)


def test_load_config():
    cfg = load_config()
    assert cfg.window.k == 2 * cfg.window.half_width + 1
    assert cfg.n_meth_types >= 1
    assert any(t.name == "none" for t in cfg.methylation_types)


@pytest.mark.parametrize("k", [21])
def test_generator_forward(k):
    """Tiny generator forward pass on CPU: no NaN, correct output shape."""
    torch = pytest.importorskip("torch")
    from kinsim_NN.models.generator import TransformerGenerator

    g = TransformerGenerator(
        k=k,
        n_meth_types=4,
        d_model=32,
        n_layers=2,
        n_heads=4,
        z_dim=16,
        pos_embed_dim=8,
        drop_rate=0.0,
    )
    g.eval()
    B = 2
    z = g.sample_z(B, device="cpu")
    base_fwd = torch.zeros(B, k, 4); base_fwd[..., 0] = 1.0
    base_rev = torch.zeros(B, k, 4); base_rev[..., 3] = 1.0
    meth_fwd = torch.zeros(B, k, 4); meth_fwd[..., 0] = 1.0
    meth_rev = torch.zeros(B, k, 4); meth_rev[..., 0] = 1.0
    with torch.no_grad():
        out = g(z, base_fwd, base_rev, meth_fwd, meth_rev)
    assert out.shape == (B, k, 4)
    assert torch.isfinite(out).all()


def test_discriminator_forward():
    """Tiny discriminator forward pass on CPU."""
    torch = pytest.importorskip("torch")
    from kinsim_NN.models.discriminator import TransformerDiscriminator

    k = 21
    d = TransformerDiscriminator(
        k=k,
        n_meth_types=4,
        d_model=32,
        n_layers=2,
        n_heads=4,
        pos_embed_dim=8,
        spectral_norm=True,
        drop_rate=0.0,
    )
    d.eval()
    B = 2
    signal = torch.randn(B, k, 4)
    base_fwd = torch.zeros(B, k, 4); base_fwd[..., 0] = 1.0
    base_rev = torch.zeros(B, k, 4); base_rev[..., 3] = 1.0
    meth_fwd = torch.zeros(B, k, 4); meth_fwd[..., 0] = 1.0
    meth_rev = torch.zeros(B, k, 4); meth_rev[..., 0] = 1.0
    with torch.no_grad():
        out = d(signal, base_fwd, base_rev, meth_fwd, meth_rev)
    assert out.shape == (B,)
    assert torch.isfinite(out).all()
