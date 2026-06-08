"""kinsim — transformer generator trained with a direct distributional loss.

Successor to kinsim_NN/. Same shard format and architecture family as the
v6 generator, but the WGAN-GP critic is replaced by a direct distributional
loss on the generator's output — bucketed energy distance + per-position
1-D Wasserstein (sorted-L1) + a tail-quantile penalty (see kinsim/losses.py).
No adversarial training, no critic.

Why: empirically (see thesis §5.4), the v6 critic shrank the marginal W₁
on the central channel but did not learn to discriminate on the spatial
autocorrelation across positions, so the generator was never pushed to
reproduce it and downstream motif recovery failed. A direct loss that
explicitly compares the joint distribution over the full kinetic tile
removes the indirection: the training signal is, by construction,
sensitive to the structure the downstream chain probes.
"""
__version__ = "0.1.0"
