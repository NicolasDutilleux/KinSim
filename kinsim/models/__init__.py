"""Neural model implementations for KinSim.

Each sub-package is a self-contained model mode:

  cgan/   — Conditional WGAN-GP (Generator + Discriminator)
  mlp/    — Supervised MLP with heteroscedastic Gaussian output

All modes share the same data pipeline from kinsim.common:
  kinsim.common.extract   — BAM extraction + shard merging
  kinsim.common.dataset   — KmerSignalDataset, log_transform, inv_log_transform
"""
