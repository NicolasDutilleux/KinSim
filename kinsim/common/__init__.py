"""Shared infrastructure for all neural KinSim modes (MLP, cGAN, future models).

This package provides the data pipeline that every neural mode consumes:

  extract.py  — extract raw (IPD, PW) samples from BAMs; merge shards
                CLI: kinsim data extract / kinsim data merge

  dataset.py  — log_transform, inv_log_transform, KmerSignalDataset
                (PyTorch Dataset wrapping the merged .pkl file)

Both MLP and cGAN training import from here.  dictionary/ mode does not
use this package — it maintains its own running-accumulator format.
"""
