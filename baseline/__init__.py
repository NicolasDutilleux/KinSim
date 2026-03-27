"""Baseline models for KinSim comparison.

Three baselines:
  1. global_gaussian  — 4 Gaussians (one per meth type, no kmer context)
  2. kmer_gaussian    — per-kmer Gaussian + IPD ratio shift for methylation
  3. conv_no_film     — ConvPredictor without FiLM (post-hoc ratio shift)
"""
