"""MLP mode — supervised regression for kinetic signal prediction.

This mode trains a Multi-Layer Perceptron (MLP) on real PacBio BAM data
to predict IPD and PW signals from 11-mer sequence context and methylation state.

Unlike the cGAN, the MLP predicts both the mean (μ) and variance (σ²) of the
signal distribution for each context, then samples from N(μ, σ²) at generation
time to preserve biological stochasticity.

Data preparation reuses the cGAN pipeline:
    kinsim cgan extract reads.bam motifs output.pkl
    kinsim cgan merge shards/ master_data.pkl

Training:
    kinsim mlp train master_data.pkl checkpoints_mlp/

Generation:
    kinsim mlp generate pbsim3_output/ checkpoints_mlp/checkpoint_epoch50.pt motifs out/
"""
