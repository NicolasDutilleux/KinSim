# KinSim - Project Context

## PROJECT ARCHITECTURE: Rescuing DNA Kinetic Signatures for Metagenomic Binning

### 1. The Core Problem: Wasted Kinetic Data

In current bioinformatics pipelines, researchers typically discard IPD (Inter-Pulse Distance) and PW (Pulse Width) data after basecalling. This kinetic information is a "hidden dimension" of the genome that captures DNA methylation patterns and sequence-specific enzyme behavior.

- **The Opportunity:** Methylation patterns (the "methylome") are species-specific. They can act as natural barcodes to group DNA fragments (Binning) more accurately than using Tetranucleotide Frequency (TNF) or coverage alone.
- **The Barrier:** There is a massive lack of "Ground Truth" data where kinetics and methylation are perfectly mapped. This makes it difficult to prove the utility of kinetics in metagenomic binning.

### 2. Our Mission: The Kinetic Rescue & Simulation

We are proving that kinetic data is useful by "re-injecting" realistic IPD and PW signals into simulated datasets.

**The Strategy:**
1. Extract real kinetic signatures (mu, sigma) from 58+ known bacterial strains.
2. Generate synthetic reads from reference genomes using PBSIM3.
3. Augment these reads by using our Statistical Lookup Table to "paint" realistic IPD/PW values onto the synthetic sequences.

**The Ultimate Goal:** Prove that binning algorithms perform significantly better when they "see" the methylome.

### 3. Modeling Hierarchy: Why Statistics First?

We are following a "Complexity Gradient" for signal generation:

- **Current Solution (Statistical):** We use a simple Gaussian sampling approach based on the Mean (mu) and Standard Deviation (sigma) calculated for each k-mer context. This is computationally efficient and provides a solid baseline.
- **Future Solution (cGAN):** If the statistical approach fails to capture the full nuance of the signal (e.g., non-Gaussian distributions or complex dependencies), we will implement a Conditional Generative Adversarial Network (cGAN) to generate the kinetic signals. The statistical tables we are building now will serve as the training foundation for that model.

### 4. Biological Engine: 11-mer Kinetic Signatures

- **The Source:** PacBio SMRT sequencing.
- **The Context:** The signal at any position is determined by the 11-mer (the target base plus 5 bases upstream and 5 bases downstream).
- **Motifs & Ambiguity:** Methylation occurs at motifs defined by IUPAC codes (e.g., GANTC, RGATCY).
  - A single motif like GANTC can represent 4 different 11-mers.
  - Our extraction system maps these back to specific chemical modification IDs: m6A (1), m4C (2), m5C (3), or None (0).

### 5. Engineering Strategy: The "Binary Master Table"

To process 58+ strains (20GB+ BAM files each) on an HPC cluster, we use a high-performance, memory-efficient approach:

#### A. Bit-Packing (22-bit Encoding)

Strings are too heavy for high-speed lookups. We convert 11-mers into 22-bit integers:
- A: 00, C: 01, G: 10, T: 11
- **Advantage:** 4^11 possible contexts are stored as discrete integer keys, enabling a minimal RAM footprint and O(1) lookup time.

#### B. Incremental Statistical Aggregation

We do not store raw values. We store Accumulators in a dictionary. For every unique (k-mer, methylation_state), we maintain a vector of 5 values:
1. **n:** Count of observations (Coverage).
2. **sum_X:** Sum of IPDs.
3. **sum_X2:** Sum of squared IPDs.
4. **sum_Y:** Sum of PWs.
5. **sum_Y2:** Sum of squared PWs.

**The Math:** This allows us to calculate the Mean (mu) and Standard Deviation (sigma) with perfect precision. Crucially, this structure is additive, meaning we can merge data from different strains simply by summing their accumulators.

### 6. Implementation & Data Structure

- **Storage:** Python Pickle (.pkl) files (Binary serialization).
- **Parallelization:** Slurm Job Arrays.
- **Dictionary Key:** `(int_kmer, meth_id)`
- **Dictionary Value:** `np.array([n, sum_x, sum_sq_x, sum_y, sum_sq_y], dtype=np.float64)`
