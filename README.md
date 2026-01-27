# **Perseus**: refining Kraken2 taxonomic classifications of long reads and contigs

<img src="img/logo.png" alt="logo" width="250" align="left" style="margin-right: 30px"/> 

Perseus is a post-processing framework for **refining Kraken2 taxonomic classifications**, with a focus on **long-read metagenomics** (PacBio HiFi, ONT). While Kraken2’s exact k-mer matching enables fast and sensitive classification, it can produce **overconfident fine-rank calls** when evidence is sparse, conserved, or partially novel. Perseus addresses this limitation by **distinguishing trustworthy from spurious taxonomic predictions** using structured k-mer evidence already present in the Kraken2 output. Perseus is designed to reduce false positive fine-rank calls arising from conserved regions, sparse k-mer support, and reference database incompleteness—failure modes that are common in long-read and high-novelty metagenomes.

Perseus assigns **confidence probabilities** to each Kraken2 classification at **every canonical taxonomic rank**, enabling informed decisions to **confirm assignments, back off to higher, lineage-consistent ranks, or convert predictions to unclassified**. 

Perseus is built on a multi-headed 1D convolutional neural network that operates directly on features derived from Kraken2 output. The workflow constructs a lineage-aware feature matrix from a standard Kraken2 output file, then performs inference to produce a Kraken2-compatible output augmented with per-rank confidence probabilities for each assignment. Perseus operates strictly as a downstream confidence filter and does not perform reclassification, alignment, or novel taxon discovery.

---

## Installation

### Conda installation (recommended)

### pip installation

### Docker/Singularity

### Build from scratch

## Quick start