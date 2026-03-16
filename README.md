[![Release](https://img.shields.io/github/v/release/matnguyen/perseus)](https://github.com/matnguyen/perseus/releases) [![bioRxiv](https://img.shields.io/badge/bioRxiv-preprint-orange)](https://doi.org/10.64898/2026.03.06.710148) [![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![CI](https://github.com/matnguyen/perseus/actions/workflows/tests.yml/badge.svg)](https://github.com/matnguyen/perseus/actions) [![codecov](https://codecov.io/github/matnguyen/perseus/branch/main/graph/badge.svg)](https://codecov.io/github/matnguyen/perseus)

# **Perseus**: refining Kraken2 taxonomic classifications of long reads and contigs

<img src="img/logo.png" alt="logo" width="250" align="left" style="margin-right: 30px"/> 

Perseus is a post-processing framework for **refining Kraken2 taxonomic classifications**, with a focus on **long-read metagenomics** (PacBio HiFi, ONT). While Kraken2’s exact k-mer matching enables fast and sensitive classification, it can produce **overconfident fine-rank calls** when evidence is sparse, conserved, or partially novel. Perseus addresses this limitation by **distinguishing trustworthy from spurious taxonomic predictions** using structured k-mer evidence already present in the Kraken2 output. Perseus is designed to reduce false positive fine-rank calls arising from conserved regions, sparse k-mer support, and reference database incompleteness—failure modes that are common in long-read and high-novelty metagenomes.

Perseus assigns **confidence probabilities** to each Kraken2 classification at **every canonical taxonomic rank**, enabling informed decisions to **confirm assignments, back off to higher, lineage-consistent ranks, or convert predictions to unclassified**.

Perseus is built on a multi-headed 1D convolutional neural network that operates directly on features derived from Kraken2 output. The workflow constructs a lineage-aware feature matrix from a standard Kraken2 output file, then performs inference to produce a Kraken2-compatible output augmented with per-rank confidence probabilities for each assignment. Perseus operates strictly as a downstream confidence filter and does not perform reclassification, alignment, or novel taxon discovery.

---

## Installation

### Conda installation (recommended)

Perseus is available through conda. We recommend creating a new environment:

```bash
conda create -n perseus -c matnguyen -c conda-forge -c pytorch perseus
conda activate perseus
```

### pip installation

Perseus is available on PyPi and can be installed through pip. There may be issues with installing ETE3 and PyTorch through pip, so we recommend using a new conda or virtual environment:

```bash
conda create -n perseus ete3 pytorch
pip install perseus-metagenomics
```

## Getting started

### Feature extraction

Perseus will perform feature extraction on a Kraken2 output file and output a directory of sharded parquets containing the features.

`perseus extract <kraken_file> <output_shards_directory>`

### Filtering

Perseus takes in the directory of sharded parquets and the Kraken2 output file for filtering.

`perseus filter <shards_directory> <kraken_file> <output_path>`

The output file will be similar to the Kraken2 output file, but without the string of k-mer matches, and with the following additional columns:

1. perseus_taxid - the taxonomic ID assigned by Perseus
2. prob_{rank} - the assignment probability at a canonical {rank}
3. chosen_rank - the final chosen rank assigned by Perseus
4. chosen_prob_at_rank - the probability at the final chosen rank

## Testing Data

We provide some data for testing Perseus. They can be found under `tests/test_data`. The Kraken2 output file is `tests/test_data/test_kraken`, the shards are in `tests/test_data/test_shards`, and the expected Perseus output file is `tests/test_data/filtered.txt`.

## Testing the Installation

### Quick Example

Run Perseus on the included test data:

```bash
perseus extract tests/test_data/test_kraken.txt example_extract
perseus filter example_extract tests/test_data/test_kraken.txt example_filtered.txt
```

This should produce an output file `example_filtered.txt`.

Because Perseus uses floating-point operations (PyTorch), small numerical differences may occur across platforms. Therefore, the output may not match the reference file exactly with a simple `diff`.

To compare the output with the expected results using a numerical tolerance:

`python scripts/compare_outputs.py example_filtered.txt tests/test_data/filtered.txt`

### Running the Full Test Suite (optional)

For a full reproducibility check, run the included test suite.

Install the testing dependency:

`pip install pytest`

Then run:

`pytest -q`

This runs unit tests and end-to-end pipeline tests used during development.

## Citing Perseus

Our preprint can be found here: [https://www.biorxiv.org/content/10.64898/2026.03.06.710148v1](https://www.biorxiv.org/content/10.64898/2026.03.06.710148v1)

## Data Generation Scripts

Scripts for generating the inclusion/exclusion simulated data are found here: [https://github.com/matnguyen/perseus-scripts](https://github.com/matnguyen/perseus-scripts)
