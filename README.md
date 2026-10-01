[![Release](https://img.shields.io/github/v/release/matnguyen/perseus)](https://github.com/matnguyen/perseus/releases) [![Bioinformatics](https://img.shields.io/badge/Bioinformatics-10.1093%2Fbioinformatics%2Fbtag687-blue)](https://doi.org/10.1093/bioinformatics/btag687) [![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![CI](https://github.com/matnguyen/perseus/actions/workflows/tests.yml/badge.svg)](https://github.com/matnguyen/perseus/actions) [![codecov](https://codecov.io/github/matnguyen/perseus/branch/main/graph/badge.svg)](https://codecov.io/github/matnguyen/perseus)

# **Perseus**: refining Kraken2 taxonomic classifications of long reads and contigs

<img src="img/logo.png" alt="logo" width="250" align="left" style="margin-right: 30px"/> 

Perseus is a post-processing framework for **refining Kraken2 taxonomic classifications**, with a focus on **long-read metagenomics** (PacBio HiFi, ONT). While Kraken2’s exact k-mer matching enables fast and sensitive classification, it can produce **overconfident fine-rank calls** when evidence is sparse, conserved, or partially novel. Perseus addresses this limitation by **distinguishing trustworthy from spurious taxonomic predictions** using structured k-mer evidence already present in the Kraken2 output. Perseus is designed to reduce false positive fine-rank calls arising from conserved regions, sparse k-mer support, and reference database incompleteness—failure modes that are common in long-read and high-novelty metagenomes.

Perseus assigns **confidence probabilities** to each Kraken2 classification at **every canonical taxonomic rank**, enabling informed decisions to **confirm assignments, back off to higher, lineage-consistent ranks, or convert predictions to unclassified**.

Perseus is built on a multi-headed 1D convolutional neural network that operates directly on features derived from Kraken2 output. The workflow constructs a lineage-aware feature matrix from a standard Kraken2 output file, then performs inference to produce a Kraken2-compatible output augmented with per-rank confidence probabilities for each assignment. Perseus operates strictly as a downstream confidence filter and does not perform reclassification, alignment, or novel taxon discovery.

---

## Table of Contents

- [Installation](#installation)
  - [Conda installation](#conda-installation-recommended)
  - [pip installation](#pip-installation)
  - [Containers (Docker/Apptainer/Singularity)](#containers-dockerapptainersingularity)
  - [Galaxy](#galaxy)
- [Getting started](#getting-started)
  - [Setup taxonomy database](#setup-taxonomy-database)
  - [Feature extraction](#feature-extraction)
  - [Filtering](#filtering)
- [Testing Data](#testing-data)
- [Testing the Installation](#testing-the-installation)
- [Citing Perseus](#citing-perseus)
- [Data Generation Scripts](#data-generation-scripts)
- [Support](#support)
- [License](#license)

## Installation

### Conda installation (recommended)

Perseus is available through Bioconda. We recommend creating a new environment:

```bash
conda create -n perseus -c conda-forge -c bioconda perseus
conda activate perseus
```

Alternatively, if Bioconda is already configured in your Conda channels:

```bash
conda create -n perseus perseus
conda activate perseus
```

### pip installation

Perseus is available on PyPI and can be installed through pip. There may be issues with installing ETE3 and PyTorch through pip, so we recommend using a new conda or virtual environment:

```bash
conda create -n perseus ete3 pytorch
pip install perseus-metagenomics
```

### Containers (Docker/Apptainer/Singularity) 

Perseus is available as a Docker container and can also be used with Singularity/Apptainer.

#### Docker

Pull the latest stable release:

```bash
docker pull matnguyen/perseus:latest
```

For reproducible analyses, we recommend using a specific version:

```bash
docker pull matnguyen/perseus:1.2.0
```

Check that Perseus is available:

```bash
docker run --rm matnguyen/perseus:1.2.0 --help
```

To run Perseus on local files, mount your working directory to `/data` inside the container:

```bash
docker run --rm \
    -v "$(pwd):/data" \
    matnguyen/perseus:1.2.0 \
    setup /data/ete3_db
```

Feature extraction can then be run with:

```bash
docker run --rm \
    -v "$(pwd):/data" \
    matnguyen/perseus:1.2.0 \
    extract \
    /data/kraken_output.txt \
    /data/perseus_shards \
    /data/ete3_db
```

and filtering with:

```bash
docker run --rm \
    -v "$(pwd):/data" \
    matnguyen/perseus:1.2.0 \
    filter \
    /data/perseus_shards \
    /data/kraken_output.txt \
    /data/perseus_output.txt \
    /data/ete3_db
```

#### Singularity/Apptainer

Perseus can also be used on systems that provide Singularity or Apptainer, which is common on HPC systems.

With Apptainer, pull the Docker image and convert it to a Singularity Image Format (`.sif`) file:

```bash
apptainer pull perseus_1.2.0.sif docker://matnguyen/perseus:1.2.0
```

For systems using the older `singularity` command:

```bash
singularity pull perseus_1.2.0.sif docker://matnguyen/perseus:1.2.0
```

Check the installation:

```bash
apptainer run perseus_1.2.0.sif --help
```

or:

```bash
singularity run perseus_1.2.0.sif --help
```

Run the taxonomy database setup:

```bash
apptainer run \
    --bind "$(pwd):/data" \
    perseus_1.2.0.sif \
    setup /data/ete3_db
```

Run feature extraction:

```bash
apptainer run \
    --bind "$(pwd):/data" \
    perseus_1.2.0.sif \
    extract \
    /data/kraken_output.txt \
    /data/perseus_shards \
    /data/ete3_db
```

Run filtering:

```bash
apptainer run \
    --bind "$(pwd):/data" \
    perseus_1.2.0.sif \
    filter \
    /data/perseus_shards \
    /data/kraken_output.txt \
    /data/perseus_output.txt \
    /data/ete3_db
```

If your system uses `singularity` instead of `apptainer`, replace `apptainer` with `singularity` in the commands above.

For reproducible analyses, we recommend using a versioned image such as `1.2.0` rather than `latest`.

### Galaxy

Perseus is also available through Galaxy, allowing users to run the workflow through a graphical web interface without installing Perseus locally.

Search for **Perseus** in your Galaxy instance's tool panel to use the available Perseus tools.

## Getting started

### Input format

Perseus takes standard Kraken2 output as input. The input must include the per-sequence k-mer/minimizer assignment string generated by Kraken2, as this information is used to construct the lineage-aware feature representation.

Perseus should be run on the original Kraken2 output file rather than the Kraken2 report file.

### Setup taxonomy database

Perseus will download an ETE3 taxonomy database.

`perseus setup <db_path>`

### Feature extraction

Perseus will perform feature extraction on a Kraken2 output file and output a directory of sharded parquets containing the features.

`perseus extract <kraken_file> <output_shards_directory> <db_path>`

### Filtering

Perseus takes in the directory of sharded parquets and the Kraken2 output file for filtering.

`perseus filter <shards_directory> <kraken_file> <output_path> <db_path>`

The output file will be similar to the Kraken2 output file, but without the string of k-mer matches, and with the following additional columns:

1. perseus_taxid - the taxonomic ID assigned by Perseus
2. perseus_taxonomy - the taxonomic name assigned by Perseus
3. chosen_rank - the final chosen rank assigned by Perseus
4. chosen_prob_at_rank - the probability at the final chosen rank
5. prob_{rank} - the assignment probability at a canonical {rank}

## Testing Data

We provide some data for testing Perseus. They can be found under `tests/test_data`. The Kraken2 output file is `tests/test_data/test_kraken`, the shards are in `tests/test_data/test_shards`, and the expected Perseus output file is `tests/test_data/filtered.txt`.

## Testing the Installation

### Quick Example

Run Perseus on the included test data:

```bash
perseus setup ete3_db
perseus extract tests/test_data/test_kraken.txt example_extract ete3_db
perseus filter example_extract tests/test_data/test_kraken.txt example_filtered.txt ete3_db
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

If you use Perseus in your work, please cite:

Nguyen MH, Schatz MC. Perseus: refining Kraken2 taxonomic classifications of long reads and contigs.
*Bioinformatics*. https://doi.org/10.1093/bioinformatics/btag687

```bibtex
@article{nguyen_perseus,
  author  = {Nguyen, Matthew H. and Schatz, Michael C.},
  title   = {Perseus: refining Kraken2 taxonomic classifications of long reads and contigs},
  journal = {Bioinformatics},
  doi     = {10.1093/bioinformatics/btag687}
}
```

## Data Generation Scripts

Scripts for generating the inclusion/exclusion simulated data are found here: [https://github.com/matnguyen/perseus-scripts](https://github.com/matnguyen/perseus-scripts)

## Support

For bug reports, feature requests, or questions, please open an
[issue](https://github.com/matnguyen/perseus/issues).

## License

Perseus is released under the [MIT License](LICENSE).
