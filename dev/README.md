# Development scripts

This directory contains research and development scripts used to train, evaluate, and generate data for the Perseus model. 

These scripts are **not part of the CLI interface**. They are intended for internal use, experimentation and reproducibility of the results presented in the manuscript.

---

## Directory overview

`extract_train_data.py`

Generates training data from Kraken2 outputs of simulated MeSS reads and outputs Perseus feature representations. Requires the following files from MeSS:

* Input file for generating MeSS reads
* File mapping each read ID to the reference genome

---

`train.py`

Train a Perseus model using the precomputed training dataset. Will split into a training and validation set stratified by species. 

---

`evaluate.py`

Evaluate trained models on held-out datasets or benchmarking runs.