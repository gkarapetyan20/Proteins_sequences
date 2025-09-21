# Protein Amino Acid Prediction with ESM-2

This project focuses on protein amino acid analysis using the [ESM-2](https://github.com/facebookresearch/esm) model.  
The goal is to predict features from protein sequences with ESM-2, and then improve the results by training additional models on top of its outputs.

---

## Repository Structure

- `requirements.txt` – list of required packages.  
- `test_one_sequence.py` – example script for running prediction on a single protein sequence.  
- `work_description.pdf` – detailed description of the project and methodology.  
- Source code files – training and evaluation scripts.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/gkarapetyan20/Proteins_sequences
cd Proteins_sequences

```
## Install dependencies

```bash
pip install -r requirements.txt

```

## Run prediction on one sequence

You can test the pipeline with the provided script:

```bash
python test_one_sequence.py
```

## Documentation

For a complete explanation of the workflow, model improvements, and experimental results, please read:

```bash
work_description.pdf
``` 