# Comparative Empirical Framework

**Title:** Serial vs. Parallel Hybrid AI: A Comparative Framework for Architecture Selection Based on Task Characteristics*

**Target Journal:** Applied Intelligence (Springer)

## Overview

This code implements the comparative experiments on Hybrid AI. It compares Semi-symbolic (serial pipeline) and Concurrency (parallel voting) architectures across multiple datasets to derive a contingency framework for architecture selection.

## Directory Structure

- 📂 my_results/
- 📄 run_comparison.py
- 📂 figures/
- 📄 generate_figures.py
- 📂 shared/
  - 📄 datasets
  - 📄 metrics
  - 📄 .ipynb_checkpoints
  - 📄 architectures

## Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Run


```bash
python experiments/run_comparison.py --datasets heart_disease,nsl_kdd --runs 20 --output ./my_results
```

Command Line Arguments

Argument	Description	Default
--datasets	Dataset names (comma-separated) or "all"	"all"
--runs	Number of runs per configuration	10
--output	Output directory	"./results"
Output Files

## File	Description
comparison_results.csv	Raw results for all runs
comparison_summary.csv	Aggregated statistics per (dataset, architecture)
contingency_analysis.csv	Contingency factors and accuracy differences

## Datasets

The following datasets are included (from shared/datasets/loader.py):

Dataset	Domain	Task Type
UCI_Heart_Disease	Healthcare	Classification
NSL_KDD	Cybersecurity	Classification
CMAPSS	Industrial	Regression
Extending

To add a new dataset:

Create a new class in shared/datasets/loader.py inheriting from BaseDataset
Implement load(), get_symbolic_rules(), and get_characteristics()
Add to get_all_datasets() function

# Citation

tbd
