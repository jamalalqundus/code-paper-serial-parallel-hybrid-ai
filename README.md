# Serial vs. Parallel Hybrid AI: A Contingency Framework for Architecture Selection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Journal](https://img.shields.io/badge/Target-Applied%20Intelligence-red.svg)](https://www.springer.com/journal/10489)

This repository contains the complete implementation for the paper:

> **"Serial vs. Parallel Hybrid AI: A Contingency Framework for Architecture Selection Based on Task Characteristics"**  
> *Target Journal: Applied Intelligence (Springer)*

The code compares **Semi-symbolic (serial pipeline)** and **Concurrency (parallel voting)** architectures across multiple datasets to derive a contingency framework for architecture selection based on task characteristics: dependency ($\delta$), error tolerance ($\varepsilon$), and ambiguity ($\alpha$).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Output Files](#output-files)
- [Extending the Framework](#extending-the-framework)
- [Results Summary](#results-summary)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## Overview

Hybrid AI systems that integrate symbolic reasoning with subsymbolic learning have demonstrated superior performance over pure approaches. However, two distinct hybrid architectures exist: **Semi-symbolic (serial pipeline)** and **Concurrency (parallel voting)**. Current practice lacks systematic guidance for choosing between them.

This code provides:

- Implementation of both architectures
- Experiments on 8 datasets across 5 domains
- Statistical analysis with 20 runs per condition
- Contingency framework validation
- Reproducible results for the paper

---

## Directory Structure
- 📂 my_results/
- 📄 run_comparison.py
- 📂 figures/
- 📄 generate_figures.py
- 📂 shared/
  - 📄 datasets
  - 📄 metrics
  - 📄 architectures


---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/jamalalqundus/code-paper-serial-parallel-hybrid-ai.git
cd code-paper-serial-parallel-hybrid-ai
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

#### Requirements (requirements.txt):
- numpy>=1.21.0
- pandas>=1.3.0
- scikit-learn>=1.0.0
- xgboost>=1.5.0
- torch>=1.9.0
- torchvision>=0.10.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- tqdm>=4.62.0
- scipy>=1.7.0
- ucimlrepo>=0.0.3

## Usage
### Run Experiments
```bash
cd experiments
python run_comparison.py --datasets all --runs 20 --output ./my_results
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--datasets` | Dataset names (comma-separated) or "all" | `"all"` |
| `--runs` | Number of runs per configuration | `20` |
| `--epochs` | Training epochs per run | `5` |
| `--output` | Output directory | `"./my_results"` |
| `--list` | List available datasets and exit | `False` |


#### Examples
```bash
# Run all datasets
python run_comparison.py --datasets all --runs 20 --output ./my_results

# Run specific datasets
python run_comparison.py --datasets heart_disease,nsl_kdd,adult_income --runs 10 --output ./my_results

# List available datasets
python run_comparison.py --list

# Quick test run
python run_comparison.py --datasets mnist --runs 5 --epochs 3 --output ./test_results
```

### Generate Figures
After running experiments, generate the paper figures:
```bash
python generate_figures.py --results ./my_results --output ./figures
```

## Datasets

The following datasets are included (from `shared/datasets/loader.py`):

| Dataset | Domain | Task Type | Samples | Features |
|---------|--------|-----------|---------|----------|
| UCI Heart Disease | Healthcare | Classification | 297 | 13 |
| NSL-KDD | Cybersecurity | Classification | 5,000 | 41 |
| Adult Income | Social Science | Classification | 10,000 | 14 |
| Credit Card Fraud | Finance | Classification | 5,412 | 30 |
| MNIST | Computer Vision | Classification | 10,000 | 784 |
| HighDep_LowAmb | Synthetic | Classification | 1,000 | 12 |
| LowDep_HighAmb | Synthetic | Classification | 1,000 | 15 |
| Medium_Balanced | Synthetic | Classification | 1,000 | 13 |

**Note:** The synthetic datasets are designed to validate the contingency framework with controlled characteristics.

---

## Output Files

| File | Description |
|------|-------------|
| `comparison_results.csv` | Raw results for all runs (320 rows) |
| `comparison_summary.csv` | Aggregated statistics per (dataset, architecture) |
| `contingency_analysis.csv` | Contingency factors and accuracy differences |
| `accuracy_comparison.png/pdf` | Bar chart comparing architectures |
| `latency_comparison.png/pdf` | Latency comparison chart |
| `explainability_comparison.png/pdf` | Explainability scores chart |
| `confidence_comparison.png/pdf` | Confidence calibration chart |
| `contingency_analysis.png/pdf` | Accuracy difference vs ambiguity plot |



## Extending the Framework


### Adding a New Dataset
#### 1. Create a new class in shared/datasets/loader.py inheriting from BaseDataset:
```bash
class MyNewDataset(BaseDataset):
    def __init__(self):
        super().__init__(
            name='My_Dataset',
            domain='my_domain',
            task_type='classification'  # or 'regression'
        )
    
    def _load_raw(self):
        # Load your data
        X, y = load_my_data()
        return X, y
    
    def get_characteristics(self):
        return {
            'dependency': 3.0,
            'error_tolerance': 3.0,
            'ambiguity': 3.0
        }
```
#### 2.Add to get_all_datasets() function:
```bash
def get_all_datasets():
    return {
        # ... existing datasets ...
        'my_dataset': MyNewDataset(),
    }
  ```


### Adding a New Architecture

Create a new class in `shared/architectures/base.py` inheriting from `BaseHybridArchitecture`.

---

## Results Summary

| Finding | Result |
|---------|--------|
| Concurrency win rate | 5/8 datasets (62.5%) |
| Improvement on high-ambiguity tasks | +0.2% to +0.9% absolute accuracy |
| Explainability (Concurrency) | 3.0 vs 2.4 (5-point scale) |
| Latency overhead | 5-6× slower |
| Contingency prediction accuracy | 75% (6/8 datasets) |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{alqundus2025serial,
  title={Serial vs. Parallel Hybrid AI: A Contingency Framework for Architecture Selection Based on Task Characteristics},
  author={Al Qundus, Jamal},
  journal={Applied Intelligence},
  year={2025},
  publisher={Springer}
}
```

## License

This project is licensed under the MIT License

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Jamal Al Qundus**  
German Jordanian University  
Email: jamal.alqundus@gju.edu.jo  
GitHub: [jamalalqundus](https://github.com/jamalalqundus)

---

## Acknowledgments

This research was supported by the Business Intelligence and Data Analytics department at German Jordanian University.