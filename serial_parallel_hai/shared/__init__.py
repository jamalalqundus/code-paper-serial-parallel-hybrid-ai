"""
Shared modules for Hybrid AI publication suite.
"""

from hai.github_repos.serial_parallel_hai.shared.architectures.base import SemiSymbolic, Concurrency
from hai.github_repos.serial_parallel_hai.shared.datasets.loader import get_all_datasets, UCIHeartDisease, NSLKDD, CMAPSS
from hai.github_repos.serial_parallel_hai.shared.metrics.compute import compute_metrics, compute_uncertainty_metrics, compute_explainability_score

__all__ = [
    'SemiSymbolic',
    'Concurrency', 
    'get_all_datasets',
    'UCIHeartDisease',
    'NSLKDD',
    'CMAPSS',
    'compute_metrics',
    'compute_uncertainty_metrics',
    'compute_explainability_score'
]