"""
Evaluation metrics for all three manuscripts.
Fixed to handle None values and regression properly.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from typing import Dict, Any, List, Optional


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                    task_type: str = 'classification') -> Dict[str, float]:
    """
    Core performance metrics.
    
    Handles None values in predictions by treating them as incorrect.
    Handles regression tasks (RMSE, MAE) vs classification (accuracy, F1).
    """
    metrics = {}
    
    # Convert to numpy arrays if not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Handle None values in predictions
    # Replace None with a sentinel value that won't match any true label
    # For classification, use -1 (assuming true labels are 0,1,2,...)
    # For regression, use NaN (we'll handle separately)
    
    if task_type == 'classification':
        # Replace None with -1 (will be counted as incorrect)
        y_pred_clean = np.array([-1 if p is None else p for p in y_pred])
        
        # Also handle any non-integer or out-of-range values
        unique_true = np.unique(y_true)
        valid_mask = np.array([p in unique_true for p in y_pred_clean])
        
        # Calculate accuracy: only count valid predictions
        if len(y_true) > 0:
            correct = np.sum((y_pred_clean == y_true) & valid_mask)
            metrics['accuracy'] = correct / len(y_true)
        else:
            metrics['accuracy'] = 0.0
        
        # Calculate F1 scores (need to filter out None/invalid predictions)
        valid_indices = valid_mask
        if np.sum(valid_indices) > 0:
            y_true_valid = y_true[valid_indices]
            y_pred_valid = y_pred_clean[valid_indices]
            
            # Ensure integer type
            y_true_valid = y_true_valid.astype(int)
            y_pred_valid = y_pred_valid.astype(int)
            
            metrics['f1_macro'] = f1_score(y_true_valid, y_pred_valid, 
                                           average='macro', zero_division=0)
            metrics['f1_weighted'] = f1_score(y_true_valid, y_pred_valid, 
                                              average='weighted', zero_division=0)
        else:
            metrics['f1_macro'] = 0.0
            metrics['f1_weighted'] = 0.0
    
    else:  # regression
        # Replace None with the mean of non-None predictions
        y_pred_clean = np.array([p if p is not None else np.nan for p in y_pred])
        
        # Remove NaN values for calculation
        valid_mask = ~np.isnan(y_pred_clean) & ~np.isnan(y_true)
        
        if np.sum(valid_mask) > 0:
            y_true_valid = y_true[valid_mask]
            y_pred_valid = y_pred_clean[valid_mask]
            
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
            metrics['mae'] = np.mean(np.abs(y_true_valid - y_pred_valid))
        else:
            metrics['rmse'] = float('inf')
            metrics['mae'] = float('inf')
    
    return metrics


def compute_uncertainty_metrics(vote_distribution: Dict[Any, int]) -> Dict[str, float]:
    """Compute uncertainty metrics from vote distribution."""
    total = sum(vote_distribution.values())
    if total == 0:
        return {'entropy': 1.0, 'dempster_conflict': 1.0, 'consensus_ratio': 0.0}
    
    probs = [count / total for count in vote_distribution.values()]
    
    entropy = -sum(p * np.log(p + 1e-10) for p in probs)
    max_entropy = np.log(len(probs)) if len(probs) > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    
    max_prob = max(probs)
    dempster_conflict = 1.0 - max_prob
    consensus_ratio = max_prob
    
    return {
        'entropy': normalized_entropy,
        'dempster_conflict': dempster_conflict,
        'consensus_ratio': consensus_ratio
    }


def compute_failure_metrics(architecture_output: Dict[str, Any]) -> Dict[str, Any]:
    """Compute failure-related metrics."""
    failure_metrics = {
        'failure_occurred': architecture_output.get('failure_flag', False),
        'failure_type': None,
        'warning_signals': {}
    }
    
    if not failure_metrics['failure_occurred']:
        intermediate = architecture_output.get('intermediate_signals', {})
        
        if 'trace_length' in intermediate:
            if intermediate['trace_length'] < 2:
                failure_metrics['warning_signals']['symbolic_bottleneck_risk'] = True
                
        if 'entropy' in intermediate:
            if intermediate['entropy'] > 0.8:
                failure_metrics['warning_signals']['high_entropy'] = True
            if intermediate.get('vote_distribution') and len(intermediate['vote_distribution']) > 2:
                failure_metrics['warning_signals']['fragmented_vote'] = True
    
    return failure_metrics


def compute_explainability_score(trace: List[str]) -> float:
    """Compute explainability score (0-5)."""
    if not trace:
        return 0.0
    
    score = 0.0
    score += min(2.0, len(trace) / 5.0)
    
    step_types = set()
    for step in trace:
        if 'symbolic' in step.lower():
            step_types.add('symbolic')
        if 'feature' in step.lower():
            step_types.add('feature')
        if 'prediction' in step.lower():
            step_types.add('prediction')
        if 'vote' in step.lower():
            step_types.add('vote')
    score += min(2.0, len(step_types) / 2.0)
    
    if all(len(step) < 200 for step in trace):
        score += 1.0
    
    return min(5.0, score)


def aggregate_results(all_results: List[Dict]) -> pd.DataFrame:
    """Aggregate results from multiple runs into a DataFrame."""
    df = pd.DataFrame(all_results)
    
    # Determine which metrics are available
    agg_dict = {}
    
    if 'accuracy' in df.columns:
        agg_dict['accuracy'] = ['mean', 'std']
    if 'rmse' in df.columns:
        agg_dict['rmse'] = ['mean', 'std']
    if 'mae' in df.columns:
        agg_dict['mae'] = ['mean', 'std']
    if 'f1_macro' in df.columns:
        agg_dict['f1_macro'] = ['mean', 'std']
    if 'latency_ms' in df.columns:
        agg_dict['latency_ms'] = ['mean', 'std']
    if 'explainability_score' in df.columns:
        agg_dict['explainability_score'] = 'mean'
    if 'failure_rate' in df.columns:
        agg_dict['failure_rate'] = 'mean'
    if 'avg_confidence' in df.columns:
        agg_dict['avg_confidence'] = 'mean'
    
    aggregated = df.groupby(['dataset', 'architecture']).agg(agg_dict).round(4)
    
    return aggregated