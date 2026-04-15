#!/usr/bin/env python
"""
Comparative Empirical Framework - Main Experiment Runner

This script runs experiments comparing Semi-symbolic and Concurrency architectures
across multiple datasets. Uses REAL datasets and CNN for image data.

Usage:
    python run_comparison.py --datasets all --runs 20 --output ./my_results
    python run_comparison.py --list  # List available datasets
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import shared modules
from hai.github_repos.serial_parallel_hai.shared.architectures.base import SemiSymbolic, Concurrency
from hai.github_repos.serial_parallel_hai.shared.metrics.compute import compute_metrics, compute_explainability_score
from hai.github_repos.serial_parallel_hai.shared.datasets.loader import get_all_datasets, list_available_datasets

# Import real ML models
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from collections import Counter

# PyTorch imports for CNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# CNN MODEL FOR MNIST
# ============================================================================

class SimpleCNN(nn.Module):
    """Simple CNN for MNIST-like 28x28 images."""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # After two poolings: 28 -> 14 -> 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class CNNClassifier:
    """Wrapper for PyTorch CNN to match sklearn interface."""
    
    def __init__(self, input_shape=(28, 28), num_classes=10, random_seed=42):
        torch.manual_seed(random_seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SimpleCNN(num_classes=num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.trained = False
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.training_epochs = 0
    
    def _preprocess(self, X):
        """Convert numpy array to torch tensor with correct shape."""
        if isinstance(X, np.ndarray):
            if X.ndim == 2:
                # Reshape (n_samples, 784) to (n_samples, 1, 28, 28)
                side = int(np.sqrt(X.shape[1]))
                X = X.reshape(-1, 1, side, side)
            X = torch.FloatTensor(X)
        return X.to(self.device)
    
    def fit(self, X, y, epochs=10, batch_size=64):
        """Train the CNN."""
        self.training_epochs = epochs
        X = self._preprocess(X)
        y = torch.LongTensor(y).to(self.device)
        
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
        
        self.trained = True
    
    def predict(self, x):
        """Predict class."""
        if not self.trained:
            return 0
        self.model.eval()
        with torch.no_grad():
            if x.ndim == 1:
                x = x.reshape(1, -1)
            x_tensor = self._preprocess(x)
            outputs = self.model(x_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()[0]
    
    def predict_proba(self, x):
        """Return prediction probability."""
        if not self.trained:
            return 0.5
        self.model.eval()
        with torch.no_grad():
            if x.ndim == 1:
                x = x.reshape(1, -1)
            x_tensor = self._preprocess(x)
            outputs = self.model(x_tensor)
            probs = torch.softmax(outputs, dim=1)
            return torch.max(probs, 1)[0].cpu().numpy()[0]
    
    def extract_features(self, x):
        """Extract features from the penultimate layer."""
        if not self.trained:
            return x
        self.model.eval()
        with torch.no_grad():
            if x.ndim == 1:
                x = x.reshape(1, -1)
            x_tensor = self._preprocess(x)
            # Forward to fc1 layer
            x_tensor = self.model.pool(torch.relu(self.model.conv1(x_tensor)))
            x_tensor = self.model.pool(torch.relu(self.model.conv2(x_tensor)))
            x_tensor = self.model.dropout1(x_tensor)
            x_tensor = x_tensor.view(-1, 64 * 7 * 7)
            features = torch.relu(self.model.fc1(x_tensor))
            return features.cpu().numpy()


class MNISTCNNClassifier(CNNClassifier):
    """Specialized CNN for MNIST."""
    
    def __init__(self, input_dim=None, random_seed=42):
        super().__init__(input_shape=(28, 28), num_classes=10, random_seed=random_seed)


# ============================================================================
# CLASSIFICATION MODELS FOR TABULAR DATA
# ============================================================================

class SymbolicClassifier:
    """Symbolic engine using DecisionTreeClassifier (interpretable)."""
    
    def __init__(self, random_seed=42):
        self.model = DecisionTreeClassifier(max_depth=3, random_state=random_seed)
        self.trained = False
    
    def fit(self, X, y, epochs=5):
        self.model.fit(X, y)
        self.trained = True
    
    def predict(self, x):
        if not self.trained:
            return 0
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return int(self.model.predict(x)[0])
    
    def predict_proba(self, x):
        if not self.trained:
            return 0.5
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return float(max(self.model.predict_proba(x)[0]))
    
    def reason(self, x):
        return {'prediction': self.predict(x), 'confidence': 0.7, 'rules_fired': []}
    
    def query(self, predicates):
        return 0


class SubsymbolicClassifier:
    """Subsymbolic model using XGBoost for tabular classification."""
    
    def __init__(self, input_dim, random_seed=42):
        self.model = xgb.XGBClassifier(
            n_estimators=50, 
            max_depth=3, 
            random_state=random_seed,
            eval_metric='logloss',
            use_label_encoder=False
        )
        self.trained = False
    
    def fit(self, X, y, epochs=5):
        self.model.fit(X, y)
        self.trained = True
    
    def predict(self, x):
        if not self.trained:
            return 0
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return int(self.model.predict(x)[0])
    
    def predict_proba(self, x):
        if not self.trained:
            return 0.5
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return float(max(self.model.predict_proba(x)[0]))
    
    def extract_features(self, x):
        return x


class SimpleSymbolicForImages:
    """Simple symbolic engine for image data (uses flattened pixels)."""
    
    def __init__(self, random_seed=42):
        self.model = DecisionTreeClassifier(max_depth=3, random_state=random_seed)
        self.trained = False
    
    def fit(self, X, y, epochs=5):
        # Flatten images for decision tree
        if X.ndim > 2:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X
        self.model.fit(X_flat, y)
        self.trained = True
    
    def predict(self, x):
        if not self.trained:
            return 0
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return int(self.model.predict(x)[0])
    
    def predict_proba(self, x):
        if not self.trained:
            return 0.5
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return float(max(self.model.predict_proba(x)[0]))
    
    def reason(self, x):
        return {'prediction': self.predict(x), 'confidence': 0.7, 'rules_fired': []}
    
    def query(self, predicates):
        return 0


# ============================================================================
# ARCHITECTURE CREATION
# ============================================================================

def create_architectures(input_dim: int, task_type: str = 'classification', 
                         dataset_name: str = '', random_seed: int = 42):
    """
    Create architecture instances with appropriate models for task type and dataset.
    
    Uses CNN for image datasets (MNIST), XGBoost for tabular data.
    """
    is_image_dataset = dataset_name in ['MNIST', 'Fashion_MNIST', 'CIFAR10']
    
    if task_type == 'classification':
        if is_image_dataset:
            # Use CNN for image datasets
            symbolic = SimpleSymbolicForImages(random_seed=random_seed)
            subsymbolic = MNISTCNNClassifier(random_seed=random_seed)
            
            voters = [
                ('symbolic_v1', SimpleSymbolicForImages(random_seed=random_seed + 1)),
                ('symbolic_v2', SimpleSymbolicForImages(random_seed=random_seed + 2)),
                ('cnn_v1', MNISTCNNClassifier(random_seed=random_seed + 3)),
                ('cnn_v2', MNISTCNNClassifier(random_seed=random_seed + 4)),
                ('hybrid_voter', MNISTCNNClassifier(random_seed=random_seed + 5)),
            ]
        else:
            # Use XGBoost for tabular data
            symbolic = SymbolicClassifier(random_seed=random_seed)
            subsymbolic = SubsymbolicClassifier(input_dim, random_seed=random_seed)
            
            voters = [
                ('symbolic_v1', SymbolicClassifier(random_seed=random_seed + 1)),
                ('symbolic_v2', SymbolicClassifier(random_seed=random_seed + 2)),
                ('subsymbolic_v1', SubsymbolicClassifier(input_dim, random_seed=random_seed + 3)),
                ('subsymbolic_v2', SubsymbolicClassifier(input_dim, random_seed=random_seed + 4)),
                ('hybrid_voter', SubsymbolicClassifier(input_dim, random_seed=random_seed + 5)),
            ]
    else:
        # Regression placeholder (if needed)
        symbolic = SymbolicClassifier(random_seed=random_seed)
        subsymbolic = SubsymbolicClassifier(input_dim, random_seed=random_seed)
        
        voters = [
            ('symbolic_v1', SymbolicClassifier(random_seed=random_seed + 1)),
            ('symbolic_v2', SymbolicClassifier(random_seed=random_seed + 2)),
            ('subsymbolic_v1', SubsymbolicClassifier(input_dim, random_seed=random_seed + 3)),
            ('subsymbolic_v2', SubsymbolicClassifier(input_dim, random_seed=random_seed + 4)),
            ('hybrid_voter', SubsymbolicClassifier(input_dim, random_seed=random_seed + 5)),
        ]
    
    semi = SemiSymbolic(symbolic, subsymbolic, mode='symbolic_first')
    semi.name = 'SemiSymbolic'
    
    concurrency = Concurrency(voters, tie_breaker='confidence')
    concurrency.name = 'Concurrency'
    
    return semi, concurrency


def train_architecture(architecture, X_train, y_train, task_type: str = 'classification', epochs: int = 5):
    """Train an architecture on training data."""
    if hasattr(architecture, 'subsymbolic') and hasattr(architecture.subsymbolic, 'fit'):
        architecture.subsymbolic.fit(X_train, y_train, epochs=epochs)
    
    if hasattr(architecture, 'symbolic') and hasattr(architecture.symbolic, 'fit'):
        architecture.symbolic.fit(X_train, y_train, epochs=epochs)
    
    if hasattr(architecture, 'voters') and architecture.voters is not None:
        for name, voter in architecture.voters:
            if hasattr(voter, 'fit'):
                try:
                    voter.fit(X_train, y_train, epochs=epochs)
                except Exception as e:
                    print(f"Warning: Voter {name} failed to train: {e}")


def run_single_experiment(dataset, architecture, run_id: int, train_epochs: int = 5):
    """Run a single experiment (one train/test split)."""
    X, y = dataset.load()
    task_type = dataset.task_type
    dataset_name = dataset.name
    
    split = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    X_shuffled, y_shuffled = X[indices], y[indices]
    
    X_train, X_test = X_shuffled[:split], X_shuffled[split:]
    y_train, y_test = y_shuffled[:split], y_shuffled[split:]
    
    # Only standardize for non-image datasets
    is_image = dataset_name in ['MNIST', 'Fashion_MNIST', 'CIFAR10']
    if not is_image:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    train_architecture(architecture, X_train, y_train, task_type, epochs=train_epochs)
    
    predictions = []
    latencies = []
    traces = []
    failure_flags = []
    confidences = []
    
    if task_type == 'classification':
        fallback_value = int(Counter(y_train).most_common(1)[0][0])
    else:
        fallback_value = float(np.mean(y_train))
    
    for x in X_test:
        output = architecture.process(x.reshape(1, -1))
        
        pred = output.get('prediction')
        if pred is None:
            pred = fallback_value
            output['prediction'] = pred
        
        predictions.append(pred)
        latencies.append(output['latency_ms'])
        traces.append(output['trace'])
        failure_flags.append(output['failure_flag'])
        confidences.append(output.get('confidence', 0.5))
    
    perf_metrics = compute_metrics(np.array(y_test), np.array(predictions), task_type)
    
    chars = dataset.get_characteristics()
    
    result = {
        'dataset': dataset.name,
        'domain': dataset.domain,
        'architecture': architecture.name,
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        **perf_metrics,
        'latency_ms': np.mean(latencies) / 1000.0,
        'latency_std': np.std(latencies) / 1000.0,
        'failure_rate': np.mean(failure_flags),
        'explainability_score': np.mean([compute_explainability_score(t) for t in traces]),
        'avg_confidence': np.mean(confidences),
        'dependency': chars.get('dependency', 3.0),
        'error_tolerance': chars.get('error_tolerance', 3.0),
        'ambiguity': chars.get('ambiguity', 3.0)
    }
    
    return result


def run_experiments(dataset, num_runs: int = 20, train_epochs: int = 5):
    """Run all experiments for a dataset."""
    X, _ = dataset.load()
    input_dim = X.shape[1]
    task_type = dataset.task_type
    dataset_name = dataset.name
    
    all_results = []
    
    for run_id in range(num_runs):
        semi, concurrency = create_architectures(input_dim, task_type, dataset_name)
        
        result_semi = run_single_experiment(dataset, semi, run_id, train_epochs)
        all_results.append(result_semi)
        
        result_conc = run_single_experiment(dataset, concurrency, run_id, train_epochs)
        all_results.append(result_conc)
    
    return all_results


def save_contingency_analysis(df, output_path):
    """Generate and save contingency analysis CSV."""
    
    contingency_data = []
    
    for dataset_name in df['dataset'].unique():
        subset = df[df['dataset'] == dataset_name]
        
        semi_subset = subset[subset['architecture'] == 'SemiSymbolic']
        conc_subset = subset[subset['architecture'] == 'Concurrency']
        
        if 'accuracy' in df.columns and not semi_subset['accuracy'].isna().all():
            semi_metric = semi_subset['accuracy'].mean()
            conc_metric = conc_subset['accuracy'].mean()
            metric_name = 'accuracy'
        else:
            semi_metric = None
            conc_metric = None
            metric_name = 'none'
        
        chars = {
            'dependency': subset['dependency'].iloc[0] if len(subset) > 0 else 3.0,
            'error_tolerance': subset['error_tolerance'].iloc[0] if len(subset) > 0 else 3.0,
            'ambiguity': subset['ambiguity'].iloc[0] if len(subset) > 0 else 3.0
        }
        
        if semi_metric is not None and conc_metric is not None and not np.isnan(semi_metric):
            diff = conc_metric - semi_metric
            winner = 'Concurrency' if diff > 0 else 'SemiSymbolic'
            semi_display = semi_metric
            conc_display = conc_metric
            diff_display = diff
        else:
            semi_display = None
            conc_display = None
            diff_display = None
            winner = 'N/A'
        
        contingency_data.append({
            'dataset': dataset_name,
            'domain': subset['domain'].iloc[0] if len(subset) > 0 else 'unknown',
            'semi_metric': semi_display,
            'concurrency_metric': conc_display,
            'metric_used': metric_name,
            'difference': diff_display,
            'winner': winner,
            'dependency': chars['dependency'],
            'error_tolerance': chars['error_tolerance'],
            'ambiguity': chars['ambiguity']
        })
    
    contingency_df = pd.DataFrame(contingency_data)
    csv_path = os.path.join(output_path, 'contingency_analysis.csv')
    contingency_df.to_csv(csv_path, index=False)
    print(f"\n💾 Contingency analysis saved to {csv_path}")
    
    # Print readable summary
    print("\n📊 CONTINGENCY ANALYSIS SUMMARY")
    print("-" * 90)
    print("{:<25s} {:>6s} {:>6s} {:>6s} {:>12s} {:>12s} {:>10s} {:>12s}".format(
        'Dataset', 'δ', 'ε', 'α', 'Semi', 'Conc', 'Diff', 'Winner'
    ))
    print("-" * 90)
    
    for _, row in contingency_df.iterrows():
        if row['semi_metric'] is not None:
            winner_symbol = '🔵 Semi' if row['winner'] == 'SemiSymbolic' else '🟠 Conc'
            diff_str = f"+{row['difference']:.4f}" if row['difference'] > 0 else f"{row['difference']:.4f}"
            print("{:<25s} {:>6.2f} {:>6.2f} {:>6.2f} {:>12.4f} {:>12.4f} {:>10s} {:>12s}".format(
                row['dataset'][:24], row['dependency'], row['error_tolerance'], 
                row['ambiguity'], row['semi_metric'], row['concurrency_metric'], 
                diff_str, winner_symbol
            ))
    
    return contingency_df


def print_results_summary(df):
    """Print a formatted summary of results."""
    print("\n" + "=" * 110)
    print(" " * 45 + "EXPERIMENT RESULTS SUMMARY")
    print("=" * 110)
    
    if 'accuracy' in df.columns:
        class_df = df[df['accuracy'].notna()].copy()
        
        if len(class_df) > 0:
            print("\n📊 CLASSIFICATION TASKS")
            print("-" * 110)
            print("{:<25s} {:<14s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
                'Dataset', 'Architecture', 'Accuracy', 'Latency(s)', 'Explain', 'Confidence', 'Winner'
            ))
            print("-" * 110)
            
            for dataset in class_df['dataset'].unique():
                semi_acc = class_df[(class_df['dataset'] == dataset) & (class_df['architecture'] == 'SemiSymbolic')]['accuracy'].mean()
                conc_acc = class_df[(class_df['dataset'] == dataset) & (class_df['architecture'] == 'Concurrency')]['accuracy'].mean()
                
                for arch in ['SemiSymbolic', 'Concurrency']:
                    subset = class_df[(class_df['dataset'] == dataset) & (class_df['architecture'] == arch)]
                    if len(subset) > 0:
                        acc_mean = subset['accuracy'].mean()
                        acc_std = subset['accuracy'].std()
                        latency = subset['latency_ms'].mean()
                        explain = subset['explainability_score'].mean()
                        confidence = subset['avg_confidence'].mean()
                        
                        winner = ''
                        if arch == 'Concurrency' and conc_acc > semi_acc + 0.005:
                            winner = '🏆'
                        elif arch == 'SemiSymbolic' and semi_acc > conc_acc + 0.005:
                            winner = '🏆'
                        
                        print("{:<25s} {:<14s} {:>6.4f}±{:<5.4f} {:>11.4f} {:>11.1f} {:>11.2f} {:>6s}".format(
                            dataset[:24], arch, acc_mean, acc_std, latency, explain, confidence, winner
                        ))
                print("-" * 110)
    
    print("\n" + "=" * 110)


def main():
    parser = argparse.ArgumentParser(description='Run comparative experiments for Hybrid AI architectures')
    parser.add_argument('--datasets', type=str, default='all',
                        help='Dataset names (comma-separated) or "all"')
    parser.add_argument('--runs', type=int, default=20,
                        help='Number of runs per configuration')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Training epochs per run')
    parser.add_argument('--output', type=str, default='./my_results',
                        help='Output directory for results')
    parser.add_argument('--list', action='store_true',
                        help='List available datasets and exit')
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable datasets:")
        print("-" * 40)
        for name in list_available_datasets():
            print(f"  • {name}")
        print()
        return
    
    os.makedirs(args.output, exist_ok=True)
    
    all_datasets = get_all_datasets()
    
    # Select datasets
    if args.datasets == 'all':
        datasets = all_datasets
    else:
        dataset_names = [d.strip() for d in args.datasets.split(',')]
        datasets = {k: v for k, v in all_datasets.items() if k in dataset_names}
    
    if not datasets:
        print("Error: No valid datasets specified.")
        print(f"Available datasets: {list_available_datasets()}")
        return
    
    print(f"\n{'='*60}")
    print(f"  Running experiments on {len(datasets)} datasets")
    print(f"  {args.runs} runs per architecture")
    print(f"  {args.epochs} training epochs")
    print(f"{'='*60}\n")
    
    all_results = []
    
    for name, dataset in tqdm(datasets.items(), desc="Processing datasets"):
        print(f"\n📊 Dataset: {name}")
        print(f"   Domain: {dataset.domain}")
        print(f"   Task: {dataset.task_type}")
        
        chars = dataset.get_characteristics()
        print(f"   Characteristics: δ={chars['dependency']}, ε={chars['error_tolerance']}, α={chars['ambiguity']}")
        
        try:
            results = run_experiments(dataset, args.runs, args.epochs)
            all_results.extend(results)
            print(f"   ✅ Completed {len(results)} runs ({args.runs} per architecture)")
        except Exception as e:
            print(f"   ❌ Error on {name}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_results:
        print("No results collected. Exiting.")
        return
    
    df = pd.DataFrame(all_results)
    
    # Save raw results
    csv_path = os.path.join(args.output, 'comparison_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Raw results saved to {csv_path}")
    
    # Print summary
    print_results_summary(df)
    
    # Save contingency analysis
    save_contingency_analysis(df, args.output)
    
    # Save summary statistics
    agg_dict = {}
    if 'accuracy' in df.columns:
        agg_dict['accuracy'] = ['mean', 'std']
    if 'latency_ms' in df.columns:
        agg_dict['latency_ms'] = 'mean'
    if 'explainability_score' in df.columns:
        agg_dict['explainability_score'] = 'mean'
    if 'failure_rate' in df.columns:
        agg_dict['failure_rate'] = 'mean'
    if 'avg_confidence' in df.columns:
        agg_dict['avg_confidence'] = 'mean'
    
    if agg_dict:
        summary = df.groupby(['dataset', 'architecture']).agg(agg_dict).round(4)
        summary_path = os.path.join(args.output, 'comparison_summary.csv')
        summary.to_csv(summary_path)
        print(f"\n💾 Summary saved to {summary_path}")
    
    print(f"\n{'='*60}")
    print(f"  ✅ Experiment complete!")
    print(f"  Results saved to {args.output}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()