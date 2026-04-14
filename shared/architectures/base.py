"""
Base classes for hybrid AI architectures.
Shared across all three manuscripts.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from collections import Counter
import time

class BaseHybridArchitecture(ABC):
    """Abstract base class for hybrid AI architectures."""
    
    def __init__(self, name: str):
        self.name = name
        self.trace = []
        
    @abstractmethod
    def process(self, input_data: np.ndarray, **kwargs) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def explain(self) -> str:
        pass


class SemiSymbolic(BaseHybridArchitecture):
    """Semi-symbolic (serial pipeline) architecture."""
    
    def __init__(self, symbolic_engine, subsymbolic_model, mode='symbolic_first'):
        super().__init__(name='SemiSymbolic')
        self.symbolic = symbolic_engine
        self.subsymbolic = subsymbolic_model
        self.mode = mode
        self.validation_loop = False
        
    def process(self, input_data: np.ndarray, quick_pass: bool = False, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        self.trace = []
        failure_flag = False
        
        try:
            if self.mode == 'symbolic_first':
                symbolic_output = self.symbolic.reason(input_data)
                self.trace.append(f"Symbolic reasoning produced: {symbolic_output}")
                
                features = self._symbolic_to_features(symbolic_output, input_data)
                prediction = self.subsymbolic.predict(features)
                self.trace.append(f"Subsymbolic prediction: {prediction}")
                
                if self.validation_loop:
                    validated = self._validate(prediction, symbolic_output)
                    if validated != prediction:
                        self.trace.append(f"Validation overrode prediction: {validated}")
                        prediction = validated
            else:
                subsymbolic_output = self.subsymbolic.extract_features(input_data)
                self.trace.append(f"Subsymbolic feature extraction: {subsymbolic_output.shape}")
                predicates = self._features_to_predicates(subsymbolic_output)
                prediction = self.symbolic.query(predicates)
                self.trace.append(f"Symbolic query result: {prediction}")
                
        except Exception as e:
            failure_flag = True
            prediction = None
            self.trace.append(f"FAILURE: {str(e)}")
        
        latency_ms = (time.time() - start_time) * 1000
        confidence = self._compute_confidence(prediction, self.trace) if not failure_flag else 0.0
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'trace': self.trace,
            'latency_ms': latency_ms,
            'failure_flag': failure_flag,
            'intermediate_signals': {
                'mode': self.mode,
                'trace_length': len(self.trace),
                'validation_used': self.validation_loop
            }
        }
    
    def _symbolic_to_features(self, symbolic_output, raw_input):
        return raw_input
    
    def _features_to_predicates(self, features):
        return []
    
    def _validate(self, prediction, symbolic_output):
        return prediction
    
    def _compute_confidence(self, prediction, trace):
        return min(0.95, 0.5 + 0.05 * len(trace))
    
    def explain(self) -> str:
        return "\n".join(self.trace)


class Concurrency(BaseHybridArchitecture):
    """Concurrency (parallel voting) architecture - FIXED to never return None."""
    
    def __init__(self, voters: List, weights: Optional[List[float]] = None, 
                 tie_breaker='random', diversity_penalty=False):
        super().__init__(name='Concurrency')
        self.voters = voters
        self.weights = weights if weights else [1.0] * len(voters)
        self.tie_breaker = tie_breaker
        self.diversity_penalty = diversity_penalty
        
    def process(self, input_data: np.ndarray, fast_subsample: bool = False, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        self.trace = []
        failure_flag = False
        
        # Fast subsample: use only first 3 voters
        voters_to_use = self.voters[:3] if fast_subsample else self.voters
        weights_to_use = self.weights[:3] if fast_subsample else self.weights
        
        # DEFAULT prediction (fallback) - never return None
        default_prediction = 0  # For classification, default to class 0
        
        try:
            votes = []
            vote_confidences = []
            
            for i, (voter_name, voter_model) in enumerate(voters_to_use):
                try:
                    result = voter_model.predict(input_data)
                    # Ensure result is not None
                    if result is None:
                        result = default_prediction
                    votes.append(result)
                    
                    # Get confidence
                    if hasattr(voter_model, 'predict_proba'):
                        conf = voter_model.predict_proba(input_data)
                        if isinstance(conf, (list, np.ndarray)):
                            conf = max(conf) if len(conf) > 0 else 0.5
                        elif conf is None:
                            conf = 0.5
                    else:
                        conf = 0.5
                    
                    vote_confidences.append(float(conf))
                    self.trace.append(f"Voter '{voter_name}' voted: {result} (conf={vote_confidences[-1]:.2f})")
                except Exception as e:
                    # Voter failed, use default
                    votes.append(default_prediction)
                    vote_confidences.append(0.0)
                    self.trace.append(f"Voter '{voter_name}' FAILED: {str(e)} -> default {default_prediction}")
            
            if self.diversity_penalty:
                votes, vote_confidences = self._apply_diversity_penalty(votes, vote_confidences)
            
            # Weighted voting
            weighted_votes = {}
            for vote, weight, conf in zip(votes, weights_to_use, vote_confidences):
                if vote is not None:
                    weighted_votes[vote] = weighted_votes.get(vote, 0) + weight * conf
            
            if weighted_votes:
                max_weight = max(weighted_votes.values())
                winners = [v for v, w in weighted_votes.items() if w == max_weight]
                
                if len(winners) > 1:
                    if self.tie_breaker == 'random':
                        prediction = np.random.choice(winners)
                        self.trace.append(f"Tie broken randomly: chose {prediction}")
                    elif self.tie_breaker == 'serial_fallback':
                        prediction = votes[0] if votes[0] is not None else default_prediction
                        self.trace.append(f"Tie broken by serial fallback: {prediction}")
                    else:  # confidence tie-breaker
                        winner_confidences = [vote_confidences[i] for i, v in enumerate(votes) if v in winners]
                        if winner_confidences:
                            prediction = winners[np.argmax(winner_confidences)]
                        else:
                            prediction = default_prediction
                        self.trace.append(f"Tie broken by confidence: {prediction}")
                else:
                    prediction = winners[0]
            else:
                # No votes cast - use default
                prediction = default_prediction
                failure_flag = True
                self.trace.append(f"WARNING: No votes cast. Using default {default_prediction}")
            
            # Ensure prediction is not None
            if prediction is None:
                prediction = default_prediction
                self.trace.append(f"WARNING: Prediction was None. Using default {default_prediction}")
            
            # Compute vote distribution and entropy
            vote_distribution = {}
            for v in votes:
                if v is not None:
                    vote_distribution[v] = vote_distribution.get(v, 0) + 1
            
            total_votes = len(votes)
            if total_votes > 0 and vote_distribution:
                probs = [count / total_votes for count in vote_distribution.values()]
                entropy = -sum(p * np.log(p + 1e-10) for p in probs)
                max_entropy = np.log(len(vote_distribution)) if len(vote_distribution) > 1 else 1.0
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            else:
                normalized_entropy = 1.0
            
        except Exception as e:
            failure_flag = True
            prediction = default_prediction
            normalized_entropy = 1.0
            self.trace.append(f"CRITICAL FAILURE: {str(e)} -> default {default_prediction}")
        
        latency_ms = (time.time() - start_time) * 1000
        confidence = 1.0 - normalized_entropy if not failure_flag else 0.1
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'trace': self.trace,
            'latency_ms': latency_ms,
            'failure_flag': failure_flag,
            'intermediate_signals': {
                'vote_distribution': vote_distribution if 'vote_distribution' in dir() else {},
                'entropy': normalized_entropy,
                'num_voters': len(voters_to_use),
                'tie_breaker': self.tie_breaker
            }
        }
    
    def _apply_diversity_penalty(self, votes, confidences):
        return votes, confidences
    
    def explain(self) -> str:
        return "\n".join(self.trace)    

# ============================================================================
# ENHANCED MOCK MODELS FOR REALISTIC SIMULATION
# ============================================================================

class EnhancedMockSymbolicEngine:
    """
    Enhanced symbolic engine that simulates realistic reasoning behavior.
    Returns predictions based on interpretable rules with some noise.
    """
    
    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)
        self.rules = []
        
    def reason(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Simulate symbolic reasoning by applying heuristic rules.
        
        For demonstration, uses simple decision rules based on feature values.
        """
        # Ensure x is 1D
        if x.ndim > 1:
            x = x.flatten()
        
        # Heuristic rules (simulate symbolic reasoning)
        # Rule 1: If first feature > median,倾向于 class 1
        rule1_fired = x[0] > 0
        # Rule 2: If second feature < -0.5,倾向于 class 0
        rule2_fired = x[1] < -0.5 if len(x) > 1 else False
        # Rule 3: Sum of first 5 features
        sum_features = np.sum(x[:5]) if len(x) >= 5 else np.sum(x)
        
        # Combine rules with some stochasticity
        score = 0.0
        if rule1_fired:
            score += 0.4
        if rule2_fired:
            score -= 0.3
        score += 0.1 * np.tanh(sum_features / 5.0)
        
        # Add small noise (symbolic systems have some uncertainty)
        score += self.rng.normal(0, 0.05)
        
        prediction = 1 if score > 0 else 0
        
        return {
            'prediction': prediction,
            'confidence': 0.5 + 0.4 * abs(score),
            'rules_fired': [rule1_fired, rule2_fired],
            'score': score
        }
    
    def query(self, predicates: list) -> int:
        """Query the symbolic engine with predicates."""
        # Simple predicate evaluation
        score = sum(1 for p in predicates if p) / max(1, len(predicates))
        return 1 if score > 0.5 else 0
    
    def predict(self, x: np.ndarray) -> int:
        """Predict class for input."""
        result = self.reason(x)
        return result['prediction']
    
    def predict_proba(self, x: np.ndarray) -> float:
        """Return prediction confidence."""
        result = self.reason(x)
        return result['confidence']
    
    def add_rule(self, rule: str):
        self.rules.append(rule)
    
    def remove_rule(self, rule: str):
        if rule in self.rules:
            self.rules.remove(rule)
    
    def get_rules(self) -> list:
        return self.rules.copy()


class EnhancedMockSubsymbolicModel:
    """
    Enhanced subsymbolic model that simulates neural network behavior.
    Uses a small internal MLP for realistic predictions.
    Supports both classification and regression tasks.
    """
    
    def __init__(self, input_dim: int = 13, hidden_dim: int = 32, 
                 task_type: str = 'classification', random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)
        self.input_dim = input_dim
        self.task_type = task_type  # 'classification' or 'regression'
        
        # Initialize random weights (simulate a trained model)
        self.W1 = self.rng.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        
        if task_type == 'classification':
            self.W2 = self.rng.randn(hidden_dim, 1) * 0.1
            self.b2 = 0.0
        else:  # regression
            self.W2 = self.rng.randn(hidden_dim, 1) * 0.1
            self.b2 = 0.0
        
        # Training flag
        self.is_trained = False
        
    def _forward(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Simple forward pass through simulated network."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Ensure input dimension matches
        if x.shape[1] != self.input_dim:
            # Pad or truncate
            if x.shape[1] < self.input_dim:
                x = np.pad(x, ((0, 0), (0, self.input_dim - x.shape[1])), mode='constant')
            else:
                x = x[:, :self.input_dim]
        
        h = np.tanh(x @ self.W1 + self.b1)
        output = (h @ self.W2 + self.b2).flatten()
        
        # Clip to prevent overflow
        output = np.clip(output, -50, 50)
        
        if self.task_type == 'classification':
            prob = 1.0 / (1.0 + np.exp(-output))  # sigmoid
            return prob[0], h
        else:  # regression
            return output[0], h
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10):
        """Simulate training by adjusting weights slightly."""
        self.is_trained = True
        
        # Simple simulated training: adjust weights toward data distribution
        for epoch in range(epochs):
            for i, x in enumerate(X):
                prob, _ = self._forward(x)
                if self.task_type == 'classification':
                    error = y[i] - prob
                else:
                    # For regression, normalize error
                    error = (y[i] - prob) / (np.std(y) + 1e-8)
                # Tiny weight update (simulate learning)
                self.W2 += 0.001 * error * self.rng.randn(*self.W2.shape)
    
    def predict(self, x: np.ndarray):
        """Predict class (classification) or value (regression)."""
        output, _ = self._forward(x)
        if self.task_type == 'classification':
            return 1 if output > 0.5 else 0
        else:  # regression
            # Scale to reasonable range (0-100 for RUL prediction)
            return max(0, min(100, output * 100))
    
    def predict_proba(self, x: np.ndarray) -> float:
        """Return prediction probability (classification only)."""
        if self.task_type == 'regression':
            # For regression, return normalized confidence based on prediction stability
            output, _ = self._forward(x)
            return min(0.95, max(0.05, 1.0 - abs(output) / 10.0))
        output, _ = self._forward(x)
        return output
    
    def extract_features(self, x: np.ndarray) -> np.ndarray:
        """Extract hidden layer features."""
        _, h = self._forward(x)
        return h


class RealisticDataset:
    """
    Realistic dataset generator with controlled difficulty and characteristics.
    Supports both classification and regression tasks.
    """
    
    def __init__(self, name: str, domain: str, task_type: str = 'classification',
                 n_samples: int = 1000, n_features: int = 13, 
                 noise_level: float = 0.1, class_sep: float = 1.0, 
                 random_seed: int = 42):
        self.name = name
        self.domain = domain
        self.task_type = task_type
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise_level = noise_level
        self.class_sep = class_sep
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        
        # Initialize attributes
        self.X = None
        self.y = None
        
        # Generate data
        self._generate_data()
        
    def _generate_data(self):
        """Generate synthetic data with controlled characteristics."""
        if self.task_type == 'classification':
            from sklearn.datasets import make_classification
            
            # Base classification data
            self.X, self.y = make_classification(
                n_samples=self.n_samples,
                n_features=self.n_features,
                n_informative=max(1, self.n_features - 2),
                n_redundant=1,
                n_clusters_per_class=1,
                flip_y=self.noise_level,
                class_sep=self.class_sep,
                random_state=self.random_seed
            )
        else:  # regression
            from sklearn.datasets import make_regression
            
            # Base regression data
            self.X, self.y = make_regression(
                n_samples=self.n_samples,
                n_features=self.n_features,
                n_informative=max(1, self.n_features - 2),
                noise=self.noise_level,
                random_state=self.random_seed
            )
            # Scale to positive range (e.g., for RUL prediction)
            self.y = np.abs(self.y) * 10
        
        # Add domain-specific patterns
        if self.domain == 'healthcare':
            # Healthcare: add clinical pattern (non-linear)
            if self.X.shape[1] > 0:
                self.X[:, 0] = np.sin(self.X[:, 0]) * 2
        elif self.domain == 'cybersecurity':
            # Cybersecurity: add outliers
            outlier_idx = self.rng.choice(self.n_samples, int(0.05 * self.n_samples), replace=False)
            self.X[outlier_idx] += self.rng.randn(len(outlier_idx), self.n_features) * 3
        elif self.domain == 'industrial':
            # Industrial: add trend
            trend = np.linspace(0, 1, self.n_samples).reshape(-1, 1)
            if self.X.shape[1] > 0:
                self.X[:, :1] += trend * 0.5
    
    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return features and labels."""
        return self.X, self.y
    
    def get_symbolic_rules(self) -> list:
        """Return symbolic rules for the domain."""
        rules = {
            'healthcare': [
                "IF feature_0 > 0 AND feature_1 < 0 THEN high_risk",
                "IF feature_2 > 1.5 THEN requires_followup",
            ],
            'cybersecurity': [
                "IF feature_3 < -1 AND feature_4 > 0 THEN attack_detected",
                "IF feature_5 > 2 THEN potential_breach",
            ],
            'industrial': [
                "IF feature_6 > 1 AND temperature_rising THEN degradation",
                "IF vibration > threshold THEN maintenance_required",
            ]
        }
        return rules.get(self.domain, ["IF condition THEN conclusion"])
    
    def get_characteristics(self) -> Dict[str, float]:
        """
        Calculate actual characteristics from the generated data.
        This provides realistic δ, ε, α values based on data properties.
        """
        # Check if data exists
        if self.X is None or self.y is None:
            return {
                'dependency': 3.0,
                'error_tolerance': 3.0,
                'ambiguity': 3.0
            }
        
        # DEPENDENCY (δ): How structured/predictable is the data?
        # Higher = more sequential dependency between features
        dependency = 3.0
        try:
            # Calculate correlation between consecutive features as a proxy
            if self.X.shape[1] > 1:
                corrs = []
                for i in range(min(5, self.X.shape[1] - 1)):
                    corr = np.corrcoef(self.X[:, i], self.X[:, i+1])[0, 1]
                    if not np.isnan(corr):
                        corrs.append(abs(corr))
                if corrs:
                    dependency = min(5.0, max(1.0, np.mean(corrs) * 5))
        except Exception:
            dependency = 3.0
        
        # ERROR TOLERANCE (ε): How robust to label noise?
        # Lower = less tolerant (errors are costly)
        try:
            if self.task_type == 'classification':
                error_tolerance = max(1.0, min(5.0, 5 - self.class_sep * 1.5))
            else:
                error_tolerance = max(1.0, min(5.0, 5 - self.noise_level * 10))
        except Exception:
            error_tolerance = 3.0
        
        # AMBIGUITY (α): How uncertain is the mapping?
        # Higher = more ambiguity
        try:
            if self.task_type == 'classification':
                ambiguity = min(5.0, max(1.0, (self.noise_level * 12) + (1.2 - self.class_sep)))
            else:
                ambiguity = min(5.0, max(1.0, self.noise_level * 15))
        except Exception:
            ambiguity = 3.0
        
        return {
            'dependency': round(dependency, 2),
            'error_tolerance': round(error_tolerance, 2),
            'ambiguity': round(ambiguity, 2)
        }


def create_realistic_datasets():
    """
    Create datasets with CONTRASTING characteristics to demonstrate 
    the contingency framework.
    
    Expected outcomes:
    - High dependency + Low ambiguity → SemiSymbolic wins
    - Low dependency + High ambiguity → Concurrency wins
    """
    return {
        # DATASET 1: High Dependency, Low Ambiguity (SemiSymbolic should win)
        'high_dependency_low_ambiguity': RealisticDataset(
            name='HighDep_LowAmb',
            domain='industrial',
            task_type='classification',
            n_samples=800,
            n_features=12,
            noise_level=0.08,      # Low noise = low ambiguity
            class_sep=1.2,         # High separation = low ambiguity
            random_seed=42
        ),
        
        # DATASET 2: Low Dependency, High Ambiguity (Concurrency should win)
        'low_dependency_high_ambiguity': RealisticDataset(
            name='LowDep_HighAmb',
            domain='cybersecurity',
            task_type='classification',
            n_samples=800,
            n_features=15,
            noise_level=0.35,      # High noise = high ambiguity
            class_sep=0.4,         # Low separation = high ambiguity
            random_seed=43         # Different seed for contrast
        ),
        
        # DATASET 3: Medium Dependency, Medium Ambiguity (Tie or small margin)
        'medium_balance': RealisticDataset(
            name='Medium_Balanced',
            domain='healthcare',
            task_type='classification',
            n_samples=800,
            n_features=13,
            noise_level=0.20,      # Medium noise
            class_sep=0.8,         # Medium separation
            random_seed=44
        ),
        
        # DATASET 4: Original heart_disease style
        'heart_disease': RealisticDataset(
            name='UCI_Heart_Disease',
            domain='healthcare',
            task_type='classification',
            n_samples=1000,
            n_features=13,
            noise_level=0.15,
            class_sep=0.9,
            random_seed=42
        ),
        
        # DATASET 5: Original nsl_kdd style
        'nsl_kdd': RealisticDataset(
            name='NSL_KDD',
            domain='cybersecurity',
            task_type='classification',
            n_samples=2000,
            n_features=41,
            noise_level=0.25,
            class_sep=0.7,
            random_seed=42
        ),
        
        # DATASET 6: Regression dataset for CMAPSS style
        'cmapss': RealisticDataset(
            name='CMAPSS',
            domain='industrial',
            task_type='regression',
            n_samples=1000,
            n_features=24,
            noise_level=0.15,
            class_sep=0.0,  # Not used for regression
            random_seed=42
        ),
    }


# Helper function to get dataset by name
def get_dataset(name: str):
    """Get a specific dataset by name."""
    datasets = create_realistic_datasets()
    return datasets.get(name)


# Helper function to list all available datasets
def list_datasets() -> List[str]:
    """List all available dataset names."""
    return list(create_realistic_datasets().keys())