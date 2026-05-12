"""
Metrics and Evaluation Utilities

Provides comprehensive metrics calculation and visualization
for evaluating phishing detection models.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, average_precision_score
)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    average_precision: float
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'roc_auc': self.roc_auc,
            'average_precision': self.average_precision,
            'confusion_matrix': self.confusion_matrix.tolist(),
            'classification_report': self.classification_report
        }
    
    def summary(self) -> str:
        """Get formatted summary string."""
        return (
            f"Accuracy: {self.accuracy:.4f}\n"
            f"Precision: {self.precision:.4f}\n"
            f"Recall: {self.recall:.4f}\n"
            f"F1-Score: {self.f1_score:.4f}\n"
            f"ROC-AUC: {self.roc_auc:.4f}\n"
            f"Average Precision: {self.average_precision:.4f}"
        )


class MetricsCalculator:
    """
    Calculator for phishing detection evaluation metrics.
    
    Computes various classification metrics and provides
    visualization tools for model evaluation.
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize the metrics calculator.
        
        Args:
            class_names: Names for classes (default: ['legitimate', 'phishing'])
        """
        self.class_names = class_names or ['legitimate', 'phishing']
    
    def calculate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> EvaluationMetrics:
        """
        Calculate all evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            
        Returns:
            EvaluationMetrics object
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Probability-based metrics
        if y_proba is not None:
            y_proba = np.array(y_proba)
            roc_auc = roc_auc_score(y_true, y_proba)
            avg_precision = average_precision_score(y_true, y_proba)
        else:
            roc_auc = 0.0
            avg_precision = 0.0
        
        # Confusion matrix and report
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            average_precision=avg_precision,
            confusion_matrix=cm,
            classification_report=report
        )
    
    def calculate_threshold_metrics(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: Optional[List[float]] = None
    ) -> List[Tuple[float, Dict[str, float]]]:
        """
        Calculate metrics at different thresholds.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            thresholds: List of thresholds to evaluate
            
        Returns:
            List of (threshold, metrics_dict) tuples
        """
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        results = []
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0)
            }
            results.append((threshold, metrics))
        
        return results
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        metric: str = 'f1_score'
    ) -> Tuple[float, float]:
        """
        Find optimal threshold for a given metric.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            metric: Metric to optimize ('f1_score', 'precision', 'recall')
            
        Returns:
            Tuple of (optimal_threshold, metric_value)
        """
        thresholds = np.linspace(0.1, 0.9, 81)
        best_threshold = 0.5
        best_value = 0.0
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1_score':
                value = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                value = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                value = recall_score(y_true, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if value > best_value:
                best_value = value
                best_threshold = threshold
        
        return best_threshold, best_value
    
    def compare_models(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        probabilities: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, EvaluationMetrics]:
        """
        Compare multiple models.
        
        Args:
            y_true: True labels
            predictions: Dictionary of model_name -> predictions
            probabilities: Optional dictionary of model_name -> probabilities
            
        Returns:
            Dictionary of model_name -> EvaluationMetrics
        """
        results = {}
        
        for model_name, y_pred in predictions.items():
            y_proba = probabilities.get(model_name) if probabilities else None
            results[model_name] = self.calculate(y_true, y_pred, y_proba)
        
        return results
    
    def get_roc_curve_data(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get ROC curve data.
        
        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        return roc_curve(y_true, y_proba)
    
    def get_pr_curve_data(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get Precision-Recall curve data.
        
        Returns:
            Tuple of (precision, recall, thresholds)
        """
        return precision_recall_curve(y_true, y_proba)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "Blues",
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: Names for classes
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure or None
    """
    if not PLOTTING_AVAILABLE:
        print("Matplotlib/Seaborn not available. Skipping plot.")
        return None
    
    class_names = class_names or ['Legitimate', 'Phishing']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure or None
    """
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available. Skipping plot.")
        return None
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "Precision-Recall Curve",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure or None
    """
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available. Skipping plot.")
        return None
    
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap_score = average_precision_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(recall, precision, 'b-', linewidth=2, label=f'PR curve (AP = {ap_score:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_metrics_comparison(
    metrics: Dict[str, EvaluationMetrics],
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Plot comparison of multiple models.
    
    Args:
        metrics: Dictionary of model_name -> EvaluationMetrics
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure or None
    """
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available. Skipping plot.")
        return None
    
    model_names = list(metrics.keys())
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    data = []
    for model_name in model_names:
        model_metrics = metrics[model_name]
        data.append([
            model_metrics.accuracy,
            model_metrics.precision,
            model_metrics.recall,
            model_metrics.f1_score,
            model_metrics.roc_auc
        ])
    
    data = np.array(data)
    x = np.arange(len(metric_names))
    width = 0.8 / len(model_names)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, model_name in enumerate(model_names):
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, data[i], width, label=model_name)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_threshold_analysis(
    threshold_metrics: List[Tuple[float, Dict[str, float]]],
    title: str = "Threshold Analysis",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Plot metrics across different thresholds.
    
    Args:
        threshold_metrics: List of (threshold, metrics) tuples
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure or None
    """
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available. Skipping plot.")
        return None
    
    thresholds = [t for t, _ in threshold_metrics]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
        values = [m[metric_name] for _, m in threshold_metrics]
        ax.plot(thresholds, values, 'o-', label=metric_name, linewidth=2, markersize=4)
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    # Example usage
    print("Metrics Calculator Module Test")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    
    y_true = np.random.randint(0, 2, n_samples)
    y_proba = np.clip(y_true + np.random.randn(n_samples) * 0.3, 0, 1)
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Calculate metrics
    calculator = MetricsCalculator()
    metrics = calculator.calculate(y_true, y_pred, y_proba)
    
    print("\nEvaluation Metrics:")
    print(metrics.summary())
    
    print("\nConfusion Matrix:")
    print(metrics.confusion_matrix)
    
    # Find optimal threshold
    optimal_threshold, best_f1 = calculator.find_optimal_threshold(y_true, y_proba, 'f1_score')
    print(f"\nOptimal threshold for F1: {optimal_threshold:.3f} (F1={best_f1:.4f})")
