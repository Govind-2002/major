#!/usr/bin/env python
"""
Evaluation Script for Hybrid Phishing Detection System

Evaluates trained models on test datasets and generates reports.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hybrid_detector.detector import HybridPhishingDetector
from src.utils.data_loader import DataLoader, ScreenshotDatasetLoader
from src.utils.metrics import (
    MetricsCalculator, 
    plot_confusion_matrix, 
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_threshold_analysis
)


def evaluate_url_model(
    model_path: str,
    test_data_path: str = None,
    output_dir: str = "evaluation"
) -> dict:
    """
    Evaluate the URL phishing classification model.
    
    Args:
        model_path: Path to trained model
        test_data_path: Path to test data CSV
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with evaluation results
    """
    print("\n" + "=" * 60)
    print("URL MODEL EVALUATION")
    print("=" * 60)
    
    from src.url_analysis.url_model import URLPhishingClassifier
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    classifier = URLPhishingClassifier.from_pretrained(model_path)
    
    # Load test data
    loader = DataLoader()
    
    if test_data_path and Path(test_data_path).exists():
        print(f"Loading test data from {test_data_path}...")
        dataset = loader.load_csv(test_data_path)
    else:
        print("Generating synthetic test data...")
        dataset = loader.generate_synthetic_dataset(n_samples=500, phishing_ratio=0.5)
    
    # Evaluate
    print(f"\nEvaluating on {len(dataset)} samples...")
    
    predictions = classifier.predict(dataset.urls)
    probabilities = classifier.predict_proba(dataset.urls)[:, 1]
    
    # Calculate metrics
    calculator = MetricsCalculator()
    metrics = calculator.calculate(
        y_true=dataset.labels,
        y_pred=predictions,
        y_proba=probabilities
    )
    
    print("\nEvaluation Results:")
    print(metrics.summary())
    
    # Threshold analysis
    print("\nThreshold Analysis:")
    threshold_metrics = calculator.calculate_threshold_metrics(
        y_true=np.array(dataset.labels),
        y_proba=probabilities
    )
    
    for threshold, m in threshold_metrics:
        print(f"  Threshold {threshold:.1f}: F1={m['f1_score']:.4f}, "
              f"Precision={m['precision']:.4f}, Recall={m['recall']:.4f}")
    
    # Find optimal threshold
    opt_threshold, opt_f1 = calculator.find_optimal_threshold(
        y_true=np.array(dataset.labels),
        y_proba=probabilities,
        metric='f1_score'
    )
    print(f"\nOptimal Threshold: {opt_threshold:.3f} (F1={opt_f1:.4f})")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'model_path': str(model_path),
        'test_samples': len(dataset),
        'metrics': metrics.to_dict(),
        'optimal_threshold': opt_threshold,
        'optimal_f1': opt_f1,
        'evaluation_date': datetime.now().isoformat()
    }
    
    results_path = output_path / "url_model_evaluation.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_path}")
    
    # Generate plots
    print("\nGenerating evaluation plots...")
    
    plot_confusion_matrix(
        metrics.confusion_matrix,
        save_path=str(output_path / "url_confusion_matrix.png")
    )
    
    plot_roc_curve(
        y_true=np.array(dataset.labels),
        y_proba=probabilities,
        save_path=str(output_path / "url_roc_curve.png")
    )
    
    plot_precision_recall_curve(
        y_true=np.array(dataset.labels),
        y_proba=probabilities,
        save_path=str(output_path / "url_pr_curve.png")
    )
    
    plot_threshold_analysis(
        threshold_metrics,
        save_path=str(output_path / "url_threshold_analysis.png")
    )
    
    print(f"Plots saved to {output_path}")
    
    return results


def evaluate_hybrid_detector(
    model_dir: str = "models",
    test_data_path: str = None,
    output_dir: str = "evaluation"
) -> dict:
    """
    Evaluate the hybrid phishing detection system.
    
    Args:
        model_dir: Directory containing trained models
        test_data_path: Path to test data CSV
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with evaluation results
    """
    print("\n" + "=" * 60)
    print("HYBRID DETECTOR EVALUATION")
    print("=" * 60)
    
    model_dir = Path(model_dir)
    
    # Initialize detector
    url_model_path = model_dir / "url_model.pkl"
    visual_model_path = model_dir / "visual_model.pt"
    
    print(f"\nInitializing hybrid detector...")
    print(f"  URL model: {url_model_path} (exists: {url_model_path.exists()})")
    print(f"  Visual model: {visual_model_path} (exists: {visual_model_path.exists()})")
    
    detector = HybridPhishingDetector(
        url_model_path=str(url_model_path) if url_model_path.exists() else None,
        visual_model_path=str(visual_model_path) if visual_model_path.exists() else None,
        llm_provider="mock",
        enable_screenshot=False
    )
    
    # Load test data
    loader = DataLoader()
    
    if test_data_path and Path(test_data_path).exists():
        print(f"\nLoading test data from {test_data_path}...")
        dataset = loader.load_csv(test_data_path)
    else:
        print("\nGenerating synthetic test data...")
        dataset = loader.generate_synthetic_dataset(n_samples=200, phishing_ratio=0.5)
    
    # Evaluate
    print(f"\nEvaluating on {len(dataset)} samples...")
    
    predictions = []
    scores = []
    analysis_times = []
    
    for i, (url, label) in enumerate(zip(dataset.urls, dataset.labels)):
        result = detector.analyze(url, capture_screenshot=False, generate_explanation=False)
        predictions.append(1 if result.is_phishing else 0)
        scores.append(result.combined_score)
        analysis_times.append(result.analysis_time)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} samples...")
    
    # Calculate metrics
    calculator = MetricsCalculator()
    metrics = calculator.calculate(
        y_true=dataset.labels,
        y_pred=predictions,
        y_proba=scores
    )
    
    print("\nEvaluation Results:")
    print(metrics.summary())
    
    print(f"\nPerformance Statistics:")
    print(f"  Average analysis time: {np.mean(analysis_times):.3f}s")
    print(f"  Total analysis time: {sum(analysis_times):.1f}s")
    print(f"  Throughput: {len(dataset) / sum(analysis_times):.1f} URLs/second")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'model_dir': str(model_dir),
        'test_samples': len(dataset),
        'metrics': metrics.to_dict(),
        'avg_analysis_time': float(np.mean(analysis_times)),
        'total_analysis_time': float(sum(analysis_times)),
        'evaluation_date': datetime.now().isoformat()
    }
    
    results_path = output_path / "hybrid_evaluation.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_path}")
    
    # Generate plots
    print("\nGenerating evaluation plots...")
    
    plot_confusion_matrix(
        metrics.confusion_matrix,
        save_path=str(output_path / "hybrid_confusion_matrix.png")
    )
    
    plot_roc_curve(
        y_true=np.array(dataset.labels),
        y_proba=np.array(scores),
        save_path=str(output_path / "hybrid_roc_curve.png")
    )
    
    print(f"Plots saved to {output_path}")
    
    detector.close()
    
    return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate Hybrid Phishing Detection Models"
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--url-model",
        help="Path to URL model (overrides model-dir)"
    )
    parser.add_argument(
        "--test-data",
        help="Path to test data CSV"
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--eval-url",
        action="store_true",
        help="Evaluate URL model only"
    )
    parser.add_argument(
        "--eval-hybrid",
        action="store_true",
        help="Evaluate hybrid detector"
    )
    parser.add_argument(
        "--eval-all",
        action="store_true",
        help="Run all evaluations"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HYBRID PHISHING DETECTION SYSTEM - EVALUATION")
    print("=" * 60)
    
    results = {}
    
    # Evaluate URL model
    if args.eval_url or args.eval_all:
        url_model = args.url_model or f"{args.model_dir}/url_model.pkl"
        if Path(url_model).exists():
            results['url_model'] = evaluate_url_model(
                model_path=url_model,
                test_data_path=args.test_data,
                output_dir=args.output_dir
            )
        else:
            print(f"\nURL model not found at {url_model}")
    
    # Evaluate hybrid detector
    if args.eval_hybrid or args.eval_all:
        results['hybrid'] = evaluate_hybrid_detector(
            model_dir=args.model_dir,
            test_data_path=args.test_data,
            output_dir=args.output_dir
        )
    
    if not (args.eval_url or args.eval_hybrid or args.eval_all):
        print("\nNo evaluation task specified. Use --eval-url, --eval-hybrid, or --eval-all")
        print("\nExample usage:")
        print("  python scripts/evaluate.py --eval-all")
        print("  python scripts/evaluate.py --eval-url --url-model models/url_model.pkl")
    
    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    
    for eval_name, eval_results in results.items():
        print(f"\n{eval_name}:")
        if 'metrics' in eval_results:
            m = eval_results['metrics']
            print(f"  Accuracy: {m['accuracy']:.4f}")
            print(f"  F1-Score: {m['f1_score']:.4f}")
            print(f"  ROC-AUC: {m['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
