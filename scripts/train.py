#!/usr/bin/env python
"""
Training Script for Hybrid Phishing Detection System

Trains both URL and Visual analysis models on provided datasets.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.url_analysis.url_model import URLPhishingClassifier
from src.visual_analysis.visual_model import VisualPhishingClassifier
from src.utils.data_loader import DataLoader, ScreenshotDatasetLoader
from src.utils.metrics import MetricsCalculator, plot_confusion_matrix, plot_roc_curve


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_url_model(
    config: dict,
    data_path: str = None,
    output_dir: str = "models",
    use_synthetic: bool = False
) -> dict:
    """
    Train the URL phishing classification model.
    
    Args:
        config: Configuration dictionary
        data_path: Path to training data CSV
        output_dir: Directory to save trained model
        use_synthetic: Whether to use synthetic data for demo
        
    Returns:
        Dictionary with training results
    """
    print("\n" + "=" * 60)
    print("URL MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    loader = DataLoader()
    
    if use_synthetic or data_path is None:
        print("\nGenerating synthetic training data...")
        dataset = loader.generate_synthetic_dataset(
            n_samples=2000,
            phishing_ratio=0.5
        )
    else:
        print(f"\nLoading data from {data_path}...")
        dataset = loader.load_csv(data_path)
    
    # Print dataset statistics
    stats = dataset.get_statistics()
    print(f"\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Split data
    train_set, test_set = dataset.split(test_size=0.2)
    print(f"\nTrain set: {len(train_set)} samples")
    print(f"Test set: {len(test_set)} samples")
    
    # Create and train model
    url_config = config.get('url_analysis', {})
    model_type = url_config.get('model_type', 'xgboost')
    model_params = url_config.get('model_params', {})
    
    print(f"\nTraining {model_type} model...")
    
    classifier = URLPhishingClassifier(
        model_type=model_type,
        model_params=model_params
    )
    
    # Train
    train_metrics = classifier.fit(
        train_set.urls,
        train_set.labels,
        validation_split=0.15
    )
    
    print("\nTraining Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = classifier.evaluate(test_set.urls, test_set.labels)
    
    print("\nTest Metrics:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
    
    # Print top features
    print("\nTop 10 Important Features:")
    for feature, importance in classifier.get_feature_importance(10):
        print(f"  {feature}: {importance:.4f}")
    
    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_path = output_path / "url_model.pkl"
    classifier.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save metrics
    results = {
        'model_type': model_type,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'dataset_stats': stats,
        'training_date': datetime.now().isoformat()
    }
    
    metrics_path = output_path / "url_model_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def train_visual_model(
    config: dict,
    data_dir: str = "data/screenshots",
    output_dir: str = "models"
) -> dict:
    """
    Train the visual phishing classification model.
    
    Args:
        config: Configuration dictionary
        data_dir: Directory containing screenshot images
        output_dir: Directory to save trained model
        
    Returns:
        Dictionary with training results
    """
    print("\n" + "=" * 60)
    print("VISUAL MODEL TRAINING")
    print("=" * 60)
    
    # Check if data directory exists and has images
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"\nWarning: Data directory {data_dir} does not exist.")
        print("Please provide screenshot images in:")
        print(f"  {data_dir}/phishing/")
        print(f"  {data_dir}/legitimate/")
        print("\nSkipping visual model training.")
        return {'status': 'skipped', 'reason': 'no_data'}
    
    # Load screenshot dataset
    screenshot_loader = ScreenshotDatasetLoader(data_dir)
    image_paths, labels = screenshot_loader.load_from_directory()
    
    if len(image_paths) == 0:
        print("\nNo images found in the data directory.")
        print("Please organize screenshots as follows:")
        print(f"  {data_dir}/phishing/*.png")
        print(f"  {data_dir}/legitimate/*.png")
        print("\nSkipping visual model training.")
        return {'status': 'skipped', 'reason': 'no_images'}
    
    print(f"\nLoaded {len(image_paths)} images")
    print(f"  Phishing: {sum(labels)}")
    print(f"  Legitimate: {len(labels) - sum(labels)}")
    
    # Split dataset
    splits = screenshot_loader.split_dataset(image_paths, labels)
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(splits['train'][0])} images")
    print(f"  Val: {len(splits['val'][0])} images")
    print(f"  Test: {len(splits['test'][0])} images")
    
    # Create and train model
    visual_config = config.get('visual_analysis', {})
    model_type = visual_config.get('model_type', 'resnet50')
    
    print(f"\nTraining {model_type} model...")
    
    classifier = VisualPhishingClassifier(
        model_type=model_type,
        image_size=visual_config.get('image_size', 224),
        pretrained=visual_config.get('pretrained', True)
    )
    
    # Train
    training_config = visual_config.get('training', {})
    history = classifier.fit(
        train_images=splits['train'][0],
        train_labels=splits['train'][1],
        val_images=splits['val'][0],
        val_labels=splits['val'][1],
        epochs=training_config.get('epochs', 50),
        batch_size=visual_config.get('batch_size', 32),
        learning_rate=training_config.get('learning_rate', 0.001)
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = classifier.evaluate(
        splits['test'][0],
        splits['test'][1]
    )
    
    print("\nTest Metrics:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
    
    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_path = output_path / "visual_model.pt"
    classifier.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save metrics
    results = {
        'model_type': model_type,
        'training_history': {k: [float(v) for v in vals] for k, vals in history.items()},
        'test_metrics': test_metrics,
        'training_date': datetime.now().isoformat()
    }
    
    metrics_path = output_path / "visual_model_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train Hybrid Phishing Detection Models"
    )
    parser.add_argument(
        "--config", 
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--url-data",
        help="Path to URL training data CSV"
    )
    parser.add_argument(
        "--screenshot-dir",
        default="data/screenshots",
        help="Directory containing screenshot images"
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--train-url",
        action="store_true",
        help="Train URL analysis model"
    )
    parser.add_argument(
        "--train-visual",
        action="store_true",
        help="Train visual analysis model"
    )
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Train all models"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for URL model training"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(args.config)
    else:
        print(f"Warning: Config file {args.config} not found. Using defaults.")
        config = {}
    
    print("=" * 60)
    print("HYBRID PHISHING DETECTION SYSTEM - TRAINING")
    print("=" * 60)
    print(f"\nConfiguration: {args.config}")
    print(f"Output directory: {args.output_dir}")
    
    results = {}
    
    # Train URL model
    if args.train_url or args.train_all:
        results['url_model'] = train_url_model(
            config=config,
            data_path=args.url_data,
            output_dir=args.output_dir,
            use_synthetic=args.synthetic or (args.url_data is None)
        )
    
    # Train Visual model
    if args.train_visual or args.train_all:
        results['visual_model'] = train_visual_model(
            config=config,
            data_dir=args.screenshot_dir,
            output_dir=args.output_dir
        )
    
    if not (args.train_url or args.train_visual or args.train_all):
        print("\nNo training task specified. Use --train-url, --train-visual, or --train-all")
        print("\nExample usage:")
        print("  python scripts/train.py --train-all --synthetic")
        print("  python scripts/train.py --train-url --url-data data/urls.csv")
        print("  python scripts/train.py --train-visual --screenshot-dir data/screenshots")
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        if isinstance(model_results, dict):
            status = model_results.get('status', 'completed')
            print(f"  Status: {status}")
            if 'test_metrics' in model_results:
                print(f"  Test Accuracy: {model_results['test_metrics'].get('accuracy', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
