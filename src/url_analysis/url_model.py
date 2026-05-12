"""
URL Phishing Classification Model

Implements machine learning models for URL-based phishing detection
using features extracted from URLs. Supports multiple model types
including Random Forest, XGBoost, and LightGBM.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

from .feature_extractor import URLFeatureExtractor, URLFeatures


class URLPhishingClassifier:
    """
    Machine learning classifier for URL-based phishing detection.
    
    Supports multiple model types and provides methods for training,
    prediction, and evaluation.
    """
    
    SUPPORTED_MODELS = {'random_forest', 'xgboost', 'lightgbm'}
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        model_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the URL Phishing Classifier.
        
        Args:
            model_type: Type of model to use ('random_forest', 'xgboost', 'lightgbm')
            model_params: Optional dictionary of model hyperparameters
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model type must be one of {self.SUPPORTED_MODELS}")
        
        self.model_type = model_type
        self.model_params = model_params or self._get_default_params(model_type)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_extractor = URLFeatureExtractor()
        self.is_fitted = False
        self.feature_names = URLFeatureExtractor.get_feature_names()
        self.feature_importances_ = None
    
    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default hyperparameters for each model type."""
        defaults = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            },
            'lightgbm': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'random_state': 42,
                'verbose': -1
            }
        }
        return defaults.get(model_type, {})
    
    def _create_model(self):
        """Create the ML model based on model_type."""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(**self.model_params)
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(**self.model_params)
        elif self.model_type == 'lightgbm':
            return lgb.LGBMClassifier(**self.model_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _extract_features(self, urls: List[str]) -> np.ndarray:
        """Extract features from URLs."""
        return self.feature_extractor.to_feature_matrix(urls)
    
    def fit(
        self,
        urls: List[str],
        labels: List[int],
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the model on labeled URL data.
        
        Args:
            urls: List of URL strings
            labels: List of labels (0 for legitimate, 1 for phishing)
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary containing training and validation metrics
        """
        # Extract features
        X = self._extract_features(urls)
        y = np.array(labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Store feature importances
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
        
        # Calculate metrics
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)
        val_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'val_accuracy': accuracy_score(y_val, val_pred),
            'val_precision': precision_score(y_val, val_pred),
            'val_recall': recall_score(y_val, val_pred),
            'val_f1': f1_score(y_val, val_pred),
            'val_roc_auc': roc_auc_score(y_val, val_proba)
        }
        
        return metrics
    
    def fit_from_dataframe(
        self,
        df: pd.DataFrame,
        url_column: str = 'url',
        label_column: str = 'label',
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the model from a pandas DataFrame.
        
        Args:
            df: DataFrame containing URLs and labels
            url_column: Name of the column containing URLs
            label_column: Name of the column containing labels
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary containing training metrics
        """
        urls = df[url_column].tolist()
        labels = df[label_column].tolist()
        return self.fit(urls, labels, validation_split)
    
    def predict(self, urls: Union[str, List[str]]) -> np.ndarray:
        """
        Predict whether URLs are phishing or legitimate.
        
        Args:
            urls: Single URL string or list of URLs
            
        Returns:
            Array of predictions (0 for legitimate, 1 for phishing)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if isinstance(urls, str):
            urls = [urls]
        
        X = self._extract_features(urls)
        X_scaled = self.scaler.transform(X) if self.scaler else X
        return self.model.predict(X_scaled)
    
    def predict_proba(self, urls: Union[str, List[str]]) -> np.ndarray:
        """
        Predict probability of URLs being phishing.
        
        Args:
            urls: Single URL string or list of URLs
            
        Returns:
            Array of probabilities for each class [legitimate, phishing]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if isinstance(urls, str):
            urls = [urls]
        
        X = self._extract_features(urls)
        X_scaled = self.scaler.transform(X) if self.scaler else X
        return self.model.predict_proba(X_scaled)
    
    def predict_with_features(
        self,
        url: str
    ) -> Tuple[int, float, Dict[str, Any]]:
        """
        Predict with detailed feature information.
        
        Args:
            url: URL string to analyze
            
        Returns:
            Tuple of (prediction, confidence, features_dict)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        features = self.feature_extractor.extract_features(url)
        X = features.to_array().reshape(1, -1)
        X_scaled = self.scaler.transform(X) if self.scaler else X
        
        prediction = self.model.predict(X_scaled)[0]
        proba = self.model.predict_proba(X_scaled)[0]
        confidence = proba[prediction]
        
        return prediction, confidence, features.to_dict()
    
    def cross_validate(
        self,
        urls: List[str],
        labels: List[int],
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation on the data.
        
        Args:
            urls: List of URL strings
            labels: List of labels
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary containing cross-validation scores
        """
        X = self._extract_features(urls)
        X_scaled = self.scaler.fit_transform(X)
        y = np.array(labels)
        
        model = self._create_model()
        
        scores = {
            'accuracy': cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy'),
            'precision': cross_val_score(model, X_scaled, y, cv=cv, scoring='precision'),
            'recall': cross_val_score(model, X_scaled, y, cv=cv, scoring='recall'),
            'f1': cross_val_score(model, X_scaled, y, cv=cv, scoring='f1'),
            'roc_auc': cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
        }
        
        return {
            f'{metric}_mean': np.mean(values)
            for metric, values in scores.items()
        } | {
            f'{metric}_std': np.std(values)
            for metric, values in scores.items()
        }
    
    def evaluate(
        self,
        urls: List[str],
        labels: List[int]
    ) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            urls: List of URL strings
            labels: List of true labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        y_true = np.array(labels)
        y_pred = self.predict(urls)
        y_proba = self.predict_proba(urls)[:, 1]
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
    
    def get_feature_importance(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get the most important features for classification.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            List of (feature_name, importance) tuples sorted by importance
        """
        if self.feature_importances_ is None:
            raise ValueError("Model must be fitted first")
        
        sorted_features = sorted(
            self.feature_importances_.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:top_n]
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'feature_names': self.feature_names,
            'feature_importances': self.feature_importances_
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: Union[str, Path]) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Handle both formats: direct model or dictionary
        if isinstance(model_data, dict):
            self.model = model_data['model']
            self.scaler = model_data.get('scaler')
            self.model_type = model_data.get('model_type', 'xgboost')
            self.model_params = model_data.get('model_params', {})
            self.feature_names = model_data.get('feature_names', [])
            self.feature_importances_ = model_data.get('feature_importances', {})
        else:
            # Direct model object (from joblib/pickle)
            self.model = model_data
            self.scaler = None
            self.model_type = 'xgboost'
            self.model_params = {}
            self.feature_names = []
            if hasattr(model_data, 'feature_importances_'):
                self.feature_importances_ = dict(enumerate(model_data.feature_importances_))
            else:
                self.feature_importances_ = {}
        self.is_fitted = True
    
    @classmethod
    def from_pretrained(cls, filepath: Union[str, Path]) -> 'URLPhishingClassifier':
        """
        Load a pretrained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            URLPhishingClassifier instance with loaded model
        """
        classifier = cls()
        classifier.load(filepath)
        return classifier


if __name__ == "__main__":
    # Example usage with synthetic data
    import random
    
    # Generate synthetic training data
    legitimate_patterns = [
        "https://www.{}.com/",
        "https://{}.org/page",
        "https://shop.{}.com/products",
        "https://www.{}.net/about",
        "https://app.{}.io/dashboard"
    ]
    
    phishing_patterns = [
        "http://192.168.1.{}/login.php",
        "http://secure-{}-login.xyz/verify",
        "http://{}-account-verify.tk/update",
        "https://www.{}.com.suspicious.link/signin",
        "http://xn--{}.com/auth?token=abc"
    ]
    
    words = ['google', 'facebook', 'microsoft', 'amazon', 'apple', 'netflix', 'paypal']
    
    urls = []
    labels = []
    
    for _ in range(100):
        word = random.choice(words)
        pattern = random.choice(legitimate_patterns)
        urls.append(pattern.format(word))
        labels.append(0)
    
    for _ in range(100):
        word = random.choice(words)
        num = random.randint(1, 255)
        pattern = random.choice(phishing_patterns)
        urls.append(pattern.format(word if '{}' in pattern and 'ip' not in pattern.lower() else num))
        labels.append(1)
    
    # Train and evaluate
    classifier = URLPhishingClassifier(model_type='xgboost')
    metrics = classifier.fit(urls, labels)
    
    print("URL Phishing Classifier Training Results")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTop 10 Important Features:")
    for feature, importance in classifier.get_feature_importance(10):
        print(f"  {feature}: {importance:.4f}")
    
    # Test prediction
    test_url = "http://secure-paypal-login.xyz/verify?token=abc123"
    pred, conf, features = classifier.predict_with_features(test_url)
    print(f"\nTest URL: {test_url}")
    print(f"  Prediction: {'Phishing' if pred == 1 else 'Legitimate'}")
    print(f"  Confidence: {conf:.4f}")
