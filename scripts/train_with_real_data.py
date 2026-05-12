"""
Train URL model with real PhishTank data.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.url_analysis.feature_extractor import URLFeatureExtractor
from src.url_analysis.url_model import URLPhishingClassifier
from src.utils.metrics import MetricsCalculator

def main():
    print("=" * 60)
    print("TRAINING URL MODEL WITH REAL DATA")
    print("=" * 60)
    
    # Load real URLs
    print("\n📥 Loading real URLs...")
    with open("data/real_urls.json", "r") as f:
        data = json.load(f)
    
    phishing_urls = [item['url'] for item in data['phishing_urls']]
    legitimate_urls = [item['url'] for item in data['legitimate_urls']]
    
    print(f"  Phishing URLs available: {len(phishing_urls)}")
    print(f"  Legitimate URLs available: {len(legitimate_urls)}")
    
    # Use ALL available data for training
    # Sample phishing URLs to balance with legitimate (max 50:1 ratio to avoid extreme imbalance)
    n_legitimate = len(legitimate_urls)
    n_phishing = len(phishing_urls)  # Use all phishing URLs
    
    print(f"  Using ALL {n_phishing} phishing URLs")
    print(f"  Using ALL {n_legitimate} legitimate URLs")
    
    # Create labels
    all_urls = phishing_urls + legitimate_urls
    labels = [1] * len(phishing_urls) + [0] * len(legitimate_urls)
    
    print(f"\n📊 Total samples: {len(all_urls)}")
    
    # Extract features
    print("\n🔍 Extracting URL features (this may take a while for large datasets)...")
    extractor = URLFeatureExtractor()
    
    features_list = []
    valid_indices = []
    
    for i, url in enumerate(all_urls):
        try:
            features = extractor.extract_features(url)
            features_list.append(features.to_array())
            valid_indices.append(i)
        except Exception as e:
            print(f"  Skip {url[:50]}: {str(e)[:30]}")
    
    X = np.array(features_list)
    y = np.array([labels[i] for i in valid_indices])
    
    print(f"  Extracted features for {len(X)} URLs")
    print(f"  Feature dimension: {X.shape[1]}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Train set: {len(X_train)} samples")
    print(f"📊 Test set: {len(X_test)} samples")
    
    # Train model
    print("\n🎯 Training XGBoost model...")
    from xgboost import XGBClassifier
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\n📈 Evaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    calc = MetricsCalculator()
    metrics = calc.calculate(y_test, y_pred, y_proba)
    
    print("\nTest Results:")
    print(f"  Accuracy:  {metrics.accuracy:.4f}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall:    {metrics.recall:.4f}")
    print(f"  F1-Score:  {metrics.f1_score:.4f}")
    print(f"  ROC-AUC:   {metrics.roc_auc:.4f}")
    
    # Save model
    import joblib
    model_path = "models/url_model_real.pkl"
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Show feature importance
    print("\n🔝 Top 10 Important Features:")
    feature_names = URLFeatureExtractor.get_feature_names()
    importance = dict(zip(feature_names, model.feature_importances_))
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for name, score in sorted_importance[:10]:
        print(f"  {name}: {score:.4f}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
