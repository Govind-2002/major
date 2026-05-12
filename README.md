# Hybrid Multi-Modal Phishing Detection System

A comprehensive phishing detection system that combines **URL Analysis**, **Visual Analysis**, and **LLM-based Explainability** to provide accurate, interpretable phishing detection results.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Training](#training)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [References](#references)

## 🎯 Overview

Phishing attacks are among the most serious cybersecurity threats, where attackers create fraudulent websites to steal sensitive information. This project implements a **hybrid multi-modal approach** that addresses the limitations of single-source detection methods:

### Problem Statement

Traditional phishing detection approaches face several limitations:
- Dependence on single data source (URL-only or content-only)
- Lack of interpretability in detection decisions
- Poor performance on advanced phishing techniques
- Limited adaptability to new attack patterns

### Our Solution

This system combines three complementary analysis modules:

1. **URL Analysis Module**: Extracts 30+ features from URLs using ML models (XGBoost/Random Forest)
2. **Visual Analysis Module**: Analyzes webpage screenshots using deep learning (ResNet/ViT)
3. **LLM Explainability Module**: Generates human-readable explanations using GPT-4 or local models

## ✨ Features

- **Multi-Modal Detection**: Combines URL, visual, and semantic analysis
- **Explainable AI**: Provides clear reasoning for each detection
- **Real-Time Analysis**: Fast URL checking with < 100ms latency
- **Batch Processing**: Efficient processing of large URL lists
- **REST API**: Easy integration with existing systems
- **Configurable Fusion**: Multiple strategies for combining predictions
- **Screenshot Capture**: Automated webpage screenshot using Selenium
- **Extensible Architecture**: Easy to add new features or models

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Input URL                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  URL Analysis   │ │ Visual Analysis │ │  Screenshot     │
│     Module      │ │     Module      │ │   Capture       │
├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│ • URL Length    │ │ • ResNet50/ViT  │ │ • Selenium      │
│ • Domain Props  │ │ • CNN Features  │ │ • Chrome/Firefox│
│ • Entropy       │ │ • Logo Detection│ │ • Headless Mode │
│ • Sensitive     │ │ • Layout        │ │                 │
│   Words         │ │   Analysis      │ │                 │
│ • XGBoost/RF    │ │                 │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
          ┌─────────────────────────────────────────┐
          │         Fusion Module                    │
          │  (Weighted Average / Voting / Stacking) │
          └─────────────────────────────────────────┘
                              │
                              ▼
          ┌─────────────────────────────────────────┐
          │      LLM Explainability Module          │
          │   (GPT-4 / Local LLM / HuggingFace)     │
          └─────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Detection Result                              │
│  • Classification: Phishing / Legitimate                        │
│  • Confidence Score: 0.0 - 1.0                                  │
│  • Risk Level: Low / Medium / High / Critical                   │
│  • Explanation: Human-readable reasoning                        │
│  • Recommendations: Security advice                             │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Chrome or Firefox browser (for screenshot capture)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-repo/phishing-detection.git
cd phishing-detection
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/MacOS
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

```bash
# Copy environment template
copy .env.example .env

# Edit .env with your API keys (optional)
# OPENAI_API_KEY=your_key_here
```

## ⚡ Quick Start

### Option 1: Run Demo Script

```bash
python scripts/demo.py --mode all
```

### Option 2: Interactive Mode

```bash
python scripts/demo.py --mode interactive
```

### Option 3: Python API

```python
from src.hybrid_detector.detector import HybridPhishingDetector

# Initialize detector
detector = HybridPhishingDetector(
    llm_provider="mock",  # Use "openai" for real explanations
    enable_screenshot=False
)

# Analyze URL
result = detector.analyze("http://suspicious-login-paypal.xyz/verify")

print(f"Classification: {'PHISHING' if result.is_phishing else 'LEGITIMATE'}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Risk Level: {result.risk_level}")

# Quick check
is_phishing, confidence, risk = detector.quick_check("https://www.google.com")
```

## 📖 Usage

### URL Analysis Only

```python
from src.url_analysis.url_model import URLPhishingClassifier
from src.url_analysis.feature_extractor import URLFeatureExtractor

# Extract features
extractor = URLFeatureExtractor()
features = extractor.extract_features("http://example.com/login")
print(f"URL Length: {features.url_length}")
print(f"Has HTTPS: {features.has_https}")
print(f"Sensitive Words: {features.num_sensitive_words}")

# Use trained model
classifier = URLPhishingClassifier.from_pretrained("models/url_model.pkl")
prediction = classifier.predict("http://suspicious-site.com")
```

### Visual Analysis Only

```python
from src.visual_analysis.visual_model import VisualPhishingClassifier
from src.visual_analysis.screenshot_capture import ScreenshotCapture

# Capture screenshot
with ScreenshotCapture(headless=True) as capturer:
    result = capturer.capture("https://example.com")
    if result.success:
        print(f"Screenshot saved: {result.screenshot_path}")

# Analyze with trained model
classifier = VisualPhishingClassifier.from_pretrained("models/visual_model.pt")
prediction = classifier.predict("screenshot.png")
```

### Batch Processing

```python
from src.hybrid_detector.detector import HybridPhishingDetector

detector = HybridPhishingDetector(llm_provider="mock")

urls = [
    "https://www.google.com",
    "https://www.github.com",
    "http://phishing-site.xyz"
]

results = detector.analyze_batch(urls, capture_screenshots=False)

for result in results:
    status = "🚫 PHISHING" if result.is_phishing else "✅ SAFE"
    print(f"{status}: {result.url} (confidence: {result.confidence:.1%})")
```

## 🌐 API Reference

### Start the API Server

```bash
python -m api.app --host 0.0.0.0 --port 8000
```

Or use uvicorn directly:

```bash
uvicorn api.app:app --reload
```

### API Endpoints

#### Health Check
```bash
GET /health
```

#### Analyze Single URL
```bash
POST /analyze
Content-Type: application/json

{
    "url": "http://suspicious-site.com/login",
    "capture_screenshot": false,
    "generate_explanation": true
}
```

#### Quick Check
```bash
POST /quick-check
Content-Type: application/json

{
    "url": "https://www.google.com"
}
```

#### Batch Analysis
```bash
POST /batch-analyze
Content-Type: application/json

{
    "urls": ["https://google.com", "http://phishing.xyz"],
    "capture_screenshots": false,
    "generate_explanations": false
}
```

### API Documentation

Access interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🎓 Training

### Train URL Model

```bash
# Using synthetic data
python scripts/train.py --train-url --synthetic

# Using custom dataset
python scripts/train.py --train-url --url-data data/urls.csv
```

### Train Visual Model

```bash
# Requires screenshot dataset
python scripts/train.py --train-visual --screenshot-dir data/screenshots
```

### Train All Models

```bash
python scripts/train.py --train-all --synthetic
```

### Dataset Format

For URL training data (CSV):
```csv
url,label
https://www.google.com,0
http://phishing-site.xyz,1
```

For visual training data:
```
data/screenshots/
├── phishing/
│   ├── screenshot1.png
│   └── screenshot2.png
└── legitimate/
    ├── screenshot3.png
    └── screenshot4.png
```

## 📊 Evaluation

### Evaluate Models

```bash
# Evaluate all models
python scripts/evaluate.py --eval-all

# Evaluate URL model only
python scripts/evaluate.py --eval-url --url-model models/url_model.pkl

# Evaluate hybrid detector
python scripts/evaluate.py --eval-hybrid
```

### Evaluation Metrics

The system evaluates models using:
- **Accuracy**: Overall correctness of predictions
- **Precision**: Correctness of phishing predictions
- **Recall**: Ability to detect actual phishing attacks
- **F1-Score**: Balance between precision and recall
- **ROC-AUC**: Area under the ROC curve

### Sample Results

| Model Type | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------------|----------|-----------|--------|----------|---------|
| URL Only   | 0.92     | 0.91      | 0.93   | 0.92     | 0.96    |
| Visual Only| 0.88     | 0.87      | 0.89   | 0.88     | 0.93    |
| Hybrid     | 0.95     | 0.94      | 0.96   | 0.95     | 0.98    |

## 📁 Project Structure

```
project/
├── config/
│   └── config.yaml           # Configuration file
├── data/
│   ├── raw/                   # Raw datasets
│   ├── processed/             # Processed data
│   └── screenshots/           # Captured screenshots
├── models/                    # Trained models
├── src/
│   ├── __init__.py
│   ├── url_analysis/          # URL analysis module
│   │   ├── __init__.py
│   │   ├── feature_extractor.py
│   │   └── url_model.py
│   ├── visual_analysis/       # Visual analysis module
│   │   ├── __init__.py
│   │   ├── screenshot_capture.py
│   │   └── visual_model.py
│   ├── llm_explainer/         # LLM explanation module
│   │   ├── __init__.py
│   │   └── explainer.py
│   ├── hybrid_detector/       # Hybrid detection module
│   │   ├── __init__.py
│   │   └── detector.py
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       ├── data_loader.py
│       └── metrics.py
├── api/
│   ├── __init__.py
│   └── app.py                 # FastAPI application
├── scripts/
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   └── demo.py                # Demo script
├── tests/
│   ├── __init__.py
│   ├── test_url_analysis.py
│   ├── test_visual_analysis.py
│   └── test_hybrid_detector.py
├── requirements.txt           # Dependencies
├── .env.example               # Environment template
├── .gitignore
└── README.md                  # This file
```

## ⚙️ Configuration

### config/config.yaml

```yaml
# URL Analysis Settings
url_analysis:
  model_type: xgboost       # Options: random_forest, xgboost, lightgbm
  model_params:
    n_estimators: 100
    max_depth: 10

# Visual Analysis Settings
visual_analysis:
  model_type: resnet50      # Options: resnet50, vit_base, efficientnet
  image_size: 224
  pretrained: true

# LLM Explainability Settings
llm_explainer:
  provider: openai          # Options: openai, huggingface, mock
  model: gpt-4
  temperature: 0.3

# Hybrid Detector Settings
hybrid_detector:
  fusion_method: weighted_average
  weights:
    url_analysis: 0.4
    visual_analysis: 0.4
    llm_reasoning: 0.2
  threshold: 0.5
```

### Environment Variables

```bash
# OpenAI API (for LLM explanations)
OPENAI_API_KEY=sk-...

# HuggingFace (optional)
HUGGINGFACE_TOKEN=hf_...

# API Settings
API_SECRET_KEY=your_secret
API_DEBUG=true
```

## 🧪 Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_url_analysis.py -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 References

### Research Papers

1. **CLASP**: Cost-Optimized LLM-based Agentic System for Phishing Detection
   - [arXiv:2510.18585](https://arxiv.org/abs/2510.18585)

2. **PhishDebate**: Multi-Agent LLM Framework for Phishing Detection
   - [arXiv:2506.15656](https://arxiv.org/abs/2506.15656)

3. **Hybrid Spear Phishing Detection using LLM and Machine Learning**
   - [ResearchGate Publication](https://www.researchgate.net/publication/398759739)

4. **Detection of Phishing Website Using Machine Learning and Features Extraction**
   - [ResearchGate Publication](https://www.researchgate.net/publication/388220390)

5. **Phishing Email Detection Using Large Language Models**
   - [arXiv:2512.10104v2](https://arxiv.org/html/2512.10104v2)

6. **AI Powered Image Analysis for Phishing Detection**
   - [arXiv:2604.13555](https://arxiv.org/pdf/2604.13555)

### Datasets

- [PhishTank](https://www.phishtank.com/) - Phishing URL database
- [Alexa Top Sites](https://www.alexa.com/topsites) - Legitimate URLs
- [Kaggle Phishing Datasets](https://www.kaggle.com/datasets) - Various phishing datasets

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-4 API
- HuggingFace for Transformers library
- PyTorch team for deep learning framework
- FastAPI team for the web framework
- All researchers who contributed to the referenced papers

---

**Built with ❤️ for a safer internet**
