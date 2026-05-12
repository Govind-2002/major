"""
Hybrid Multi-Modal Phishing Detection System
Main Package Initialization
"""

__version__ = "1.0.0"
__author__ = "Phishing Detection Team"
__description__ = "A hybrid system combining URL analysis, visual features, and LLM reasoning for phishing detection"

from .hybrid_detector.detector import HybridPhishingDetector
from .url_analysis.url_model import URLPhishingClassifier
from .visual_analysis.visual_model import VisualPhishingClassifier
from .llm_explainer.explainer import LLMExplainer

__all__ = [
    "HybridPhishingDetector",
    "URLPhishingClassifier",
    "VisualPhishingClassifier",
    "LLMExplainer"
]
