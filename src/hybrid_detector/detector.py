"""
Hybrid Phishing Detection System

Combines URL analysis, visual analysis, and LLM-based reasoning
to provide comprehensive phishing detection with explainable results.
"""

import os
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import logging
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from ..url_analysis.feature_extractor import URLFeatureExtractor
from ..url_analysis.url_model import URLPhishingClassifier
from ..visual_analysis.screenshot_capture import ScreenshotCapture, ScreenshotResult
from ..visual_analysis.visual_model import VisualPhishingClassifier
from ..llm_explainer.explainer import LLMExplainer, ExplanationResult


# Whitelist of known legitimate domains
TRUSTED_DOMAINS = {
    # Tech giants
    'google.com', 'microsoft.com', 'apple.com', 'amazon.com', 'facebook.com',
    'meta.com', 'twitter.com', 'x.com', 'linkedin.com', 'github.com',
    'gitlab.com', 'bitbucket.org', 'stackoverflow.com',
    
    # Cloud providers
    'aws.amazon.com', 'azure.microsoft.com', 'cloud.google.com',
    
    # Banking & Finance
    'paypal.com', 'stripe.com', 'chase.com', 'bankofamerica.com',
    'wellsfargo.com', 'citibank.com', 'capitalone.com',
    
    # E-commerce
    'ebay.com', 'walmart.com', 'target.com', 'bestbuy.com', 'etsy.com',
    
    # Social Media
    'instagram.com', 'tiktok.com', 'snapchat.com', 'pinterest.com',
    'reddit.com', 'discord.com', 'twitch.tv', 'youtube.com',
    
    # Email providers
    'gmail.com', 'outlook.com', 'yahoo.com', 'protonmail.com',
    
    # Enterprise
    'salesforce.com', 'slack.com', 'zoom.us', 'dropbox.com', 'box.com',
    'atlassian.com', 'jira.com', 'confluence.com', 'notion.so',
    
    # Media & Entertainment
    'netflix.com', 'spotify.com', 'hulu.com', 'disneyplus.com',
    'hbomax.com', 'primevideo.com',
    
    # Others
    'wikipedia.org', 'adobe.com', 'oracle.com', 'ibm.com', 'intel.com',
    'nvidia.com', 'amd.com', 'dell.com', 'hp.com', 'lenovo.com',
    'zeiss.com', 'samsung.com', 'sony.com', 'lg.com', 'panasonic.com',
}


class FusionMethod(Enum):
    """Methods for combining multi-modal predictions."""
    WEIGHTED_AVERAGE = "weighted_average"
    VOTING = "voting"
    STACKING = "stacking"
    MAX_CONFIDENCE = "max_confidence"
    ATTENTION = "attention"


@dataclass
class DetectionResult:
    """Comprehensive phishing detection result."""
    url: str
    is_phishing: bool
    confidence: float
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    
    # Component scores
    url_score: float = 0.0
    visual_score: float = 0.0
    combined_score: float = 0.0
    
    # Detailed information
    url_features: Dict[str, Any] = field(default_factory=dict)
    visual_features: Dict[str, Any] = field(default_factory=dict)
    screenshot_path: Optional[str] = None
    
    # Explanation
    explanation: Optional[ExplanationResult] = None
    
    # Metadata
    analysis_time: float = 0.0
    modules_used: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = asdict(self)
        if self.explanation:
            result['explanation'] = {
                'summary': self.explanation.summary,
                'detailed_explanation': self.explanation.detailed_explanation,
                'indicators': [asdict(ind) for ind in self.explanation.indicators],
                'recommendations': self.explanation.recommendations,
                'risk_score': self.explanation.risk_score
            }
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class HybridPhishingDetector:
    """
    Hybrid Multi-Modal Phishing Detection System.
    
    Combines three analysis modules:
    1. URL Analysis: Extracts and analyzes URL features using ML models
    2. Visual Analysis: Analyzes webpage screenshots using deep learning
    3. LLM Explainability: Generates human-readable explanations
    
    The system uses configurable fusion methods to combine predictions
    from multiple modalities for improved accuracy.
    """
    
    def __init__(
        self,
        url_model_path: Optional[str] = None,
        visual_model_path: Optional[str] = None,
        fusion_method: str = "weighted_average",
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5,
        llm_provider: str = "mock",
        llm_api_key: Optional[str] = None,
        enable_screenshot: bool = True,
        screenshot_config: Optional[Dict[str, Any]] = None,
        device: str = "auto"
    ):
        """
        Initialize the Hybrid Phishing Detector.
        
        Args:
            url_model_path: Path to pretrained URL model
            visual_model_path: Path to pretrained visual model
            fusion_method: Method for combining predictions
            weights: Weights for each modality in fusion
            threshold: Classification threshold
            llm_provider: LLM provider for explanations
            llm_api_key: API key for LLM provider
            enable_screenshot: Whether to capture screenshots
            screenshot_config: Configuration for screenshot capture
            device: Device for visual model ('auto', 'cuda', 'cpu')
        """
        self.fusion_method = FusionMethod(fusion_method)
        self.weights = weights or {
            'url': 0.8,  # Higher weight for URL (trained on real data)
            'visual': 0.2,  # Lower weight for visual (trained on synthetic data)
            'llm': 0.0
        }
        self.threshold = threshold
        self.enable_screenshot = enable_screenshot
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize URL Analysis Module
        self.url_classifier = URLPhishingClassifier(model_type='xgboost')
        if url_model_path and Path(url_model_path).exists():
            self.url_classifier.load(url_model_path)
            self.logger.info(f"Loaded URL model from {url_model_path}")
        
        # Initialize Visual Analysis Module
        self.visual_classifier = VisualPhishingClassifier(
            model_type='resnet50',
            device=device if device != 'auto' else None
        )
        if visual_model_path and Path(visual_model_path).exists():
            self.visual_classifier.load(visual_model_path)
            self.logger.info(f"Loaded visual model from {visual_model_path}")
        
        # Initialize Screenshot Capture
        screenshot_config = screenshot_config or {}
        self.screenshot_capturer = ScreenshotCapture(
            headless=screenshot_config.get('headless', True),
            width=screenshot_config.get('width', 1920),
            height=screenshot_config.get('height', 1080),
            timeout=screenshot_config.get('timeout', 30)
        )
        
        # Initialize LLM Explainer
        self.llm_explainer = LLMExplainer(
            provider=llm_provider,
            api_key=llm_api_key
        )
        
        # Feature extractor for URL analysis
        self.url_feature_extractor = URLFeatureExtractor()
    
    def _is_trusted_domain(self, url: str) -> bool:
        """
        Check if URL belongs to a trusted/whitelisted domain.
        
        Args:
            url: URL to check
            
        Returns:
            True if domain is trusted, False otherwise
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Check exact match
            if domain in TRUSTED_DOMAINS:
                return True
            
            # Check if it's a subdomain of a trusted domain
            for trusted in TRUSTED_DOMAINS:
                if domain.endswith('.' + trusted):
                    return True
                    
            return False
        except Exception:
            return False
    
    def analyze(
        self,
        url: str,
        capture_screenshot: Optional[bool] = None,
        generate_explanation: bool = True,
        screenshot_image: Optional[Union[str, Image.Image, np.ndarray]] = None
    ) -> DetectionResult:
        """
        Perform comprehensive phishing analysis on a URL.
        
        Args:
            url: URL to analyze
            capture_screenshot: Override for screenshot capture setting
            generate_explanation: Whether to generate LLM explanation
            screenshot_image: Pre-captured screenshot image (optional)
            
        Returns:
            DetectionResult with comprehensive analysis
        """
        import time
        start_time = time.time()
        
        modules_used = []
        
        # Ensure URL has scheme
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        # Check if domain is whitelisted - return safe result immediately
        if self._is_trusted_domain(url):
            url_features = self.url_feature_extractor.extract_features(url).to_dict()
            return DetectionResult(
                url=url,
                is_phishing=False,
                confidence=0.99,
                risk_level="low",
                url_score=0.01,
                visual_score=0.0,
                combined_score=0.01,
                url_features=url_features,
                visual_features={},
                screenshot_path=None,
                explanation=None,
                analysis_time=time.time() - start_time,
                modules_used=['whitelist']
            )
        
        # 1. URL Analysis
        url_score, url_features = self._analyze_url(url)
        modules_used.append('url_analysis')
        
        # 2. Visual Analysis
        visual_score = 0.0
        visual_features = {}
        screenshot_path = None
        
        should_capture = capture_screenshot if capture_screenshot is not None else self.enable_screenshot
        
        if screenshot_image is not None:
            # Use provided screenshot
            visual_score, visual_features = self._analyze_visual(screenshot_image)
            modules_used.append('visual_analysis')
        elif should_capture and self.visual_classifier.is_fitted:
            # Capture and analyze screenshot
            screenshot_result = self.screenshot_capturer.capture(url)
            if screenshot_result.success and screenshot_result.image:
                visual_score, visual_features = self._analyze_visual(screenshot_result.image)
                screenshot_path = screenshot_result.screenshot_path
                visual_features.update(screenshot_result.metadata or {})
                modules_used.append('visual_analysis')
        
        # 3. Combine Scores
        combined_score = self._fuse_predictions(url_score, visual_score)
        
        # 4. Determine classification
        is_phishing = combined_score >= self.threshold
        classification = "phishing" if is_phishing else "legitimate"
        confidence = combined_score if is_phishing else (1 - combined_score)
        risk_level = self._calculate_risk_level(combined_score)
        
        # 5. Generate Explanation
        explanation = None
        if generate_explanation:
            try:
                explanation = self.llm_explainer.explain(
                    url=url,
                    url_score=url_score,
                    url_features=url_features,
                    visual_score=visual_score,
                    visual_features=visual_features,
                    classification=classification,
                    confidence=confidence
                )
                modules_used.append('llm_explainer')
            except Exception as e:
                self.logger.warning(f"Failed to generate explanation: {e}")
        
        analysis_time = time.time() - start_time
        
        return DetectionResult(
            url=url,
            is_phishing=is_phishing,
            confidence=confidence,
            risk_level=risk_level,
            url_score=url_score,
            visual_score=visual_score,
            combined_score=combined_score,
            url_features=url_features,
            visual_features=visual_features,
            screenshot_path=screenshot_path,
            explanation=explanation,
            analysis_time=analysis_time,
            modules_used=modules_used
        )
    
    def _analyze_url(self, url: str) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze URL and return score and features.
        
        Args:
            url: URL to analyze
            
        Returns:
            Tuple of (phishing_score, features_dict)
        """
        # Extract features
        features = self.url_feature_extractor.extract_features(url)
        features_dict = features.to_dict()
        
        # Get prediction if model is fitted
        if self.url_classifier.is_fitted:
            proba = self.url_classifier.predict_proba(url)
            score = proba[0][1]  # Probability of phishing
        else:
            # Heuristic score based on features
            score = self._heuristic_url_score(features_dict)
        
        # Apply rule-based detection for patterns the model might miss
        rule_boost = self._detect_suspicious_patterns(url)
        if rule_boost > 0:
            # Boost score but cap at 0.99
            score = min(0.99, score + rule_boost * (1 - score))
            self.logger.info(f"Rule-based boost applied: +{rule_boost:.2f} -> score={score:.2f}")
        
        return score, features_dict
    
    def _detect_suspicious_patterns(self, url: str) -> float:
        """
        Detect suspicious patterns that ML model might miss.
        
        Returns a boost value (0.0 to 1.0) to add to phishing score.
        """
        import re
        import tldextract
        
        boost = 0.0
        
        try:
            extracted = tldextract.extract(url)
            domain = extracted.domain.lower()
            subdomain = extracted.subdomain.lower() if extracted.subdomain else ""
            suffix = extracted.suffix.lower()
            full_domain = f"{subdomain}.{domain}" if subdomain else domain
            
            # 1. Suspicious TLDs
            suspicious_tlds = {
                'xyz', 'top', 'work', 'click', 'link', 'gq', 'ml', 'cf', 'tk', 'ga',
                'shop', 'store', 'online', 'site', 'website', 'space', 'fun', 'icu',
                'buzz', 'rest', 'fit', 'life', 'live', 'pw', 'cc', 'ws', 'su', 'info',
                'biz', 'mobi', 'pro', 'lat', 'cam', 'cyou', 'cfd', 'sbs'
            }
            if suffix in suspicious_tlds:
                boost += 0.3
                self.logger.debug(f"Suspicious TLD detected: .{suffix}")
            
            # 2. Brand names in domain/subdomain that aren't the brand's actual domain
            brand_patterns = {
                'allegro': 'allegro.pl',
                'paypal': 'paypal.com',
                'amazon': 'amazon.',
                'microsoft': 'microsoft.com',
                'apple': 'apple.com',
                'google': 'google.',
                'facebook': 'facebook.com',
                'netflix': 'netflix.com',
                'instagram': 'instagram.com',
                'linkedin': 'linkedin.com',
                'twitter': 'twitter.com',
                'ebay': 'ebay.',
                'dhl': 'dhl.',
                'fedex': 'fedex.com',
                'ups': 'ups.com',
                'chase': 'chase.com',
                'wellsfargo': 'wellsfargo.com',
                'bankofamerica': 'bankofamerica.com',
                'citibank': 'citi.',
                'olx': 'olx.',
                'inpost': 'inpost.',
                'whatsapp': 'whatsapp.com',
                'telegram': 'telegram.org',
                'spotify': 'spotify.com',
                'steam': 'steampowered.com',
                'roblox': 'roblox.com',
            }
            
            for brand, legit_domain in brand_patterns.items():
                if brand in full_domain and legit_domain not in url.lower():
                    boost += 0.5
                    self.logger.debug(f"Brand impersonation detected: {brand}")
                    break
            
            # 3. Random-looking domain patterns (many digits mixed with letters)
            # Pattern like "smar094235823012-y34" is highly suspicious
            digit_count = sum(c.isdigit() for c in domain)
            if digit_count >= 5:
                boost += 0.3
                self.logger.debug(f"Random digit pattern in domain: {digit_count} digits")
            
            # 4. Long hyphenated domains are suspicious
            if domain.count('-') >= 2 or (subdomain and subdomain.count('-') >= 2):
                boost += 0.2
                self.logger.debug("Multiple hyphens in domain/subdomain")
            
            # 5. Very long subdomain (often used for phishing)
            if len(subdomain) > 30:
                boost += 0.2
                self.logger.debug(f"Very long subdomain: {len(subdomain)} chars")
            
            # 6. Random hex-like patterns in domain
            if re.search(r'[a-f0-9]{8,}', full_domain, re.I):
                boost += 0.2
                self.logger.debug("Hex-like pattern in domain")
            
        except Exception as e:
            self.logger.debug(f"Pattern detection error: {e}")
        
        return min(boost, 0.9)  # Cap total boost
        
        return score, features_dict
    
    def _heuristic_url_score(self, features: Dict[str, Any]) -> float:
        """
        Calculate heuristic URL score when model is not fitted.
        
        Args:
            features: URL features dictionary
            
        Returns:
            Phishing probability score
        """
        score = 0.0
        
        # IP address in URL
        if features.get('has_ip_address'):
            score += 0.25
        
        # No HTTPS
        if not features.get('has_https'):
            score += 0.15
        
        # Sensitive words
        sensitive_count = features.get('num_sensitive_words', 0)
        score += min(sensitive_count * 0.1, 0.3)
        
        # High entropy
        if features.get('entropy', 0) > 4.0:
            score += 0.1
        
        # Suspicious params
        if features.get('has_suspicious_params'):
            score += 0.1
        
        # Long URL
        if features.get('url_length', 0) > 100:
            score += 0.05
        
        # At symbol
        if features.get('has_at_symbol'):
            score += 0.15
        
        return min(score, 1.0)
    
    def _analyze_visual(
        self,
        image: Union[str, Image.Image, np.ndarray]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze screenshot and return score and features.
        
        Args:
            image: Screenshot image
            
        Returns:
            Tuple of (phishing_score, features_dict)
        """
        if not self.visual_classifier.is_fitted:
            return 0.5, {}  # Neutral score if not fitted
        
        proba = self.visual_classifier.predict_proba(image)
        score = proba[0][1]  # Probability of phishing
        
        # Extract visual features
        features = self.visual_classifier.predict_with_features(image)
        features_dict = {
            'visual_confidence': features.confidence,
            'predicted_class': features.predicted_class
        }
        
        return score, features_dict
    
    def _fuse_predictions(
        self,
        url_score: float,
        visual_score: float
    ) -> float:
        """
        Combine predictions from multiple modalities.
        
        Args:
            url_score: Score from URL analysis
            visual_score: Score from visual analysis
            
        Returns:
            Combined phishing score
        """
        if self.fusion_method == FusionMethod.WEIGHTED_AVERAGE:
            # Weighted average
            url_weight = self.weights.get('url', 0.5)
            visual_weight = self.weights.get('visual', 0.5)
            
            # Normalize weights if visual is not available
            if visual_score == 0.0:
                return url_score
            
            total_weight = url_weight + visual_weight
            combined = (url_score * url_weight + visual_score * visual_weight) / total_weight
            return combined
        
        elif self.fusion_method == FusionMethod.MAX_CONFIDENCE:
            # Use the score with higher confidence (further from 0.5)
            url_confidence = abs(url_score - 0.5)
            visual_confidence = abs(visual_score - 0.5)
            
            if url_confidence > visual_confidence:
                return url_score
            return visual_score
        
        elif self.fusion_method == FusionMethod.VOTING:
            # Majority voting with threshold
            url_vote = 1 if url_score >= 0.5 else 0
            visual_vote = 1 if visual_score >= 0.5 else 0
            
            votes = url_vote + visual_vote
            return votes / 2.0
        
        else:
            # Default to simple average
            return (url_score + visual_score) / 2
    
    def _calculate_risk_level(self, score: float) -> str:
        """
        Calculate risk level from combined score.
        
        Args:
            score: Combined phishing score
            
        Returns:
            Risk level string
        """
        if score >= 0.85:
            return "critical"
        elif score >= 0.65:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def analyze_batch(
        self,
        urls: List[str],
        capture_screenshots: bool = False,
        generate_explanations: bool = False,
        progress_callback: Optional[callable] = None
    ) -> List[DetectionResult]:
        """
        Analyze multiple URLs.
        
        Args:
            urls: List of URLs to analyze
            capture_screenshots: Whether to capture screenshots
            generate_explanations: Whether to generate explanations
            progress_callback: Callback for progress updates
            
        Returns:
            List of DetectionResult objects
        """
        results = []
        total = len(urls)
        
        for i, url in enumerate(urls):
            result = self.analyze(
                url=url,
                capture_screenshot=capture_screenshots,
                generate_explanation=generate_explanations
            )
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total, url, result.is_phishing)
        
        return results
    
    def quick_check(self, url: str) -> Tuple[bool, float, str]:
        """
        Perform quick URL-only analysis.
        
        Args:
            url: URL to check
            
        Returns:
            Tuple of (is_phishing, confidence, risk_level)
        """
        # Check if domain is whitelisted
        if self._is_trusted_domain(url):
            return False, 0.99, "low"
        
        url_score, _ = self._analyze_url(url)
        is_phishing = url_score >= self.threshold
        confidence = url_score if is_phishing else (1 - url_score)
        risk_level = self._calculate_risk_level(url_score)
        
        return is_phishing, confidence, risk_level
    
    def train_url_model(
        self,
        urls: List[str],
        labels: List[int],
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the URL classification model.
        
        Args:
            urls: List of training URLs
            labels: List of labels (0=legitimate, 1=phishing)
            validation_split: Validation split ratio
            
        Returns:
            Training metrics
        """
        return self.url_classifier.fit(urls, labels, validation_split)
    
    def train_visual_model(
        self,
        image_paths: List[str],
        labels: List[int],
        val_images: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        Train the visual classification model.
        
        Args:
            image_paths: List of training image paths
            labels: List of labels
            val_images: Optional validation images
            val_labels: Optional validation labels
            **kwargs: Additional training arguments
            
        Returns:
            Training history
        """
        return self.visual_classifier.fit(
            image_paths, labels,
            val_images, val_labels,
            **kwargs
        )
    
    def save_models(self, directory: Union[str, Path]) -> None:
        """
        Save all trained models.
        
        Args:
            directory: Directory to save models
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        if self.url_classifier.is_fitted:
            self.url_classifier.save(directory / "url_model.pkl")
        
        if self.visual_classifier.is_fitted:
            self.visual_classifier.save(directory / "visual_model.pt")
    
    def load_models(self, directory: Union[str, Path]) -> None:
        """
        Load all trained models.
        
        Args:
            directory: Directory containing saved models
        """
        directory = Path(directory)
        
        url_model_path = directory / "url_model.pkl"
        if url_model_path.exists():
            self.url_classifier.load(url_model_path)
        
        visual_model_path = directory / "visual_model.pt"
        if visual_model_path.exists():
            self.visual_classifier.load(visual_model_path)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics and status."""
        return {
            'url_model_fitted': self.url_classifier.is_fitted,
            'visual_model_fitted': self.visual_classifier.is_fitted,
            'fusion_method': self.fusion_method.value,
            'weights': self.weights,
            'threshold': self.threshold,
            'screenshot_enabled': self.enable_screenshot
        }
    
    def close(self) -> None:
        """Release resources."""
        self.screenshot_capturer.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


if __name__ == "__main__":
    # Example usage
    print("Hybrid Phishing Detection System")
    print("=" * 50)
    
    # Initialize detector with mock LLM
    detector = HybridPhishingDetector(
        llm_provider="mock",
        enable_screenshot=False  # Disable for testing
    )
    
    print(f"\nSystem Statistics:")
    stats = detector.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test URLs
    test_urls = [
        "https://www.google.com",
        "http://192.168.1.1/login.php",
        "http://secure-paypal-login.xyz/verify?token=abc123",
        "https://www.microsoft.com/account"
    ]
    
    print("\nAnalyzing URLs:")
    print("-" * 50)
    
    for url in test_urls:
        is_phishing, confidence, risk_level = detector.quick_check(url)
        status = "🚫 PHISHING" if is_phishing else "✅ LEGITIMATE"
        print(f"\n{url}")
        print(f"  Status: {status}")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Risk Level: {risk_level.upper()}")
    
    # Full analysis example
    print("\n" + "=" * 50)
    print("Full Analysis Example:")
    print("=" * 50)
    
    result = detector.analyze(
        "http://secure-paypal-login.xyz/verify?token=abc123",
        capture_screenshot=False,
        generate_explanation=True
    )
    
    print(f"\nURL: {result.url}")
    print(f"Classification: {'PHISHING' if result.is_phishing else 'LEGITIMATE'}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Risk Level: {result.risk_level.upper()}")
    print(f"URL Score: {result.url_score:.2f}")
    print(f"Analysis Time: {result.analysis_time:.2f}s")
    print(f"Modules Used: {', '.join(result.modules_used)}")
    
    if result.explanation:
        print(f"\nExplanation Summary:")
        print(f"  {result.explanation.summary}")
