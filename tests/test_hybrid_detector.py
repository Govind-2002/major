"""
Unit Tests for Hybrid Phishing Detector

Tests the complete hybrid detection system including
URL analysis, visual analysis, and LLM explanations.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hybrid_detector.detector import (
    HybridPhishingDetector,
    DetectionResult,
    FusionMethod
)


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""
    
    def test_detection_result_creation(self):
        """Test DetectionResult creation."""
        result = DetectionResult(
            url="https://example.com",
            is_phishing=False,
            confidence=0.95,
            risk_level="low"
        )
        
        assert result.url == "https://example.com"
        assert not result.is_phishing
        assert result.confidence == 0.95
        assert result.risk_level == "low"
    
    def test_to_dict(self):
        """Test DetectionResult to dictionary conversion."""
        result = DetectionResult(
            url="https://example.com",
            is_phishing=True,
            confidence=0.85,
            risk_level="high"
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['url'] == "https://example.com"
        assert result_dict['is_phishing'] is True
    
    def test_to_json(self):
        """Test DetectionResult to JSON conversion."""
        result = DetectionResult(
            url="https://example.com",
            is_phishing=False,
            confidence=0.9,
            risk_level="low"
        )
        
        json_str = result.to_json()
        
        assert isinstance(json_str, str)
        assert "example.com" in json_str


class TestFusionMethod:
    """Tests for FusionMethod enum."""
    
    def test_fusion_methods_exist(self):
        """Test that all fusion methods are defined."""
        assert FusionMethod.WEIGHTED_AVERAGE is not None
        assert FusionMethod.VOTING is not None
        assert FusionMethod.MAX_CONFIDENCE is not None
    
    def test_fusion_method_values(self):
        """Test fusion method string values."""
        assert FusionMethod.WEIGHTED_AVERAGE.value == "weighted_average"
        assert FusionMethod.VOTING.value == "voting"


class TestHybridPhishingDetector:
    """Tests for HybridPhishingDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance with mock LLM."""
        return HybridPhishingDetector(
            llm_provider="mock",
            enable_screenshot=False
        )
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = HybridPhishingDetector(
            llm_provider="mock",
            enable_screenshot=False
        )
        
        assert detector is not None
        assert detector.threshold == 0.5
    
    def test_custom_weights(self):
        """Test custom fusion weights."""
        weights = {'url': 0.6, 'visual': 0.3, 'llm': 0.1}
        detector = HybridPhishingDetector(
            llm_provider="mock",
            weights=weights
        )
        
        assert detector.weights == weights
    
    def test_custom_threshold(self):
        """Test custom classification threshold."""
        detector = HybridPhishingDetector(
            llm_provider="mock",
            threshold=0.7
        )
        
        assert detector.threshold == 0.7
    
    def test_analyze_legitimate_url(self, detector):
        """Test analysis of legitimate URL."""
        result = detector.analyze(
            url="https://www.google.com",
            capture_screenshot=False,
            generate_explanation=False
        )
        
        assert isinstance(result, DetectionResult)
        assert result.url == "https://www.google.com"
        assert result.risk_level in ['low', 'medium', 'high', 'critical']
    
    def test_analyze_phishing_url(self, detector):
        """Test analysis of phishing-like URL."""
        result = detector.analyze(
            url="http://secure-paypal-login.xyz/verify?token=abc",
            capture_screenshot=False,
            generate_explanation=False
        )
        
        assert isinstance(result, DetectionResult)
        # Phishing-like URL should have higher risk
        assert result.url_score > 0.3
    
    def test_quick_check(self, detector):
        """Test quick URL check."""
        is_phishing, confidence, risk_level = detector.quick_check(
            "https://www.microsoft.com"
        )
        
        assert isinstance(is_phishing, bool)
        assert 0 <= confidence <= 1
        assert risk_level in ['low', 'medium', 'high', 'critical']
    
    def test_analyze_with_explanation(self, detector):
        """Test analysis with explanation generation."""
        result = detector.analyze(
            url="http://malicious-site.tk/login",
            capture_screenshot=False,
            generate_explanation=True
        )
        
        assert result.explanation is not None
    
    def test_batch_analyze(self, detector):
        """Test batch URL analysis."""
        urls = [
            "https://www.google.com",
            "https://www.github.com",
            "http://phishing-site.xyz"
        ]
        
        results = detector.analyze_batch(
            urls=urls,
            capture_screenshots=False,
            generate_explanations=False
        )
        
        assert len(results) == 3
        assert all(isinstance(r, DetectionResult) for r in results)
    
    def test_url_without_scheme(self, detector):
        """Test analysis of URL without scheme."""
        result = detector.analyze(
            url="www.example.com",
            capture_screenshot=False
        )
        
        assert "http" in result.url
    
    def test_heuristic_url_score(self, detector):
        """Test heuristic URL scoring."""
        features = {
            'has_ip_address': True,
            'has_https': False,
            'num_sensitive_words': 2,
            'entropy': 4.5,
            'has_suspicious_params': True
        }
        
        score = detector._heuristic_url_score(features)
        
        assert 0 <= score <= 1
        assert score > 0.3  # Should be relatively high
    
    def test_risk_level_calculation(self, detector):
        """Test risk level calculation."""
        assert detector._calculate_risk_level(0.9) == "critical"
        assert detector._calculate_risk_level(0.7) == "high"
        assert detector._calculate_risk_level(0.5) == "medium"
        assert detector._calculate_risk_level(0.2) == "low"
    
    def test_fuse_predictions_weighted_average(self, detector):
        """Test weighted average fusion."""
        detector.fusion_method = FusionMethod.WEIGHTED_AVERAGE
        
        combined = detector._fuse_predictions(0.8, 0.6)
        
        assert 0 <= combined <= 1
    
    def test_fuse_predictions_max_confidence(self, detector):
        """Test max confidence fusion."""
        detector.fusion_method = FusionMethod.MAX_CONFIDENCE
        
        # URL score is further from 0.5
        combined = detector._fuse_predictions(0.9, 0.6)
        assert combined == 0.9
        
        # Visual score is further from 0.5
        combined = detector._fuse_predictions(0.55, 0.2)
        assert combined == 0.2
    
    def test_fuse_predictions_voting(self, detector):
        """Test voting fusion."""
        detector.fusion_method = FusionMethod.VOTING
        
        # Both predict phishing
        combined = detector._fuse_predictions(0.8, 0.7)
        assert combined == 1.0
        
        # Both predict legitimate
        combined = detector._fuse_predictions(0.3, 0.2)
        assert combined == 0.0
        
        # Disagreement
        combined = detector._fuse_predictions(0.8, 0.3)
        assert combined == 0.5
    
    def test_get_statistics(self, detector):
        """Test getting system statistics."""
        stats = detector.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'url_model_fitted' in stats
        assert 'fusion_method' in stats
        assert 'threshold' in stats
    
    def test_context_manager(self):
        """Test detector as context manager."""
        with HybridPhishingDetector(llm_provider="mock") as detector:
            result = detector.quick_check("https://example.com")
            assert result is not None


class TestHybridDetectorModules:
    """Tests for individual detector modules."""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return HybridPhishingDetector(
            llm_provider="mock",
            enable_screenshot=False
        )
    
    def test_url_analysis_module(self, detector):
        """Test URL analysis module."""
        score, features = detector._analyze_url("https://example.com")
        
        assert 0 <= score <= 1
        assert isinstance(features, dict)
        assert 'url_length' in features
    
    def test_url_analysis_ip_address(self, detector):
        """Test URL analysis with IP address."""
        score, features = detector._analyze_url("http://192.168.1.1/login")
        
        assert features['has_ip_address'] == 1
        assert score > 0.2  # Should increase score


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_full_analysis_flow(self):
        """Test complete analysis flow."""
        detector = HybridPhishingDetector(
            llm_provider="mock",
            enable_screenshot=False
        )
        
        # Analyze multiple URLs
        test_cases = [
            ("https://www.google.com", False),  # Expected legitimate
            ("http://192.168.1.1/admin", True),  # Expected suspicious
            ("http://paypal-secure-login.tk/verify", True),  # Expected phishing
        ]
        
        for url, expected_suspicious in test_cases:
            result = detector.analyze(url, capture_screenshot=False)
            
            assert result.url == url
            assert isinstance(result.confidence, float)
            
            # High-risk URLs should have higher scores
            if expected_suspicious:
                assert result.url_score > 0.2
        
        detector.close()
    
    def test_performance_batch_processing(self):
        """Test performance with batch processing."""
        import time
        
        detector = HybridPhishingDetector(
            llm_provider="mock",
            enable_screenshot=False
        )
        
        urls = [f"https://example{i}.com/page" for i in range(20)]
        
        start = time.time()
        results = detector.analyze_batch(urls, capture_screenshots=False, generate_explanations=False)
        elapsed = time.time() - start
        
        assert len(results) == 20
        assert elapsed < 30  # Should complete in reasonable time
        
        detector.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
