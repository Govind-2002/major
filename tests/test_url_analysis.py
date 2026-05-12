"""
Unit Tests for URL Analysis Module

Tests URL feature extraction and classification functionality.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.url_analysis.feature_extractor import URLFeatureExtractor, URLFeatures
from src.url_analysis.url_model import URLPhishingClassifier


class TestURLFeatureExtractor:
    """Tests for URLFeatureExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return URLFeatureExtractor()
    
    def test_basic_url_extraction(self, extractor):
        """Test basic URL feature extraction."""
        url = "https://www.example.com/page"
        features = extractor.extract_features(url)
        
        assert isinstance(features, URLFeatures)
        assert features.url_length > 0
        assert features.has_https == 1
        assert features.num_dots >= 2
    
    def test_ip_address_detection(self, extractor):
        """Test IP address detection in URL."""
        ip_url = "http://192.168.1.1/login"
        features = extractor.extract_features(ip_url)
        
        assert features.has_ip_address == 1
    
    def test_no_ip_address(self, extractor):
        """Test URL without IP address."""
        url = "https://www.google.com"
        features = extractor.extract_features(url)
        
        assert features.has_ip_address == 0
    
    def test_https_detection(self, extractor):
        """Test HTTPS detection."""
        https_url = "https://secure.example.com"
        http_url = "http://insecure.example.com"
        
        https_features = extractor.extract_features(https_url)
        http_features = extractor.extract_features(http_url)
        
        assert https_features.has_https == 1
        assert http_features.has_https == 0
    
    def test_sensitive_words_detection(self, extractor):
        """Test detection of sensitive words."""
        phishing_url = "http://secure-login-paypal-verify.com"
        features = extractor.extract_features(phishing_url)
        
        assert features.num_sensitive_words > 0
    
    def test_url_without_scheme(self, extractor):
        """Test URL without http/https scheme."""
        url = "www.example.com/page"
        features = extractor.extract_features(url)
        
        assert features.url_length > 0
    
    def test_empty_url(self, extractor):
        """Test empty URL handling."""
        features = extractor.extract_features("")
        
        assert features.url_length == 0
    
    def test_special_characters_count(self, extractor):
        """Test counting of special characters."""
        url = "https://example.com/path?query=value&other=123"
        features = extractor.extract_features(url)
        
        assert features.num_question_marks == 1
        assert features.num_ampersands == 1
        assert features.num_equals == 2
    
    def test_subdomain_counting(self, extractor):
        """Test subdomain counting."""
        url = "https://sub1.sub2.example.com"
        features = extractor.extract_features(url)
        
        assert features.num_subdomains >= 1
    
    def test_entropy_calculation(self, extractor):
        """Test entropy calculation."""
        simple_url = "https://aaa.com"
        complex_url = "https://abc123xyz789def456.com"
        
        simple_features = extractor.extract_features(simple_url)
        complex_features = extractor.extract_features(complex_url)
        
        assert complex_features.entropy > simple_features.entropy
    
    def test_hex_encoding_detection(self, extractor):
        """Test hex encoding detection."""
        encoded_url = "http://example.com/%2F%2F"
        features = extractor.extract_features(encoded_url)
        
        assert features.has_hex_encoding == 1
    
    def test_at_symbol_detection(self, extractor):
        """Test @ symbol detection."""
        at_url = "http://user@example.com/page"
        features = extractor.extract_features(at_url)
        
        assert features.has_at_symbol == 1
        assert features.num_at_symbols == 1
    
    def test_batch_extraction(self, extractor):
        """Test batch feature extraction."""
        urls = [
            "https://google.com",
            "http://facebook.com",
            "https://github.com"
        ]
        
        features_list = extractor.extract_batch(urls)
        
        assert len(features_list) == 3
        assert all(isinstance(f, URLFeatures) for f in features_list)
    
    def test_to_feature_matrix(self, extractor):
        """Test conversion to feature matrix."""
        urls = ["https://example.com", "http://test.org"]
        
        matrix = extractor.to_feature_matrix(urls)
        
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape[0] == 2
    
    def test_features_to_dict(self, extractor):
        """Test features to dictionary conversion."""
        features = extractor.extract_features("https://example.com")
        features_dict = features.to_dict()
        
        assert isinstance(features_dict, dict)
        assert 'url_length' in features_dict
        assert 'has_https' in features_dict
    
    def test_features_to_array(self, extractor):
        """Test features to numpy array conversion."""
        features = extractor.extract_features("https://example.com")
        array = features.to_array()
        
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float32
    
    def test_feature_names(self):
        """Test getting feature names."""
        names = URLFeatureExtractor.get_feature_names()
        
        assert isinstance(names, list)
        assert len(names) > 0
        assert 'url_length' in names


class TestURLPhishingClassifier:
    """Tests for URLPhishingClassifier class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        legitimate_urls = [
            "https://www.google.com/search",
            "https://www.facebook.com/home",
            "https://www.amazon.com/products",
            "https://www.microsoft.com/windows",
            "https://github.com/features"
        ] * 10
        
        phishing_urls = [
            "http://192.168.1.1/login.php",
            "http://secure-paypal.xyz/verify",
            "http://google-login.tk/auth",
            "http://facebook.com.malicious.ml/signin",
            "http://amazon-deals.click/offer"
        ] * 10
        
        urls = legitimate_urls + phishing_urls
        labels = [0] * len(legitimate_urls) + [1] * len(phishing_urls)
        
        return urls, labels
    
    def test_classifier_initialization(self):
        """Test classifier initialization."""
        classifier = URLPhishingClassifier(model_type='xgboost')
        
        assert classifier.model_type == 'xgboost'
        assert not classifier.is_fitted
    
    def test_supported_models(self):
        """Test that all supported models can be initialized."""
        for model_type in URLPhishingClassifier.SUPPORTED_MODELS:
            classifier = URLPhishingClassifier(model_type=model_type)
            assert classifier.model_type == model_type
    
    def test_invalid_model_type(self):
        """Test initialization with invalid model type."""
        with pytest.raises(ValueError):
            URLPhishingClassifier(model_type='invalid_model')
    
    def test_fit_model(self, sample_data):
        """Test model training."""
        urls, labels = sample_data
        classifier = URLPhishingClassifier(model_type='random_forest')
        
        metrics = classifier.fit(urls, labels)
        
        assert classifier.is_fitted
        assert 'train_accuracy' in metrics
        assert 'val_accuracy' in metrics
    
    def test_predict(self, sample_data):
        """Test prediction."""
        urls, labels = sample_data
        classifier = URLPhishingClassifier(model_type='random_forest')
        classifier.fit(urls, labels)
        
        predictions = classifier.predict(["https://www.google.com"])
        
        assert len(predictions) == 1
        assert predictions[0] in [0, 1]
    
    def test_predict_proba(self, sample_data):
        """Test probability prediction."""
        urls, labels = sample_data
        classifier = URLPhishingClassifier(model_type='random_forest')
        classifier.fit(urls, labels)
        
        probas = classifier.predict_proba(["https://www.google.com"])
        
        assert probas.shape == (1, 2)
        assert np.allclose(probas.sum(axis=1), 1.0)
    
    def test_predict_single_url(self, sample_data):
        """Test prediction with single URL string."""
        urls, labels = sample_data
        classifier = URLPhishingClassifier()
        classifier.fit(urls, labels)
        
        prediction = classifier.predict("https://example.com")
        
        assert len(prediction) == 1
    
    def test_predict_without_fitting(self):
        """Test prediction without fitting raises error."""
        classifier = URLPhishingClassifier()
        
        with pytest.raises(ValueError):
            classifier.predict(["https://example.com"])
    
    def test_predict_with_features(self, sample_data):
        """Test prediction with feature information."""
        urls, labels = sample_data
        classifier = URLPhishingClassifier()
        classifier.fit(urls, labels)
        
        pred, conf, features = classifier.predict_with_features("https://example.com")
        
        assert pred in [0, 1]
        assert 0 <= conf <= 1
        assert isinstance(features, dict)
    
    def test_evaluate(self, sample_data):
        """Test model evaluation."""
        urls, labels = sample_data
        classifier = URLPhishingClassifier()
        classifier.fit(urls[:80], labels[:80])
        
        metrics = classifier.evaluate(urls[80:], labels[80:])
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
    
    def test_feature_importance(self, sample_data):
        """Test feature importance retrieval."""
        urls, labels = sample_data
        classifier = URLPhishingClassifier()
        classifier.fit(urls, labels)
        
        importance = classifier.get_feature_importance(5)
        
        assert len(importance) == 5
        assert all(isinstance(item, tuple) for item in importance)
    
    def test_save_and_load(self, sample_data, tmp_path):
        """Test model save and load."""
        urls, labels = sample_data
        classifier = URLPhishingClassifier()
        classifier.fit(urls, labels)
        
        # Save
        model_path = tmp_path / "test_model.pkl"
        classifier.save(model_path)
        
        assert model_path.exists()
        
        # Load
        loaded_classifier = URLPhishingClassifier()
        loaded_classifier.load(model_path)
        
        assert loaded_classifier.is_fitted
        
        # Compare predictions
        original_pred = classifier.predict(["https://example.com"])
        loaded_pred = loaded_classifier.predict(["https://example.com"])
        
        assert original_pred[0] == loaded_pred[0]
    
    def test_from_pretrained(self, sample_data, tmp_path):
        """Test loading from pretrained model."""
        urls, labels = sample_data
        classifier = URLPhishingClassifier()
        classifier.fit(urls, labels)
        
        model_path = tmp_path / "test_model.pkl"
        classifier.save(model_path)
        
        loaded = URLPhishingClassifier.from_pretrained(model_path)
        
        assert loaded.is_fitted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
