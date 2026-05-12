"""
Unit Tests for Visual Analysis Module

Tests screenshot capture and visual classification functionality.
"""

import pytest
import numpy as np
from PIL import Image
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visual_analysis.visual_model import (
    VisualPhishingClassifier,
    PhishingScreenshotDataset,
    VisualFeatures
)


class TestPhishingScreenshotDataset:
    """Tests for PhishingScreenshotDataset class."""
    
    @pytest.fixture
    def sample_images(self, tmp_path):
        """Create sample images for testing."""
        image_paths = []
        labels = []
        
        for i in range(10):
            # Create dummy image
            img = Image.new('RGB', (224, 224), color=(i * 25, i * 25, i * 25))
            path = tmp_path / f"image_{i}.png"
            img.save(path)
            image_paths.append(str(path))
            labels.append(i % 2)  # Alternate labels
        
        return image_paths, labels
    
    def test_dataset_length(self, sample_images):
        """Test dataset length."""
        image_paths, labels = sample_images
        dataset = PhishingScreenshotDataset(image_paths, labels)
        
        assert len(dataset) == 10
    
    def test_dataset_getitem(self, sample_images):
        """Test dataset item retrieval."""
        image_paths, labels = sample_images
        dataset = PhishingScreenshotDataset(image_paths, labels)
        
        image, label = dataset[0]
        
        assert isinstance(image, np.ndarray) or hasattr(image, 'numpy')
        assert label in [0, 1]
    
    def test_dataset_without_labels(self, sample_images):
        """Test dataset without labels."""
        image_paths, _ = sample_images
        dataset = PhishingScreenshotDataset(image_paths)
        
        image, label = dataset[0]
        
        assert label == -1


class TestVisualPhishingClassifier:
    """Tests for VisualPhishingClassifier class."""
    
    @pytest.fixture
    def classifier(self):
        """Create classifier instance."""
        return VisualPhishingClassifier(
            model_type='resnet50',
            pretrained=False  # Use False for faster tests
        )
    
    @pytest.fixture
    def sample_image(self, tmp_path):
        """Create a sample image."""
        img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        path = tmp_path / "test_image.png"
        img.save(path)
        return str(path)
    
    @pytest.fixture
    def sample_pil_image(self):
        """Create a sample PIL image."""
        return Image.new('RGB', (224, 224), color=(128, 128, 128))
    
    @pytest.fixture
    def sample_numpy_image(self):
        """Create a sample numpy image."""
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    def test_classifier_initialization(self):
        """Test classifier initialization."""
        classifier = VisualPhishingClassifier(model_type='resnet50')
        
        assert classifier.model_type == 'resnet50'
        assert not classifier.is_fitted
    
    def test_supported_models(self):
        """Test that supported models are listed."""
        assert 'resnet50' in VisualPhishingClassifier.SUPPORTED_MODELS
        assert 'vit_base' in VisualPhishingClassifier.SUPPORTED_MODELS
    
    def test_invalid_model_type(self):
        """Test initialization with invalid model type."""
        with pytest.raises(ValueError):
            VisualPhishingClassifier(model_type='invalid_model')
    
    def test_image_size_setting(self):
        """Test image size configuration."""
        classifier = VisualPhishingClassifier(image_size=256)
        
        assert classifier.image_size == 256
    
    def test_prepare_images_from_path(self, classifier, sample_image):
        """Test image preparation from file path."""
        tensors = classifier._prepare_images(sample_image)
        
        assert tensors.shape[0] == 1
        assert tensors.shape[2] == classifier.image_size
        assert tensors.shape[3] == classifier.image_size
    
    def test_prepare_images_from_pil(self, classifier, sample_pil_image):
        """Test image preparation from PIL Image."""
        tensors = classifier._prepare_images(sample_pil_image)
        
        assert tensors.shape[0] == 1
    
    def test_prepare_images_from_numpy(self, classifier, sample_numpy_image):
        """Test image preparation from numpy array."""
        tensors = classifier._prepare_images(sample_numpy_image)
        
        assert tensors.shape[0] == 1
    
    def test_prepare_images_list(self, classifier, sample_pil_image):
        """Test image preparation from list."""
        images = [sample_pil_image, sample_pil_image]
        tensors = classifier._prepare_images(images)
        
        assert tensors.shape[0] == 2
    
    def test_predict_without_fitting(self, classifier, sample_image):
        """Test prediction without fitting raises error."""
        with pytest.raises(ValueError):
            classifier.predict(sample_image)
    
    def test_create_model(self, classifier):
        """Test model creation."""
        model = classifier._create_model(num_classes=2)
        
        assert model is not None
    
    def test_visual_features_dataclass(self):
        """Test VisualFeatures dataclass."""
        features = VisualFeatures(
            prediction=1,
            confidence=0.95,
            feature_vector=np.array([1, 2, 3]),
            predicted_class='phishing'
        )
        
        assert features.prediction == 1
        assert features.confidence == 0.95
        assert features.predicted_class == 'phishing'


class TestScreenshotCapture:
    """Tests for ScreenshotCapture class (mocked)."""
    
    def test_screenshot_result_dataclass(self):
        """Test ScreenshotResult dataclass."""
        from src.visual_analysis.screenshot_capture import ScreenshotResult
        
        result = ScreenshotResult(
            success=True,
            url="https://example.com",
            page_title="Example"
        )
        
        assert result.success
        assert result.url == "https://example.com"
    
    @patch('src.visual_analysis.screenshot_capture.webdriver')
    def test_capture_initialization(self, mock_webdriver):
        """Test ScreenshotCapture initialization."""
        from src.visual_analysis.screenshot_capture import ScreenshotCapture
        
        capturer = ScreenshotCapture(
            headless=True,
            width=1920,
            height=1080
        )
        
        assert capturer.headless
        assert capturer.width == 1920
        assert capturer.height == 1080
    
    def test_generate_filename(self):
        """Test filename generation."""
        from src.visual_analysis.screenshot_capture import ScreenshotCapture
        
        capturer = ScreenshotCapture()
        filename = capturer._generate_filename("https://example.com")
        
        assert filename.startswith("screenshot_")
        assert filename.endswith(".png")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
