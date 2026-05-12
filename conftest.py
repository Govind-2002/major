"""
Pytest configuration and fixtures for the test suite.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent


@pytest.fixture(scope="session")
def data_dir(project_root):
    """Return the data directory."""
    return project_root / "data"


@pytest.fixture(scope="session")
def models_dir(project_root):
    """Return the models directory."""
    return project_root / "models"


@pytest.fixture
def sample_legitimate_urls():
    """Return sample legitimate URLs for testing."""
    return [
        "https://www.google.com/search?q=test",
        "https://www.github.com/features",
        "https://www.microsoft.com/en-us/windows",
        "https://www.amazon.com/products",
        "https://www.apple.com/iphone",
        "https://www.linkedin.com/feed",
        "https://www.netflix.com/browse",
        "https://docs.python.org/3/tutorial",
    ]


@pytest.fixture
def sample_phishing_urls():
    """Return sample phishing-like URLs for testing."""
    return [
        "http://192.168.1.1/login.php",
        "http://secure-paypal-login.xyz/verify",
        "http://google-signin.tk/auth",
        "http://facebook.com.malicious.ml/login",
        "http://amazon-deals.click/offer?id=123",
        "http://microsoft-support.work/help",
        "http://apple-id-verify.ga/confirm",
        "http://netflix-billing.cf/update",
    ]


@pytest.fixture
def sample_mixed_urls(sample_legitimate_urls, sample_phishing_urls):
    """Return mixed sample URLs with labels."""
    urls = sample_legitimate_urls + sample_phishing_urls
    labels = [0] * len(sample_legitimate_urls) + [1] * len(sample_phishing_urls)
    return urls, labels


@pytest.fixture
def mock_detector():
    """Create a mock detector for testing."""
    from src.hybrid_detector.detector import HybridPhishingDetector
    
    detector = HybridPhishingDetector(
        llm_provider="mock",
        enable_screenshot=False
    )
    yield detector
    detector.close()
