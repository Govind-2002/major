"""
URL Feature Extractor Module

Extracts various features from URLs for phishing detection including:
- Structural features (length, special characters, etc.)
- Domain-based features (TLD, subdomain properties)
- Lexical features (entropy, digit ratio, etc.)
- Security indicators (HTTPS, IP address presence)
"""

import re
import math
import socket
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse, parse_qs
from dataclasses import dataclass, asdict

import tldextract
import numpy as np

# Configure tldextract to use bundled snapshot (no network needed)
_tld_extractor = tldextract.TLDExtract(suffix_list_urls=None)


@dataclass
class URLFeatures:
    """Data class containing all extracted URL features."""
    
    # Basic URL properties
    url_length: int = 0
    domain_length: int = 0
    subdomain_length: int = 0
    path_length: int = 0
    query_length: int = 0
    fragment_length: int = 0
    
    # Character counts
    num_dots: int = 0
    num_hyphens: int = 0
    num_underscores: int = 0
    num_slashes: int = 0
    num_at_symbols: int = 0
    num_question_marks: int = 0
    num_ampersands: int = 0
    num_equals: int = 0
    num_digits: int = 0
    num_special_chars: int = 0
    
    # Ratios
    digit_ratio: float = 0.0
    letter_ratio: float = 0.0
    special_char_ratio: float = 0.0
    
    # Security indicators
    has_https: int = 0
    has_ip_address: int = 0
    has_port: int = 0
    has_at_symbol: int = 0
    
    # Domain properties
    domain_in_path: int = 0
    domain_in_subdomain: int = 0
    tld_length: int = 0
    num_subdomains: int = 0
    
    # Lexical features
    entropy: float = 0.0
    longest_word_length: int = 0
    avg_word_length: float = 0.0
    num_words: int = 0
    
    # Suspicious patterns
    has_double_slash_redirect: int = 0
    has_hex_encoding: int = 0
    has_punycode: int = 0
    num_sensitive_words: int = 0
    
    # Query parameters
    num_params: int = 0
    has_suspicious_params: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert features to dictionary."""
        return asdict(self)
    
    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for ML models."""
        return np.array(list(asdict(self).values()), dtype=np.float32)


class URLFeatureExtractor:
    """
    Extracts comprehensive features from URLs for phishing detection.
    
    Features include structural, lexical, domain-based, and security indicators
    that are commonly used to distinguish phishing URLs from legitimate ones.
    """
    
    # Common legitimate TLDs
    COMMON_TLDS = {'com', 'org', 'net', 'edu', 'gov', 'co', 'io', 'app', 'dev'}
    
    # Suspicious TLDs often used in phishing
    SUSPICIOUS_TLDS = {
        'xyz', 'top', 'work', 'click', 'link', 'gq', 'ml', 'cf', 'tk', 'ga',
        'shop', 'store', 'online', 'site', 'website', 'space', 'fun', 'icu',
        'buzz', 'rest', 'fit', 'life', 'live', 'pw', 'cc', 'ws', 'su', 'info',
        'biz', 'mobi', 'pro', 'asia', 'lat', 'cam', 'cyou', 'cfd', 'sbs'
    }
    
    # Sensitive/phishing-related words
    SENSITIVE_WORDS = {
        # Login/Auth terms
        'login', 'signin', 'sign-in', 'logon', 'log-in', 'signon', 'sign-on',
        'authenticate', 'authentication', 'credential', 'credentials', 'password',
        # Account terms
        'bank', 'banking', 'secure', 'account', 'myaccount', 'my-account',
        'update', 'verify', 'verification', 'confirm', 'confirmation',
        'suspend', 'suspended', 'restrict', 'restricted', 'unlock', 'validation',
        'wallet', 'billing', 'payment', 'invoice', 'transaction', 'refund',
        # Major US brands
        'paypal', 'apple', 'microsoft', 'google', 'amazon', 'netflix', 'facebook',
        'instagram', 'twitter', 'linkedin', 'dropbox', 'icloud', 'yahoo', 'ebay',
        'chase', 'wellsfargo', 'bankofamerica', 'citibank', 'usbank', 'capitalone',
        # European brands
        'allegro', 'olx', 'inpost', 'blik', 'mbank', 'ing', 'santander', 'pko',
        'ceneo', 'empik', 'mediamarkt', 'saturn', 'zalando', 'aboutyou',
        # Global brands
        'dhl', 'fedex', 'ups', 'usps', 'royalmail', 'laposte', 'dpd', 'gls',
        'whatsapp', 'telegram', 'signal', 'viber', 'wechat', 'tiktok',
        'spotify', 'steam', 'origin', 'epicgames', 'blizzard', 'roblox',
        # Urgency terms
        'support', 'security', 'alert', 'warning', 'urgent', 'immediate',
        'required', 'action', 'expire', 'expired', 'expiring', 'limited',
        'prize', 'winner', 'won', 'reward', 'gift', 'bonus', 'free', 'claim'
    }
    
    # Suspicious query parameters
    SUSPICIOUS_PARAMS = {
        'token', 'session', 'redirect', 'return', 'callback', 'next',
        'continue', 'dest', 'destination', 'url', 'link', 'goto'
    }
    
    def __init__(self):
        """Initialize the URL Feature Extractor."""
        self._ip_pattern = re.compile(
            r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})|'
            r'(0x[0-9a-fA-F]{1,8})'
        )
        self._hex_pattern = re.compile(r'%[0-9a-fA-F]{2}')
        self._special_chars_pattern = re.compile(r'[^a-zA-Z0-9]')
    
    def extract_features(self, url: str) -> URLFeatures:
        """
        Extract all features from a URL.
        
        Args:
            url: The URL string to analyze
            
        Returns:
            URLFeatures object containing all extracted features
        """
        features = URLFeatures()
        
        if not url:
            return features
        
        # Ensure URL has a scheme
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        try:
            parsed = urlparse(url)
            extracted = _tld_extractor(url)
        except Exception:
            return features
        
        # Extract all feature categories
        self._extract_basic_features(url, parsed, features)
        self._extract_character_counts(url, features)
        self._extract_ratios(url, features)
        self._extract_security_indicators(url, parsed, features)
        self._extract_domain_features(url, parsed, extracted, features)
        self._extract_lexical_features(url, features)
        self._extract_suspicious_patterns(url, parsed, features)
        self._extract_query_features(parsed, features)
        
        return features
    
    def _extract_basic_features(self, url: str, parsed, features: URLFeatures) -> None:
        """Extract basic URL structure features."""
        features.url_length = len(url)
        features.domain_length = len(parsed.netloc) if parsed.netloc else 0
        features.path_length = len(parsed.path) if parsed.path else 0
        features.query_length = len(parsed.query) if parsed.query else 0
        features.fragment_length = len(parsed.fragment) if parsed.fragment else 0
    
    def _extract_character_counts(self, url: str, features: URLFeatures) -> None:
        """Count occurrences of various characters."""
        features.num_dots = url.count('.')
        features.num_hyphens = url.count('-')
        features.num_underscores = url.count('_')
        features.num_slashes = url.count('/')
        features.num_at_symbols = url.count('@')
        features.num_question_marks = url.count('?')
        features.num_ampersands = url.count('&')
        features.num_equals = url.count('=')
        features.num_digits = sum(c.isdigit() for c in url)
        features.num_special_chars = len(self._special_chars_pattern.findall(url))
    
    def _extract_ratios(self, url: str, features: URLFeatures) -> None:
        """Calculate character ratios."""
        url_len = len(url) if url else 1
        features.digit_ratio = features.num_digits / url_len
        features.letter_ratio = sum(c.isalpha() for c in url) / url_len
        features.special_char_ratio = features.num_special_chars / url_len
    
    def _extract_security_indicators(self, url: str, parsed, features: URLFeatures) -> None:
        """Extract security-related indicators."""
        features.has_https = 1 if parsed.scheme == 'https' else 0
        features.has_ip_address = 1 if self._ip_pattern.search(parsed.netloc or '') else 0
        features.has_port = 1 if ':' in (parsed.netloc or '') and not parsed.netloc.startswith('[') else 0
        features.has_at_symbol = 1 if '@' in url else 0
    
    def _extract_domain_features(self, url: str, parsed, extracted, features: URLFeatures) -> None:
        """Extract domain-related features."""
        domain = extracted.domain
        subdomain = extracted.subdomain
        suffix = extracted.suffix
        
        features.subdomain_length = len(subdomain) if subdomain else 0
        features.tld_length = len(suffix) if suffix else 0
        features.num_subdomains = subdomain.count('.') + 1 if subdomain else 0
        
        # Check if domain name appears in path (possible phishing indicator)
        path = parsed.path.lower() if parsed.path else ''
        features.domain_in_path = 1 if any(
            brand in path for brand in ['google', 'facebook', 'apple', 'microsoft', 'amazon', 'paypal']
        ) else 0
        
        # Check if popular brand appears in subdomain
        features.domain_in_subdomain = 1 if subdomain and any(
            brand in subdomain.lower() for brand in ['google', 'facebook', 'apple', 'microsoft', 'amazon', 'paypal']
        ) else 0
    
    def _extract_lexical_features(self, url: str, features: URLFeatures) -> None:
        """Extract lexical features like entropy and word statistics."""
        # Calculate Shannon entropy
        features.entropy = self._calculate_entropy(url)
        
        # Extract words and calculate statistics
        words = re.findall(r'[a-zA-Z]+', url)
        if words:
            features.longest_word_length = max(len(w) for w in words)
            features.avg_word_length = sum(len(w) for w in words) / len(words)
            features.num_words = len(words)
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of a string."""
        if not text:
            return 0.0
        
        # Count character frequencies
        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1
        
        # Calculate entropy
        length = len(text)
        entropy = 0.0
        for count in freq.values():
            probability = count / length
            entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _extract_suspicious_patterns(self, url: str, parsed, features: URLFeatures) -> None:
        """Detect suspicious patterns commonly used in phishing URLs."""
        # Double slash in path (redirect trick)
        path = parsed.path or ''
        features.has_double_slash_redirect = 1 if '//' in path else 0
        
        # Hex encoding
        features.has_hex_encoding = 1 if self._hex_pattern.search(url) else 0
        
        # Punycode (internationalized domain names)
        features.has_punycode = 1 if 'xn--' in url.lower() else 0
        
        # Count sensitive words
        url_lower = url.lower()
        features.num_sensitive_words = sum(1 for word in self.SENSITIVE_WORDS if word in url_lower)
    
    def _extract_query_features(self, parsed, features: URLFeatures) -> None:
        """Extract features from query parameters."""
        if not parsed.query:
            return
        
        try:
            params = parse_qs(parsed.query)
            features.num_params = len(params)
            
            # Check for suspicious parameter names
            param_names = set(k.lower() for k in params.keys())
            features.has_suspicious_params = 1 if param_names & self.SUSPICIOUS_PARAMS else 0
        except Exception:
            pass
    
    def extract_batch(self, urls: List[str]) -> List[URLFeatures]:
        """
        Extract features from multiple URLs.
        
        Args:
            urls: List of URL strings
            
        Returns:
            List of URLFeatures objects
        """
        return [self.extract_features(url) for url in urls]
    
    def to_feature_matrix(self, urls: List[str]) -> np.ndarray:
        """
        Convert list of URLs to a feature matrix for ML models.
        
        Args:
            urls: List of URL strings
            
        Returns:
            2D numpy array of shape (n_urls, n_features)
        """
        features_list = self.extract_batch(urls)
        return np.vstack([f.to_array() for f in features_list])
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get list of feature names in order."""
        return list(URLFeatures().__dict__.keys())


if __name__ == "__main__":
    # Example usage and testing
    extractor = URLFeatureExtractor()
    
    test_urls = [
        "https://www.google.com/search?q=test",
        "http://192.168.1.1/login.php",
        "https://secure-login-paypal.suspicious-domain.xyz/verify?token=abc123",
        "https://www.microsoft.com/en-us/account",
        "http://xn--googl-7pa.com/signin"
    ]
    
    print("URL Feature Extraction Examples")
    print("=" * 60)
    
    for url in test_urls:
        features = extractor.extract_features(url)
        print(f"\nURL: {url}")
        print(f"  Length: {features.url_length}")
        print(f"  Has HTTPS: {features.has_https}")
        print(f"  Has IP: {features.has_ip_address}")
        print(f"  Entropy: {features.entropy:.3f}")
        print(f"  Sensitive words: {features.num_sensitive_words}")
        print(f"  Suspicious params: {features.has_suspicious_params}")
