#!/usr/bin/env python
"""
Demo Script for Hybrid Phishing Detection System

Demonstrates the capabilities of the system with interactive examples.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hybrid_detector.detector import HybridPhishingDetector
from src.url_analysis.feature_extractor import URLFeatureExtractor
from src.llm_explainer.explainer import LLMExplainer


def print_banner():
    """Print the demo banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     HYBRID MULTI-MODAL PHISHING DETECTION SYSTEM              ║
    ║                                                               ║
    ║     URL Analysis + Visual Analysis + LLM Reasoning            ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def demo_url_features():
    """Demonstrate URL feature extraction."""
    print("\n" + "=" * 60)
    print("DEMO 1: URL Feature Extraction")
    print("=" * 60)
    
    extractor = URLFeatureExtractor()
    
    demo_urls = [
        ("https://www.google.com/search?q=python", "Legitimate - Google Search"),
        ("https://www.github.com/openai/gpt-4", "Legitimate - GitHub Repository"),
        ("http://192.168.1.100/login.php", "Suspicious - IP-based URL"),
        ("http://secure-paypal-login.xyz/verify?token=abc", "Phishing - Fake PayPal"),
        ("http://faceb00k.com.malicious-site.tk/login", "Phishing - Typosquatting"),
    ]
    
    for url, description in demo_urls:
        print(f"\n{'-' * 50}")
        print(f"URL: {url}")
        print(f"Description: {description}")
        
        features = extractor.extract_features(url)
        
        print(f"\nKey Features:")
        print(f"  URL Length: {features.url_length}")
        print(f"  Has HTTPS: {'Yes' if features.has_https else 'No'}")
        print(f"  Has IP Address: {'Yes' if features.has_ip_address else 'No'}")
        print(f"  Sensitive Words: {features.num_sensitive_words}")
        print(f"  Entropy: {features.entropy:.3f}")
        print(f"  Num Dots: {features.num_dots}")
        print(f"  Has Suspicious Params: {'Yes' if features.has_suspicious_params else 'No'}")


def demo_quick_check():
    """Demonstrate quick URL checking."""
    print("\n" + "=" * 60)
    print("DEMO 2: Quick URL Check")
    print("=" * 60)
    
    detector = HybridPhishingDetector(
        llm_provider="mock",
        enable_screenshot=False
    )
    
    test_urls = [
        "https://www.microsoft.com/account",
        "https://www.amazon.com/products/laptop",
        "http://secure-bank-login.tk/verify",
        "http://192.168.0.1/admin/login.php",
        "https://paypal-security.verify-account.xyz/confirm",
        "https://docs.python.org/3/tutorial/index.html",
    ]
    
    print(f"\nAnalyzing {len(test_urls)} URLs...\n")
    print(f"{'URL':<55} {'Status':<12} {'Confidence':<10} {'Risk'}")
    print("-" * 90)
    
    for url in test_urls:
        is_phishing, confidence, risk_level = detector.quick_check(url)
        status = "🚫 PHISHING" if is_phishing else "✅ SAFE"
        url_display = url[:52] + "..." if len(url) > 55 else url
        print(f"{url_display:<55} {status:<12} {confidence:.1%}       {risk_level.upper()}")
    
    detector.close()


def demo_full_analysis():
    """Demonstrate full URL analysis with explanation."""
    print("\n" + "=" * 60)
    print("DEMO 3: Full Analysis with Explanation")
    print("=" * 60)
    
    detector = HybridPhishingDetector(
        llm_provider="mock",
        enable_screenshot=False
    )
    
    test_url = "http://secure-paypal-login.xyz/verify?token=abc123&redirect=home"
    
    print(f"\nAnalyzing: {test_url}")
    print("-" * 60)
    
    result = detector.analyze(
        url=test_url,
        capture_screenshot=False,
        generate_explanation=True
    )
    
    print(f"\n📊 Analysis Results:")
    print(f"   Classification: {'🚫 PHISHING' if result.is_phishing else '✅ LEGITIMATE'}")
    print(f"   Confidence: {result.confidence:.1%}")
    print(f"   Risk Level: {result.risk_level.upper()}")
    print(f"   Analysis Time: {result.analysis_time:.3f}s")
    
    print(f"\n📈 Component Scores:")
    print(f"   URL Score: {result.url_score:.3f}")
    print(f"   Visual Score: {result.visual_score:.3f}")
    print(f"   Combined Score: {result.combined_score:.3f}")
    
    print(f"\n🔍 Key URL Features:")
    features = result.url_features
    print(f"   - URL Length: {features.get('url_length', 'N/A')}")
    print(f"   - Has HTTPS: {'No' if not features.get('has_https') else 'Yes'}")
    print(f"   - Sensitive Words: {features.get('num_sensitive_words', 0)}")
    print(f"   - Has Suspicious Params: {'Yes' if features.get('has_suspicious_params') else 'No'}")
    
    if result.explanation:
        print(f"\n💡 Explanation:")
        print(f"   {result.explanation.summary}")
        
        if result.explanation.recommendations:
            print(f"\n📋 Recommendations:")
            for i, rec in enumerate(result.explanation.recommendations, 1):
                print(f"   {i}. {rec}")
    
    detector.close()


def demo_batch_analysis():
    """Demonstrate batch URL analysis."""
    print("\n" + "=" * 60)
    print("DEMO 4: Batch Analysis")
    print("=" * 60)
    
    detector = HybridPhishingDetector(
        llm_provider="mock",
        enable_screenshot=False
    )
    
    urls = [
        "https://www.google.com",
        "https://www.facebook.com/login",
        "http://malicious-login.tk/verify",
        "https://github.com/features",
        "http://192.168.1.1/admin",
        "https://signin-apple.com.fake-domain.xyz",
        "https://www.linkedin.com/feed",
        "http://secure-banking-portal.ml/login",
        "https://www.netflix.com/browse",
        "http://amazon-deal.click/offer?id=123",
    ]
    
    print(f"\nProcessing {len(urls)} URLs...")
    
    def progress_callback(current, total, url, is_phishing):
        status = "🚫" if is_phishing else "✅"
        print(f"  [{current}/{total}] {status} {url[:50]}...")
    
    results = detector.analyze_batch(
        urls=urls,
        capture_screenshots=False,
        generate_explanations=False,
        progress_callback=progress_callback
    )
    
    # Summary
    phishing_count = sum(1 for r in results if r.is_phishing)
    legitimate_count = len(results) - phishing_count
    
    print(f"\n📊 Batch Analysis Summary:")
    print(f"   Total URLs: {len(results)}")
    print(f"   Phishing: {phishing_count} ({phishing_count/len(results)*100:.0f}%)")
    print(f"   Legitimate: {legitimate_count} ({legitimate_count/len(results)*100:.0f}%)")
    
    # Risk distribution
    risk_counts = {}
    for r in results:
        risk_counts[r.risk_level] = risk_counts.get(r.risk_level, 0) + 1
    
    print(f"\n📈 Risk Distribution:")
    for level in ['critical', 'high', 'medium', 'low']:
        count = risk_counts.get(level, 0)
        bar = '█' * count
        print(f"   {level.upper():10s}: {bar} ({count})")
    
    detector.close()


def interactive_mode():
    """Run interactive URL checking mode."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("\nEnter URLs to analyze (type 'quit' to exit)")
    
    detector = HybridPhishingDetector(
        llm_provider="mock",
        enable_screenshot=False
    )
    
    while True:
        try:
            url = input("\n🔗 Enter URL: ").strip()
            
            if url.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not url:
                continue
            
            result = detector.analyze(
                url=url,
                capture_screenshot=False,
                generate_explanation=True
            )
            
            status = "🚫 PHISHING DETECTED!" if result.is_phishing else "✅ URL appears SAFE"
            print(f"\n{status}")
            print(f"   Confidence: {result.confidence:.1%}")
            print(f"   Risk Level: {result.risk_level.upper()}")
            
            if result.explanation:
                print(f"\n   💡 {result.explanation.summary}")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error analyzing URL: {e}")
    
    detector.close()


def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Demo for Hybrid Phishing Detection System"
    )
    parser.add_argument(
        "--mode",
        choices=['all', 'features', 'quick', 'full', 'batch', 'interactive'],
        default='all',
        help="Demo mode to run"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.mode == 'interactive':
        interactive_mode()
    elif args.mode == 'features':
        demo_url_features()
    elif args.mode == 'quick':
        demo_quick_check()
    elif args.mode == 'full':
        demo_full_analysis()
    elif args.mode == 'batch':
        demo_batch_analysis()
    else:  # all
        demo_url_features()
        demo_quick_check()
        demo_full_analysis()
        demo_batch_analysis()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETE")
        print("=" * 60)
        print("\nTo try the interactive mode, run:")
        print("  python scripts/demo.py --mode interactive")


if __name__ == "__main__":
    main()
