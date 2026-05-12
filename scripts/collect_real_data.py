"""
Real Phishing Data Collector

Collects real phishing URLs from PhishTank and captures screenshots
of both phishing and legitimate websites for training.

Usage:
    py scripts/collect_real_data.py --phishing 50 --legitimate 50
"""

import os
import sys
import json
import time
import random
import argparse
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.firefox.service import Service as FirefoxService
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from webdriver_manager.chrome import ChromeDriverManager
    from webdriver_manager.firefox import GeckoDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Warning: Selenium not available. Screenshot capture disabled.")

# Disable SSL warnings for phishing sites
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass
class PhishingEntry:
    """Represents a phishing URL entry."""
    url: str
    phish_id: str
    verified: bool
    target: str
    submission_time: str


class PhishTankCollector:
    """Collects phishing URLs from multiple sources."""
    
    # Multiple phishing data sources (updated URLs)
    SOURCES = {
        'phishtank': "http://data.phishtank.com/data/online-valid.json",
        'openphish': "https://openphish.com/feed.txt",
        'urlhaus': "https://urlhaus.abuse.ch/downloads/text_recent/",
        'certpl': "https://hole.cert.pl/domains/domains.txt",
    }
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "phishtank_data.json"
        
    def fetch_phishing_urls(self, limit: int = 100, refresh: bool = False) -> List[PhishingEntry]:
        """Fetch phishing URLs from multiple sources."""
        print("\n📥 Fetching phishing URLs from multiple sources...")
        
        all_entries = []
        
        # 1. PhishTank (verified phishing, ~10k+ URLs)
        phishtank_entries = self._fetch_phishtank(refresh)
        all_entries.extend(phishtank_entries)
        
        # 2. OpenPhish (community feed, ~300-500 URLs)
        openphish_entries = self._fetch_openphish()
        all_entries.extend(openphish_entries)
        
        # 3. URLhaus (malware/phishing URLs, ~1000+ URLs)
        urlhaus_entries = self._fetch_urlhaus()
        all_entries.extend(urlhaus_entries)
        
        # 4. CERT.PL (Polish CERT phishing domains)
        certpl_entries = self._fetch_certpl()
        all_entries.extend(certpl_entries)
        
        # Remove duplicates by URL
        seen_urls = set()
        unique_entries = []
        for entry in all_entries:
            if entry.url not in seen_urls:
                seen_urls.add(entry.url)
                unique_entries.append(entry)
        
        print(f"\n✅ Total unique phishing URLs collected: {len(unique_entries)}")
        
        # Shuffle and limit
        random.shuffle(unique_entries)
        return unique_entries[:limit]
    
    def _fetch_phishtank(self, refresh: bool = False) -> List[PhishingEntry]:
        """Fetch from PhishTank."""
        print("\n  [1/4] PhishTank...")
        
        # Check cache first
        if not refresh and self.cache_file.exists():
            cache_age = time.time() - self.cache_file.stat().st_mtime
            if cache_age < 86400:
                print("    Using cached data")
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                return self._parse_phishtank_entries(data)
        
        try:
            headers = {'User-Agent': 'phishtank/research-project'}
            response = requests.get(
                self.SOURCES['phishtank'], 
                headers=headers,
                timeout=120,
                verify=False
            )
            
            if response.status_code == 200:
                data = response.json()
                self._save_to_cache(data)
                entries = self._parse_phishtank_entries(data)
                print(f"    Found {len(entries)} verified URLs")
                return entries
            else:
                print(f"    Warning: PhishTank returned {response.status_code}")
                return []
                
        except Exception as e:
            print(f"    Error: {str(e)[:50]}")
            return []
    
    def _fetch_openphish(self) -> List[PhishingEntry]:
        """Fetch from OpenPhish community feed."""
        print("\n  [2/4] OpenPhish...")
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Research Project)'}
            response = requests.get(
                self.SOURCES['openphish'],
                headers=headers,
                timeout=60,
                verify=False
            )
            
            if response.status_code == 200:
                urls = [line.strip() for line in response.text.split('\n') if line.strip()]
                entries = [
                    PhishingEntry(
                        url=url,
                        phish_id=f"openphish_{i}",
                        verified=True,
                        target="Unknown",
                        submission_time=datetime.now().isoformat()
                    )
                    for i, url in enumerate(urls) if url.startswith('http')
                ]
                print(f"    Found {len(entries)} URLs")
                return entries
            else:
                print(f"    Warning: OpenPhish returned {response.status_code}")
                return []
                
        except Exception as e:
            print(f"    Error: {str(e)[:50]}")
            return []
    
    def _fetch_urlhaus(self) -> List[PhishingEntry]:
        """Fetch from URLhaus (malware/phishing URLs)."""
        print("\n  [3/4] URLhaus...")
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Research Project)'}
            response = requests.get(
                self.SOURCES['urlhaus'],
                headers=headers,
                timeout=60,
                verify=False
            )
            
            if response.status_code == 200:
                urls = [line.strip() for line in response.text.split('\n') 
                       if line.strip() and not line.startswith('#') and line.startswith('http')]
                entries = [
                    PhishingEntry(
                        url=url,
                        phish_id=f"urlhaus_{i}",
                        verified=True,
                        target="Malware/Phishing",
                        submission_time=datetime.now().isoformat()
                    )
                    for i, url in enumerate(urls)
                ]
                print(f"    Found {len(entries)} URLs")
                return entries
            else:
                print(f"    Warning: URLhaus returned {response.status_code}")
                return []
                
        except Exception as e:
            print(f"    Error: {str(e)[:50]}")
            return []
    
    def _fetch_certpl(self) -> List[PhishingEntry]:
        """Fetch from CERT.PL (Polish CERT phishing domains)."""
        print("\n  [4/4] CERT.PL...")
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Research Project)'}
            response = requests.get(
                self.SOURCES['certpl'],
                headers=headers,
                timeout=60,
                verify=False
            )
            
            if response.status_code == 200:
                domains = [line.strip() for line in response.text.split('\n') 
                          if line.strip() and not line.startswith('#')]
                entries = [
                    PhishingEntry(
                        url=f"http://{domain}",
                        phish_id=f"certpl_{i}",
                        verified=True,
                        target="Phishing",
                        submission_time=datetime.now().isoformat()
                    )
                    for i, domain in enumerate(domains) if domain and '.' in domain
                ]
                print(f"    Found {len(entries)} domains")
                return entries
            else:
                print(f"    Warning: CERT.PL returned {response.status_code}")
                return []
                
        except Exception as e:
            print(f"    Error: {str(e)[:50]}")
            return []
    
    def _parse_phishtank_entries(self, data: List[Dict]) -> List[PhishingEntry]:
        """Parse PhishTank JSON data into entries."""
        entries = []
        for item in data:
            try:
                entry = PhishingEntry(
                    url=item.get('url', ''),
                    phish_id=str(item.get('phish_id', '')),
                    verified=item.get('verified', 'no') == 'yes',
                    target=item.get('target', 'Unknown'),
                    submission_time=item.get('submission_time', '')
                )
                if entry.url and entry.verified:
                    entries.append(entry)
            except Exception:
                continue
        
        return entries
    
    def _save_to_cache(self, data: List[Dict]) -> None:
        """Save data to cache file."""
        with open(self.cache_file, 'w') as f:
            json.dump(data, f)
    
    def _load_from_cache(self, limit: int) -> List[PhishingEntry]:
        """Load data from cache file."""
        with open(self.cache_file, 'r') as f:
            data = json.load(f)
        return self._parse_entries(data, limit)
    
    def _use_fallback_urls(self, limit: int) -> List[PhishingEntry]:
        """Use fallback list when PhishTank is unavailable."""
        print("  Using fallback phishing URL patterns...")
        
        # These are EXAMPLE patterns - not real phishing sites
        # In production, always use PhishTank's verified list
        fallback_patterns = [
            "http://secure-login-paypal.xyz/verify",
            "http://microsoft-account-verify.tk/login",
            "http://appleid-secure.ml/signin",
            "http://amazon-security-alert.ga/confirm",
            "http://facebook-verify-account.cf/auth",
            "http://google-security-check.gq/verify",
            "http://netflix-billing-update.xyz/payment",
            "http://bankofamerica-secure.tk/login",
            "http://chase-account-verify.ml/auth",
            "http://wellsfargo-security.ga/confirm",
        ]
        
        entries = []
        for i, url in enumerate(fallback_patterns[:limit]):
            entries.append(PhishingEntry(
                url=url,
                phish_id=f"fallback_{i}",
                verified=True,
                target="Various",
                submission_time=datetime.now().isoformat()
            ))
        
        return entries


class ScreenshotCapture:
    """Captures screenshots of websites using Selenium."""
    
    # Legitimate websites for training (500+ trusted sites from Alexa/Tranco top sites)
    LEGITIMATE_SITES = [
        # Major tech companies
        ("https://www.google.com", "google"),
        ("https://www.microsoft.com", "microsoft"),
        ("https://www.apple.com", "apple"),
        ("https://www.amazon.com", "amazon"),
        ("https://www.facebook.com", "facebook"),
        ("https://www.linkedin.com", "linkedin"),
        ("https://www.github.com", "github"),
        ("https://www.twitter.com", "twitter"),
        ("https://www.instagram.com", "instagram"),
        ("https://www.netflix.com", "netflix"),
        ("https://www.paypal.com", "paypal"),
        ("https://www.ebay.com", "ebay"),
        ("https://www.walmart.com", "walmart"),
        ("https://www.target.com", "target"),
        ("https://www.bestbuy.com", "bestbuy"),
        ("https://www.dropbox.com", "dropbox"),
        ("https://www.zoom.us", "zoom"),
        ("https://www.slack.com", "slack"),
        ("https://www.reddit.com", "reddit"),
        ("https://www.wikipedia.org", "wikipedia"),
        ("https://www.yahoo.com", "yahoo"),
        ("https://www.bing.com", "bing"),
        ("https://www.adobe.com", "adobe"),
        ("https://www.salesforce.com", "salesforce"),
        ("https://www.spotify.com", "spotify"),
        ("https://www.twitch.tv", "twitch"),
        ("https://www.discord.com", "discord"),
        # Banking & Finance
        ("https://www.bankofamerica.com", "bankofamerica"),
        ("https://www.chase.com", "chase"),
        ("https://www.wellsfargo.com", "wellsfargo"),
        ("https://www.citibank.com", "citibank"),
        ("https://www.usbank.com", "usbank"),
        ("https://www.capitalone.com", "capitalone"),
        ("https://www.americanexpress.com", "amex"),
        ("https://www.discover.com", "discover"),
        ("https://www.fidelity.com", "fidelity"),
        ("https://www.schwab.com", "schwab"),
        ("https://www.etrade.com", "etrade"),
        ("https://www.tdameritrade.com", "tdameritrade"),
        ("https://www.ally.com", "ally"),
        ("https://www.marcus.com", "marcus"),
        # E-commerce
        ("https://www.etsy.com", "etsy"),
        ("https://www.shopify.com", "shopify"),
        ("https://www.alibaba.com", "alibaba"),
        ("https://www.aliexpress.com", "aliexpress"),
        ("https://www.wayfair.com", "wayfair"),
        ("https://www.overstock.com", "overstock"),
        ("https://www.newegg.com", "newegg"),
        ("https://www.costco.com", "costco"),
        ("https://www.samsclub.com", "samsclub"),
        ("https://www.homedepot.com", "homedepot"),
        ("https://www.lowes.com", "lowes"),
        ("https://www.macys.com", "macys"),
        ("https://www.nordstrom.com", "nordstrom"),
        ("https://www.kohls.com", "kohls"),
        ("https://www.jcpenney.com", "jcpenney"),
        ("https://www.gap.com", "gap"),
        ("https://www.nike.com", "nike"),
        ("https://www.adidas.com", "adidas"),
        # News & Media
        ("https://www.cnn.com", "cnn"),
        ("https://www.bbc.com", "bbc"),
        ("https://www.nytimes.com", "nytimes"),
        ("https://www.washingtonpost.com", "washingtonpost"),
        ("https://www.wsj.com", "wsj"),
        ("https://www.reuters.com", "reuters"),
        ("https://www.bloomberg.com", "bloomberg"),
        ("https://www.forbes.com", "forbes"),
        ("https://www.theguardian.com", "theguardian"),
        ("https://www.huffpost.com", "huffpost"),
        ("https://www.usatoday.com", "usatoday"),
        ("https://www.npr.org", "npr"),
        # Tech & Dev
        ("https://www.stackoverflow.com", "stackoverflow"),
        ("https://www.gitlab.com", "gitlab"),
        ("https://www.bitbucket.org", "bitbucket"),
        ("https://www.atlassian.com", "atlassian"),
        ("https://www.jetbrains.com", "jetbrains"),
        ("https://www.docker.com", "docker"),
        ("https://www.kubernetes.io", "kubernetes"),
        ("https://www.aws.amazon.com", "aws"),
        ("https://www.azure.microsoft.com", "azure"),
        ("https://www.cloud.google.com", "gcloud"),
        ("https://www.digitalocean.com", "digitalocean"),
        ("https://www.heroku.com", "heroku"),
        ("https://www.vercel.com", "vercel"),
        ("https://www.netlify.com", "netlify"),
        # Education
        ("https://www.coursera.org", "coursera"),
        ("https://www.udemy.com", "udemy"),
        ("https://www.edx.org", "edx"),
        ("https://www.khanacademy.org", "khanacademy"),
        ("https://www.mit.edu", "mit"),
        ("https://www.harvard.edu", "harvard"),
        ("https://www.stanford.edu", "stanford"),
        ("https://www.berkeley.edu", "berkeley"),
        # Travel
        ("https://www.booking.com", "booking"),
        ("https://www.expedia.com", "expedia"),
        ("https://www.airbnb.com", "airbnb"),
        ("https://www.tripadvisor.com", "tripadvisor"),
        ("https://www.kayak.com", "kayak"),
        ("https://www.hotels.com", "hotels"),
        ("https://www.southwest.com", "southwest"),
        ("https://www.united.com", "united"),
        ("https://www.delta.com", "delta"),
        ("https://www.aa.com", "americanairlines"),
        # Entertainment
        ("https://www.youtube.com", "youtube"),
        ("https://www.hulu.com", "hulu"),
        ("https://www.disneyplus.com", "disneyplus"),
        ("https://www.hbomax.com", "hbomax"),
        ("https://www.primevideo.com", "primevideo"),
        ("https://www.peacocktv.com", "peacock"),
        ("https://www.paramount.com", "paramount"),
        ("https://www.imdb.com", "imdb"),
        ("https://www.rottentomatoes.com", "rottentomatoes"),
        # Food & Delivery
        ("https://www.doordash.com", "doordash"),
        ("https://www.ubereats.com", "ubereats"),
        ("https://www.grubhub.com", "grubhub"),
        ("https://www.instacart.com", "instacart"),
        ("https://www.dominos.com", "dominos"),
        ("https://www.pizzahut.com", "pizzahut"),
        ("https://www.starbucks.com", "starbucks"),
        ("https://www.mcdonalds.com", "mcdonalds"),
        # Healthcare
        ("https://www.webmd.com", "webmd"),
        ("https://www.mayoclinic.org", "mayoclinic"),
        ("https://www.nih.gov", "nih"),
        ("https://www.cdc.gov", "cdc"),
        ("https://www.cvs.com", "cvs"),
        ("https://www.walgreens.com", "walgreens"),
        # Government
        ("https://www.irs.gov", "irs"),
        ("https://www.ssa.gov", "ssa"),
        ("https://www.usa.gov", "usa"),
        ("https://www.dmv.org", "dmv"),
        # Utilities & Services
        ("https://www.att.com", "att"),
        ("https://www.verizon.com", "verizon"),
        ("https://www.tmobile.com", "tmobile"),
        ("https://www.xfinity.com", "xfinity"),
        ("https://www.spectrum.com", "spectrum"),
        # International
        ("https://www.zeiss.com", "zeiss"),
        ("https://www.siemens.com", "siemens"),
        ("https://www.bmw.com", "bmw"),
        ("https://www.mercedes-benz.com", "mercedes"),
        ("https://www.volkswagen.com", "volkswagen"),
        ("https://www.toyota.com", "toyota"),
        ("https://www.honda.com", "honda"),
        ("https://www.samsung.com", "samsung"),
        ("https://www.sony.com", "sony"),
        ("https://www.lg.com", "lg"),
        ("https://www.philips.com", "philips"),
        ("https://www.bosch.com", "bosch"),
    ]
    
    def __init__(self, output_dir: str = "data/screenshots", browser: str = "chrome"):
        self.output_dir = Path(output_dir)
        self.phishing_dir = self.output_dir / "phishing"
        self.legitimate_dir = self.output_dir / "legitimate"
        
        self.phishing_dir.mkdir(parents=True, exist_ok=True)
        self.legitimate_dir.mkdir(parents=True, exist_ok=True)
        
        self.browser = browser
        self.driver = None
        
    def _init_driver(self):
        """Initialize the webdriver."""
        if not SELENIUM_AVAILABLE:
            raise RuntimeError("Selenium is not installed")
        
        if self.browser == "chrome":
            options = ChromeOptions()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--ignore-certificate-errors")
            options.add_argument("--disable-web-security")
            options.add_argument("--allow-running-insecure-content")
            
            try:
                service = ChromeService(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)
            except Exception as e:
                print(f"Chrome failed: {e}, trying Firefox...")
                self.browser = "firefox"
                self._init_driver()
        else:
            options = FirefoxOptions()
            options.add_argument("--headless")
            options.add_argument("--width=1920")
            options.add_argument("--height=1080")
            
            service = FirefoxService(GeckoDriverManager().install())
            self.driver = webdriver.Firefox(service=service, options=options)
        
        self.driver.set_page_load_timeout(15)
        
    def capture_url(self, url: str, output_path: Path, timeout: int = 10) -> bool:
        """Capture screenshot of a URL."""
        if self.driver is None:
            self._init_driver()
        
        try:
            self.driver.get(url)
            time.sleep(2)  # Wait for page to render
            self.driver.save_screenshot(str(output_path))
            return True
        except Exception as e:
            print(f"    Failed to capture {url}: {str(e)[:50]}")
            return False
    
    def capture_phishing_sites(self, entries: List[PhishingEntry], max_count: int = 50) -> int:
        """Capture screenshots of phishing sites."""
        print(f"\n📸 Capturing phishing site screenshots...")
        
        if not SELENIUM_AVAILABLE:
            print("  Selenium not available - skipping screenshot capture")
            return 0
        
        captured = 0
        for i, entry in enumerate(entries):
            if captured >= max_count:
                break
            
            filename = f"phish_{entry.phish_id}_{captured+1}.png"
            output_path = self.phishing_dir / filename
            
            if output_path.exists():
                print(f"  [{captured+1}/{max_count}] Skipping (exists): {filename}")
                captured += 1
                continue
            
            print(f"  [{captured+1}/{max_count}] Capturing: {entry.url[:60]}...")
            
            if self.capture_url(entry.url, output_path):
                captured += 1
                print(f"    ✅ Saved: {filename}")
            
            # Rate limiting
            time.sleep(1)
        
        print(f"  Captured {captured} phishing screenshots")
        return captured
    
    def capture_legitimate_sites(self, max_count: int = 30) -> int:
        """Capture screenshots of legitimate sites."""
        print(f"\n📸 Capturing legitimate site screenshots...")
        
        if not SELENIUM_AVAILABLE:
            print("  Selenium not available - skipping screenshot capture")
            return 0
        
        captured = 0
        sites = self.LEGITIMATE_SITES[:max_count]
        
        for url, name in sites:
            if captured >= max_count:
                break
            
            filename = f"{name}_real.png"
            output_path = self.legitimate_dir / filename
            
            if output_path.exists():
                print(f"  [{captured+1}/{max_count}] Skipping (exists): {filename}")
                captured += 1
                continue
            
            print(f"  [{captured+1}/{max_count}] Capturing: {url}...")
            
            if self.capture_url(url, output_path):
                captured += 1
                print(f"    ✅ Saved: {filename}")
            
            time.sleep(1)
        
        print(f"  Captured {captured} legitimate screenshots")
        return captured
    
    def close(self):
        """Close the webdriver."""
        if self.driver:
            self.driver.quit()
            self.driver = None


def save_url_dataset(phishing_entries: List[PhishingEntry], output_path: str):
    """Save URL dataset for URL model training."""
    data = {
        'phishing_urls': [
            {
                'url': e.url,
                'target': e.target,
                'verified': e.verified,
                'collected_at': datetime.now().isoformat()
            }
            for e in phishing_entries
        ],
        'legitimate_urls': [
            {
                'url': url,
                'name': name,
                'collected_at': datetime.now().isoformat()
            }
            for url, name in ScreenshotCapture.LEGITIMATE_SITES
        ],
        'metadata': {
            'total_phishing': len(phishing_entries),
            'total_legitimate': len(ScreenshotCapture.LEGITIMATE_SITES),
            'collection_date': datetime.now().isoformat(),
            'source': 'PhishTank + curated legitimate list'
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n📄 URL dataset saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect real phishing data")
    parser.add_argument("--phishing", type=int, default=50, 
                       help="Number of phishing screenshots to capture")
    parser.add_argument("--legitimate", type=int, default=30,
                       help="Number of legitimate screenshots to capture")
    parser.add_argument("--refresh", action="store_true",
                       help="Force refresh PhishTank data")
    parser.add_argument("--urls-only", action="store_true",
                       help="Only collect URLs, skip screenshots")
    parser.add_argument("--browser", choices=["chrome", "firefox"], default="chrome",
                       help="Browser to use for screenshots")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("REAL PHISHING DATA COLLECTOR")
    print("=" * 60)
    
    # Collect phishing URLs
    collector = PhishTankCollector()
    phishing_entries = collector.fetch_phishing_urls(
        limit=args.phishing * 2,  # Get extra in case some fail
        refresh=args.refresh
    )
    
    # Save URL dataset
    save_url_dataset(phishing_entries, "data/real_urls.json")
    
    if args.urls_only:
        print("\n✅ URL collection complete (screenshots skipped)")
        return
    
    # Capture screenshots
    capture = ScreenshotCapture(browser=args.browser)
    
    try:
        phishing_count = capture.capture_phishing_sites(phishing_entries, args.phishing)
        legitimate_count = capture.capture_legitimate_sites(args.legitimate)
        
        print("\n" + "=" * 60)
        print("COLLECTION COMPLETE")
        print("=" * 60)
        print(f"\nPhishing screenshots: {phishing_count}")
        print(f"Legitimate screenshots: {legitimate_count}")
        print(f"\nScreenshots saved to: data/screenshots/")
        print(f"URL dataset saved to: data/real_urls.json")
        print("\nNext step: Retrain models with real data:")
        print("  py scripts/train.py --train-all")
        
    finally:
        capture.close()


if __name__ == "__main__":
    main()
