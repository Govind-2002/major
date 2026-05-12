"""
Screenshot Capture Module

Captures screenshots of websites using Selenium WebDriver
for visual analysis in phishing detection.
"""

import os
import time
import hashlib
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from dataclasses import dataclass
import io

from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, WebDriverException, 
    InvalidSessionIdException
)
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager


@dataclass
class ScreenshotResult:
    """Result of a screenshot capture operation."""
    success: bool
    screenshot_path: Optional[str] = None
    image: Optional[Image.Image] = None
    url: str = ""
    final_url: str = ""
    page_title: str = ""
    load_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ScreenshotCapture:
    """
    Captures screenshots of websites using Selenium WebDriver.
    
    Supports both Chrome and Firefox browsers with configurable
    options for headless mode, viewport size, and timeouts.
    """
    
    def __init__(
        self,
        browser: str = 'chrome',
        headless: bool = True,
        width: int = 1920,
        height: int = 1080,
        timeout: int = 30,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the Screenshot Capture module.
        
        Args:
            browser: Browser to use ('chrome' or 'firefox')
            headless: Whether to run in headless mode
            width: Viewport width in pixels
            height: Viewport height in pixels
            timeout: Page load timeout in seconds
            output_dir: Directory to save screenshots
        """
        self.browser = browser.lower()
        self.headless = headless
        self.width = width
        self.height = height
        self.timeout = timeout
        self.output_dir = Path(output_dir) if output_dir else Path("data/screenshots")
        self.driver = None
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_chrome_driver(self) -> webdriver.Chrome:
        """Initialize Chrome WebDriver with options."""
        options = ChromeOptions()
        
        if self.headless:
            options.add_argument('--headless=new')
        
        options.add_argument(f'--window-size={self.width},{self.height}')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-infobars')
        options.add_argument('--ignore-certificate-errors')
        options.add_argument('--ignore-ssl-errors')
        options.add_argument('--disable-web-security')
        options.add_argument('--allow-running-insecure-content')
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # Disable logging
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        options.add_argument('--log-level=3')
        
        # Try to use system chromedriver first (if in PATH)
        try:
            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(self.timeout)
            return driver
        except Exception:
            pass
        
        # Fallback to webdriver_manager
        try:
            service = ChromeService(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            driver.set_page_load_timeout(self.timeout)
            return driver
        except Exception as e:
            raise WebDriverException(f"Failed to initialize Chrome: {e}")
    
    def _init_firefox_driver(self) -> webdriver.Firefox:
        """Initialize Firefox WebDriver with options."""
        options = FirefoxOptions()
        
        if self.headless:
            options.add_argument('--headless')
        
        options.add_argument(f'--width={self.width}')
        options.add_argument(f'--height={self.height}')
        
        # Set preferences
        options.set_preference('network.http.connection-timeout', self.timeout)
        options.set_preference('dom.max_script_run_time', self.timeout)
        
        # Try to use system geckodriver first (if in PATH)
        try:
            driver = webdriver.Firefox(options=options)
            driver.set_page_load_timeout(self.timeout)
            return driver
        except Exception:
            pass
        
        # Fallback to webdriver_manager
        try:
            service = FirefoxService(GeckoDriverManager().install())
            driver = webdriver.Firefox(service=service, options=options)
            driver.set_page_load_timeout(self.timeout)
            return driver
        except Exception as e:
            raise WebDriverException(f"Failed to initialize Firefox: {e}")
    
    def _init_driver(self) -> None:
        """Initialize the appropriate WebDriver."""
        if self.driver is not None:
            return
        
        if self.browser == 'chrome':
            self.driver = self._init_chrome_driver()
        elif self.browser == 'firefox':
            self.driver = self._init_firefox_driver()
        else:
            raise ValueError(f"Unsupported browser: {self.browser}")
    
    def _generate_filename(self, url: str) -> str:
        """Generate a unique filename for the screenshot based on URL hash."""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        timestamp = int(time.time())
        return f"screenshot_{url_hash}_{timestamp}.png"
    
    def capture(
        self,
        url: str,
        save: bool = True,
        filename: Optional[str] = None
    ) -> ScreenshotResult:
        """
        Capture a screenshot of a website.
        
        Args:
            url: URL to capture
            save: Whether to save the screenshot to disk
            filename: Optional custom filename for the screenshot
            
        Returns:
            ScreenshotResult object containing capture results
        """
        # Ensure URL has a scheme
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        start_time = time.time()
        
        try:
            self._init_driver()
            
            # Navigate to URL
            self.driver.get(url)
            
            # Wait for page to load
            WebDriverWait(self.driver, self.timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Small delay for dynamic content
            time.sleep(1)
            
            # Get page info
            final_url = self.driver.current_url
            page_title = self.driver.title
            
            # Capture screenshot
            screenshot_png = self.driver.get_screenshot_as_png()
            image = Image.open(io.BytesIO(screenshot_png))
            
            # Resize to standard size if needed
            if image.size != (self.width, self.height):
                image = image.resize((self.width, self.height), Image.Resampling.LANCZOS)
            
            load_time = time.time() - start_time
            
            # Save screenshot if requested
            screenshot_path = None
            if save:
                filename = filename or self._generate_filename(url)
                screenshot_path = str(self.output_dir / filename)
                image.save(screenshot_path)
            
            # Collect metadata
            metadata = self._collect_metadata()
            
            return ScreenshotResult(
                success=True,
                screenshot_path=screenshot_path,
                image=image,
                url=url,
                final_url=final_url,
                page_title=page_title,
                load_time=load_time,
                metadata=metadata
            )
            
        except TimeoutException:
            return ScreenshotResult(
                success=False,
                url=url,
                error_message=f"Page load timeout after {self.timeout} seconds"
            )
        except WebDriverException as e:
            return ScreenshotResult(
                success=False,
                url=url,
                error_message=f"WebDriver error: {str(e)}"
            )
        except Exception as e:
            return ScreenshotResult(
                success=False,
                url=url,
                error_message=f"Unexpected error: {str(e)}"
            )
    
    def _collect_metadata(self) -> Dict[str, Any]:
        """Collect additional metadata from the page."""
        metadata = {}
        
        try:
            # Count elements
            metadata['num_links'] = len(self.driver.find_elements(By.TAG_NAME, "a"))
            metadata['num_forms'] = len(self.driver.find_elements(By.TAG_NAME, "form"))
            metadata['num_inputs'] = len(self.driver.find_elements(By.TAG_NAME, "input"))
            metadata['num_images'] = len(self.driver.find_elements(By.TAG_NAME, "img"))
            metadata['num_scripts'] = len(self.driver.find_elements(By.TAG_NAME, "script"))
            
            # Check for password fields
            password_inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='password']")
            metadata['has_password_field'] = len(password_inputs) > 0
            
            # Check for login forms
            login_keywords = ['login', 'signin', 'sign-in', 'auth', 'authenticate']
            page_source_lower = self.driver.page_source.lower()
            metadata['has_login_form'] = any(kw in page_source_lower for kw in login_keywords)
            
            # Get page dimensions
            metadata['page_width'] = self.driver.execute_script("return document.body.scrollWidth")
            metadata['page_height'] = self.driver.execute_script("return document.body.scrollHeight")
            
        except Exception:
            pass
        
        return metadata
    
    def capture_batch(
        self,
        urls: list,
        save: bool = True,
        progress_callback: Optional[callable] = None
    ) -> list:
        """
        Capture screenshots of multiple URLs.
        
        Args:
            urls: List of URLs to capture
            save: Whether to save screenshots to disk
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of ScreenshotResult objects
        """
        results = []
        total = len(urls)
        
        for i, url in enumerate(urls):
            result = self.capture(url, save=save)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total, url, result.success)
        
        return results
    
    def capture_to_array(self, url: str) -> Optional[Tuple[Any, ScreenshotResult]]:
        """
        Capture screenshot and return as numpy array.
        
        Args:
            url: URL to capture
            
        Returns:
            Tuple of (numpy array, ScreenshotResult) or None if capture failed
        """
        import numpy as np
        
        result = self.capture(url, save=False)
        
        if result.success and result.image:
            # Convert to RGB if necessary
            if result.image.mode != 'RGB':
                result.image = result.image.convert('RGB')
            
            array = np.array(result.image)
            return array, result
        
        return None
    
    def close(self) -> None:
        """Close the WebDriver and release resources."""
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            finally:
                self.driver = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Destructor to ensure driver is closed."""
        self.close()


if __name__ == "__main__":
    # Example usage
    test_urls = [
        "https://www.google.com",
        "https://www.github.com",
        "https://www.microsoft.com"
    ]
    
    print("Screenshot Capture Module Test")
    print("=" * 50)
    
    with ScreenshotCapture(headless=True) as capturer:
        for url in test_urls:
            print(f"\nCapturing: {url}")
            result = capturer.capture(url)
            
            if result.success:
                print(f"  Success! Saved to: {result.screenshot_path}")
                print(f"  Final URL: {result.final_url}")
                print(f"  Page Title: {result.page_title}")
                print(f"  Load Time: {result.load_time:.2f}s")
                if result.metadata:
                    print(f"  Links: {result.metadata.get('num_links', 'N/A')}")
                    print(f"  Forms: {result.metadata.get('num_forms', 'N/A')}")
            else:
                print(f"  Failed: {result.error_message}")
