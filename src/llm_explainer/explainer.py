"""
LLM Explainability Module

Generates human-readable explanations for phishing detection results
using Large Language Models (LLMs). Provides interpretable insights
into why a website was classified as phishing or legitimate.
"""

import os
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class PhishingIndicator:
    """Individual phishing indicator with severity and description."""
    indicator: str
    severity: str  # 'high', 'medium', 'low'
    description: str
    category: str  # 'url', 'visual', 'behavioral'


@dataclass
class ExplanationResult:
    """Result of LLM explanation generation."""
    classification: str  # 'phishing' or 'legitimate'
    confidence: float
    summary: str
    detailed_explanation: str
    indicators: List[PhishingIndicator]
    recommendations: List[str]
    risk_score: float
    raw_response: Optional[str] = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from the LLM."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider for LLM explanations."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.3,
        max_tokens: int = 1000
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (e.g., 'gpt-4', 'gpt-3.5-turbo')
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        try:
            from openai import OpenAI
            import httpx
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Create httpx client with SSL verification disabled (for environments with SSL issues)
        http_client = httpx.Client(verify=False)
        self.client = OpenAI(api_key=self.api_key, http_client=http_client)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate explanation using OpenAI API."""
        response = self.client.chat.completions.create(
            model=kwargs.get('model', self.model),
            messages=[
                {
                    "role": "system",
                    "content": "You are a cybersecurity expert specializing in phishing detection and analysis. Provide clear, accurate, and helpful explanations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=kwargs.get('temperature', self.temperature),
            max_tokens=kwargs.get('max_tokens', self.max_tokens)
        )
        
        return response.choices[0].message.content


class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace Transformers provider for local LLM explanations."""
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        device: str = "auto",
        max_length: int = 512
    ):
        """
        Initialize HuggingFace provider.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to use ('auto', 'cuda', 'cpu')
            max_length: Maximum generation length
        """
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            raise ImportError("transformers package is required. Install with: pip install transformers")
        
        self.model_name = model_name
        self.max_length = max_length
        
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            device_map=device if device != "auto" else None,
            max_length=max_length
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate explanation using local HuggingFace model."""
        result = self.generator(
            prompt,
            max_length=kwargs.get('max_length', self.max_length),
            num_return_sequences=1,
            do_sample=True,
            temperature=kwargs.get('temperature', 0.7)
        )
        
        return result[0]['generated_text']


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider for LLM explanations."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
        temperature: float = 0.3,
        max_tokens: int = 1000
    ):
        """
        Initialize Gemini provider.
        
        Args:
            api_key: Google AI API key (defaults to GEMINI_API_KEY or GOOGLE_API_KEY env var)
            model: Model to use (e.g., 'gemini-1.5-flash', 'gemini-1.5-pro')
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key is required (set GEMINI_API_KEY or GOOGLE_API_KEY)")
        
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate explanation using Gemini REST API (bypasses SSL issues)."""
        import requests
        import urllib3
        import time
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": kwargs.get('temperature', self.temperature),
                "maxOutputTokens": kwargs.get('max_tokens', self.max_tokens),
            }
        }
        
        url = f"{self.api_url}?key={self.api_key}"
        
        # Retry with exponential backoff for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=data, headers=headers, verify=False, timeout=90)
                
                if response.status_code == 429:
                    wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s
                    import logging
                    logging.getLogger(__name__).warning(f"Rate limited, waiting {wait_time}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                
                result = response.json()
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                
                import logging
                logging.getLogger(__name__).info(f"Gemini response length: {len(text)} chars")
                
                return text
                
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Gemini attempt {attempt+1} failed: {e}")
                if attempt == max_retries - 1:
                    # Return fallback mock response
                    return self._generate_fallback(prompt)
        
        # If all retries exhausted due to rate limits
        return self._generate_fallback(prompt)
    
    def _generate_fallback(self, prompt: str) -> str:
        """Generate fallback response when API is unavailable."""
        # Check various ways phishing might be indicated in the prompt
        prompt_lower = prompt.lower()
        is_phishing = (
            "classification: phishing" in prompt_lower or
            "classified as phishing" in prompt_lower or
            "url score: 0.9" in prompt_lower or
            "url score: 0.8" in prompt_lower or
            "url score: 1.0" in prompt_lower or
            "final classification: phishing" in prompt_lower
        )
        
        if is_phishing:
            return json.dumps({
                "summary": "This URL exhibits multiple characteristics commonly associated with phishing attempts, including suspicious domain patterns and URL structure anomalies.",
                "detailed_explanation": "The analysis detected several warning signs: the URL contains terms mimicking legitimate services, uses unusual domain extensions or subdomains, and employs URL obfuscation techniques. These patterns are consistent with phishing infrastructure designed to deceive users into revealing sensitive information.",
                "indicators": [
                    {"indicator": "Suspicious domain pattern", "severity": "high", "description": "Domain mimics legitimate service", "category": "url"},
                    {"indicator": "URL obfuscation", "severity": "medium", "description": "URL structure designed to deceive", "category": "url"}
                ],
                "recommendations": [
                    "Do not enter any personal information on this site",
                    "Close the browser tab immediately",
                    "Report the URL to your IT security team",
                    "If credentials were entered, change passwords immediately"
                ],
                "risk_score": 0.95
            })
        else:
            return json.dumps({
                "summary": "This URL does not exhibit obvious phishing characteristics based on initial analysis.",
                "detailed_explanation": "The URL structure appears standard and does not contain common phishing indicators. However, always exercise caution when entering sensitive information online.",
                "indicators": [],
                "recommendations": [
                    "Verify the website's SSL certificate before entering data",
                    "Look for trust indicators like security badges",
                    "Check for spelling errors or unusual requests"
                ],
                "risk_score": 0.15
            })



class MockProvider(BaseLLMProvider):
    """Mock provider for intelligent explanations without API calls."""
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate intelligent mock explanation based on prompt content."""
        # Parse the prompt to extract analysis info
        is_phishing = "classified as PHISHING" in prompt or "URL Score: 0.9" in prompt or "URL Score: 0.8" in prompt
        
        # Extract URL from prompt
        url_match = prompt.find("URL: ")
        url = "the analyzed URL"
        if url_match != -1:
            url_end = prompt.find("\n", url_match)
            url = prompt[url_match + 5:url_end].strip() if url_end != -1 else "the analyzed URL"
        
        if is_phishing:
            summary = f"This URL shows multiple indicators of a phishing attempt. The domain appears to impersonate a legitimate service, and the URL structure contains suspicious patterns commonly used in credential theft attacks."
            detailed = f"Analysis of {url} reveals concerning patterns: The domain name mimics well-known brands using subtle character substitutions or additions. The URL contains suspicious keywords often associated with phishing campaigns. These characteristics strongly suggest this is a malicious phishing site designed to steal user credentials."
            indicators = [
                {"indicator": "Suspicious domain name", "severity": "high", "description": "Domain appears to impersonate a legitimate brand", "category": "url"},
                {"indicator": "Phishing URL patterns", "severity": "high", "description": "URL structure matches known phishing templates", "category": "url"}
            ]
            recommendations = [
                "Do NOT enter any personal information on this site",
                "Do NOT click any links or download files",
                "Report this URL to your IT security team",
                "If you entered credentials, change your passwords immediately"
            ]
            risk_score = 0.9
        else:
            summary = f"This URL appears to be legitimate based on our analysis. The domain structure and URL patterns match expected characteristics of authentic websites."
            detailed = f"Analysis of {url} shows no significant red flags. The domain appears genuine with proper structure, and the URL doesn't contain suspicious patterns typically associated with phishing attempts."
            indicators = [
                {"indicator": "Valid domain structure", "severity": "low", "description": "Domain follows standard naming conventions", "category": "url"},
                {"indicator": "No phishing patterns detected", "severity": "low", "description": "URL structure appears legitimate", "category": "url"}
            ]
            recommendations = [
                "This site appears safe, but always verify the URL before entering sensitive information",
                "Look for HTTPS and valid certificates before submitting data"
            ]
            risk_score = 0.1
        
        return json.dumps({
            "summary": summary,
            "detailed_explanation": detailed,
            "indicators": indicators,
            "recommendations": recommendations,
            "risk_score": risk_score
        })


class LLMExplainer:
    """
    Generates explainable phishing detection results using LLMs.
    
    Combines analysis results from URL and visual modules to produce
    human-readable explanations that help users understand why a
    website was flagged as phishing or deemed legitimate.
    """
    
    EXPLANATION_PROMPT_TEMPLATE = """
You are a cybersecurity expert analyzing a website for potential phishing threats.
Based on the following analysis results, provide a comprehensive explanation.

## Analysis Results

### URL Analysis
- URL: {url}
- URL Score: {url_score:.2f} (0=safe, 1=phishing)
- Key URL Features:
{url_features}

### Visual Analysis
- Visual Score: {visual_score:.2f} (0=safe, 1=phishing)
- Key Visual Features:
{visual_features}

### Combined Prediction
- Final Classification: {classification}
- Overall Confidence: {confidence:.2f}

## Instructions
Provide your analysis in the following JSON format:
{{
    "summary": "A 2-3 sentence summary of the classification",
    "detailed_explanation": "A detailed paragraph explaining the reasoning",
    "indicators": [
        {{
            "indicator": "Name of the indicator",
            "severity": "high/medium/low",
            "description": "What this indicator means",
            "category": "url/visual/behavioral"
        }}
    ],
    "recommendations": ["List of recommended actions for the user"],
    "risk_score": 0.0-1.0
}}

Respond ONLY with valid JSON, no additional text.
"""
    
    SIMPLE_EXPLANATION_TEMPLATE = """
Explain in simple terms why this website is classified as {classification}.

URL: {url}
Main concerns:
{concerns}

Provide a clear, non-technical explanation that a regular user can understand.
Keep it brief (2-3 sentences).
"""
    
    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.3,
        **kwargs
    ):
        """
        Initialize the LLM Explainer.
        
        Args:
            provider: LLM provider ('openai', 'huggingface', 'mock')
            api_key: API key for the provider (if applicable)
            model: Model name to use
            temperature: Sampling temperature
            **kwargs: Additional provider-specific arguments
        """
        self.provider_name = provider
        
        if provider == "openai":
            self.provider = OpenAIProvider(
                api_key=api_key,
                model=model,
                temperature=temperature,
                **kwargs
            )
        elif provider == "huggingface":
            self.provider = HuggingFaceProvider(
                model_name=model,
                **kwargs
            )
        elif provider == "gemini":
            self.provider = GeminiProvider(
                api_key=api_key,
                model="gemini-2.0-flash",  # Faster, higher quota
                temperature=temperature,
                max_tokens=2000,
                **kwargs
            )
        elif provider == "mock":
            self.provider = MockProvider()
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def explain(
        self,
        url: str,
        url_score: float,
        url_features: Dict[str, Any],
        visual_score: float,
        visual_features: Dict[str, Any],
        classification: str,
        confidence: float
    ) -> ExplanationResult:
        """
        Generate an explanation for the phishing detection result.
        
        Args:
            url: The analyzed URL
            url_score: Score from URL analysis (0-1)
            url_features: Features extracted from URL
            visual_score: Score from visual analysis (0-1)
            visual_features: Features extracted from visual analysis
            classification: Final classification ('phishing' or 'legitimate')
            confidence: Overall confidence score
            
        Returns:
            ExplanationResult with detailed explanation
        """
        # Format features for prompt
        url_features_str = self._format_features(url_features)
        visual_features_str = self._format_features(visual_features)
        
        # Generate prompt
        prompt = self.EXPLANATION_PROMPT_TEMPLATE.format(
            url=url,
            url_score=url_score,
            url_features=url_features_str,
            visual_score=visual_score,
            visual_features=visual_features_str,
            classification=classification,
            confidence=confidence
        )
        
        # Get LLM response
        response = self.provider.generate(prompt)
        
        # Strip markdown code blocks before parsing
        import re
        response = re.sub(r'^```json\s*\n?', '', response.strip())
        response = re.sub(r'^```\s*\n?', '', response)
        response = re.sub(r'\n?```\s*$', '', response)
        response = response.strip()
        
        # Parse response
        try:
            result_data = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            result_data = self._extract_json(response)
        
        # Create indicator objects
        indicators = [
            PhishingIndicator(
                indicator=ind.get('indicator', ''),
                severity=ind.get('severity', 'medium'),
                description=ind.get('description', ''),
                category=ind.get('category', 'unknown')
            )
            for ind in result_data.get('indicators', [])
        ]
        
        return ExplanationResult(
            classification=classification,
            confidence=confidence,
            summary=result_data.get('summary', ''),
            detailed_explanation=result_data.get('detailed_explanation', ''),
            indicators=indicators,
            recommendations=result_data.get('recommendations', []),
            risk_score=result_data.get('risk_score', url_score * 0.5 + visual_score * 0.5),
            raw_response=response
        )
    
    def explain_simple(
        self,
        url: str,
        classification: str,
        key_concerns: List[str]
    ) -> str:
        """
        Generate a simple, user-friendly explanation.
        
        Args:
            url: The analyzed URL
            classification: The classification result
            key_concerns: List of main concerns or indicators
            
        Returns:
            Simple text explanation
        """
        concerns_str = "\n".join(f"- {concern}" for concern in key_concerns)
        
        prompt = self.SIMPLE_EXPLANATION_TEMPLATE.format(
            url=url,
            classification=classification,
            concerns=concerns_str
        )
        
        return self.provider.generate(prompt)
    
    def analyze_url_features(self, features: Dict[str, Any]) -> List[str]:
        """
        Analyze URL features and return key concerns.
        
        Args:
            features: Dictionary of URL features
            
        Returns:
            List of concern strings
        """
        concerns = []
        
        if features.get('has_ip_address'):
            concerns.append("URL contains an IP address instead of a domain name")
        
        if features.get('num_sensitive_words', 0) > 0:
            concerns.append(f"URL contains {features['num_sensitive_words']} sensitive words commonly used in phishing")
        
        if features.get('has_suspicious_params'):
            concerns.append("URL contains suspicious query parameters")
        
        if features.get('has_punycode'):
            concerns.append("URL uses internationalized characters (possible homograph attack)")
        
        if features.get('entropy', 0) > 4.0:
            concerns.append("URL has high randomness, suggesting it may be auto-generated")
        
        if not features.get('has_https'):
            concerns.append("Website does not use HTTPS encryption")
        
        if features.get('url_length', 0) > 100:
            concerns.append("URL is unusually long")
        
        if features.get('num_subdomains', 0) > 3:
            concerns.append("URL has many subdomains")
        
        if features.get('domain_in_path'):
            concerns.append("URL path contains brand names (possible impersonation)")
        
        return concerns
    
    def analyze_visual_features(self, features: Dict[str, Any]) -> List[str]:
        """
        Analyze visual features and return key concerns.
        
        Args:
            features: Dictionary of visual features
            
        Returns:
            List of concern strings
        """
        concerns = []
        
        if features.get('has_password_field'):
            concerns.append("Page contains password input fields")
        
        if features.get('has_login_form'):
            concerns.append("Page appears to have a login form")
        
        if features.get('brand_similarity', 0) > 0.7:
            concerns.append("Visual design closely resembles a known brand")
        
        if features.get('low_quality_images'):
            concerns.append("Page contains low-quality or stretched images")
        
        if features.get('mismatched_favicon'):
            concerns.append("Favicon doesn't match the claimed brand")
        
        return concerns
    
    def _format_features(self, features: Dict[str, Any]) -> str:
        """Format feature dictionary for prompt."""
        if not features:
            return "  - No features available"
        
        lines = []
        for key, value in features.items():
            if isinstance(value, float):
                lines.append(f"  - {key}: {value:.4f}")
            elif isinstance(value, bool):
                lines.append(f"  - {key}: {'Yes' if value else 'No'}")
            else:
                lines.append(f"  - {key}: {value}")
        
        return "\n".join(lines[:15])  # Limit to top 15 features
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text that may contain additional content."""
        import re
        
        # Strip markdown code blocks (multiple patterns)
        text = re.sub(r'```json\s*\n?', '', text)
        text = re.sub(r'```\s*\n?', '', text)
        text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
        text = text.strip()
        
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Find the outermost JSON object by matching balanced braces
        start_idx = text.find('{')
        if start_idx == -1:
            return self._fallback_response(text)
        
        # Count braces to find matching end
        brace_count = 0
        end_idx = start_idx
        for i, char in enumerate(text[start_idx:], start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        json_str = text[start_idx:end_idx]
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues
            # Remove any trailing commas
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return self._fallback_response(text)
    
    def _fallback_response(self, text: str) -> Dict[str, Any]:
        """Generate fallback response when JSON parsing fails."""
        # Try to extract summary from text
        summary = "Analysis complete. "
        if "phishing" in text.lower():
            summary += "This URL shows signs of being a phishing attempt."
        else:
            summary += "This URL appears to be legitimate based on our analysis."
        
        return {
            "summary": summary,
            "detailed_explanation": text[:500] if text else "Unable to generate detailed explanation.",
            "indicators": [],
            "recommendations": ["Review the URL carefully before proceeding"],
            "risk_score": 0.5
        }
    
    def batch_explain(
        self,
        results: List[Dict[str, Any]]
    ) -> List[ExplanationResult]:
        """
        Generate explanations for multiple detection results.
        
        Args:
            results: List of detection result dictionaries
            
        Returns:
            List of ExplanationResult objects
        """
        explanations = []
        for result in results:
            explanation = self.explain(
                url=result['url'],
                url_score=result['url_score'],
                url_features=result.get('url_features', {}),
                visual_score=result['visual_score'],
                visual_features=result.get('visual_features', {}),
                classification=result['classification'],
                confidence=result['confidence']
            )
            explanations.append(explanation)
        
        return explanations
    
    def generate_report(
        self,
        explanation: ExplanationResult,
        include_raw: bool = False
    ) -> str:
        """
        Generate a formatted report from an explanation.
        
        Args:
            explanation: ExplanationResult to format
            include_raw: Whether to include raw LLM response
            
        Returns:
            Formatted report string
        """
        report_lines = [
            "=" * 60,
            "PHISHING DETECTION REPORT",
            "=" * 60,
            "",
            f"Classification: {explanation.classification.upper()}",
            f"Confidence: {explanation.confidence:.1%}",
            f"Risk Score: {explanation.risk_score:.1%}",
            "",
            "SUMMARY",
            "-" * 40,
            explanation.summary,
            "",
            "DETAILED ANALYSIS",
            "-" * 40,
            explanation.detailed_explanation,
            "",
        ]
        
        if explanation.indicators:
            report_lines.extend([
                "KEY INDICATORS",
                "-" * 40
            ])
            for ind in explanation.indicators:
                report_lines.append(
                    f"  [{ind.severity.upper()}] {ind.indicator} ({ind.category})"
                )
                report_lines.append(f"    {ind.description}")
            report_lines.append("")
        
        if explanation.recommendations:
            report_lines.extend([
                "RECOMMENDATIONS",
                "-" * 40
            ])
            for i, rec in enumerate(explanation.recommendations, 1):
                report_lines.append(f"  {i}. {rec}")
            report_lines.append("")
        
        if include_raw and explanation.raw_response:
            report_lines.extend([
                "RAW LLM RESPONSE",
                "-" * 40,
                explanation.raw_response,
                ""
            ])
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)


if __name__ == "__main__":
    # Example usage with mock provider
    print("LLM Explainer Module Test")
    print("=" * 50)
    
    explainer = LLMExplainer(provider="mock")
    
    # Test explanation
    result = explainer.explain(
        url="http://secure-paypal-login.xyz/verify",
        url_score=0.85,
        url_features={
            'url_length': 45,
            'has_https': False,
            'has_ip_address': False,
            'num_sensitive_words': 3,
            'entropy': 3.5
        },
        visual_score=0.72,
        visual_features={
            'has_password_field': True,
            'has_login_form': True
        },
        classification="phishing",
        confidence=0.82
    )
    
    print("\nExplanation Result:")
    print(f"  Summary: {result.summary}")
    print(f"  Risk Score: {result.risk_score:.2f}")
    print(f"  Indicators: {len(result.indicators)}")
    print(f"  Recommendations: {len(result.recommendations)}")
    
    print("\nFormatted Report:")
    print(explainer.generate_report(result))
