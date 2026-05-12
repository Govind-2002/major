"""
FastAPI REST API for Hybrid Phishing Detection System

Provides RESTful endpoints for phishing detection, batch analysis,
and system management.
"""

import os
import time
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, HttpUrl
import uvicorn

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hybrid_detector.detector import HybridPhishingDetector, DetectionResult


# Global detector instance
detector: Optional[HybridPhishingDetector] = None


# Pydantic models for request/response
class URLAnalysisRequest(BaseModel):
    """Request model for URL analysis."""
    url: str = Field(..., description="URL to analyze for phishing")
    capture_screenshot: bool = Field(default=False, description="Whether to capture webpage screenshot")
    generate_explanation: bool = Field(default=True, description="Whether to generate LLM explanation")


class BatchAnalysisRequest(BaseModel):
    """Request model for batch URL analysis."""
    urls: List[str] = Field(..., description="List of URLs to analyze")
    capture_screenshots: bool = Field(default=False, description="Whether to capture screenshots")
    generate_explanations: bool = Field(default=False, description="Whether to generate explanations")


class QuickCheckRequest(BaseModel):
    """Request model for quick URL check."""
    url: str = Field(..., description="URL to check")


class AnalysisResponse(BaseModel):
    """Response model for analysis results."""
    url: str
    is_phishing: bool
    confidence: float
    risk_level: str
    url_score: float
    visual_score: float
    combined_score: float
    analysis_time: float
    explanation: Optional[Dict[str, Any]] = None
    url_features: Optional[Dict[str, Any]] = None


class QuickCheckResponse(BaseModel):
    """Response model for quick check."""
    url: str
    is_phishing: bool
    confidence: float
    risk_level: str


class BatchAnalysisResponse(BaseModel):
    """Response model for batch analysis."""
    total_urls: int
    phishing_count: int
    legitimate_count: int
    results: List[AnalysisResponse]
    total_time: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    url_model_loaded: bool
    visual_model_loaded: bool
    uptime: float


class PhishingAPI:
    """
    REST API for the Hybrid Phishing Detection System.
    
    Provides endpoints for:
    - Single URL analysis
    - Batch URL analysis
    - Quick URL checking
    - System health monitoring
    """
    
    def __init__(
        self,
        url_model_path: Optional[str] = None,
        visual_model_path: Optional[str] = None,
        llm_provider: str = "mock",
        llm_api_key: Optional[str] = None
    ):
        """
        Initialize the API.
        
        Args:
            url_model_path: Path to pretrained URL model
            visual_model_path: Path to pretrained visual model
            llm_provider: LLM provider for explanations
            llm_api_key: API key for LLM provider
        """
        self.url_model_path = url_model_path
        self.visual_model_path = visual_model_path
        self.llm_provider = llm_provider
        self.llm_api_key = llm_api_key
        self.start_time = time.time()
        self.detector: Optional[HybridPhishingDetector] = None
    
    def initialize(self) -> None:
        """Initialize the detector."""
        self.detector = HybridPhishingDetector(
            url_model_path=self.url_model_path,
            visual_model_path=self.visual_model_path,
            llm_provider=self.llm_provider,
            llm_api_key=self.llm_api_key,
            enable_screenshot=True
        )
        self.start_time = time.time()
    
    def shutdown(self) -> None:
        """Cleanup resources."""
        if self.detector:
            self.detector.close()
    
    def analyze_url(
        self,
        url: str,
        capture_screenshot: bool = False,
        generate_explanation: bool = True
    ) -> DetectionResult:
        """Analyze a single URL."""
        if not self.detector:
            raise RuntimeError("Detector not initialized")
        
        return self.detector.analyze(
            url=url,
            capture_screenshot=capture_screenshot,
            generate_explanation=generate_explanation
        )
    
    def quick_check(self, url: str) -> tuple:
        """Quick URL check without full analysis."""
        if not self.detector:
            raise RuntimeError("Detector not initialized")
        
        return self.detector.quick_check(url)
    
    def batch_analyze(
        self,
        urls: List[str],
        capture_screenshots: bool = False,
        generate_explanations: bool = False
    ) -> List[DetectionResult]:
        """Analyze multiple URLs."""
        if not self.detector:
            raise RuntimeError("Detector not initialized")
        
        return self.detector.analyze_batch(
            urls=urls,
            capture_screenshots=capture_screenshots,
            generate_explanations=generate_explanations
        )
    
    def get_health(self) -> Dict[str, Any]:
        """Get system health status."""
        stats = self.detector.get_statistics() if self.detector else {}
        
        return {
            'status': 'healthy' if self.detector else 'not_initialized',
            'version': '1.0.0',
            'url_model_loaded': stats.get('url_model_fitted', False),
            'visual_model_loaded': stats.get('visual_model_fitted', False),
            'uptime': time.time() - self.start_time
        }


# Get model paths relative to project root
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_url_model_path = os.path.join(_project_root, "models", "url_model_real.pkl")
_visual_model_path = os.path.join(_project_root, "models", "visual_model.pt")

# Fall back to synthetic model if real model doesn't exist
if not os.path.exists(_url_model_path):
    _url_model_path = os.path.join(_project_root, "models", "url_model.pkl")
if not os.path.exists(_url_model_path):
    _url_model_path = None
if not os.path.exists(_visual_model_path):
    _visual_model_path = None

# Load API keys from environment
from dotenv import load_dotenv
load_dotenv()
_gemini_api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
_openai_api_key = os.getenv('OPENAI_API_KEY')

# Use Gemini if available, else OpenAI, else mock
if _gemini_api_key:
    _llm_provider = "gemini"
    _llm_api_key = _gemini_api_key
elif _openai_api_key:
    _llm_provider = "openai"
    _llm_api_key = _openai_api_key
else:
    _llm_provider = "mock"
    _llm_api_key = None

# Create API instance with model paths
api_instance = PhishingAPI(
    url_model_path=_url_model_path,
    visual_model_path=_visual_model_path,
    llm_provider=_llm_provider,
    llm_api_key=_llm_api_key
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    api_instance.initialize()
    yield
    # Shutdown
    api_instance.shutdown()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Hybrid Phishing Detection API",
        description="""
        A multi-modal phishing detection system combining URL analysis,
        visual analysis, and LLM-based reasoning.
        
        ## Features
        - URL-based phishing detection using ML models
        - Visual analysis of webpage screenshots
        - Explainable AI with LLM-generated insights
        - Batch processing support
        - Real-time quick checks
        """,
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Health check endpoint
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """Check system health and status."""
        return api_instance.get_health()
    
    @app.get("/", tags=["System"])
    async def root():
        """Root endpoint - serve the web UI."""
        frontend_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "frontend", "index.html"
        )
        if os.path.exists(frontend_path):
            return FileResponse(frontend_path, media_type="text/html")
        return {
            "name": "Hybrid Phishing Detection API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health"
        }
    
    @app.get("/api", tags=["System"])
    async def api_info():
        """API information endpoint."""
        return {
            "name": "Hybrid Phishing Detection API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health"
        }
    
    # Analysis endpoints
    @app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
    async def analyze_url(request: URLAnalysisRequest):
        """
        Analyze a single URL for phishing.
        
        Performs comprehensive analysis using URL features, visual analysis,
        and optionally generates LLM-based explanations.
        """
        try:
            result = api_instance.analyze_url(
                url=request.url,
                capture_screenshot=request.capture_screenshot,
                generate_explanation=request.generate_explanation
            )
            
            response_data = {
                "url": result.url,
                "is_phishing": result.is_phishing,
                "confidence": result.confidence,
                "risk_level": result.risk_level,
                "url_score": result.url_score,
                "visual_score": result.visual_score,
                "combined_score": result.combined_score,
                "analysis_time": result.analysis_time,
                "url_features": result.url_features
            }
            
            if result.explanation:
                response_data["explanation"] = {
                    "summary": result.explanation.summary,
                    "detailed": result.explanation.detailed_explanation,
                    "recommendations": result.explanation.recommendations,
                    "risk_score": result.explanation.risk_score
                }
            
            return response_data
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/quick-check", response_model=QuickCheckResponse, tags=["Analysis"])
    async def quick_check(request: QuickCheckRequest):
        """
        Perform quick URL-only analysis.
        
        Fast check using only URL features without screenshot capture
        or visual analysis. Suitable for high-volume screening.
        """
        try:
            is_phishing, confidence, risk_level = api_instance.quick_check(request.url)
            
            return {
                "url": request.url,
                "is_phishing": is_phishing,
                "confidence": confidence,
                "risk_level": risk_level
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/batch-analyze", response_model=BatchAnalysisResponse, tags=["Analysis"])
    async def batch_analyze(request: BatchAnalysisRequest):
        """
        Analyze multiple URLs in batch.
        
        Process multiple URLs efficiently with optional screenshot
        capture and explanation generation.
        """
        try:
            start_time = time.time()
            
            results = api_instance.batch_analyze(
                urls=request.urls,
                capture_screenshots=request.capture_screenshots,
                generate_explanations=request.generate_explanations
            )
            
            response_results = []
            phishing_count = 0
            
            for result in results:
                if result.is_phishing:
                    phishing_count += 1
                
                response_data = {
                    "url": result.url,
                    "is_phishing": result.is_phishing,
                    "confidence": result.confidence,
                    "risk_level": result.risk_level,
                    "url_score": result.url_score,
                    "visual_score": result.visual_score,
                    "combined_score": result.combined_score,
                    "analysis_time": result.analysis_time
                }
                response_results.append(response_data)
            
            return {
                "total_urls": len(request.urls),
                "phishing_count": phishing_count,
                "legitimate_count": len(request.urls) - phishing_count,
                "results": response_results,
                "total_time": time.time() - start_time
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/statistics", tags=["System"])
    async def get_statistics():
        """Get system statistics and configuration."""
        if api_instance.detector:
            return api_instance.detector.get_statistics()
        return {"error": "Detector not initialized"}
    
    return app


# Create app instance
app = create_app()


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False
):
    """Run the API server."""
    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Phishing Detection API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print(f"Starting Phishing Detection API on http://{args.host}:{args.port}")
    print(f"API Documentation: http://{args.host}:{args.port}/docs")
    
    run_server(host=args.host, port=args.port, reload=args.reload)
