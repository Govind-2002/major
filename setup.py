"""
Setup script for Hybrid Phishing Detection System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hybrid-phishing-detector",
    version="1.0.0",
    author="Phishing Detection Team",
    author_email="team@example.com",
    description="A hybrid multi-modal phishing detection system using URL analysis, visual features, and LLM reasoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/phishing-detection",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "phishing-detect=scripts.demo:main",
            "phishing-train=scripts.train:main",
            "phishing-eval=scripts.evaluate:main",
            "phishing-api=api.app:run_server",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
