"""
Data Loading Utilities

Provides utilities for loading and preprocessing phishing detection datasets
from various sources including PhishTank, Kaggle, and custom datasets.
"""

import os
import csv
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass
import random

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm


@dataclass
class PhishingDataset:
    """Container for phishing detection dataset."""
    urls: List[str]
    labels: List[int]
    metadata: Optional[Dict[str, Any]] = None
    
    def __len__(self) -> int:
        return len(self.urls)
    
    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.urls[idx], self.labels[idx]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame({
            'url': self.urls,
            'label': self.labels
        })
    
    def split(
        self,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple['PhishingDataset', 'PhishingDataset']:
        """Split dataset into train and test sets."""
        from sklearn.model_selection import train_test_split
        
        urls_train, urls_test, labels_train, labels_test = train_test_split(
            self.urls, self.labels,
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels
        )
        
        return (
            PhishingDataset(urls=urls_train, labels=labels_train),
            PhishingDataset(urls=urls_test, labels=labels_test)
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        labels = np.array(self.labels)
        return {
            'total_samples': len(self.urls),
            'phishing_count': int(np.sum(labels == 1)),
            'legitimate_count': int(np.sum(labels == 0)),
            'phishing_ratio': float(np.mean(labels)),
            'legitimate_ratio': float(1 - np.mean(labels))
        }


class DataLoader:
    """
    Data loader for phishing detection datasets.
    
    Supports loading from various sources:
    - CSV files
    - JSON files
    - PhishTank API
    - Kaggle datasets
    - Custom URL lists
    """
    
    PHISHTANK_API_URL = "http://data.phishtank.com/data/online-valid.json"
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Base directory for data files
        """
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_csv(
        self,
        filepath: Union[str, Path],
        url_column: str = 'url',
        label_column: str = 'label',
        delimiter: str = ','
    ) -> PhishingDataset:
        """
        Load dataset from CSV file.
        
        Args:
            filepath: Path to CSV file
            url_column: Name of URL column
            label_column: Name of label column
            delimiter: CSV delimiter
            
        Returns:
            PhishingDataset object
        """
        df = pd.read_csv(filepath, delimiter=delimiter)
        
        return PhishingDataset(
            urls=df[url_column].tolist(),
            labels=df[label_column].astype(int).tolist(),
            metadata={'source': str(filepath), 'format': 'csv'}
        )
    
    def load_json(
        self,
        filepath: Union[str, Path],
        url_key: str = 'url',
        label_key: str = 'label'
    ) -> PhishingDataset:
        """
        Load dataset from JSON file.
        
        Args:
            filepath: Path to JSON file
            url_key: Key for URL in JSON objects
            label_key: Key for label in JSON objects
            
        Returns:
            PhishingDataset object
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            urls = [item[url_key] for item in data]
            labels = [item[label_key] for item in data]
        else:
            urls = data[url_key]
            labels = data[label_key]
        
        return PhishingDataset(
            urls=urls,
            labels=[int(l) for l in labels],
            metadata={'source': str(filepath), 'format': 'json'}
        )
    
    def load_url_list(
        self,
        filepath: Union[str, Path],
        label: int
    ) -> PhishingDataset:
        """
        Load URLs from a text file (one URL per line).
        
        Args:
            filepath: Path to text file
            label: Label to assign to all URLs
            
        Returns:
            PhishingDataset object
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        return PhishingDataset(
            urls=urls,
            labels=[label] * len(urls),
            metadata={'source': str(filepath), 'format': 'url_list'}
        )
    
    def download_phishtank(
        self,
        api_key: Optional[str] = None,
        limit: Optional[int] = None,
        save: bool = True
    ) -> PhishingDataset:
        """
        Download phishing URLs from PhishTank.
        
        Args:
            api_key: PhishTank API key (optional)
            limit: Maximum number of URLs to download
            save: Whether to save downloaded data
            
        Returns:
            PhishingDataset with phishing URLs (label=1)
        """
        url = self.PHISHTANK_API_URL
        if api_key:
            url += f"?application_key={api_key}"
        
        print("Downloading PhishTank data...")
        response = requests.get(url, headers={'User-Agent': 'PhishingDetector/1.0'})
        response.raise_for_status()
        
        data = response.json()
        
        urls = []
        for item in data[:limit] if limit else data:
            urls.append(item['url'])
        
        if save:
            save_path = self.data_dir / 'raw' / 'phishtank_urls.json'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump({'urls': urls}, f)
        
        return PhishingDataset(
            urls=urls,
            labels=[1] * len(urls),
            metadata={'source': 'phishtank', 'count': len(urls)}
        )
    
    def load_alexa_top_sites(
        self,
        filepath: Optional[Union[str, Path]] = None,
        limit: int = 1000
    ) -> PhishingDataset:
        """
        Load legitimate URLs from Alexa Top Sites or similar list.
        
        Args:
            filepath: Path to Alexa top sites file
            limit: Maximum number of sites to load
            
        Returns:
            PhishingDataset with legitimate URLs (label=0)
        """
        if filepath and Path(filepath).exists():
            with open(filepath, 'r') as f:
                urls = [line.strip() for line in f if line.strip()][:limit]
        else:
            # Generate common legitimate URLs as fallback
            domains = [
                'google.com', 'youtube.com', 'facebook.com', 'amazon.com',
                'wikipedia.org', 'twitter.com', 'instagram.com', 'linkedin.com',
                'microsoft.com', 'apple.com', 'github.com', 'netflix.com',
                'reddit.com', 'yahoo.com', 'ebay.com', 'bing.com',
                'office.com', 'live.com', 'twitch.tv', 'spotify.com'
            ]
            
            urls = []
            for domain in domains:
                urls.extend([
                    f"https://www.{domain}",
                    f"https://{domain}",
                    f"https://www.{domain}/",
                ])
            urls = urls[:limit]
        
        return PhishingDataset(
            urls=urls,
            labels=[0] * len(urls),
            metadata={'source': 'alexa_top_sites', 'count': len(urls)}
        )
    
    def create_balanced_dataset(
        self,
        phishing_urls: List[str],
        legitimate_urls: List[str],
        balance_ratio: float = 1.0
    ) -> PhishingDataset:
        """
        Create a balanced dataset from phishing and legitimate URLs.
        
        Args:
            phishing_urls: List of phishing URLs
            legitimate_urls: List of legitimate URLs
            balance_ratio: Ratio of legitimate to phishing (1.0 = equal)
            
        Returns:
            Balanced PhishingDataset
        """
        n_phishing = len(phishing_urls)
        n_legitimate = int(n_phishing * balance_ratio)
        
        # Sample if needed
        if len(legitimate_urls) > n_legitimate:
            legitimate_urls = random.sample(legitimate_urls, n_legitimate)
        elif len(legitimate_urls) < n_legitimate:
            n_phishing = int(len(legitimate_urls) / balance_ratio)
            phishing_urls = random.sample(phishing_urls, n_phishing)
            legitimate_urls = legitimate_urls
        
        # Combine and shuffle
        urls = phishing_urls + legitimate_urls
        labels = [1] * len(phishing_urls) + [0] * len(legitimate_urls)
        
        combined = list(zip(urls, labels))
        random.shuffle(combined)
        urls, labels = zip(*combined)
        
        return PhishingDataset(
            urls=list(urls),
            labels=list(labels),
            metadata={
                'phishing_count': len(phishing_urls),
                'legitimate_count': len(legitimate_urls),
                'balance_ratio': balance_ratio
            }
        )
    
    def generate_synthetic_dataset(
        self,
        n_samples: int = 1000,
        phishing_ratio: float = 0.5
    ) -> PhishingDataset:
        """
        Generate synthetic phishing detection dataset.
        
        Args:
            n_samples: Total number of samples
            phishing_ratio: Ratio of phishing samples
            
        Returns:
            Synthetic PhishingDataset
        """
        n_phishing = int(n_samples * phishing_ratio)
        n_legitimate = n_samples - n_phishing
        
        # Legitimate URL patterns
        legitimate_domains = [
            'google.com', 'facebook.com', 'amazon.com', 'microsoft.com',
            'apple.com', 'github.com', 'linkedin.com', 'twitter.com'
        ]
        legitimate_paths = [
            '', '/', '/home', '/about', '/contact', '/products',
            '/services', '/login', '/account', '/search'
        ]
        
        # Phishing URL patterns
        phishing_tlds = ['.xyz', '.tk', '.ml', '.ga', '.cf', '.top', '.work']
        phishing_keywords = [
            'secure', 'login', 'verify', 'update', 'confirm', 'account',
            'password', 'signin', 'auth', 'wallet', 'bank'
        ]
        
        urls = []
        labels = []
        
        # Generate legitimate URLs
        for _ in range(n_legitimate):
            domain = random.choice(legitimate_domains)
            path = random.choice(legitimate_paths)
            urls.append(f"https://www.{domain}{path}")
            labels.append(0)
        
        # Generate phishing URLs
        for _ in range(n_phishing):
            pattern = random.choice([
                # IP-based
                lambda: f"http://{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}/login.php",
                # Suspicious domain
                lambda: f"http://{''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=8))}{random.choice(phishing_tlds)}/{random.choice(phishing_keywords)}",
                # Brand impersonation
                lambda: f"http://{random.choice(phishing_keywords)}-{random.choice(legitimate_domains).split('.')[0]}{random.choice(phishing_tlds)}/verify",
                # Subdomain abuse
                lambda: f"http://{random.choice(legitimate_domains).split('.')[0]}.suspicious-site.com/{random.choice(phishing_keywords)}",
            ])
            urls.append(pattern())
            labels.append(1)
        
        # Shuffle
        combined = list(zip(urls, labels))
        random.shuffle(combined)
        urls, labels = zip(*combined)
        
        return PhishingDataset(
            urls=list(urls),
            labels=list(labels),
            metadata={
                'synthetic': True,
                'n_samples': n_samples,
                'phishing_ratio': phishing_ratio
            }
        )
    
    def save_dataset(
        self,
        dataset: PhishingDataset,
        filepath: Union[str, Path],
        format: str = 'csv'
    ) -> None:
        """
        Save dataset to file.
        
        Args:
            dataset: Dataset to save
            filepath: Output file path
            format: Output format ('csv' or 'json')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            df = dataset.to_dataframe()
            df.to_csv(filepath, index=False)
        elif format == 'json':
            data = {
                'urls': dataset.urls,
                'labels': dataset.labels,
                'metadata': dataset.metadata
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")


class ScreenshotDatasetLoader:
    """Loader for screenshot datasets."""
    
    def __init__(self, data_dir: str = "data/screenshots"):
        """Initialize the screenshot dataset loader."""
        self.data_dir = Path(data_dir)
    
    def load_from_directory(
        self,
        phishing_dir: Optional[str] = None,
        legitimate_dir: Optional[str] = None
    ) -> Tuple[List[str], List[int]]:
        """
        Load screenshot paths from directory structure.
        
        Expected structure:
        - data_dir/phishing/*.png
        - data_dir/legitimate/*.png
        
        Returns:
            Tuple of (image_paths, labels)
        """
        image_paths = []
        labels = []
        
        # Load phishing screenshots
        phishing_path = Path(phishing_dir) if phishing_dir else self.data_dir / 'phishing'
        if phishing_path.exists():
            for img_path in phishing_path.glob('*.png'):
                image_paths.append(str(img_path))
                labels.append(1)
        
        # Load legitimate screenshots
        legitimate_path = Path(legitimate_dir) if legitimate_dir else self.data_dir / 'legitimate'
        if legitimate_path.exists():
            for img_path in legitimate_path.glob('*.png'):
                image_paths.append(str(img_path))
                labels.append(0)
        
        return image_paths, labels
    
    def split_dataset(
        self,
        image_paths: List[str],
        labels: List[int],
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Dict[str, Tuple[List[str], List[int]]]:
        """
        Split screenshot dataset into train/val/test sets.
        
        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            image_paths, labels,
            test_size=test_size,
            stratify=labels,
            random_state=42
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_ratio,
            stratify=y_trainval,
            random_state=42
        )
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }


if __name__ == "__main__":
    # Example usage
    print("Data Loader Module Test")
    print("=" * 50)
    
    loader = DataLoader()
    
    # Generate synthetic dataset
    print("\nGenerating synthetic dataset...")
    dataset = loader.generate_synthetic_dataset(n_samples=200, phishing_ratio=0.5)
    
    print(f"Dataset Statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Split dataset
    train_set, test_set = dataset.split(test_size=0.2)
    print(f"\nTrain set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")
    
    # Show sample URLs
    print("\nSample URLs:")
    for i in range(min(5, len(dataset))):
        url, label = dataset[i]
        label_str = "PHISHING" if label == 1 else "LEGITIMATE"
        print(f"  [{label_str}] {url}")
