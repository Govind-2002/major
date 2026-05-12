"""
Visual Phishing Classification Model

Implements deep learning models for visual-based phishing detection
using webpage screenshots. Supports multiple architectures including
ResNet, Vision Transformer (ViT), and EfficientNet.
"""

import os
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import timm
from tqdm import tqdm


@dataclass
class VisualFeatures:
    """Visual features extracted from screenshot analysis."""
    prediction: int
    confidence: float
    feature_vector: np.ndarray
    attention_map: Optional[np.ndarray] = None
    predicted_class: str = ""


class PhishingScreenshotDataset(Dataset):
    """Dataset class for phishing screenshot images."""
    
    def __init__(
        self,
        image_paths: List[str],
        labels: Optional[List[int]] = None,
        transform: Optional[transforms.Compose] = None,
        image_size: int = 224
    ):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of paths to screenshot images
            labels: Optional list of labels (0=legitimate, 1=phishing)
            transform: Optional torchvision transforms
            image_size: Target image size
        """
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        
        # Load and convert image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        if self.labels is not None:
            return image, self.labels[idx]
        return image, -1


class VisualPhishingClassifier:
    """
    Deep learning classifier for visual-based phishing detection.
    
    Analyzes webpage screenshots using CNN or Vision Transformer
    architectures to detect phishing patterns.
    """
    
    SUPPORTED_MODELS = {
        'resnet50': 'resnet50',
        'resnet101': 'resnet101',
        'efficientnet_b0': 'efficientnet_b0',
        'efficientnet_b3': 'efficientnet_b3',
        'vit_base': 'vit_base_patch16_224',
        'vit_small': 'vit_small_patch16_224',
        'swin_base': 'swin_base_patch4_window7_224'
    }
    
    def __init__(
        self,
        model_type: str = 'resnet50',
        image_size: int = 224,
        pretrained: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the Visual Phishing Classifier.
        
        Args:
            model_type: Type of model architecture
            image_size: Input image size
            pretrained: Whether to use pretrained weights
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model type must be one of {list(self.SUPPORTED_MODELS.keys())}")
        
        self.model_type = model_type
        self.image_size = image_size
        self.pretrained = pretrained
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.is_fitted = False
        self.classes = ['legitimate', 'phishing']
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.eval_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _create_model(self, num_classes: int = 2) -> nn.Module:
        """Create the neural network model."""
        model_name = self.SUPPORTED_MODELS[self.model_type]
        
        if self.model_type.startswith('resnet'):
            # Use torchvision ResNet
            if self.model_type == 'resnet50':
                model = models.resnet50(pretrained=self.pretrained)
            else:
                model = models.resnet101(pretrained=self.pretrained)
            
            # Replace final layer
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, num_classes)
            )
        
        elif self.model_type.startswith('efficientnet'):
            # Use timm EfficientNet
            model = timm.create_model(
                model_name,
                pretrained=self.pretrained,
                num_classes=num_classes
            )
        
        elif self.model_type.startswith('vit') or self.model_type.startswith('swin'):
            # Use timm Vision Transformer / Swin
            model = timm.create_model(
                model_name,
                pretrained=self.pretrained,
                num_classes=num_classes
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model.to(self.device)
    
    def fit(
        self,
        train_images: List[str],
        train_labels: List[int],
        val_images: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train the model on labeled screenshot data.
        
        Args:
            train_images: List of paths to training images
            train_labels: List of training labels
            val_images: Optional list of validation image paths
            val_labels: Optional list of validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Dictionary containing training history
        """
        # Create datasets
        train_dataset = PhishingScreenshotDataset(
            train_images, train_labels,
            transform=self.train_transform,
            image_size=self.image_size
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = None
        if val_images and val_labels:
            val_dataset = PhishingScreenshotDataset(
                val_images, val_labels,
                transform=self.eval_transform,
                image_size=self.image_size
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # Create model
        self.model = self._create_model(num_classes=2)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            train_loss /= len(train_loader)
            train_acc = 100. * train_correct / train_total
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation phase
            if val_loader:
                val_loss, val_acc = self._evaluate(val_loader, criterion)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, '
                      f'Train Acc={train_acc:.2f}%, Val Loss={val_loss:.4f}, '
                      f'Val Acc={val_acc:.2f}%')
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f'Early stopping at epoch {epoch+1}')
                        break
            else:
                print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, '
                      f'Train Acc={train_acc:.2f}%')
            
            scheduler.step()
        
        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        self.is_fitted = True
        return history
    
    def _evaluate(
        self,
        data_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Evaluate model on a data loader."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(data_loader), 100. * correct / total
    
    def predict(
        self,
        images: Union[str, List[str], Image.Image, List[Image.Image], np.ndarray]
    ) -> np.ndarray:
        """
        Predict whether screenshots show phishing or legitimate websites.
        
        Args:
            images: Single image or list of images (path, PIL Image, or numpy array)
            
        Returns:
            Array of predictions (0=legitimate, 1=phishing)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare images
        tensors = self._prepare_images(images)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tensors.to(self.device))
            _, predictions = outputs.max(1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(
        self,
        images: Union[str, List[str], Image.Image, List[Image.Image], np.ndarray]
    ) -> np.ndarray:
        """
        Predict probabilities for each class.
        
        Args:
            images: Single image or list of images
            
        Returns:
            Array of probabilities [legitimate, phishing]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        tensors = self._prepare_images(images)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tensors.to(self.device))
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def predict_with_features(
        self,
        image: Union[str, Image.Image, np.ndarray]
    ) -> VisualFeatures:
        """
        Predict with detailed feature information.
        
        Args:
            image: Single image to analyze
            
        Returns:
            VisualFeatures object with prediction details
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        tensor = self._prepare_images(image)
        
        self.model.eval()
        
        # Get features from penultimate layer
        features = None
        def hook_fn(module, input, output):
            nonlocal features
            features = output.detach()
        
        # Register hook based on model type
        if hasattr(self.model, 'fc'):
            handle = self.model.fc[0].register_forward_hook(hook_fn) if isinstance(self.model.fc, nn.Sequential) else None
        
        with torch.no_grad():
            outputs = self.model(tensor.to(self.device))
            probabilities = torch.softmax(outputs, dim=1)
            prediction = outputs.argmax(dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        if handle:
            handle.remove()
        
        return VisualFeatures(
            prediction=prediction,
            confidence=confidence,
            feature_vector=features.cpu().numpy().flatten() if features is not None else np.array([]),
            predicted_class=self.classes[prediction]
        )
    
    def _prepare_images(
        self,
        images: Union[str, List[str], Image.Image, List[Image.Image], np.ndarray]
    ) -> torch.Tensor:
        """Convert various image formats to tensor batch."""
        # Handle single image
        if isinstance(images, (str, Image.Image)):
            images = [images]
        elif isinstance(images, np.ndarray):
            if len(images.shape) == 3:
                images = [Image.fromarray(images)]
            else:
                images = [Image.fromarray(img) for img in images]
        
        tensors = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img).convert('RGB')
            elif not isinstance(img, Image.Image):
                raise ValueError(f"Unsupported image type: {type(img)}")
            else:
                img = img.convert('RGB')
            
            tensor = self.eval_transform(img)
            tensors.append(tensor)
        
        return torch.stack(tensors)
    
    def extract_features(
        self,
        images: Union[str, List[str], Image.Image, List[Image.Image]]
    ) -> np.ndarray:
        """
        Extract feature vectors from images without classification.
        
        Args:
            images: Images to extract features from
            
        Returns:
            Feature matrix of shape (n_images, feature_dim)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        tensors = self._prepare_images(images)
        
        # Create feature extractor model
        if hasattr(self.model, 'fc'):
            # ResNet-style models
            feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        else:
            # For timm models, use forward_features
            feature_extractor = self.model
        
        feature_extractor.eval()
        
        with torch.no_grad():
            if hasattr(self.model, 'forward_features'):
                features = self.model.forward_features(tensors.to(self.device))
                if len(features.shape) > 2:
                    features = features.mean(dim=[2, 3]) if len(features.shape) == 4 else features.mean(dim=1)
            else:
                features = feature_extractor(tensors.to(self.device))
                features = features.flatten(start_dim=1)
        
        return features.cpu().numpy()
    
    def evaluate(
        self,
        images: List[str],
        labels: List[int],
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            images: List of image paths
            labels: List of true labels
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix
        )
        
        predictions = []
        probabilities = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_pred = self.predict(batch_images)
            batch_proba = self.predict_proba(batch_images)[:, 1]
            predictions.extend(batch_pred)
            probabilities.extend(batch_proba)
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        labels = np.array(labels)
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions),
            'recall': recall_score(labels, predictions),
            'f1_score': f1_score(labels, predictions),
            'roc_auc': roc_auc_score(labels, probabilities),
            'confusion_matrix': confusion_matrix(labels, predictions).tolist()
        }
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save the trained model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'image_size': self.image_size,
            'classes': self.classes
        }, filepath)
    
    def load(self, filepath: Union[str, Path]) -> None:
        """Load a trained model from disk."""
        filepath = Path(filepath)
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model_type = checkpoint['model_type']
        self.image_size = checkpoint['image_size']
        self.classes = checkpoint['classes']
        
        self.model = self._create_model(num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.is_fitted = True
    
    @classmethod
    def from_pretrained(cls, filepath: Union[str, Path], device: Optional[str] = None) -> 'VisualPhishingClassifier':
        """Load a pretrained model from disk."""
        classifier = cls(device=device)
        classifier.load(filepath)
        return classifier


if __name__ == "__main__":
    # Example usage
    print("Visual Phishing Classifier")
    print("=" * 50)
    print(f"Supported models: {list(VisualPhishingClassifier.SUPPORTED_MODELS.keys())}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Create classifier
    classifier = VisualPhishingClassifier(
        model_type='resnet50',
        pretrained=True
    )
    
    print(f"\nModel type: {classifier.model_type}")
    print(f"Device: {classifier.device}")
    print(f"Image size: {classifier.image_size}x{classifier.image_size}")
