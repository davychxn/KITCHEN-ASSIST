"""
Train a deep learning classifier for pan/pot states
Uses transfer learning with ResNet/EfficientNet for 4-class classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt


class PanPotDataset(Dataset):
    """Dataset for pan/pot state classification"""
    
    def __init__(self, image_paths, labels, transform=None, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment
        
        # Additional augmentation transforms
        if augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply augmentation if enabled
        if self.augment and self.aug_transform:
            image = self.aug_transform(image)
        
        # Apply main transform
        if self.transform:
            image = self.transform(image)
        
        return image, label


class StateClassifier(nn.Module):
    """
    Deep learning classifier for pan/pot states
    Uses transfer learning with pre-trained backbone
    """
    
    def __init__(self, num_classes=4, backbone='resnet50', pretrained=True):
        super(StateClassifier, self).__init__()
        
        self.backbone_name = backbone
        
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class StateClassifierTrainer:
    """Trainer for the state classifier"""
    
    def __init__(self, num_classes=4, backbone='resnet18', device=None):
        self.num_classes = num_classes
        self.backbone = backbone
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Class names
        self.class_names = ['boiling', 'normal', 'on_fire', 'smoking']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
        
        # Model
        self.model = None
        
        # Transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Using device: {self.device}")
    
    def parse_filename(self, filename):
        """
        Parse filename to extract label
        Format: kitchenware_state_number.jpg
        e.g., cooking-pot_boiling_01.jpg -> boiling
        e.g., frying-pan_on-fire_01.jpg -> on_fire
        """
        stem = Path(filename).stem
        parts = stem.split('_')
        
        if len(parts) >= 2:
            state = parts[1]
            # Handle on-fire (convert hyphen to underscore)
            if state == 'on-fire':
                state = 'on_fire'
            # Map 'normal' to 'still' for consistency
            if state == 'normal':
                state = 'normal'  # Keep as normal, will map to 'still' in class names
            return state
        return None
    
    def load_dataset(self, data_dir, test_size=0.2):
        """Load and split dataset"""
        data_dir = Path(data_dir)
        
        # Get all image files
        image_files = list(data_dir.glob('*.jpg')) + list(data_dir.glob('*.jpeg')) + list(data_dir.glob('*.png'))
        
        if len(image_files) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        print(f"Found {len(image_files)} images")
        
        # Parse labels from filenames
        image_paths = []
        labels = []
        label_counts = {}
        
        for img_path in image_files:
            state = self.parse_filename(img_path.name)
            if state and state in self.class_to_idx:
                image_paths.append(str(img_path))
                labels.append(self.class_to_idx[state])
                label_counts[state] = label_counts.get(state, 0) + 1
        
        print(f"\nDataset distribution:")
        for state, count in label_counts.items():
            print(f"  {state}: {count}")
        
        # Split into train and validation
        # For very small datasets, use all data for both training and validation
        min_samples_for_split = max(10, len(self.class_names) * 2)
        
        if len(image_paths) < min_samples_for_split:
            print(f"\nWarning: Small dataset ({len(image_paths)} images). Using all data for both training and validation.")
            print("Consider adding more labeled images for better results.")
            train_paths, val_paths = image_paths, image_paths
            train_labels, val_labels = labels, labels
        else:
            try:
                train_paths, val_paths, train_labels, val_labels = train_test_split(
                    image_paths, labels, test_size=test_size, stratify=labels, random_state=42
                )
            except ValueError:
                print(f"\nWarning: Cannot stratify with current distribution. Using all data for both training and validation.")
                train_paths, val_paths = image_paths, image_paths
                train_labels, val_labels = labels, labels
        
        print(f"\nTrain set: {len(train_paths)} images")
        print(f"Val set: {len(val_paths)} images")
        
        return train_paths, val_paths, train_labels, val_labels
    
    def create_dataloaders(self, train_paths, val_paths, train_labels, val_labels, 
                          batch_size=8, augment=True):
        """Create data loaders"""
        
        train_dataset = PanPotDataset(
            train_paths, train_labels, 
            transform=self.train_transform,
            augment=augment
        )
        
        val_dataset = PanPotDataset(
            val_paths, val_labels,
            transform=self.val_transform,
            augment=False
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, 
            shuffle=True, num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=0
        )
        
        return train_loader, val_loader
    
    def train(self, data_dir, epochs=50, batch_size=8, learning_rate=0.001, 
             save_path='best_model.pth', augment=True):
        """Train the classifier"""
        
        # Load dataset
        train_paths, val_paths, train_labels, val_labels = self.load_dataset(data_dir)
        
        # Create data loaders
        train_loader, val_loader = self.create_dataloaders(
            train_paths, val_paths, train_labels, val_labels,
            batch_size=batch_size, augment=augment
        )
        
        # Create model
        self.model = StateClassifier(
            num_classes=self.num_classes,
            backbone=self.backbone,
            pretrained=True
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        best_val_acc = 0.0
        
        print(f"\nStarting training for {epochs} epochs...")
        print("="*60)
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}]")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(save_path)
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"  ✓ Best model saved (Val Acc: {best_val_acc:.2f}%)")
        
        print("="*60)
        print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")
        
        # Plot training history
        self.plot_history(history)
        
        return history
    
    def save_model(self, path):
        """Save model and configuration"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'class_to_idx': self.class_to_idx,
            'backbone': self.backbone,
            'num_classes': self.num_classes
        }
        torch.save(checkpoint, path)
    
    def load_model(self, path):
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.class_names = checkpoint['class_names']
        self.class_to_idx = checkpoint['class_to_idx']
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        
        self.model = StateClassifier(
            num_classes=checkpoint['num_classes'],
            backbone=checkpoint['backbone'],
            pretrained=False
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {path}")
    
    def plot_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(history['train_acc'], label='Train Acc')
        ax2.plot(history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150)
        print("\nTraining history plot saved to: training_history.png")
        plt.close()
    
    def predict(self, image_path):
        """Predict state for a single image"""
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        self.model.eval()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.val_transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()
        
        # Get results
        predicted_class = self.idx_to_class[predicted_idx]
        confidence = probabilities[predicted_idx].item()
        
        # Get all class probabilities
        all_probs = {
            self.idx_to_class[i]: probabilities[i].item() 
            for i in range(len(self.class_names))
        }
        
        return predicted_class, confidence, all_probs


def main():
    """Main training function"""
    
    # Configuration
    DATA_DIR = './pics'
    EPOCHS = 100
    BATCH_SIZE = 4  # Small batch for small dataset
    LEARNING_RATE = 0.0001
    BACKBONE = 'resnet18'  # Smaller model for small dataset
    
    # Create trainer
    trainer = StateClassifierTrainer(
        num_classes=4,
        backbone=BACKBONE
    )
    
    # Train model
    print("Starting training...")
    history = trainer.train(
        data_dir=DATA_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        save_path='pan_pot_classifier.pth',
        augment=True  # Use augmentation to increase effective dataset size
    )
    
    print("\nTraining completed!")
    print("Model saved to: pan_pot_classifier.pth")


if __name__ == "__main__":
    main()
