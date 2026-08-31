"""
Tumor Classification Models
Multi-class classification for kidney tumor types:
- Clear Cell RCC
- Papillary RCC
- Chromophobe RCC
- Oncocytoma
"""

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class TumorClassifier(nn.Module):
    """
    ResNet-based classifier for tumor type identification
    
    Args:
        num_classes (int): Number of tumor types to classify
        pretrained (bool): Use ImageNet pretrained weights
        backbone (str): ResNet variant ('resnet50', 'resnet101', 'resnet152')
    """
    def __init__(self, num_classes=4, pretrained=True, backbone='resnet50'):
        super(TumorClassifier, self).__init__()
        
        # Load backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
        elif backbone == 'resnet152':
            self.backbone = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Modify first conv layer for single channel (grayscale CT)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Get feature dimension
        num_features = self.backbone.fc.in_features
        
        # Replace final fully connected layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-based classifier for better accuracy with fewer parameters
    """
    def __init__(self, num_classes=4, pretrained=True):
        super(EfficientNetClassifier, self).__init__()
        
        # Load EfficientNet-B3
        self.backbone = models.efficientnet_b3(pretrained=pretrained)
        
        # Modify first conv for grayscale
        self.backbone.features[0][0] = nn.Conv2d(1, 40, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Replace classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)


class DenseNetClassifier(nn.Module):
    """DenseNet-based classifier for feature reuse"""
    def __init__(self, num_classes=4, pretrained=True):
        super(DenseNetClassifier, self).__init__()
        
        self.backbone = models.densenet121(pretrained=pretrained)
        
        # Modify for grayscale
        self.backbone.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace classifier
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)


class AttentionClassifier(nn.Module):
    """
    Classifier with spatial attention mechanism
    Focuses on relevant tumor regions
    """
    def __init__(self, num_classes=4):
        super(AttentionClassifier, self).__init__()
        
        # Feature extractor
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove avg pool and fc
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Attention module
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Extract features
        features = self.features(x)
        
        # Apply attention
        attention_map = self.attention(features)
        attended_features = features * attention_map
        
        # Global pooling
        pooled = self.global_pool(attended_features)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Classify
        output = self.classifier(pooled)
        
        return output, attention_map


def get_classifier(model_type='resnet50', num_classes=4, pretrained=True):
    """
    Factory function to get classifier model
    
    Args:
        model_type (str): Type of classifier
        num_classes (int): Number of output classes
        pretrained (bool): Use pretrained weights
        
    Returns:
        nn.Module: Classifier model
    """
    if model_type == 'resnet50':
        return TumorClassifier(num_classes=num_classes, pretrained=pretrained, backbone='resnet50')
    elif model_type == 'resnet101':
        return TumorClassifier(num_classes=num_classes, pretrained=pretrained, backbone='resnet101')
    elif model_type == 'efficientnet':
        return EfficientNetClassifier(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'densenet':
        return DenseNetClassifier(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'attention':
        return AttentionClassifier(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test classifier
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TumorClassifier(num_classes=4)
    model = model.to(device)
    
    # Test input
    x = torch.randn(2, 1, 224, 224).to(device)
    output = model(x)
    print(f"Output shape: {output.shape}")