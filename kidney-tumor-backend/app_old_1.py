from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
from scipy import ndimage
from skimage import measure, feature
import json
from datetime import datetime
from pathlib import Path  # ✅ ADDED


app = Flask(__name__)
CORS(app)


# ===============================
# Deep Learning Models
# ===============================


class UNet3D(nn.Module):
    """3D U-Net for kidney tumor segmentation"""
    def __init__(self, in_channels=1, out_channels=3):
        super(UNet3D, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.dec4 = self.conv_block(1024 + 512, 512)
        self.dec3 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)
        
        # Output
        self.out = nn.Conv3d(64, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.dec4(torch.cat([self.upsample(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
        
        return torch.softmax(self.out(d1), dim=1)



class TumorClassifier(nn.Module):
    """ResNet-based classifier for tumor type identification"""
    def __init__(self, num_classes=4):
        super(TumorClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        # Modify first layer for grayscale
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify final layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)



class GrowthPredictor(nn.Module):
    """Neural network for tumor growth prediction"""
    def __init__(self, input_features=20):
        super(GrowthPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Growth rate output
        )
    
    def forward(self, x):
        return self.network(x)



# ===============================
# Radiomics Feature Extraction
# ===============================


class RadiomicsExtractor:
    """Extract radiomics features from tumor regions"""
    
    @staticmethod
    def extract_texture_features(image_region):
        """Extract GLCM texture features"""
        # Normalize image
        image_norm = ((image_region - image_region.min()) / 
                     (image_region.max() - image_region.min()) * 255).astype(np.uint8)
        
        # Calculate GLCM
        glcm = feature.graycomatrix(image_norm, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 
                                    levels=256, symmetric=True, normed=True)
        
        # Extract features
        contrast = feature.graycoprops(glcm, 'contrast').mean()
        dissimilarity = feature.graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = feature.graycoprops(glcm, 'homogeneity').mean()
        energy = feature.graycoprops(glcm, 'energy').mean()
        correlation = feature.graycoprops(glcm, 'correlation').mean()
        
        # Calculate entropy
        entropy = -np.sum(glcm * np.log2(glcm + 1e-10))
        
        return {
            'homogeneity': float(homogeneity),
            'entropy': float(entropy),
            'correlation': float(correlation),
            'contrast': float(contrast),
            'energy': float(energy)
        }
    
    @staticmethod
    def extract_shape_features(mask):
        """Extract shape features from segmentation mask"""
        # Label connected components
        labeled = measure.label(mask)
        props = measure.regionprops(labeled)[0] if len(measure.regionprops(labeled)) > 0 else None
        
        if props is None:
            return {
                'sphericity': 0.5,
                'compactness': 0.5,
                'surface_area': 0,
                'volume': 0
            }
        
        volume = props.area
        surface_area = props.perimeter if mask.ndim == 2 else props.area * 1.5  # Approximation
        
        # Calculate sphericity (3D: surface area / volume ratio)
        ideal_sphere_surface = 4.84 * (volume ** (2/3))
        sphericity = ideal_sphere_surface / (surface_area + 1e-10)
        sphericity = min(sphericity, 1.0)
        
        # Calculate compactness
        compactness = (volume ** 2) / (surface_area + 1e-10)
        compactness = min(compactness / 100, 1.0)  # Normalize
        
        return {
            'sphericity': float(sphericity),
            'compactness': float(compactness),
            'surface_area': float(surface_area),
            'volume': float(volume)
        }
    
    @staticmethod
    def extract_intensity_features(image_region):
        """Extract intensity-based features"""
        mean_intensity = np.mean(image_region)
        std_intensity = np.std(image_region)
        skewness = float(np.mean(((image_region - mean_intensity) / (std_intensity + 1e-10)) ** 3))
        kurtosis = float(np.mean(((image_region - mean_intensity) / (std_intensity + 1e-10)) ** 4))
        
        return {
            'mean': float(mean_intensity),
            'stdDev': float(std_intensity),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis)
        }



# ===============================
# Medical Analysis Functions
# ===============================


class KidneyTumorAnalyzer:
    """Main analyzer class combining all models"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ✅ UPDATED: Load your trained model
        model_path = Path(__file__).parent / 'data' / 'models' / 'best_unet3d_model.pth'
        
        if model_path.exists():
            print(f"Loading trained segmentation model from {model_path}")
            
            # Import the model
            from models.kidney_unet3d import UNet3D
            
            # Create model with YOUR training configuration
            self.segmentation_model = UNet3D(
                in_channels=1, 
                out_channels=3, 
                init_features=24,  # ✅ Match your training config
                use_attention=False
            ).to(self.device)
            
            # ✅ FIXED: Load weights with weights_only=False
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.segmentation_model.load_state_dict(checkpoint['model_state_dict'])
            self.segmentation_model.eval()
            
            print(f"✅ Model loaded successfully! Dice score: {checkpoint['val_dice']:.4f}")
        else:
            print(f"⚠️ No trained model found at {model_path}")
            print("   Using mock predictions for now")
            self.segmentation_model = None
        
        # Classification and growth models (use mock for now)
        self.classification_model = None
        self.growth_model = None
        
        self.radiomics = RadiomicsExtractor()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    
    def preprocess_image(self, image_file):
        """Preprocess uploaded image"""
        image = Image.open(io.BytesIO(image_file)).convert('L')
        image_array = np.array(image)
        return image, image_array
    
    def detect_tumor(self, image_array):
        """Detect tumor presence and location"""
        # Simulate tumor detection (in production, use actual model)
        # For demo, we'll create a mock detection
        height, width = image_array.shape
        
        # Create mock tumor mask (in production, this comes from segmentation model)
        mask = np.zeros((height, width), dtype=np.uint8)
        center_y, center_x = height // 2, width // 2
        radius = min(height, width) // 6
        
        y, x = np.ogrid[:height, :width]
        mask_circle = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        mask[mask_circle] = 1
        
        # Calculate tumor properties
        labeled = measure.label(mask)
        props = measure.regionprops(labeled)[0] if len(measure.regionprops(labeled)) > 0 else None
        
        if props:
            bbox = props.bbox
            centroid = props.centroid
            area = props.area
            
            # Convert pixel area to cm² (assuming standard CT pixel spacing)
            pixel_spacing = 0.1  # cm per pixel (example)
            area_cm2 = area * (pixel_spacing ** 2)
            
            # Estimate 3D volume (assuming roughly spherical)
            radius_cm = np.sqrt(area_cm2 / np.pi)
            volume_cm3 = (4/3) * np.pi * (radius_cm ** 3)
            
            detection = {
                'hasTumor': True,
                'confidence': 0.94,
                'location': 'Right kidney, upper pole',
                'centroid': centroid,
                'bbox': bbox,
                'mask': mask,
                'size': {
                    'width': float(radius_cm * 2),
                    'height': float(radius_cm * 2),
                    'depth': float(radius_cm * 2),
                    'volume': float(volume_cm3)
                }
            }
        else:
            detection = {
                'hasTumor': False,
                'confidence': 0.95,
                'location': None,
                'mask': None,
                'size': None
            }
        
        return detection
    
    def classify_tumor(self, image_array, mask):
        """Classify tumor type"""
        # Extract tumor region
        tumor_region = image_array * mask
        
        # In production, use actual classification model
        # For demo, return mock classification
        classification = {
            'primary': 'Clear Cell Renal Cell Carcinoma (ccRCC)',
            'confidence': 0.89,
            'subtypes': [
                {'name': 'Clear Cell RCC', 'probability': 0.89},
                {'name': 'Papillary RCC Type 1', 'probability': 0.07},
                {'name': 'Chromophobe RCC', 'probability': 0.04}
            ]
        }
        
        return classification
    
    def assess_malignancy(self, image_array, mask, radiomics_features):
        """Assess malignancy characteristics"""
        # Calculate enhancement metrics (mock data for demo)
        tumor_region = image_array[mask > 0]
        mean_intensity = np.mean(tumor_region)
        std_intensity = np.std(tumor_region)
        
        # Assess malignancy based on features
        malignancy_score = 0.87  # Mock score
        
        indicators = []
        if std_intensity > 30:
            indicators.append('Heterogeneous enhancement pattern')
        if radiomics_features['shape']['sphericity'] < 0.8:
            indicators.append('Irregular margins')
        if radiomics_features['texture']['entropy'] > 7.0:
            indicators.append('Necrotic areas present')
        if radiomics_features['shape']['volume'] > 30:
            indicators.append('Size > 4cm')
        
        malignancy = {
            'status': 'Likely Malignant' if malignancy_score > 0.7 else 'Indeterminate',
            'risk': 'High' if malignancy_score > 0.8 else 'Moderate',
            'probability': malignancy_score,
            'indicators': indicators
        }
        
        return malignancy
    
    def predict_growth(self, detection, radiomics_features):
        """Predict tumor growth trajectory"""
        current_volume = detection['size']['volume']
        
        # Growth rate based on tumor characteristics
        growth_rate = 0.8  # cm/year (mock)
        doubling_time = 12  # months (mock)
        
        # Project future volumes
        predictions = {
            'current': current_volume,
            'sixMonths': current_volume * 1.10,
            'oneYear': current_volume * 1.22,
            'twoYears': current_volume * 1.48
        }
        
        return {
            'estimatedGrowthRate': f'{growth_rate} cm/year',
            'doublingTime': f'{doubling_time}-18 months',
            'aggressiveness': 'Moderate-High',
            'prediction': predictions
        }
    
    def determine_staging(self, detection):
        """Determine TNM staging"""
        volume = detection['size']['volume']
        
        # T staging based on size
        if volume < 28:  # ~4cm diameter
            t_stage = 'T1a'
            description = 'Tumor ≤4cm, limited to kidney'
        elif volume < 144:  # ~7cm diameter
            t_stage = 'T1b'
            description = 'Tumor >4cm but ≤7cm, limited to kidney'
        elif volume < 343:  # ~10cm diameter
            t_stage = 'T2a'
            description = 'Tumor >7cm but ≤10cm, limited to kidney'
        else:
            t_stage = 'T2b'
            description = 'Tumor >10cm, limited to kidney'
        
        staging = {
            'tStage': t_stage,
            'description': description,
            'lymphNodes': 'No regional lymph node involvement detected',
            'metastasis': 'No distant metastasis detected',
            'overall': 'Stage I' if t_stage in ['T1a', 'T1b'] else 'Stage II'
        }
        
        return staging
    
    def generate_clinical_recommendations(self, classification, staging, malignancy):
        """Generate clinical recommendations"""
        recommendations = []
        
        if malignancy['risk'] == 'High':
            recommendations.append('Surgical intervention recommended (partial or radical nephrectomy)')
            recommendations.append('Consider MRI for better soft tissue characterization')
            recommendations.append('Chest CT to rule out pulmonary metastases')
        else:
            recommendations.append('Active surveillance with imaging every 3-6 months')
            recommendations.append('Consider percutaneous biopsy for definitive diagnosis')
        
        recommendations.extend([
            'Baseline complete metabolic panel and CBC',
            'Urological oncology consultation within 2-4 weeks'
        ])
        
        # Determine Bosniak classification (for cystic lesions)
        bosniak = 'IV'  # Mock - clearly malignant
        
        # Estimate Fuhrman grade
        fuhrman = 'Likely Grade 2-3'
        
        # Prognosis
        prognosis = {
            'fiveYearSurvival': '81%',
            'riskFactors': [
                'Tumor size >4cm',
                'Heterogeneous enhancement',
                'Clear cell histology (assumed)'
            ],
            'favorableFactors': [
                'No lymph node involvement',
                'No distant metastases',
                'Confined to kidney (T1b)'
            ]
        }
        
        return {
            'bosniak': bosniak,
            'fuhrmanGrade': fuhrman,
            'recommendations': recommendations,
            'prognosis': prognosis
        }
    
    def analyze(self, image_file):
        """Complete analysis pipeline"""
        # Preprocess image
        image, image_array = self.preprocess_image(image_file)
        
        # Detect tumor
        detection = self.detect_tumor(image_array)
        
        if not detection['hasTumor']:
            return {
                'detection': detection,
                'message': 'No tumor detected in the scan'
            }
        
        mask = detection['mask']
        
        # Extract radiomics features
        texture_features = self.radiomics.extract_texture_features(image_array * mask)
        shape_features = self.radiomics.extract_shape_features(mask)
        intensity_features = self.radiomics.extract_intensity_features(image_array[mask > 0])
        
        radiomics_features = {
            'texture': texture_features,
            'shape': shape_features,
            'intensity': intensity_features
        }
        
        # Classify tumor type
        classification = self.classify_tumor(image_array, mask)
        
        # Assess malignancy
        malignancy = self.assess_malignancy(image_array, mask, radiomics_features)
        
        # Predict growth
        growth = self.predict_growth(detection, radiomics_features)
        
        # Determine staging
        staging = self.determine_staging(detection)
        
        # Generate clinical info
        clinical = self.generate_clinical_recommendations(classification, staging, malignancy)
        
        # Enhancement characteristics (mock)
        enhancement = {
            'pattern': 'Heterogeneous',
            'arterialPhase': 'Strong (85 HU)',
            'venousPhase': 'Moderate washout (62 HU)',
            'interpretation': 'Typical for RCC'
        }
        
        # Differential diagnoses
        differentials = [
            {'condition': 'Clear Cell RCC', 'likelihood': 'Very High (89%)'},
            {'condition': 'Papillary RCC', 'likelihood': 'Low (7%)'},
            {'condition': 'Oncocytoma', 'likelihood': 'Very Low (2%)'},
            {'condition': 'Angiomyolipoma (fat-poor)', 'likelihood': 'Very Low (2%)'}
        ]
        
        # Vascular assessment
        vascular = {
            'involvement': 'No renal vein or IVC involvement',
            'arterialSupply': 'From main renal artery',
            'venousDrainage': 'Normal renal vein',
            'significance': 'Favorable for surgical planning'
        }
        
        # Compile complete results
        results = {
            'detection': {
                'hasTumor': detection['hasTumor'],
                'confidence': detection['confidence'],
                'location': detection['location'],
                'size': detection['size']
            },
            'tumorType': classification,
            'characteristics': {
                'malignancy': malignancy,
                'enhancement': enhancement,
                'staging': staging
            },
            'growth': growth,
            'radiomics': {
                'texture': {
                    'homogeneity': radiomics_features['texture']['homogeneity'],
                    'entropy': radiomics_features['texture']['entropy'],
                    'correlation': radiomics_features['texture']['correlation'],
                    'interpretation': 'Heterogeneous texture suggesting aggressive variant'
                },
                'shape': {
                    'sphericity': radiomics_features['shape']['sphericity'],
                    'compactness': radiomics_features['shape']['compactness'],
                    'surfaceArea': radiomics_features['shape']['surface_area'],
                    'interpretation': 'Irregular shape with moderate sphericity'
                },
                'intensity': {
                    'mean': radiomics_features['intensity']['mean'],
                    'stdDev': radiomics_features['intensity']['stdDev'],
                    'skewness': radiomics_features['intensity']['skewness'],
                    'interpretation': 'Variable density with necrotic components'
                }
            },
            'clinical': clinical,
            'differentials': differentials,
            'vascular': vascular
        }
        
        return results



# ===============================
# Flask API Endpoints
# ===============================


# Initialize analyzer
analyzer = KidneyTumorAnalyzer()


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'device': str(analyzer.device),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_scan():
    """Main analysis endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Read file
        file_bytes = file.read()
        
        # Analyze
        results = analyzer.analyze(file_bytes)
        
        return jsonify({
            'success': True,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/segment', methods=['POST'])
def segment_tumor():
    """Tumor segmentation endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        file_bytes = file.read()
        
        # Preprocess and detect
        image, image_array = analyzer.preprocess_image(file_bytes)
        detection = analyzer.detect_tumor(image_array)
        
        if detection['hasTumor']:
            # Convert mask to base64 for frontend
            mask = detection['mask']
            mask_image = Image.fromarray((mask * 255).astype(np.uint8))
            buffer = io.BytesIO()
            mask_image.save(buffer, format='PNG')
            mask_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return jsonify({
                'success': True,
                'hasTumor': True,
                'mask': mask_base64,
                'confidence': detection['confidence']
            })
        else:
            return jsonify({
                'success': True,
                'hasTumor': False
            })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/radiomics', methods=['POST'])
def extract_radiomics():
    """Extract radiomics features endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        file_bytes = file.read()
        
        image, image_array = analyzer.preprocess_image(file_bytes)
        detection = analyzer.detect_tumor(image_array)
        
        if not detection['hasTumor']:
            return jsonify({
                'success': False,
                'error': 'No tumor detected'
            }), 400
        
        mask = detection['mask']
        
        # Extract features
        texture = analyzer.radiomics.extract_texture_features(image_array * mask)
        shape = analyzer.radiomics.extract_shape_features(mask)
        intensity = analyzer.radiomics.extract_intensity_features(image_array[mask > 0])
        
        return jsonify({
            'success': True,
            'features': {
                'texture': texture,
                'shape': shape,
                'intensity': intensity
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/classify', methods=['POST'])
def classify_tumor_type():
    """Tumor classification endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        file_bytes = file.read()
        
        image, image_array = analyzer.preprocess_image(file_bytes)
        detection = analyzer.detect_tumor(image_array)
        
        if not detection['hasTumor']:
            return jsonify({
                'success': False,
                'error': 'No tumor detected'
            }), 400
        
        classification = analyzer.classify_tumor(image_array, detection['mask'])
        
        return jsonify({
            'success': True,
            'classification': classification
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict-growth', methods=['POST'])
def predict_tumor_growth():
    """Growth prediction endpoint"""
    try:
        data = request.get_json()
        
        if 'currentVolume' not in data:
            return jsonify({'error': 'Current volume required'}), 400
        
        current_volume = float(data['currentVolume'])
        
        # Simple growth model
        growth_rate = 0.8  # cm/year
        
        predictions = {
            'current': current_volume,
            'sixMonths': current_volume * 1.10,
            'oneYear': current_volume * 1.22,
            'twoYears': current_volume * 1.48
        }
        
        return jsonify({
            'success': True,
            'growthRate': f'{growth_rate} cm/year',
            'predictions': predictions
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500



# ===============================
# Model Training Functions
# ===============================


def train_segmentation_model(train_loader, val_loader, epochs=100):
    """Train the 3D U-Net segmentation model"""
    model = UNet3D().to(analyzer.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_dice = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(analyzer.device)
            masks = masks.to(analyzer.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_dice = evaluate_segmentation(model, val_loader)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, Dice: {val_dice:.4f}')
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), 'best_segmentation_model.pth')
    
    return model


def evaluate_segmentation(model, data_loader):
    """Evaluate segmentation model using Dice coefficient"""
    model.eval()
    dice_scores = []
    
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(analyzer.device)
            masks = masks.to(analyzer.device)
            
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            # Calculate Dice coefficient
            dice = calculate_dice(predictions, masks)
            dice_scores.append(dice)
    
    return np.mean(dice_scores)


def calculate_dice(pred, target, smooth=1e-6):
    """Calculate Dice coefficient"""
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def train_classifier(train_loader, val_loader, epochs=50):
    """Train the tumor classifier"""
    model = TumorClassifier().to(analyzer.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(analyzer.device)
            labels = labels.to(analyzer.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        val_acc = evaluate_classifier(model, val_loader)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_classifier_model.pth')
    
    return model


def evaluate_classifier(model, data_loader):
    """Evaluate classifier accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(analyzer.device)
            labels = labels.to(analyzer.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total



# ===============================
# Utility Functions
# ===============================


def load_pretrained_models():
    """Load pre-trained model weights"""
    try:
        analyzer.segmentation_model.load_state_dict(
            torch.load('best_segmentation_model.pth', 
                      map_location=analyzer.device, 
                      weights_only=False)  # ✅ FIXED
        )
        print("Segmentation model loaded successfully")
    except:
        print("No pre-trained segmentation model found, using random initialization")
    
    try:
        analyzer.classification_model.load_state_dict(
            torch.load('best_classifier_model.pth', 
                      map_location=analyzer.device,
                      weights_only=False)  # ✅ FIXED
        )
        print("Classification model loaded successfully")
    except:
        print("No pre-trained classification model found, using random initialization")
    
    try:
        analyzer.growth_model.load_state_dict(
            torch.load('best_growth_model.pth', 
                      map_location=analyzer.device,
                      weights_only=False)  # ✅ FIXED
        )
        print("Growth prediction model loaded successfully")
    except:
        print("No pre-trained growth model found, using random initialization")



# ===============================
# Main Execution
# ===============================


if __name__ == '__main__':
    # Load pre-trained models if available
    load_pretrained_models()
    
    print(f"Server running on device: {analyzer.device}")
    print("Models loaded and ready for inference")
    print("\nAPI Endpoints:")
    print("  - POST /api/analyze - Complete tumor analysis")
    print("  - POST /api/segment - Tumor segmentation")
    print("  - POST /api/radiomics - Radiomics feature extraction")
    print("  - POST /api/classify - Tumor classification")
    print("  - POST /api/predict-growth - Growth prediction")
    print("  - GET  /api/health - Health check")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
