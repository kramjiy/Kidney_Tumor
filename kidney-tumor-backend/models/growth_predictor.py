"""
Tumor Growth Prediction Models
Predicts growth rate, doubling time, and future tumor volumes
"""

import torch
import torch.nn as nn
import numpy as np


class GrowthPredictor(nn.Module):
    """
    Neural network for tumor growth prediction
    
    Input features:
    - Current tumor volume
    - Radiomics features (texture, shape, intensity)
    - Enhancement characteristics
    - Patient demographics (age, gender)
    - Tumor location
    
    Outputs:
    - Growth rate (cm/year)
    - Doubling time (months)
    - Aggressiveness score (0-1)
    """
    def __init__(self, input_features=25, hidden_size=128):
        super(GrowthPredictor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size // 4, 3)  # [growth_rate, doubling_time, aggressiveness]
        )
        
    def forward(self, x):
        return self.network(x)


class BayesianGrowthPredictor(nn.Module):
    """
    Bayesian neural network for growth prediction with uncertainty
    Provides confidence intervals for predictions
    """
    def __init__(self, input_features=25, hidden_size=128, num_samples=10):
        super(BayesianGrowthPredictor, self).__init__()
        
        self.num_samples = num_samples
        
        # Variational layers
        self.fc1_mean = nn.Linear(input_features, hidden_size)
        self.fc1_logvar = nn.Linear(input_features, hidden_size)
        
        self.fc2_mean = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2_logvar = nn.Linear(hidden_size, hidden_size // 2)
        
        self.fc3 = nn.Linear(hidden_size // 2, 3)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x, return_uncertainty=False):
        if return_uncertainty:
            # Multiple forward passes for uncertainty estimation
            predictions = []
            for _ in range(self.num_samples):
                pred = self._forward_once(x)
                predictions.append(pred)
            
            predictions = torch.stack(predictions)
            mean = predictions.mean(dim=0)
            std = predictions.std(dim=0)
            
            return mean, std
        else:
            return self._forward_once(x)
    
    def _forward_once(self, x):
        # Layer 1
        h1_mean = self.fc1_mean(x)
        h1_logvar = self.fc1_logvar(x)
        h1 = self.reparameterize(h1_mean, h1_logvar)
        h1 = self.relu(h1)
        h1 = self.dropout(h1)
        
        # Layer 2
        h2_mean = self.fc2_mean(h1)
        h2_logvar = self.fc2_logvar(h1)
        h2 = self.reparameterize(h2_mean, h2_logvar)
        h2 = self.relu(h2)
        h2 = self.dropout(h2)
        
        # Output
        output = self.fc3(h2)
        
        return output


def extract_growth_features(radiomics, tumor_info, patient_info=None):
    """
    Extract features for growth prediction model
    
    Args:
        radiomics (dict): Radiomics features
        tumor_info (dict): Tumor information (size, location, etc.)
        patient_info (dict): Patient demographics (optional)
        
    Returns:
        torch.Tensor: Feature vector for growth prediction
    """
    features = []
    
    # Current tumor volume
    features.append(tumor_info.get('volume', 0))
    
    # Radiomics - Texture
    features.append(radiomics['texture'].get('homogeneity', 0.5))
    features.append(radiomics['texture'].get('entropy', 5.0))
    features.append(radiomics['texture'].get('correlation', 0.5))
    features.append(radiomics['texture'].get('contrast', 100))
    features.append(radiomics['texture'].get('energy', 0.5))
    
    # Radiomics - Shape
    features.append(radiomics['shape'].get('sphericity', 0.5))
    features.append(radiomics['shape'].get('compactness', 0.5))
    features.append(radiomics['shape'].get('surface_area', 50))
    features.append(radiomics['shape'].get('volume', 30))
    
    # Radiomics - Intensity
    features.append(radiomics['intensity'].get('mean', 50))
    features.append(radiomics['intensity'].get('stdDev', 20))
    features.append(radiomics['intensity'].get('skewness', 0))
    features.append(radiomics['intensity'].get('kurtosis', 3))
    
    # Tumor characteristics
    features.append(1 if tumor_info.get('enhancement') == 'strong' else 0)
    features.append(1 if tumor_info.get('necrosis', False) else 0)
    features.append(1 if tumor_info.get('calcification', False) else 0)
    
    # Location encoding (upper=0, middle=1, lower=2)
    location_map = {'upper': 0, 'middle': 1, 'lower': 2}
    features.append(location_map.get(tumor_info.get('location', 'middle'), 1))
    
    # Patient info (if available)
    if patient_info:
        features.append(patient_info.get('age', 60) / 100)  # Normalized
        features.append(1 if patient_info.get('gender') == 'male' else 0)
        features.append(1 if patient_info.get('smoker', False) else 0)
        features.append(1 if patient_info.get('hypertension', False) else 0)
        features.append(1 if patient_info.get('diabetes', False) else 0)
    else:
        features.extend([0.6, 0, 0, 0, 0])  # Default values
    
    # Additional clinical factors
    features.append(tumor_info.get('growth_rate_history', 0.5))
    
    return torch.tensor(features, dtype=torch.float32)


def predict_growth_trajectory(model, current_volume, features, time_points=[0, 6, 12, 24]):
    """
    Predict tumor volume at future time points
    
    Args:
        model: Trained growth prediction model
        current_volume (float): Current tumor volume (cm³)
        features (torch.Tensor): Feature vector
        time_points (list): Time points in months
        
    Returns:
        dict: Predicted volumes and growth metrics
    """
    model.eval()
    
    with torch.no_grad():
        # Predict growth parameters
        predictions = model(features.unsqueeze(0))
        
        growth_rate = predictions[0, 0].item()  # cm/year
        doubling_time = predictions[0, 1].item()  # months
        aggressiveness = predictions[0, 2].item()  # 0-1 score
        
        # Calculate volumes at future time points
        volumes = {}
        for t in time_points:
            # Exponential growth model
            growth_factor = np.exp(growth_rate * t / 12)
            volumes[f'{t}_months'] = current_volume * growth_factor
        
        return {
            'growth_rate': f'{growth_rate:.2f} cm/year',
            'doubling_time': f'{doubling_time:.1f} months',
            'aggressiveness': 'High' if aggressiveness > 0.7 else 'Moderate' if aggressiveness > 0.4 else 'Low',
            'predicted_volumes': volumes
        }


if __name__ == "__main__":
    # Test growth predictor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GrowthPredictor(input_features=25).to(device)
    
    # Test input
    x = torch.randn(2, 25).to(device)
    output = model(x)
    print(f"Output shape: {output.shape}")