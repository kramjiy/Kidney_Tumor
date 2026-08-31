"""
Radiomics Feature Extraction
Comprehensive feature extraction for tumor characterization
"""

import numpy as np
from scipy import ndimage
from skimage import measure, feature
from skimage.feature import graycomatrix, graycoprops
import warnings
warnings.filterwarnings('ignore')


class RadiomicsExtractor:
    """Complete radiomics feature extraction pipeline"""
    
    def __init__(self, bin_width=25):
        self.bin_width = bin_width
    
    def extract_all_features(self, image, mask):
        """
        Extract all radiomics features
        
        Args:
            image (np.ndarray): CT image (2D or 3D)
            mask (np.ndarray): Binary tumor mask
            
        Returns:
            dict: All radiomics features
        """
        # Extract tumor region
        tumor_region = image[mask > 0]
        
        if len(tumor_region) == 0:
            return self._get_default_features()
        
        # Extract different feature categories
        texture_features = self.extract_texture_features(image, mask)
        shape_features = self.extract_shape_features(mask)
        intensity_features = self.extract_intensity_features(tumor_region)
        
        return {
            'texture': texture_features,
            'shape': shape_features,
            'intensity': intensity_features
        }
    
    def extract_texture_features(self, image, mask):
        """Extract GLCM texture features"""
        tumor_region = image * mask
        tumor_roi = tumor_region[mask > 0]
        
        if len(tumor_roi) == 0:
            return self._get_default_texture_features()
        
        normalized = ((tumor_roi - tumor_roi.min()) / 
                     (tumor_roi.max() - tumor_roi.min() + 1e-10) * 255).astype(np.uint8)
        
        if image.ndim == 2:
            roi_image = (tumor_region * 255 / (tumor_region.max() + 1e-10)).astype(np.uint8)
        else:
            middle_slice = image.shape[0] // 2
            roi_image = (tumor_region[middle_slice] * 255 / 
                        (tumor_region[middle_slice].max() + 1e-10)).astype(np.uint8)
        
        try:
            glcm = graycomatrix(roi_image, [1, 2], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                               levels=256, symmetric=True, normed=True)
            
            contrast = graycoprops(glcm, 'contrast').mean()
            dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            energy = graycoprops(glcm, 'energy').mean()
            correlation = graycoprops(glcm, 'correlation').mean()
            
            glcm_norm = glcm / (glcm.sum() + 1e-10)
            entropy = -np.sum(glcm_norm * np.log2(glcm_norm + 1e-10))
            
        except:
            return self._get_default_texture_features()
        
        return {
            'contrast': float(contrast),
            'dissimilarity': float(dissimilarity),
            'homogeneity': float(homogeneity),
            'energy': float(energy),
            'correlation': float(correlation),
            'entropy': float(entropy)
        }
    
    def extract_shape_features(self, mask):
        """Extract 3D shape features"""
        if mask.sum() == 0:
            return self._get_default_shape_features()
        
        labeled = measure.label(mask)
        props = measure.regionprops(labeled)
        
        if len(props) == 0:
            return self._get_default_shape_features()
        
        props = sorted(props, key=lambda x: x.area, reverse=True)[0]
        
        volume = props.area
        
        if mask.ndim == 3:
            try:
                verts, faces, normals, values = measure.marching_cubes(mask.astype(float), level=0.5)
                surface_area = measure.mesh_surface_area(verts, faces)
            except:
                surface_area = volume * 1.5
        else:
            surface_area = props.perimeter
        
        if mask.ndim == 3:
            sphere_surface = np.pi ** (1/3) * (6 * volume) ** (2/3)
        else:
            sphere_surface = 2 * np.pi * np.sqrt(volume / np.pi)
        
        sphericity = sphere_surface / (surface_area + 1e-10)
        sphericity = min(sphericity, 1.0)
        
        if mask.ndim == 3:
            compactness = (volume ** 2) / (surface_area ** 3 + 1e-10)
        else:
            compactness = (4 * np.pi * volume) / (surface_area ** 2 + 1e-10)
        
        return {
            'volume': float(volume),
            'surface_area': float(surface_area),
            'sphericity': float(sphericity),
            'compactness': float(compactness)
        }
    
    def extract_intensity_features(self, tumor_region):
        """Extract intensity-based first-order statistics"""
        if len(tumor_region) == 0:
            return self._get_default_intensity_features()
        
        mean_intensity = np.mean(tumor_region)
        std_intensity = np.std(tumor_region)
        
        skewness = self._calculate_skewness(tumor_region, mean_intensity, std_intensity)
        kurtosis = self._calculate_kurtosis(tumor_region, mean_intensity, std_intensity)
        
        return {
            'mean': float(mean_intensity),
            'stdDev': float(std_intensity),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis)
        }
    
    def _calculate_skewness(self, data, mean, std):
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data, mean, std):
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _get_default_features(self):
        return {
            'texture': self._get_default_texture_features(),
            'shape': self._get_default_shape_features(),
            'intensity': self._get_default_intensity_features()
        }
    
    def _get_default_texture_features(self):
        return {
            'contrast': 0.0,
            'dissimilarity': 0.0,
            'homogeneity': 0.5,
            'energy': 0.5,
            'correlation': 0.5,
            'entropy': 5.0
        }
    
    def _get_default_shape_features(self):
        return {
            'volume': 0.0,
            'surface_area': 0.0,
            'sphericity': 0.5,
            'compactness': 0.5
        }
    
    def _get_default_intensity_features(self):
        return {
            'mean': 0.0,
            'stdDev': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0
        }


if __name__ == "__main__":
    extractor = RadiomicsExtractor()
    
    # Test
    image = np.random.rand(100, 100) * 255
    mask = np.zeros((100, 100))
    mask[40:60, 40:60] = 1
    
    features = extractor.extract_all_features(image, mask)
    print("Radiomics features extracted successfully!")