"""
Image Preprocessing Utilities
Handles DICOM loading, normalization, augmentation, and preparation
"""

import numpy as np
import cv2
from PIL import Image
import pydicom
from scipy import ndimage
from skimage import exposure, transform
import io


class CTPreprocessor:
    """Preprocessing pipeline for CT images"""
    
    def __init__(self, target_size=(512, 512), window_center=50, window_width=400):
        """
        Args:
            target_size (tuple): Target image size
            window_center (int): HU window center (50 for soft tissue, 40 for kidney)
            window_width (int): HU window width (400 for soft tissue, 300 for kidney)
        """
        self.target_size = target_size
        self.window_center = window_center
        self.window_width = window_width
    
    def load_dicom(self, dicom_path):
        """
        Load DICOM file and convert to HU units
        
        Args:
            dicom_path (str): Path to DICOM file
            
        Returns:
            np.ndarray: Image in HU units
            dict: DICOM metadata
        """
        # Read DICOM
        ds = pydicom.dcmread(dicom_path)
        
        # Get pixel array
        image = ds.pixel_array.astype(np.float32)
        
        # Convert to HU units
        intercept = ds.RescaleIntercept if hasattr(ds, 'RescaleIntercept') else 0
        slope = ds.RescaleSlope if hasattr(ds, 'RescaleSlope') else 1
        
        image_hu = image * slope + intercept
        
        # Extract metadata
        metadata = {
            'patient_id': str(ds.PatientID) if hasattr(ds, 'PatientID') else 'Unknown',
            'study_date': str(ds.StudyDate) if hasattr(ds, 'StudyDate') else 'Unknown',
            'modality': str(ds.Modality) if hasattr(ds, 'Modality') else 'CT',
            'slice_thickness': float(ds.SliceThickness) if hasattr(ds, 'SliceThickness') else 1.0,
            'pixel_spacing': ds.PixelSpacing if hasattr(ds, 'PixelSpacing') else [1.0, 1.0],
            'rows': int(ds.Rows) if hasattr(ds, 'Rows') else image.shape[0],
            'columns': int(ds.Columns) if hasattr(ds, 'Columns') else image.shape[1]
        }
        
        return image_hu, metadata
    
    def apply_window(self, image, center=None, width=None):
        """
        Apply windowing (contrast adjustment) to CT image
        
        Args:
            image (np.ndarray): Image in HU units
            center (int): Window center
            width (int): Window width
            
        Returns:
            np.ndarray: Windowed image (0-255)
        """
        if center is None:
            center = self.window_center
        if width is None:
            width = self.window_width
        
        min_hu = center - width / 2
        max_hu = center + width / 2
        
        # Clip and normalize
        windowed = np.clip(image, min_hu, max_hu)
        windowed = (windowed - min_hu) / (max_hu - min_hu) * 255
        
        return windowed.astype(np.uint8)
    
    def normalize_image(self, image, method='minmax'):
        """
        Normalize image intensities
        
        Args:
            image (np.ndarray): Input image
            method (str): Normalization method ('minmax', 'zscore', 'percentile')
            
        Returns:
            np.ndarray: Normalized image
        """
        if method == 'minmax':
            # Min-max normalization to [0, 1]
            normalized = (image - image.min()) / (image.max() - image.min() + 1e-10)
        
        elif method == 'zscore':
            # Z-score normalization
            normalized = (image - image.mean()) / (image.std() + 1e-10)
        
        elif method == 'percentile':
            # Clip outliers using percentiles
            p2, p98 = np.percentile(image, (2, 98))
            normalized = np.clip(image, p2, p98)
            normalized = (normalized - p2) / (p98 - p2 + 1e-10)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    def resize_image(self, image, target_size=None):
        """
        Resize image to target size
        
        Args:
            image (np.ndarray): Input image
            target_size (tuple): Target size (height, width)
            
        Returns:
            np.ndarray: Resized image
        """
        if target_size is None:
            target_size = self.target_size
        
        if len(image.shape) == 2:
            # 2D image
            resized = cv2.resize(image, (target_size[1], target_size[0]), 
                                interpolation=cv2.INTER_LINEAR)
        else:
            # 3D volume
            zoom_factors = [target_size[i] / image.shape[i] for i in range(3)]
            resized = ndimage.zoom(image, zoom_factors, order=1)
        
        return resized
    
    def enhance_contrast(self, image, method='clahe'):
        """
        Enhance image contrast
        
        Args:
            image (np.ndarray): Input image (0-255)
            method (str): Enhancement method ('clahe', 'hist_eq', 'adaptive')
            
        Returns:
            np.ndarray: Enhanced image
        """
        if method == 'clahe':
            # Contrast Limited Adaptive Histogram Equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image.astype(np.uint8))
        
        elif method == 'hist_eq':
            # Global histogram equalization
            enhanced = cv2.equalizeHist(image.astype(np.uint8))
        
        elif method == 'adaptive':
            # Adaptive equalization using scikit-image
            enhanced = exposure.equalize_adapthist(image)
            enhanced = (enhanced * 255).astype(np.uint8)
        
        else:
            raise ValueError(f"Unknown enhancement method: {method}")
        
        return enhanced
    
    def denoise_image(self, image, method='gaussian'):
        """
        Denoise image
        
        Args:
            image (np.ndarray): Input image
            method (str): Denoising method ('gaussian', 'bilateral', 'nlm')
            
        Returns:
            np.ndarray: Denoised image
        """
        if method == 'gaussian':
            denoised = cv2.GaussianBlur(image, (5, 5), 0)
        
        elif method == 'bilateral':
            denoised = cv2.bilateralFilter(image.astype(np.uint8), 9, 75, 75)
        
        elif method == 'nlm':
            # Non-local means denoising
            denoised = cv2.fastNlMeansDenoising(image.astype(np.uint8), None, 10, 7, 21)
        
        else:
            raise ValueError(f"Unknown denoising method: {method}")
        
        return denoised
    
    def preprocess_pipeline(self, image, is_dicom=False, enhance=True, denoise=False):
        """
        Complete preprocessing pipeline
        
        Args:
            image: Input image (path, bytes, or array)
            is_dicom (bool): Whether input is DICOM
            enhance (bool): Apply contrast enhancement
            denoise (bool): Apply denoising
            
        Returns:
            tuple: (preprocessed_image, metadata)
        """
        metadata = {}
        
        # Load image
        if isinstance(image, str):
            if is_dicom:
                image_array, metadata = self.load_dicom(image)
            else:
                image_array = np.array(Image.open(image).convert('L'))
        elif isinstance(image, bytes):
            image_array = np.array(Image.open(io.BytesIO(image)).convert('L'))
        else:
            image_array = image
        
        # Apply windowing if DICOM
        if is_dicom or image_array.min() < -100:  # Likely HU units
            image_array = self.apply_window(image_array)
        
        # Normalize
        normalized = self.normalize_image(image_array, method='percentile')
        
        # Convert to uint8
        if normalized.max() <= 1.0:
            normalized = (normalized * 255).astype(np.uint8)
        else:
            normalized = normalized.astype(np.uint8)
        
        # Denoise if requested
        if denoise:
            normalized = self.denoise_image(normalized, method='bilateral')
        
        # Enhance contrast
        if enhance:
            normalized = self.enhance_contrast(normalized, method='clahe')
        
        # Resize
        resized = self.resize_image(normalized)
        
        metadata['preprocessed_shape'] = resized.shape
        metadata['original_shape'] = image_array.shape
        
        return resized, metadata


class DataAugmentor:
    """Data augmentation for training"""
    
    def __init__(self):
        pass
    
    def random_rotate(self, image, mask, max_angle=15):
        """Random rotation"""
        angle = np.random.uniform(-max_angle, max_angle)
        
        image_rotated = ndimage.rotate(image, angle, reshape=False, order=1)
        mask_rotated = ndimage.rotate(mask, angle, reshape=False, order=0)
        
        return image_rotated, mask_rotated
    
    def random_flip(self, image, mask):
        """Random horizontal/vertical flip"""
        if np.random.rand() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        
        if np.random.rand() > 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)
        
        return image, mask
    
    def random_zoom(self, image, mask, zoom_range=(0.9, 1.1)):
        """Random zoom"""
        zoom_factor = np.random.uniform(zoom_range[0], zoom_range[1])
        
        image_zoomed = ndimage.zoom(image, zoom_factor, order=1)
        mask_zoomed = ndimage.zoom(mask, zoom_factor, order=0)
        
        # Crop or pad to original size
        image_zoomed = self._crop_or_pad(image_zoomed, image.shape)
        mask_zoomed = self._crop_or_pad(mask_zoomed, mask.shape)
        
        return image_zoomed, mask_zoomed
    
    def random_brightness(self, image, brightness_range=(-30, 30)):
        """Random brightness adjustment"""
        brightness = np.random.uniform(brightness_range[0], brightness_range[1])
        adjusted = np.clip(image + brightness, 0, 255)
        return adjusted.astype(np.uint8)
    
    def random_contrast(self, image, contrast_range=(0.8, 1.2)):
        """Random contrast adjustment"""
        contrast = np.random.uniform(contrast_range[0], contrast_range[1])
        mean = image.mean()
        adjusted = (image - mean) * contrast + mean
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    def elastic_deformation(self, image, mask, alpha=500, sigma=20):
        """Elastic deformation"""
        random_state = np.random.RandomState(None)
        shape = image.shape
        
        dx = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        image_deformed = ndimage.map_coordinates(image, indices, order=1, mode='reflect')
        mask_deformed = ndimage.map_coordinates(mask, indices, order=0, mode='reflect')
        
        return image_deformed.reshape(shape), mask_deformed.reshape(shape)
    
    def augment(self, image, mask, augment_list=['rotate', 'flip', 'brightness']):
        """
        Apply multiple augmentations
        
        Args:
            image (np.ndarray): Input image
            mask (np.ndarray): Input mask
            augment_list (list): List of augmentations to apply
            
        Returns:
            tuple: Augmented (image, mask)
        """
        aug_image = image.copy()
        aug_mask = mask.copy()
        
        for aug_type in augment_list:
            if aug_type == 'rotate':
                aug_image, aug_mask = self.random_rotate(aug_image, aug_mask)
            elif aug_type == 'flip':
                aug_image, aug_mask = self.random_flip(aug_image, aug_mask)
            elif aug_type == 'zoom':
                aug_image, aug_mask = self.random_zoom(aug_image, aug_mask)
            elif aug_type == 'brightness':
                aug_image = self.random_brightness(aug_image)
            elif aug_type == 'contrast':
                aug_image = self.random_contrast(aug_image)
            elif aug_type == 'elastic':
                aug_image, aug_mask = self.elastic_deformation(aug_image, aug_mask)
        
        return aug_image, aug_mask
    
    def _crop_or_pad(self, array, target_shape):
        """Crop or pad array to target shape"""
        current_shape = array.shape
        
        # Calculate padding/cropping
        pad_width = []
        for current, target in zip(current_shape, target_shape):
            if current < target:
                diff = target - current
                pad_width.append((diff // 2, diff - diff // 2))
            else:
                pad_width.append((0, 0))
        
        # Pad if necessary
        if any(p[0] > 0 or p[1] > 0 for p in pad_width):
            array = np.pad(array, pad_width, mode='constant')
        
        # Crop if necessary
        slices = []
        for current, target in zip(array.shape, target_shape):
            if current > target:
                start = (current - target) // 2
                slices.append(slice(start, start + target))
            else:
                slices.append(slice(None))
        
        return array[tuple(slices)]


def load_and_preprocess(image_path, is_dicom=False):
    """
    Convenience function to load and preprocess an image
    
    Args:
        image_path (str): Path to image
        is_dicom (bool): Whether image is DICOM
        
    Returns:
        tuple: (preprocessed_image, metadata)
    """
    preprocessor = CTPreprocessor()
    return preprocessor.preprocess_pipeline(image_path, is_dicom=is_dicom)


if __name__ == "__main__":
    # Test preprocessing
    print("Testing CT Preprocessor...")
    
    # Create synthetic test image
    test_image = np.random.rand(256, 256) * 255
    test_image = test_image.astype(np.uint8)
    
    preprocessor = CTPreprocessor(target_size=(512, 512))
    processed, metadata = preprocessor.preprocess_pipeline(test_image)
    
    print(f"Original shape: {test_image.shape}")
    print(f"Processed shape: {processed.shape}")
    print(f"Metadata: {metadata}")
    
    # Test augmentation
    print("\nTesting Data Augmentor...")
    augmentor = DataAugmentor()
    test_mask = np.zeros((256, 256))
    test_mask[100:150, 100:150] = 1
    
    aug_image, aug_mask = augmentor.augment(test_image, test_mask, 
                                            ['rotate', 'flip', 'brightness'])
    print(f"Augmented image shape: {aug_image.shape}")
    print(f"Augmented mask shape: {aug_mask.shape}")