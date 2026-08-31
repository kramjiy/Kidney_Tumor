"""
Postprocessing Utilities
Refine segmentation results, format outputs, generate visualizations
"""

import numpy as np
from scipy import ndimage
from skimage import measure, morphology
import cv2


class SegmentationPostprocessor:
    """Postprocessing for segmentation masks"""
    
    def __init__(self, min_size=100):
        """
        Args:
            min_size (int): Minimum component size to keep (in pixels)
        """
        self.min_size = min_size
    
    def remove_small_components(self, mask, min_size=None):
        """
        Remove small connected components
        
        Args:
            mask (np.ndarray): Binary mask
            min_size (int): Minimum size threshold
            
        Returns:
            np.ndarray: Cleaned mask
        """
        if min_size is None:
            min_size = self.min_size
        
        # Label connected components
        labeled = measure.label(mask)
        
        # Remove small components
        cleaned = morphology.remove_small_objects(labeled, min_size=min_size)
        
        return (cleaned > 0).astype(np.uint8)
    
    def fill_holes(self, mask):
        """
        Fill holes in binary mask
        
        Args:
            mask (np.ndarray): Binary mask
            
        Returns:
            np.ndarray: Filled mask
        """
        filled = ndimage.binary_fill_holes(mask).astype(np.uint8)
        return filled
    
    def smooth_contours(self, mask, kernel_size=5):
        """
        Smooth mask contours using morphological operations
        
        Args:
            mask (np.ndarray): Binary mask
            kernel_size (int): Size of morphological kernel
            
        Returns:
            np.ndarray: Smoothed mask
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Closing to fill small holes
        smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Opening to remove small protrusions
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
        
        return smoothed
    
    def largest_component(self, mask):
        """
        Keep only the largest connected component
        
        Args:
            mask (np.ndarray): Binary mask
            
        Returns:
            np.ndarray: Mask with largest component only
        """
        labeled = measure.label(mask)
        
        if labeled.max() == 0:
            return mask
        
        # Find largest component
        largest = (labeled == np.argmax(np.bincount(labeled.flat)[1:]) + 1)
        
        return largest.astype(np.uint8)
    
    def refine_mask(self, mask, remove_small=True, fill_holes_flag=True, 
                   smooth=True, keep_largest=True):
        """
        Complete mask refinement pipeline
        
        Args:
            mask (np.ndarray): Binary mask
            remove_small (bool): Remove small components
            fill_holes_flag (bool): Fill holes
            smooth (bool): Smooth contours
            keep_largest (bool): Keep only largest component
            
        Returns:
            np.ndarray: Refined mask
        """
        refined = mask.copy()
        
        if remove_small:
            refined = self.remove_small_components(refined)
        
        if fill_holes_flag:
            refined = self.fill_holes(refined)
        
        if smooth:
            refined = self.smooth_contours(refined)
        
        if keep_largest:
            refined = self.largest_component(refined)
        
        return refined


class ResultFormatter:
    """Format analysis results for frontend"""
    
    @staticmethod
    def format_detection_result(mask, confidence, location):
        """Format tumor detection result"""
        has_tumor = mask is not None and mask.sum() > 0
        
        if has_tumor:
            # Calculate tumor properties
            props = measure.regionprops(measure.label(mask))[0]
            
            return {
                'hasTumor': True,
                'confidence': float(confidence),
                'location': location,
                'size': {
                    'area_pixels': int(props.area),
                    'bbox': [int(x) for x in props.bbox],
                    'centroid': [float(x) for x in props.centroid]
                }
            }
        else:
            return {
                'hasTumor': False,
                'confidence': float(confidence),
                'location': None,
                'size': None
            }
    
    @staticmethod
    def format_classification_result(predictions, class_names):
        """Format classification result"""
        pred_probs = predictions.softmax(dim=1)[0].cpu().numpy()
        pred_class = int(predictions.argmax(dim=1)[0])
        
        subtypes = [
            {'name': class_names[i], 'probability': float(pred_probs[i])}
            for i in range(len(class_names))
        ]
        
        # Sort by probability
        subtypes = sorted(subtypes, key=lambda x: x['probability'], reverse=True)
        
        return {
            'primary': class_names[pred_class],
            'confidence': float(pred_probs[pred_class]),
            'subtypes': subtypes
        }
    
    @staticmethod
    def format_growth_prediction(growth_params, current_volume):
        """Format growth prediction result"""
        growth_rate = float(growth_params[0])
        doubling_time = float(growth_params[1])
        aggressiveness = float(growth_params[2])
        
        # Calculate future volumes
        predictions = {
            'current': float(current_volume),
            'sixMonths': float(current_volume * (1 + growth_rate * 0.5)),
            'oneYear': float(current_volume * (1 + growth_rate)),
            'twoYears': float(current_volume * (1 + growth_rate * 2))
        }
        
        # Classify aggressiveness
        if aggressiveness > 0.7:
            aggr_class = 'High'
        elif aggressiveness > 0.4:
            aggr_class = 'Moderate-High'
        else:
            aggr_class = 'Low-Moderate'
        
        return {
            'estimatedGrowthRate': f'{growth_rate:.2f} cm/year',
            'doublingTime': f'{doubling_time:.1f} months',
            'aggressiveness': aggr_class,
            'prediction': predictions
        }
    
    @staticmethod
    def format_complete_report(detection, classification, radiomics, 
                               growth, staging, clinical):
        """Format complete analysis report"""
        return {
            'detection': detection,
            'tumorType': classification,
            'radiomics': radiomics,
            'growth': growth,
            'staging': staging,
            'clinical': clinical,
            'timestamp': np.datetime64('now').astype(str)
        }


class Visualizer:
    """Generate visualizations for results"""
    
    @staticmethod
    def overlay_mask(image, mask, alpha=0.5, color=(255, 0, 0)):
        """
        Overlay segmentation mask on image
        
        Args:
            image (np.ndarray): Grayscale image
            mask (np.ndarray): Binary mask
            alpha (float): Transparency of overlay
            color (tuple): RGB color for mask
            
        Returns:
            np.ndarray: Image with overlay
        """
        # Convert grayscale to RGB
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image.copy()
        
        # Create colored mask
        mask_rgb = np.zeros_like(image_rgb)
        mask_rgb[mask > 0] = color
        
        # Blend
        overlay = cv2.addWeighted(image_rgb, 1 - alpha, mask_rgb, alpha, 0)
        
        return overlay
    
    @staticmethod
    def draw_contours(image, mask, color=(0, 255, 0), thickness=2):
        """
        Draw contours on image
        
        Args:
            image (np.ndarray): Input image
            mask (np.ndarray): Binary mask
            color (tuple): RGB color for contours
            thickness (int): Contour thickness
            
        Returns:
            np.ndarray: Image with contours
        """
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image.copy()
        
        # Find contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours
        cv2.drawContours(image_rgb, contours, -1, color, thickness)
        
        return image_rgb
    
    @staticmethod
    def create_comparison_view(original, preprocessed, segmentation):
        """
        Create side-by-side comparison view
        
        Args:
            original (np.ndarray): Original image
            preprocessed (np.ndarray): Preprocessed image
            segmentation (np.ndarray): Segmentation mask
            
        Returns:
            np.ndarray: Concatenated comparison image
        """
        # Ensure all same size
        h, w = original.shape[:2]
        preprocessed = cv2.resize(preprocessed, (w, h))
        segmentation = cv2.resize(segmentation, (w, h))
        
        # Convert to RGB
        if len(original.shape) == 2:
            original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
        else:
            original_rgb = original
        
        if len(preprocessed.shape) == 2:
            preprocessed_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)
        else:
            preprocessed_rgb = preprocessed
        
        # Create segmentation overlay
        seg_overlay = Visualizer.overlay_mask(preprocessed, segmentation)
        
        # Concatenate horizontally
        comparison = np.hstack([original_rgb, preprocessed_rgb, seg_overlay])
        
        return comparison
    
    @staticmethod
    def add_text_annotation(image, text, position=(10, 30), 
                           font_scale=0.7, color=(255, 255, 255), thickness=2):
        """Add text annotation to image"""
        annotated = image.copy()
        cv2.putText(annotated, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, color, thickness)
        return annotated


def postprocess_segmentation(mask, image=None):
    """
    Convenience function for segmentation postprocessing
    
    Args:
        mask (np.ndarray): Raw segmentation mask
        image (np.ndarray): Original image (optional, for visualization)
        
    Returns:
        dict: Postprocessed results
    """
    processor = SegmentationPostprocessor()
    
    # Refine mask
    refined_mask = processor.refine_mask(mask)
    
    # Calculate properties
    if refined_mask.sum() > 0:
        props = measure.regionprops(measure.label(refined_mask))[0]
        
        results = {
            'mask': refined_mask,
            'area': int(props.area),
            'bbox': [int(x) for x in props.bbox],
            'centroid': [float(x) for x in props.centroid],
            'perimeter': float(props.perimeter) if hasattr(props, 'perimeter') else 0
        }
        
        # Create visualization if image provided
        if image is not None:
            visualizer = Visualizer()
            results['visualization'] = visualizer.overlay_mask(image, refined_mask)
        
    else:
        results = {
            'mask': refined_mask,
            'area': 0,
            'bbox': None,
            'centroid': None,
            'perimeter': 0
        }
    
    return results


if __name__ == "__main__":
    # Test postprocessing
    print("Testing Segmentation Postprocessor...")
    
    # Create noisy test mask
    test_mask = np.zeros((256, 256), dtype=np.uint8)
    test_mask[100:150, 100:150] = 1
    
    # Add noise
    noise = np.random.rand(256, 256) > 0.98
    test_mask[noise] = 1
    
    # Add holes
    test_mask[120:130, 120:130] = 0
    
    processor = SegmentationPostprocessor(min_size=50)
    refined = processor.refine_mask(test_mask)
    
    print(f"Original mask pixels: {test_mask.sum()}")
    print(f"Refined mask pixels: {refined.sum()}")
    
    # Test visualization
    print("\nTesting Visualizer...")
    test_image = np.random.rand(256, 256) * 255
    test_image = test_image.astype(np.uint8)
    
    visualizer = Visualizer()
    overlay = visualizer.overlay_mask(test_image, refined)
    contours = visualizer.draw_contours(test_image, refined)
    
    print(f"Overlay shape: {overlay.shape}")
    print(f"Contours shape: {contours.shape}")