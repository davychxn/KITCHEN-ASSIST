"""
Pan/Pot Detection and Classification System
Detects pans/pots on stoves and classifies their state into: boiling, still, smoking, on_fire
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional

class PanPotDetector:
    def __init__(self, model_name='yolov8n.pt', target_size=(640, 640), 
                 classifier_path: Optional[str] = None, use_deep_learning=True):
        """
        Initialize the detector
        
        Args:
            model_name: YOLO model to use (yolov8n.pt, yolov8s.pt, etc.)
            target_size: Target image size for YOLO (width, height)
            classifier_path: Path to trained classifier model (optional)
            use_deep_learning: Use deep learning classifier if available
        """
        self.model = YOLO(model_name)
        self.target_size = target_size
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Load trained classifier if available
        self.classifier = None
        self.use_deep_learning = use_deep_learning
        
        if use_deep_learning and classifier_path and Path(classifier_path).exists():
            try:
                from train_classifier import StateClassifierTrainer
                self.classifier = StateClassifierTrainer()
                self.classifier.load_model(classifier_path)
                print(f"Loaded trained classifier from {classifier_path}")
            except Exception as e:
                print(f"Warning: Could not load classifier: {e}")
                print("Falling back to heuristic classification")
        
        # Classes that might represent pans/pots in COCO dataset
        # 46: bowl, 47: cup, 78: microwave, 79: oven, etc.
        # We'll detect all objects and filter later
        
    def load_and_resize_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load image and create a resized version for YOLO
        
        Args:
            image_path: Path to the image
            
        Returns:
            Tuple of (original_image, resized_image)
        """
        # Load original image
        img_original = cv2.imread(str(image_path))
        if img_original is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize for YOLO while maintaining aspect ratio
        img_resized = self._resize_with_padding(img_original, self.target_size)
        
        return img_original, img_resized
    
    def _resize_with_padding(self, img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to target size while maintaining aspect ratio with padding
        
        Args:
            img: Input image
            target_size: Target (width, height)
            
        Returns:
            Resized and padded image
        """
        h, w = img.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Calculate padding offsets
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        
        # Place resized image in center
        padded[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized
        
        return padded
    
    def detect_pan_pot(self, image_path: str, conf_threshold: float = 0.3) -> List[Dict]:
        """
        Detect pans/pots in the image using YOLO
        
        Args:
            image_path: Path to the image
            conf_threshold: Confidence threshold for detection
            
        Returns:
            List of detected objects with bounding boxes and confidence
        """
        img_original, img_resized = self.load_and_resize_image(image_path)
        
        # Run YOLO detection
        results = self.model(img_resized, conf=conf_threshold, verbose=False)
        
        detections = []
        
        # Calculate scale factors to map back to original image
        h_orig, w_orig = img_original.shape[:2]
        h_resized, w_resized = img_resized.shape[:2]
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates (in resized image space)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = result.names[cls]
                
                # Map coordinates back to original image
                # First, account for padding
                scale = min(self.target_size[0] / w_orig, self.target_size[1] / h_orig)
                offset_x = (self.target_size[0] - int(w_orig * scale)) // 2
                offset_y = (self.target_size[1] - int(h_orig * scale)) // 2
                
                # Remove padding offset
                x1_adj = x1 - offset_x
                y1_adj = y1 - offset_y
                x2_adj = x2 - offset_x
                y2_adj = y2 - offset_y
                
                # Scale back to original size
                x1_orig = int(x1_adj / scale)
                y1_orig = int(y1_adj / scale)
                x2_orig = int(x2_adj / scale)
                y2_orig = int(y2_adj / scale)
                
                # Clip to image boundaries
                x1_orig = max(0, min(x1_orig, w_orig))
                y1_orig = max(0, min(y1_orig, h_orig))
                x2_orig = max(0, min(x2_orig, w_orig))
                y2_orig = max(0, min(y2_orig, h_orig))
                
                detections.append({
                    'bbox': (x1_orig, y1_orig, x2_orig, y2_orig),
                    'confidence': conf,
                    'class': class_name,
                    'class_id': cls
                })
        
        return detections, img_original
    
    def crop_focused_area(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                         padding_ratio: float = 0.1) -> np.ndarray:
        """
        Crop image to focused area around detected pan/pot with some padding
        
        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2)
            padding_ratio: Additional padding around the box (ratio of box size)
            
        Returns:
            Cropped image
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Calculate padding
        box_w = x2 - x1
        box_h = y2 - y1
        pad_w = int(box_w * padding_ratio)
        pad_h = int(box_h * padding_ratio)
        
        # Apply padding and clip to image boundaries
        x1_pad = max(0, x1 - pad_w)
        y1_pad = max(0, y1 - pad_h)
        x2_pad = min(w, x2 + pad_w)
        y2_pad = min(h, y2 + pad_h)
        
        # Crop
        cropped = image[y1_pad:y2_pad, x1_pad:x2_pad]
        
        return cropped
    
    def classify_state(self, cropped_image: np.ndarray, cropped_pil: Image.Image = None) -> Dict[str, float]:
        """
        Classify the state of the pan/pot into: boiling, normal, smoking, on_fire
        
        Uses trained deep learning classifier if available, otherwise falls back to heuristics
        
        Args:
            cropped_image: Cropped image of pan/pot (OpenCV format)
            cropped_pil: PIL Image version (optional, for deep learning)
            
        Returns:
            Dictionary with scores for each class
        """
        # Try deep learning classifier first
        if self.classifier is not None:
            try:
                # Convert OpenCV to PIL if needed
                if cropped_pil is None:
                    cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                    cropped_pil = Image.fromarray(cropped_rgb)
                
                # Save temporarily for prediction
                temp_path = self.results_dir / 'temp_crop.jpg'
                cropped_pil.save(temp_path)
                
                # Predict
                pred_class, confidence, all_probs = self.classifier.predict(str(temp_path))
                
                # Map 'normal' to 'still' for backward compatibility
                if 'normal' in all_probs:
                    all_probs['still'] = all_probs.pop('normal')
                if pred_class == 'normal':
                    pred_class = 'still'
                
                # Clean up
                temp_path.unlink(missing_ok=True)
                
                return all_probs
            except Exception as e:
                print(f"    Deep learning classification failed: {e}, falling back to heuristics")
        
        # Fallback to heuristic classification
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        
        # Initialize scores - use 'normal' to match classifier
        scores = {
            'normal': 0.0,
            'boiling': 0.0,
            'smoking': 0.0,
            'on_fire': 0.0
        }
        
        # Analyze brightness
        avg_brightness = np.mean(gray)
        max_brightness = np.max(gray)
        
        # Analyze color distribution
        avg_hue = np.mean(hsv[:, :, 0])
        avg_saturation = np.mean(hsv[:, :, 1])
        avg_value = np.mean(hsv[:, :, 2])
        
        # Count red/orange/yellow pixels (fire indicators)
        # Red: H in [0, 10] or [170, 180], Orange: [10, 25], Yellow: [25, 35]
        red_orange_yellow = np.sum(
            ((hsv[:, :, 0] < 35) | (hsv[:, :, 0] > 170)) & 
            (hsv[:, :, 1] > 100) & 
            (hsv[:, :, 2] > 100)
        )
        total_pixels = cropped_image.shape[0] * cropped_image.shape[1]
        fire_ratio = red_orange_yellow / total_pixels
        
        # Count white/gray pixels (smoke/steam indicators)
        smoke_pixels = np.sum(
            (hsv[:, :, 1] < 50) & 
            (hsv[:, :, 2] > 150)
        )
        smoke_ratio = smoke_pixels / total_pixels
        
        # Heuristic rules for classification
        
        # On Fire: High brightness, lots of red/orange/yellow
        if fire_ratio > 0.15 and max_brightness > 200:
            scores['on_fire'] = min(1.0, fire_ratio * 3 + (max_brightness - 200) / 55)
        
        # Smoking: Grayish/whitish regions, medium brightness
        if smoke_ratio > 0.2 and avg_saturation < 80:
            scores['smoking'] = min(1.0, smoke_ratio * 2)
        
        # Boiling: Look for bubbles (high local variance), some brightness
        # Calculate local variance
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        variance = cv2.Laplacian(blur, cv2.CV_64F).var()
        
        if variance > 100 and avg_brightness > 80:
            scores['boiling'] = min(1.0, variance / 500)
        
        # Normal/Still: Low activity, normal colors
        if max(scores.values()) < 0.3:
            scores['normal'] = 1.0 - max(scores.values())
        else:
            scores['normal'] = 0.1
        
        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v / total_score for k, v in scores.items()}
        
        # Map 'normal' to 'still' for display
        scores['still'] = scores.pop('normal')
        
        return scores
    
    def process_image(self, image_path: str, visualize: bool = True) -> Dict:
        """
        Complete pipeline: detect, crop, and classify
        
        Args:
            image_path: Path to the image
            visualize: Whether to create visualization
            
        Returns:
            Dictionary with results
        """
        print(f"\nProcessing: {Path(image_path).name}")
        
        # Detect pans/pots
        detections, img_original = self.detect_pan_pot(image_path)
        
        if not detections:
            print("  No objects detected")
            return {
                'image_path': image_path,
                'detections': [],
                'classifications': []
            }
        
        print(f"  Found {len(detections)} object(s)")
        
        classifications = []
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            
            # Crop focused area
            cropped = self.crop_focused_area(img_original, bbox)
            
            # Classify state
            scores = self.classify_state(cropped)
            predicted_class = max(scores, key=scores.get)
            
            classifications.append({
                'detection_id': i,
                'bbox': bbox,
                'detected_class': detection['class'],
                'confidence': detection['confidence'],
                'state': predicted_class,
                'state_scores': scores,
                'cropped_image': cropped
            })
            
            print(f"  Detection {i+1}: {detection['class']} (conf: {detection['confidence']:.2f})")
            print(f"    State: {predicted_class} (score: {scores[predicted_class]:.2f})")
        
        # Visualize if requested
        if visualize:
            self._visualize_results(image_path, img_original, classifications)
        
        return {
            'image_path': image_path,
            'detections': detections,
            'classifications': classifications
        }
    
    def _visualize_results(self, image_path: str, img_original: np.ndarray, 
                          classifications: List[Dict]):
        """
        Create visualization of detection and classification results
        
        Args:
            image_path: Original image path
            img_original: Original image
            classifications: Classification results
        """
        img_display = img_original.copy()
        
        # Color map for states
        color_map = {
            'still': (0, 255, 0),      # Green
            'normal': (0, 255, 0),     # Green (alias)
            'boiling': (255, 255, 0),   # Cyan
            'smoking': (128, 128, 128), # Gray
            'on_fire': (0, 0, 255)      # Red
        }
        
        for cls in classifications:
            bbox = cls['bbox']
            state = cls['state']
            color = color_map.get(state, (255, 255, 255))
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 3)
            
            # Add label
            label = f"{cls['detected_class']}: {state} ({cls['state_scores'][state]:.2f})"
            
            # Calculate text size for background
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw background rectangle for text
            cv2.rectangle(img_display, (x1, y1 - text_h - 10), 
                         (x1 + text_w, y1), color, -1)
            
            # Draw text
            cv2.putText(img_display, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save result
        result_path = self.results_dir / f"{Path(image_path).stem}_result.jpg"
        cv2.imwrite(str(result_path), img_display)
        print(f"  Saved result to: {result_path}")
        
        # Also save individual cropped images
        for i, cls in enumerate(classifications):
            crop_path = self.results_dir / f"{Path(image_path).stem}_crop_{i}_{cls['state']}.jpg"
            cv2.imwrite(str(crop_path), cls['cropped_image'])
    
    def process_directory(self, dir_path: str, pattern: str = "*.jpg") -> List[Dict]:
        """
        Process all images in a directory
        
        Args:
            dir_path: Directory containing images
            pattern: File pattern to match
            
        Returns:
            List of results for all images
        """
        dir_path = Path(dir_path)
        image_files = sorted(dir_path.glob(pattern))
        
        print(f"Found {len(image_files)} images to process")
        
        all_results = []
        for img_path in image_files:
            try:
                result = self.process_image(str(img_path), visualize=True)
                all_results.append(result)
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
        
        # Print summary
        self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results: List[Dict]):
        """Print summary statistics"""
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        total_images = len(results)
        total_detections = sum(len(r['detections']) for r in results)
        
        # Count states
        state_counts = {'still': 0, 'boiling': 0, 'smoking': 0, 'on_fire': 0}
        for r in results:
            for cls in r['classifications']:
                state_counts[cls['state']] += 1
        
        print(f"Total images processed: {total_images}")
        print(f"Total objects detected: {total_detections}")
        print(f"\nState distribution:")
        for state, count in state_counts.items():
            print(f"  {state}: {count}")
        print("="*60)


def main():
    """Main function to run the detector"""
    
    # Initialize detector with trained classifier
    print("Initializing Pan/Pot Detector...")
    
    # Check if trained model exists
    classifier_path = 'pan_pot_classifier.pth'
    if Path(classifier_path).exists():
        print(f"Using trained classifier: {classifier_path}")
        detector = PanPotDetector(
            model_name='yolov8n.pt', 
            target_size=(640, 640),
            classifier_path=classifier_path,
            use_deep_learning=True
        )
    else:
        print("No trained classifier found. Using heuristic classification.")
        print("To train a classifier, run: python train_classifier.py")
        detector = PanPotDetector(
            model_name='yolov8n.pt', 
            target_size=(640, 640),
            use_deep_learning=False
        )
    
    # Process all images in pics directory
    pics_dir = './pics'
    results = detector.process_directory(pics_dir, pattern="*.jpg")
    
    print("\nProcessing complete!")
    print(f"Results saved to: {detector.results_dir}")


if __name__ == "__main__":
    main()
