"""
Circle-based Pan/Pot Detector using Hough Circle Transform
Better suited for top-down views of circular cookware
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


class CirclePanPotDetector:
    """
    Detects pans and pots using circle detection (Hough Circle Transform)
    More accurate for circular cookware in top-down views
    """
    
    def __init__(self, 
                 min_radius=50, 
                 max_radius=300,
                 param1=50,
                 param2=30,
                 min_dist=100):
        """
        Initialize circle detector
        
        Args:
            min_radius: Minimum circle radius in pixels
            max_radius: Maximum circle radius in pixels
            param1: Higher threshold for Canny edge detector
            param2: Accumulator threshold (lower = more circles detected)
            min_dist: Minimum distance between circle centers
        """
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.param1 = param1
        self.param2 = param2
        self.min_dist = min_dist
    
    def detect_circles(self, image_path: str, visualize: bool = False) -> List[Tuple[int, int, int]]:
        """
        Detect circular pots/pans in image
        
        Args:
            image_path: Path to image file
            visualize: Whether to show detection visualization
            
        Returns:
            List of (x, y, radius) tuples for detected circles
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        detected_circles = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # Sort by radius (largest first)
            circles = sorted(circles, key=lambda c: c[2], reverse=True)
            
            for (x, y, r) in circles:
                detected_circles.append((x, y, r))
            
            if visualize:
                vis_img = img.copy()
                for (x, y, r) in detected_circles:
                    # Draw circle
                    cv2.circle(vis_img, (x, y), r, (0, 255, 0), 3)
                    # Draw center
                    cv2.circle(vis_img, (x, y), 5, (0, 0, 255), -1)
                    # Add label
                    cv2.putText(vis_img, f"r={r}", (x-20, y-r-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display
                cv2.imshow("Circle Detection", vis_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        return detected_circles
    
    def circle_to_bbox(self, x: int, y: int, radius: int, 
                       margin: float = 1.0) -> dict:
        """
        Convert circle to bounding box coordinates
        
        Args:
            x, y: Circle center coordinates
            radius: Circle radius
            margin: Multiplier for radius (1.0 = exact fit, 0.9 = tighter)
            
        Returns:
            Dictionary with x1, y1, x2, y2 coordinates
        """
        r = int(radius * margin)
        return {
            'x1': max(0, x - r),
            'y1': max(0, y - r),
            'x2': x + r,
            'y2': y + r
        }
    
    def detect_and_get_bbox(self, image_path: str, 
                           margin: float = 0.95,
                           return_largest: bool = True) -> Optional[dict]:
        """
        Detect circles and return bounding box of largest circle
        
        Args:
            image_path: Path to image
            margin: Bbox margin (0.95 = 95% of circle diameter)
            return_largest: Return only the largest circle
            
        Returns:
            Bounding box dict or None if no circles detected
        """
        circles = self.detect_circles(image_path, visualize=False)
        
        if not circles:
            return None
        
        if return_largest:
            # Return bbox of largest circle
            x, y, r = circles[0]
            bbox = self.circle_to_bbox(x, y, r, margin)
            return bbox
        
        # Return all circles
        return [self.circle_to_bbox(x, y, r, margin) for x, y, r in circles]


def test_circle_detector():
    """Test circle detector on verification images"""
    
    detector = CirclePanPotDetector(
        min_radius=50,
        max_radius=300,
        param1=50,
        param2=30,
        min_dist=100
    )
    
    veri_dir = Path('./veri_pics')
    if not veri_dir.exists():
        print(f"Error: {veri_dir} not found")
        return
    
    images = sorted(list(veri_dir.glob('*.jpg')))
    
    print("="*70)
    print("Circle-Based Pan/Pot Detection Test")
    print("="*70)
    
    for img_path in images:
        print(f"\n{img_path.name}:")
        
        circles = detector.detect_circles(str(img_path))
        
        if circles:
            print(f"  Found {len(circles)} circle(s):")
            for i, (x, y, r) in enumerate(circles, 1):
                bbox = detector.circle_to_bbox(x, y, r, margin=0.95)
                print(f"    Circle {i}: center=({x},{y}), radius={r}")
                print(f"             bbox=({bbox['x1']},{bbox['y1']}) to ({bbox['x2']},{bbox['y2']})")
        else:
            print("  No circles detected")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    test_circle_detector()
