"""
Manual Image Cropper for Pan/Pot Detection
Use this when YOLO fails to detect the pan/pot correctly
"""

import cv2
import numpy as np
from pathlib import Path
import json

class ManualCropper:
    def __init__(self, input_dir='./pics', output_dir='./pics_cropped'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.current_image = None
        self.current_image_path = None
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.crop_coords = {}
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing rectangle"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
    
    def crop_image(self, image_path):
        """Interactively crop a single image"""
        print(f"\nProcessing: {image_path.name}")
        print("Instructions:")
        print("  - Click and drag to draw a rectangle around the pan/pot")
        print("  - Press 's' to save the crop")
        print("  - Press 'r' to reset")
        print("  - Press 'n' to skip this image")
        print("  - Press 'q' to quit")
        
        self.current_image = cv2.imread(str(image_path))
        if self.current_image is None:
            print(f"Error: Could not load {image_path}")
            return False
        
        self.current_image_path = image_path
        self.start_point = None
        self.end_point = None
        self.drawing = False
        
        window_name = 'Manual Cropper - ' + image_path.name
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        while True:
            display_image = self.current_image.copy()
            
            # Draw rectangle if points are defined
            if self.start_point and self.end_point:
                cv2.rectangle(display_image, self.start_point, self.end_point, 
                            (0, 255, 0), 2)
                
                # Show dimensions
                width = abs(self.end_point[0] - self.start_point[0])
                height = abs(self.end_point[1] - self.start_point[1])
                cv2.putText(display_image, f"Size: {width}x{height}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show instructions
            cv2.putText(display_image, "s: Save | r: Reset | n: Skip | q: Quit", 
                       (10, display_image.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(window_name, display_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):  # Save
                if self.start_point and self.end_point:
                    self.save_crop()
                    cv2.destroyWindow(window_name)
                    return True
                else:
                    print("  Please draw a rectangle first!")
            
            elif key == ord('r'):  # Reset
                self.start_point = None
                self.end_point = None
                print("  Rectangle reset")
            
            elif key == ord('n'):  # Skip
                print("  Skipped")
                cv2.destroyWindow(window_name)
                return True
            
            elif key == ord('q'):  # Quit
                cv2.destroyWindow(window_name)
                return False
        
        cv2.destroyWindow(window_name)
        return True
    
    def save_crop(self):
        """Save the cropped region"""
        if not self.start_point or not self.end_point:
            return
        
        # Get crop coordinates
        x1 = min(self.start_point[0], self.end_point[0])
        y1 = min(self.start_point[1], self.end_point[1])
        x2 = max(self.start_point[0], self.end_point[0])
        y2 = max(self.start_point[1], self.end_point[1])
        
        # Ensure valid crop
        if x2 - x1 < 10 or y2 - y1 < 10:
            print("  Crop region too small!")
            return
        
        # Crop image
        cropped = self.current_image[y1:y2, x1:x2]
        
        # Save cropped image with same filename
        output_path = self.output_dir / self.current_image_path.name
        cv2.imwrite(str(output_path), cropped)
        
        # Save crop coordinates
        self.crop_coords[str(self.current_image_path)] = {
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
        }
        
        print(f"  ✓ Saved cropped image to: {output_path}")
    
    def process_directory(self):
        """Process all images in directory"""
        image_files = sorted(list(self.input_dir.glob('*.jpg')) + 
                           list(self.input_dir.glob('*.jpeg')) +
                           list(self.input_dir.glob('*.png')))
        
        if len(image_files) == 0:
            print(f"No images found in {self.input_dir}")
            return
        
        print(f"\nFound {len(image_files)} images to crop")
        print("="*60)
        
        for i, img_path in enumerate(image_files, 1):
            print(f"\nImage {i}/{len(image_files)}")
            
            # Skip if already cropped
            output_path = self.output_dir / img_path.name
            if output_path.exists():
                print(f"  {img_path.name} already cropped. Skipping.")
                print("  (Delete from pics_cropped to re-crop)")
                continue
            
            if not self.crop_image(img_path):
                print("\nCropping cancelled by user")
                break
        
        # Save crop coordinates
        coords_file = self.output_dir / 'crop_coords.json'
        with open(coords_file, 'w') as f:
            json.dump(self.crop_coords, f, indent=2)
        
        print("\n" + "="*60)
        print(f"Cropping complete!")
        print(f"Cropped images saved to: {self.output_dir}")
        print(f"Crop coordinates saved to: {coords_file}")
        print("\nNext steps:")
        print("1. Move cropped images to pics/ folder:")
        print("   - Backup your original pics/ folder first!")
        print("   - Copy images from pics_cropped/ to pics/")
        print("2. Retrain: python train_classifier.py")
        print("3. Test: python pan_pot_detector.py")


def main():
    """Main function"""
    print("="*60)
    print("Manual Image Cropper for Pan/Pot Detection")
    print("="*60)
    print("\nThis tool helps you manually crop images when YOLO")
    print("detection fails to find the pan/pot correctly.\n")
    
    cropper = ManualCropper(input_dir='./pics', output_dir='./pics_cropped')
    cropper.process_directory()


if __name__ == "__main__":
    main()
