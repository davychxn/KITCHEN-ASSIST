"""
Official Predictor for Verification Images
Predicts pan/pot states for images in veri_pics folder and marks detected areas
"""

import torch
import numpy as np
from pathlib import Path
from train_classifier import StateClassifierTrainer
import cv2
from PIL import Image
import json
from ultralytics import YOLO


def predict_veri_images(model_path='pan_pot_classifier.pth', 
                        veri_dir='./veri_pics', 
                        output_dir='./veri_results_marked',
                        crop_coords_file='./pics_cropped/crop_coords.json',
                        show_ground_truth=True,
                        output_width=1280,
                        resize_for_prediction=224):
    """
    Predict states for verification images and mark detected areas with green wireframes
    
    Args:
        model_path: Path to trained classifier model
        veri_dir: Directory containing verification images
        output_dir: Directory to save marked images
        crop_coords_file: JSON file with crop coordinates (optional)
        show_ground_truth: Whether to show ground truth labels (if filename contains state)
        output_width: Target width for output images (height auto-calculated to maintain aspect ratio)
        resize_for_prediction: Resize images to this size before prediction (to match training, 224 for MobileNet)
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Try to load crop coordinates if available
    crop_coords = {}
    crop_coords_path = Path(crop_coords_file)
    if crop_coords_path.exists():
        with open(crop_coords_path, 'r') as f:
            crop_coords = json.load(f)
        print(f"Loaded crop coordinates for {len(crop_coords)} images")
    
    # Load YOLO for detection
    print("Loading YOLO model for object detection...")
    yolo_model = YOLO('yolov8n.pt')
    
    # Load classifier
    print("Loading classifier model...")
    trainer = StateClassifierTrainer()
    trainer.load_model(model_path)
    print(f"Classifier loaded. Classes: {trainer.class_names}")
    
    # Get all verification images
    veri_dir = Path(veri_dir)
    if not veri_dir.exists():
        print(f"Error: Directory {veri_dir} does not exist!")
        return
    
    image_files = sorted(list(veri_dir.glob('*.jpg')) + 
                        list(veri_dir.glob('*.jpeg')) + 
                        list(veri_dir.glob('*.png')))
    
    if len(image_files) == 0:
        print(f"No images found in {veri_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"Processing {len(image_files)} verification images from {veri_dir}")
    print(f"{'='*70}\n")
    
    # Color map for states
    color_map = {
        'boiling': (255, 255, 0),   # Cyan
        'normal': (0, 255, 0),      # Green
        'on_fire': (0, 0, 255),     # Red
        'smoking': (128, 128, 128)  # Gray
    }
    
    processed_count = 0
    predictions = []
    
    for img_path in image_files:
        print(f"Processing: {img_path.name}")
        
        # Load original image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not load {img_path}, skipping...")
            continue
        
        # Store original size for visualization
        original_img = img.copy()
        
        # Resize for prediction to match training size
        if resize_for_prediction:
            img_for_pred = cv2.resize(img, (resize_for_prediction, resize_for_prediction))
            pred_img_path = img_path  # Still use original path, but will pass resized image
            # Save temporary resized image for prediction
            temp_pred_path = output_dir / f"temp_{img_path.name}"
            cv2.imwrite(str(temp_pred_path), img_for_pred)
            pred_img_path_str = str(temp_pred_path)
        else:
            pred_img_path_str = str(img_path)
        
        # Find crop coordinates or use YOLO detection
        coords = None
        detection_method = "YOLO"
        
        # Check if we have manual crop coordinates
        for key, value in crop_coords.items():
            if Path(key).name == img_path.name:
                coords = value
                detection_method = "Manual Crop"
                break
        
        # If no manual coords, use YOLO to detect
        if coords is None:
            try:
                yolo_results = yolo_model(str(img_path), verbose=False)
                if len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0:
                    # Use the largest detected object
                    boxes = yolo_results[0].boxes
                    areas = [(box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]) 
                             for box in boxes]
                    largest_idx = np.argmax(areas)
                    box = boxes[largest_idx]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    coords = {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)}
                    print(f"  Detected area: ({coords['x1']}, {coords['y1']}) to ({coords['x2']}, {coords['y2']})")
                else:
                    print(f"  ⚠ Warning: No objects detected by YOLO")
            except Exception as e:
                print(f"  ✗ Detection failed: {e}")
        else:
            print(f"  Using manual crop coordinates")
        
        # Predict state using resized image
        pred_state, confidence, all_probs = trainer.predict(pred_img_path_str)
        
        # Remove temporary file if created
        if resize_for_prediction and Path(pred_img_path_str).exists():
            Path(pred_img_path_str).unlink()
        
        # Use original image for visualization
        img = original_img
        
        # Scale coordinates back to original image size if we resized
        if coords and resize_for_prediction:
            h_orig, w_orig = img.shape[:2]
            scale_x = w_orig / resize_for_prediction
            scale_y = h_orig / resize_for_prediction
            coords = {
                'x1': int(coords['x1'] * scale_x),
                'y1': int(coords['y1'] * scale_y),
                'x2': int(coords['x2'] * scale_x),
                'y2': int(coords['y2'] * scale_y)
            }
        
        # Get ground truth if available (from filename)
        true_state = None
        is_correct = None
        if show_ground_truth:
            true_state = trainer.parse_filename(img_path.name)
            if true_state and true_state in trainer.class_to_idx:
                is_correct = true_state == pred_state
                status = "✓" if is_correct else "✗"
                print(f"  {status} True: {true_state:10s} | Pred: {pred_state:10s} (conf: {confidence:.3f})")
            else:
                print(f"  Pred: {pred_state:10s} (conf: {confidence:.3f})")
        else:
            print(f"  Pred: {pred_state:10s} (conf: {confidence:.3f})")
        
        # Print all probabilities
        print(f"  Probabilities: {', '.join([f'{k}: {v:.3f}' for k, v in all_probs.items()])}")
        
        # Store prediction results
        prediction_record = {
            'filename': img_path.name,
            'predicted_state': pred_state,
            'confidence': confidence,
            'all_probabilities': all_probs,
            'detection_method': detection_method
        }
        if true_state and true_state in trainer.class_to_idx:
            prediction_record['true_state'] = true_state
            prediction_record['correct'] = is_correct
        if coords:
            prediction_record['detection_bbox'] = coords
        
        predictions.append(prediction_record)
        
        # Draw green wireframe for detected area
        if coords:
            x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
            # Draw green rectangle with thick line
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Add label
            label_text = "Detected Area"
            label_y = max(y1 + 20, 20)
            cv2.putText(img, label_text, (x1 + 5, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add prediction information at top
        y_offset = 30
        
        # Background for better text visibility
        overlay = img.copy()
        bg_height = 110 if (show_ground_truth and true_state and true_state in trainer.class_to_idx) else 75
        cv2.rectangle(overlay, (0, 0), (img.shape[1], bg_height), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
        
        # Show ground truth if available
        if show_ground_truth and true_state and true_state in trainer.class_to_idx:
            cv2.putText(img, f"True: {true_state}", (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
            cv2.putText(img, f"True: {true_state}", (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_map.get(true_state, (255, 255, 255)), 2)
            y_offset += 35
        
        # Predicted state
        cv2.putText(img, f"Pred: {pred_state} ({confidence:.3f})", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        cv2.putText(img, f"Pred: {pred_state} ({confidence:.3f})", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_map.get(pred_state, (255, 255, 255)), 2)
        
        # Status (if ground truth available)
        if show_ground_truth and true_state and true_state in trainer.class_to_idx:
            y_offset += 35
            status_text = "✓ CORRECT" if is_correct else "✗ WRONG"
            status_color = (0, 255, 0) if is_correct else (0, 0, 255)
            cv2.putText(img, status_text, (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Resize to consistent width while maintaining aspect ratio
        if output_width and output_width > 0:
            h, w = img.shape[:2]
            if w != output_width:
                aspect_ratio = h / w
                new_height = int(output_width * aspect_ratio)
                img = cv2.resize(img, (output_width, new_height), interpolation=cv2.INTER_AREA)
                print(f"  Resized: {w}x{h} → {output_width}x{new_height}")
        
        # Save marked image
        output_path = output_dir / f"{img_path.stem}_marked.jpg"
        cv2.imwrite(str(output_path), img)
        processed_count += 1
        
        print(f"  ✓ Saved: {output_path.name}\n")
    
    print(f"{'='*70}")
    print(f"Processing complete!")
    print(f"{'='*70}")
    print(f"Total images processed: {processed_count}/{len(image_files)}")
    print(f"Marked images saved to: {output_dir}")
    
    # Calculate accuracy if ground truth is available
    predictions_with_truth = [p for p in predictions if 'correct' in p]
    if predictions_with_truth:
        accuracy = sum(1 for p in predictions_with_truth if p['correct']) / len(predictions_with_truth) * 100
        print(f"\nAccuracy: {accuracy:.2f}% ({sum(1 for p in predictions_with_truth if p['correct'])}/{len(predictions_with_truth)})")
        
        # Per-class statistics
        print("\nPer-class Results:")
        for class_name in trainer.class_names:
            class_preds = [p for p in predictions_with_truth if p['true_state'] == class_name]
            if class_preds:
                class_correct = sum(1 for p in class_preds if p['correct'])
                class_acc = class_correct / len(class_preds) * 100
                print(f"  {class_name:10s}: {class_acc:.2f}% ({class_correct}/{len(class_preds)})")
    
    # Save predictions to JSON
    predictions_file = output_dir / 'predictions.json'
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"\nPrediction details saved to: {predictions_file}")
    
    return predictions


def main():
    """Main prediction function"""
    
    print("\n" + "="*70)
    print("Pan/Pot State Predictor - Verification Images")
    print("="*70)
    
    try:
        predictions = predict_veri_images(
            model_path='pan_pot_classifier.pth',
            veri_dir='./veri_pics',
            output_dir='./veri_results_marked',
            crop_coords_file='./pics_cropped/crop_coords.json',
            show_ground_truth=True,
            output_width=1280,  # Consistent width for all output images
            resize_for_prediction=224  # Resize to match training dimensions
        )
        
        if predictions:
            print("\n✓ Prediction complete!")
            print("\nNext steps:")
            print("  - Review marked images in: veri_results_marked/")
            print("  - Check prediction details in: veri_results_marked/predictions.json")
        
    except Exception as e:
        print(f"\n✗ Error during prediction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
