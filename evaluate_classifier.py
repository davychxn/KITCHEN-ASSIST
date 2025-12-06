"""
Evaluate the trained classifier on the dataset
Shows accuracy, confusion matrix, and per-class metrics
"""

import torch
import numpy as np
from pathlib import Path
from train_classifier import StateClassifierTrainer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import cv2
from PIL import Image
import json
from ultralytics import YOLO


def evaluate_model(model_path='pan_pot_classifier.pth', data_dir='./pics'):
    """Evaluate the trained model"""
    
    # Load model
    trainer = StateClassifierTrainer()
    trainer.load_model(model_path)
    
    # Get all images
    data_dir = Path(data_dir)
    image_files = list(data_dir.glob('*.jpg')) + list(data_dir.glob('*.jpeg'))
    
    if len(image_files) == 0:
        print("No images found!")
        return
    
    print(f"Evaluating on {len(image_files)} images...")
    print("="*60)
    
    # Collect predictions and ground truth
    y_true = []
    y_pred = []
    y_pred_names = []
    y_true_names = []
    results = []
    
    for img_path in image_files:
        # Get ground truth from filename
        true_state = trainer.parse_filename(img_path.name)
        if true_state not in trainer.class_to_idx:
            if true_state:
                print(f"  Skipping {img_path.name}: state '{true_state}' not in trained classes {list(trainer.class_to_idx.keys())}")
            continue
        
        # Predict
        pred_state, confidence, all_probs = trainer.predict(str(img_path))
        
        # Store results
        y_true.append(trainer.class_to_idx[true_state])
        y_pred.append(trainer.class_to_idx[pred_state])
        y_true_names.append(true_state)
        y_pred_names.append(pred_state)
        
        results.append({
            'filename': img_path.name,
            'true_state': true_state,
            'pred_state': pred_state,
            'confidence': confidence,
            'correct': true_state == pred_state,
            'all_probs': all_probs
        })
        
        # Print result
        status = "✓" if true_state == pred_state else "✗"
        print(f"{status} {img_path.name:40s} True: {true_state:10s} Pred: {pred_state:10s} (conf: {confidence:.2f})")
    
    print("="*60)
    
    # Calculate accuracy
    accuracy = sum(r['correct'] for r in results) / len(results) * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    
    # Per-class accuracy
    print("\nPer-class Accuracy:")
    for class_name in trainer.class_names:
        class_results = [r for r in results if r['true_state'] == class_name]
        if class_results:
            class_acc = sum(r['correct'] for r in class_results) / len(class_results) * 100
            print(f"  {class_name:10s}: {class_acc:.2f}% ({sum(r['correct'] for r in class_results)}/{len(class_results)})")
    
    # Classification report
    print("\nClassification Report:")
    # Get unique classes that actually appear in the data
    unique_classes = sorted(set(y_true_names))
    print(classification_report(y_true_names, y_pred_names, labels=unique_classes, target_names=unique_classes))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=trainer.class_names,
                yticklabels=trainer.class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print("\nConfusion matrix saved to: confusion_matrix.png")
    plt.close()
    
    # Show misclassified examples
    misclassified = [r for r in results if not r['correct']]
    if misclassified:
        print(f"\nMisclassified Examples ({len(misclassified)}):")
        for r in misclassified:
            print(f"  {r['filename']:40s} True: {r['true_state']:10s} → Pred: {r['pred_state']:10s}")
            print(f"    Probabilities: {', '.join([f'{k}: {v:.2f}' for k, v in r['all_probs'].items()])}")
    
    return results


def visualize_predictions(model_path='pan_pot_classifier.pth', data_dir='./pics', output_dir='./evaluation_results'):
    """Create visualization of predictions"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    trainer = StateClassifierTrainer()
    trainer.load_model(model_path)
    
    # Get all images
    data_dir = Path(data_dir)
    image_files = sorted(list(data_dir.glob('*.jpg')) + list(data_dir.glob('*.jpeg')))
    
    # Color map
    color_map = {
        'boiling': (255, 255, 0),   # Cyan
        'normal': (0, 255, 0),      # Green
        'on_fire': (0, 0, 255),     # Red
        'smoking': (128, 128, 128)  # Gray
    }
    
    for img_path in image_files:
        # Get ground truth
        true_state = trainer.parse_filename(img_path.name)
        if true_state not in trainer.class_to_idx:
            continue
        
        # Predict
        pred_state, confidence, all_probs = trainer.predict(str(img_path))
        
        # Load image
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        # Determine color based on correctness
        is_correct = true_state == pred_state
        border_color = (0, 255, 0) if is_correct else (0, 0, 255)
        
        # Add border
        img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=border_color)
        
        # Add text information
        y_offset = 30
        cv2.putText(img, f"True: {true_state}", (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"True: {true_state}", (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_map.get(true_state, (255,255,255)), 1)
        
        y_offset += 30
        cv2.putText(img, f"Pred: {pred_state} ({confidence:.2f})", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Pred: {pred_state} ({confidence:.2f})", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_map.get(pred_state, (255,255,255)), 1)
        
        # Add status
        y_offset += 30
        status_text = "CORRECT" if is_correct else "WRONG"
        status_color = (0, 255, 0) if is_correct else (0, 0, 255)
        cv2.putText(img, status_text, (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Save
        output_path = output_dir / f"{img_path.stem}_eval.jpg"
        cv2.imwrite(str(output_path), img)
    
    print(f"\nVisualization saved to: {output_dir}")


def visualize_veri_pics_with_crops(model_path='pan_pot_classifier.pth', 
                                    veri_dir='./veri_pics', 
                                    output_dir='./veri_results_marked',
                                    crop_coords_file='./pics_cropped/crop_coords.json'):
    """
    Visualize veri_pics images with green wireframes marking the detected areas
    
    Args:
        model_path: Path to trained classifier model
        veri_dir: Directory containing verification images
        output_dir: Directory to save marked images
        crop_coords_file: JSON file with crop coordinates (optional)
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
    trainer = StateClassifierTrainer()
    trainer.load_model(model_path)
    
    # Get all verification images
    veri_dir = Path(veri_dir)
    image_files = sorted(list(veri_dir.glob('*.jpg')) + list(veri_dir.glob('*.jpeg')))
    
    if len(image_files) == 0:
        print(f"No images found in {veri_dir}")
        return
    
    print(f"\nProcessing {len(image_files)} verification images...")
    
    # Color map for states
    color_map = {
        'boiling': (255, 255, 0),   # Cyan
        'normal': (0, 255, 0),      # Green
        'on_fire': (0, 0, 255),     # Red
        'smoking': (128, 128, 128)  # Gray
    }
    
    processed_count = 0
    
    for img_path in image_files:
        # Get ground truth
        true_state = trainer.parse_filename(img_path.name)
        if true_state not in trainer.class_to_idx:
            print(f"Skipping {img_path.name}: unknown state")
            continue
        
        # Load original image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not load {img_path}, skipping...")
            continue
        
        # Find crop coordinates or use YOLO detection
        coords = None
        
        # Check if we have manual crop coordinates
        for key, value in crop_coords.items():
            if Path(key).name == img_path.name:
                coords = value
                break
        
        # If no manual coords, use YOLO to detect
        if coords is None:
            try:
                results = yolo_model(str(img_path), verbose=False)
                if len(results) > 0 and len(results[0].boxes) > 0:
                    # Use the largest detected object
                    boxes = results[0].boxes
                    areas = [(box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]) 
                             for box in boxes]
                    largest_idx = np.argmax(areas)
                    box = boxes[largest_idx]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    coords = {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)}
            except Exception as e:
                print(f"Warning: Detection failed for {img_path.name}: {e}")
        
        # Predict state
        pred_state, confidence, all_probs = trainer.predict(str(img_path))
        
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
        
        # Determine if prediction is correct
        is_correct = true_state == pred_state
        
        # Add prediction information at top
        y_offset = 30
        
        # Background for better text visibility
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], 110), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
        
        # True state
        cv2.putText(img, f"True: {true_state}", (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        cv2.putText(img, f"True: {true_state}", (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_map.get(true_state, (255, 255, 255)), 2)
        
        # Predicted state
        y_offset += 35
        cv2.putText(img, f"Pred: {pred_state} ({confidence:.2f})", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        cv2.putText(img, f"Pred: {pred_state} ({confidence:.2f})", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_map.get(pred_state, (255, 255, 255)), 2)
        
        # Status
        y_offset += 35
        status_text = "✓ CORRECT" if is_correct else "✗ WRONG"
        status_color = (0, 255, 0) if is_correct else (0, 0, 255)
        cv2.putText(img, status_text, (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Save marked image
        output_path = output_dir / f"{img_path.stem}_marked.jpg"
        cv2.imwrite(str(output_path), img)
        processed_count += 1
        
        print(f"  Processed: {img_path.name} → {output_path.name}")
    
    print(f"\n{processed_count} verification images marked and saved to: {output_dir}")


def main():
    """Main evaluation function"""
    
    print("Evaluating Pan/Pot State Classifier")
    print("="*60)
    
    # Evaluate
    results = evaluate_model(
        model_path='pan_pot_classifier.pth',
        data_dir='./pics'
    )
    
    # Create visualizations
    if results:
        print("\nCreating visualizations...")
        visualize_predictions(
            model_path='pan_pot_classifier.pth',
            data_dir='./pics',
            output_dir='./evaluation_results'
        )
    
    # Mark verification images with crop areas
    print("\nMarking verification images with detected areas...")
    try:
        visualize_veri_pics_with_crops(
            model_path='pan_pot_classifier.pth',
            veri_dir='./veri_pics',
            output_dir='./veri_results_marked',
            crop_coords_file='./pics_cropped/crop_coords.json'
        )
    except Exception as e:
        print(f"Error marking verification images: {e}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
