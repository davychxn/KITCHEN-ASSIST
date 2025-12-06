"""
Verify trained classifier on new test images in veri_pics folder
"""

import cv2
import numpy as np
from pathlib import Path
from train_classifier import StateClassifierTrainer
from pan_pot_detector import PanPotDetector


def verify_classifier(test_dir='./veri_pics', output_dir='./veri_results'):
    """Verify classifier on test images"""
    
    test_dir = Path(test_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load trained classifier
    print("Loading trained classifier...")
    trainer = StateClassifierTrainer()
    trainer.load_model('pan_pot_classifier.pth')
    
    # Get test images
    image_files = sorted(list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.jpeg')))
    
    if len(image_files) == 0:
        print(f"No images found in {test_dir}")
        return
    
    print(f"\nVerifying on {len(image_files)} new test images")
    print("="*70)
    
    results = []
    correct_count = 0
    
    for img_path in image_files:
        # Get ground truth from filename
        true_state = trainer.parse_filename(img_path.name)
        
        if not true_state or true_state not in trainer.class_to_idx:
            print(f"⚠ Skipping {img_path.name}: Cannot parse state or unknown class")
            continue
        
        # Predict
        pred_state, confidence, all_probs = trainer.predict(str(img_path))
        
        # Check if correct
        is_correct = (pred_state == true_state)
        if is_correct:
            correct_count += 1
        
        results.append({
            'filename': img_path.name,
            'true_state': true_state,
            'pred_state': pred_state,
            'confidence': confidence,
            'correct': is_correct,
            'all_probs': all_probs
        })
        
        # Print result
        status = "✓" if is_correct else "✗"
        color_code = "CORRECT" if is_correct else "WRONG"
        print(f"{status} {img_path.name:40s} True: {true_state:10s} → Pred: {pred_state:10s} "
              f"(conf: {confidence:.3f}) [{color_code}]")
        
        # Show all probabilities
        probs_str = " | ".join([f"{k}: {v:.3f}" for k, v in all_probs.items()])
        print(f"   Probabilities: {probs_str}")
        
        # Create visualization
        img = cv2.imread(str(img_path))
        if img is not None:
            # Add border based on correctness
            border_color = (0, 255, 0) if is_correct else (0, 0, 255)
            img = cv2.copyMakeBorder(img, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=border_color)
            
            # Add text
            y_offset = 35
            cv2.putText(img, f"True: {true_state}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.putText(img, f"True: {true_state}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            y_offset += 40
            pred_color = (0, 255, 0) if is_correct else (0, 0, 255)
            cv2.putText(img, f"Pred: {pred_state} ({confidence:.3f})", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.putText(img, f"Pred: {pred_state} ({confidence:.3f})", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, pred_color, 2)
            
            y_offset += 40
            status_text = "CORRECT" if is_correct else "WRONG"
            cv2.putText(img, status_text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, border_color, 3)
            
            # Save
            output_path = output_dir / f"{img_path.stem}_verification.jpg"
            cv2.imwrite(str(output_path), img)
    
    # Print summary
    print("="*70)
    print(f"\nVERIFICATION SUMMARY")
    print("="*70)
    
    if results:
        accuracy = (correct_count / len(results)) * 100
        print(f"Test Accuracy: {accuracy:.2f}% ({correct_count}/{len(results)})")
        
        # Per-class results
        print(f"\nPer-class Results:")
        for class_name in trainer.class_names:
            class_results = [r for r in results if r['true_state'] == class_name]
            if class_results:
                class_correct = sum(r['correct'] for r in class_results)
                class_total = len(class_results)
                class_acc = (class_correct / class_total) * 100
                print(f"  {class_name:10s}: {class_acc:6.2f}% ({class_correct}/{class_total})")
        
        # Show misclassifications
        misclassified = [r for r in results if not r['correct']]
        if misclassified:
            print(f"\nMisclassified ({len(misclassified)}):")
            for r in misclassified:
                print(f"  {r['filename']:40s} True: {r['true_state']:10s} → Pred: {r['pred_state']:10s}")
                print(f"    Confidence: {r['confidence']:.3f}")
                probs_str = " | ".join([f"{k}: {v:.3f}" for k, v in r['all_probs'].items()])
                print(f"    All probs: {probs_str}")
        else:
            print(f"\n✓ All test images classified correctly!")
        
        print(f"\nVisualization results saved to: {output_dir}")
    
    print("="*70)
    
    return results


def verify_with_detection(test_dir='./veri_pics', output_dir='./veri_results_detection'):
    """Verify full detection + classification pipeline"""
    
    test_dir = Path(test_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("FULL DETECTION + CLASSIFICATION PIPELINE VERIFICATION")
    print("="*70)
    
    # Initialize detector with trained classifier
    detector = PanPotDetector(
        model_name='yolov8n.pt',
        target_size=(640, 640),
        classifier_path='pan_pot_classifier.pth',
        use_deep_learning=True
    )
    
    # Temporarily change results directory
    original_results_dir = detector.results_dir
    detector.results_dir = output_dir
    
    # Process images
    image_files = sorted(list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.jpeg')))
    
    print(f"\nProcessing {len(image_files)} test images with YOLO detection...")
    
    for img_path in image_files:
        print(f"\n{img_path.name}:")
        result = detector.process_image(str(img_path), visualize=True)
        
        if result['detections']:
            print(f"  ✓ Detected {len(result['detections'])} object(s)")
            for cls in result['classifications']:
                print(f"    → {cls['detected_class']}: {cls['state']} (conf: {cls['state_scores'][cls['state']]:.3f})")
        else:
            print(f"  ✗ No objects detected (YOLO detection failed)")
    
    print(f"\nDetection results saved to: {output_dir}")
    
    # Restore original results directory
    detector.results_dir = original_results_dir


def main():
    """Main verification function"""
    
    print("="*70)
    print("MODEL VERIFICATION ON NEW TEST IMAGES")
    print("="*70)
    
    # Method 1: Direct classification (no YOLO detection)
    print("\n[Method 1] Direct Classification (Cropped Images)")
    print("-" * 70)
    results = verify_classifier(test_dir='./veri_pics', output_dir='./veri_results')
    
    # Method 2: Full pipeline with YOLO detection
    print("\n[Method 2] Full Pipeline (YOLO + Classification)")
    print("-" * 70)
    verify_with_detection(test_dir='./veri_pics', output_dir='./veri_results_detection')
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE!")
    print("="*70)
    print("\nCheck results:")
    print("  - veri_results/ : Direct classification results")
    print("  - veri_results_detection/ : Full pipeline with YOLO detection")


if __name__ == "__main__":
    main()
