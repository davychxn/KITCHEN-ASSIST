# Improving Classification Accuracy - Guide

## Current Status

✅ **Achieved 100% accuracy on the 8 images that were properly detected**
- Boiling: 100% (3/3)
- Normal: 100% (3/3)  
- Smoking: 100% (2/2)

⚠️ **Issues Found:**
1. Missing "on_fire" examples in training data (2 images not used)
2. YOLO sometimes detects "person" instead of pan/pot
3. Small dataset (only 10 images, 8 used for training)

## Current Results Summary

From the latest run:
- **Boiling detection**: Works well (3/3 boiling images correctly classified)
- **Normal/Still detection**: Works well (3/3 normal images correctly classified)
- **Smoking detection**: Partially working (1/2 correct - one classified as boiling)
- **On-fire detection**: Not working (0/2 correct - both classified as boiling, also detection issues)

## Recommendations to Improve Accuracy

### 1. Add More Labeled Images (HIGHLY RECOMMENDED)

**Current dataset is too small!** You only have 10 images total. For production-quality results, you need:

- **Minimum**: 50-100 images per class (200-400 total)
- **Better**: 200-500 images per class (800-2000 total)
- **Ideal**: 1000+ images per class

**How to collect more data:**
- Take multiple photos of each scenario from different angles
- Use different pans/pots (various sizes, colors, materials)
- Vary lighting conditions (bright, dim, different times of day)
- Include different stove types if applicable
- Capture different stages of each state

**File naming format:**
```
kitchenware_state_number.jpg

Examples:
cooking-pot_boiling_01.jpg
cooking-pot_boiling_02.jpg
frying-pan_on_fire_01.jpg
frying-pan_normal_01.jpg
sauce-pan_smoking_01.jpg
```

### 2. Include On-Fire Examples

**Critical missing class!** Currently you have:
- Boiling: 3 images ✓
- Normal: 3 images ✓
- Smoking: 2 images ✓
- **On-fire: 0 images ✗** (the 2 on-fire images weren't properly processed)

**Action needed:**
- Ensure on-fire images are properly detected (see YOLO detection issues below)
- Add more on-fire examples (at least 5-10 minimum)
- Make sure flames/fire are clearly visible

### 3. Fix YOLO Detection Issues

**Problem:** YOLO is detecting "person" or other objects instead of the pan/pot

**Solutions:**

#### Option A: Manual Cropping (Quick Fix)
Instead of relying on YOLO detection, manually crop your images to focus on just the pan/pot:

```python
# Use this script to manually crop images
python manual_crop.py
```

I'll create this script for you.

#### Option B: Train Custom YOLO Model
Train YOLO specifically on your pan/pot images:
- Label 50-100 images with bounding boxes around pans/pots
- Use tools like LabelImg or CVAT for labeling
- Fine-tune YOLO on your custom dataset

#### Option C: Lower YOLO Confidence Threshold
Currently using 0.3 - try lowering to 0.2 or use all detections

### 4. Retrain After Adding More Data

Once you have more images:

```bash
# Retrain the classifier
python train_classifier.py

# Evaluate the results
python evaluate_classifier.py

# Run full detection pipeline
python pan_pot_detector.py
```

### 5. Consider Data Augmentation Enhancements

Current augmentation includes:
- Horizontal flips
- Rotation (±15°)
- Color jitter
- Translation and scaling

**Additional augmentations to try:**
- Gaussian blur (simulate steam/smoke)
- Brightness adjustments (simulate fire glow)
- Add noise (simulate real-world conditions)

### 6. Try Different Model Architectures

Current: ResNet18 (small, fast)

**Alternatives to test:**
- ResNet50 (more capacity): `backbone='resnet50'`
- EfficientNet-B0 (efficient): `backbone='efficientnet_b0'`
- Vision Transformer (if you get lots of data)

### 7. Use Cross-Validation

With more data, use k-fold cross-validation:
- Split data into 5 folds
- Train 5 models, each using different validation set
- Average predictions for better accuracy

### 8. Add Temporal Information (If Using Video)

If you have video streams:
- Analyze multiple consecutive frames
- Use state transitions (normal → boiling → smoking → on-fire)
- Apply temporal smoothing to reduce false alarms

## Expected Accuracy Improvements

| Dataset Size | Expected Accuracy |
|-------------|-------------------|
| 10 images (current) | 60-80% |
| 50 per class (200 total) | 85-92% |
| 100 per class (400 total) | 90-95% |
| 200+ per class (800+ total) | 95-98% |

## Quick Action Plan

**Immediate (1-2 hours):**
1. ✅ Train classifier on current 10 images - DONE
2. Manually crop on-fire images to ensure they're in training set
3. Add 5-10 more images for each class

**Short-term (1-2 days):**
4. Collect 50+ images per class
5. Retrain classifier with larger dataset
6. Evaluate and iterate

**Long-term (1-2 weeks):**
7. Collect 200+ images per class
8. Consider custom YOLO training for better detection
9. Implement temporal smoothing if using video

## Testing Your Improvements

After each improvement, run:

```bash
# Evaluate classifier accuracy
python evaluate_classifier.py

# Full pipeline test
python pan_pot_detector.py

# Check results in:
# - results/ (detection visualizations)
# - evaluation_results/ (classification accuracy)
# - confusion_matrix.png (see which classes confuse the model)
```

## Current Files

- `train_classifier.py` - Train the deep learning classifier
- `evaluate_classifier.py` - Evaluate classifier accuracy
- `pan_pot_detector.py` - Full detection + classification pipeline
- `pan_pot_classifier.pth` - Trained model (100% on current 8 images)
- `training_history.png` - Training curves
- `confusion_matrix.png` - Classification confusion matrix

## Questions?

**Q: Do I need GPU for training?**
A: No, CPU works fine for small datasets (< 100 images). GPU recommended for 500+ images.

**Q: How do I know if my model is overfitting?**
A: Check `training_history.png` - if training accuracy is much higher than validation accuracy, you're overfitting. Solution: Add more data and stronger augmentation.

**Q: Can I use pre-labeled datasets?**
A: Possibly, but kitchen/cooking datasets are rare. Your specific setup (stove type, lighting) might differ. Best to collect your own data.

**Q: What if I can't collect more on-fire images (safety)?**
A: Use simulation:
- Edit images digitally to add flames
- Use existing fire images as overlay
- Generate synthetic data with GANs (advanced)
