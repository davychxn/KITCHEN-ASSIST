# Official Predictor for Verification Images

## Overview
`predict_veri.py` is the official standalone predictor for processing images in the `veri_pics` folder. It predicts pan/pot states and marks detected areas with green wireframes.

## Features

✅ **Focused Prediction**: Only processes images in `veri_pics` folder  
✅ **Green Wireframes**: Marks detected pan/pot areas with green rectangles  
✅ **Automatic Detection**: Uses YOLO to automatically find pan/pot objects  
✅ **Detailed Output**: Saves marked images and prediction details to JSON  
✅ **Accuracy Reporting**: Shows accuracy if ground truth is available in filenames  

## Usage

### Basic Usage
```bash
python predict_veri.py
```

This will:
1. Load the trained classifier model (`pan_pot_classifier.pth`)
2. Process all images in `veri_pics/` folder
3. Use YOLO to detect pan/pot areas
4. Mark detected areas with green wireframes
5. Save results to `veri_results_marked/` folder

### Output

#### Marked Images
Each image is saved with `_marked.jpg` suffix in `veri_results_marked/`:
- Green rectangle (3px) around detected pan/pot area
- "Detected Area" label on the green wireframe
- Prediction information overlay at the top:
  - True state (if available from filename)
  - Predicted state with confidence score
  - Correctness indicator (✓ or ✗)

#### Predictions JSON
All predictions are saved to `veri_results_marked/predictions.json` with:
```json
{
  "filename": "image_name.jpg",
  "predicted_state": "boiling",
  "confidence": 0.882,
  "all_probabilities": {
    "boiling": 0.882,
    "normal": 0.076,
    "on_fire": 0.017,
    "smoking": 0.026
  },
  "detection_method": "YOLO",
  "detection_bbox": {"x1": 4, "y1": 1, "x2": 1923, "y2": 1532},
  "true_state": "boiling",  // If available
  "correct": true           // If ground truth available
}
```

## States Detected

The classifier recognizes 4 states:
- **boiling**: Water/liquid is actively boiling
- **normal**: Pan/pot in normal heating state
- **on_fire**: Flames visible on pan/pot
- **smoking**: Smoke visible from pan/pot

## Console Output

The predictor provides detailed console output:
```
Processing: frying-pan_smoking_01.jpg
  Detected area: (242, 29) to (987, 681)
  ✓ True: smoking    | Pred: smoking    (conf: 0.935)
  Probabilities: boiling: 0.009, normal: 0.004, on_fire: 0.052, smoking: 0.935
  ✓ Saved: frying-pan_smoking_01_marked.jpg
```

## Detection Methods

1. **Manual Crop Coordinates** (Primary)
   - Loads from `pics_cropped/crop_coords.json` if available
   - Uses exact cropped regions for precise detection

2. **YOLO Detection** (Automatic Fallback)
   - Automatically detects objects in images
   - Selects the largest detected object
   - Works even without manual crop coordinates

## Requirements

- Python 3.7+
- PyTorch
- OpenCV (cv2)
- Ultralytics YOLO
- Trained classifier model (`pan_pot_classifier.pth`)

## Directory Structure

```
KITCHEN-ASSIST/
├── predict_veri.py              # Official predictor script
├── pan_pot_classifier.pth       # Trained model
├── yolov8n.pt                   # YOLO detection model
├── veri_pics/                   # Input: verification images
│   ├── image1.jpg
│   └── image2.jpg
└── veri_results_marked/         # Output: marked images + JSON
    ├── image1_marked.jpg
    ├── image2_marked.jpg
    └── predictions.json
```

## File Naming Convention

For automatic ground truth detection, name files as:
```
{pan_type}_{state}_{number}.jpg
```

Examples:
- `cooking-pot_normal_01.jpg` → state: normal
- `frying-pan_boiling_01.jpg` → state: boiling
- `frying-pan_on-fire_01.jpg` → state: on_fire
- `frying-pan_smoking_01.jpg` → state: smoking

If filenames don't match this pattern, predictions will still work but without accuracy calculation.

## Customization

Edit `predict_veri.py` to customize:

```python
predictions = predict_veri_images(
    model_path='pan_pot_classifier.pth',      # Path to trained model
    veri_dir='./veri_pics',                   # Input directory
    output_dir='./veri_results_marked',       # Output directory
    crop_coords_file='./pics_cropped/crop_coords.json',  # Optional
    show_ground_truth=True                    # Show/hide ground truth labels
)
```

## Example Output

After running the predictor:

```
======================================================================
Processing 4 verification images from veri_pics
======================================================================

Processing: cooking-pot_normal_01.jpg
  Detected area: (4, 1) to (1923, 1532)
  ✓ True: normal     | Pred: normal     (conf: 0.854)
  Probabilities: boiling: 0.094, normal: 0.854, on_fire: 0.028, smoking: 0.024
  ✓ Saved: cooking-pot_normal_01_marked.jpg

...

======================================================================
Processing complete!
======================================================================
Total images processed: 4/4
Marked images saved to: veri_results_marked

Accuracy: 100.00% (4/4)

Per-class Results:
  boiling   : 100.00% (1/1)
  normal    : 100.00% (1/1)
  on_fire   : 100.00% (1/1)
  smoking   : 100.00% (1/1)
```

## Troubleshooting

### No objects detected by YOLO
- Image might not contain clearly visible pan/pot
- Try adjusting YOLO confidence threshold
- Or manually create crop coordinates

### Low confidence predictions
- Image quality might be poor
- Object might be partially occluded
- Consider retraining with more similar examples

### Model not found
- Ensure `pan_pot_classifier.pth` is in the project root
- Run training first: `python train_classifier.py`

## Integration

To use the predictor in your own scripts:

```python
from predict_veri import predict_veri_images

predictions = predict_veri_images(
    model_path='pan_pot_classifier.pth',
    veri_dir='./my_images',
    output_dir='./my_results'
)

# Process predictions
for pred in predictions:
    print(f"{pred['filename']}: {pred['predicted_state']} ({pred['confidence']:.2f})")
```

## License

Part of the KITCHEN-ASSIST project.

---

*Last updated: December 6, 2025*
