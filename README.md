# Pan/Pot Detection and Classification System

This system uses YOLO for object detection to identify pans and pots on stoves, then classifies their state into four categories using a **trained deep learning classifier**: **boiling**, **normal/still**, **smoking**, and **on_fire**.

## Features

1. **Image Preprocessing**: Resizes images to uniform dimensions (640x640) while maintaining aspect ratio
2. **Object Detection**: Uses YOLOv8 to detect pans/pots and other objects on the stove
3. **Focused Cropping**: Extracts only the pan/pot region, excluding the rest of the stove
4. **Deep Learning Classification**: Uses a trained ResNet18 model to classify each detected pan/pot into one of four states:
   - **Normal/Still**: No activity, calm state
   - **Boiling**: Bubbling water/liquid
   - **Smoking**: Steam or smoke visible
   - **On Fire**: Flames present

## Current Performance

With the current trained model on your labeled images:
- **Overall Accuracy**: 100% (on detected objects)
- **Boiling**: 100% (3/3)
- **Normal**: 100% (3/3)
- **Smoking**: 100% (2/2)
- **On Fire**: Limited data (need more examples)

⚠️ **Note**: Results are based on a small dataset (8 training images). For production use, more data is needed.

## Installation

1. Install Python 3.8 or higher

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start (Using Trained Model)

The system now includes a trained classifier. Simply run:

```bash
python pan_pot_detector.py
```

This will:
- Load the trained classifier (`pan_pot_classifier.pth`)
- Process all images in `./pics` folder
- Save results to `./results` folder

### Training Your Own Classifier

**Important**: For better accuracy, collect more labeled images!

1. **Prepare labeled images** in `./pics` folder with naming format:
   ```
   kitchenware_state_number.jpg
   
   Examples:
   cooking-pot_boiling_01.jpg
   frying-pan_on_fire_01.jpg
   cooking-pot_normal_01.jpg
   ```

2. **Train the classifier**:
   ```bash
   python train_classifier.py
   ```

3. **Evaluate accuracy**:
   ```bash
   python evaluate_classifier.py
   ```

4. **Run full pipeline**:
   ```bash
   python pan_pot_detector.py
   ```

### Manual Cropping (When Detection Fails)

If YOLO doesn't detect the pan/pot correctly:

```bash
python manual_crop.py
```

This opens an interactive tool to manually crop each image.

## Results

The script will:
- Process all `.jpg` images in the `./pics` folder
- Detect pans/pots in each image
- Classify their state
- Save results to the `./results` folder:
  - `*_result.jpg`: Original image with bounding boxes and labels
  - `*_crop_X_STATE.jpg`: Cropped images of each detected pan/pot

### Example Output

```
Processing: Clipboard_12-05-2025_04.jpg
  Found 2 object(s)
  Detection 1: bowl (conf: 0.85)
    State: boiling (score: 0.65)
  Detection 2: bowl (conf: 0.72)
    State: still (score: 0.80)
  Saved result to: results\Clipboard_12-05-2025_04_result.jpg
```

## Classification Logic

The system uses heuristic-based classification analyzing:

- **Brightness**: Overall and maximum brightness levels
- **Color Distribution**: HSV analysis for fire indicators (red/orange/yellow)
- **Smoke Detection**: White/gray pixel ratios
- **Activity Detection**: Local variance for bubbling/boiling

### State Indicators:

- **On Fire**: High brightness + red/orange/yellow colors (>15% fire-colored pixels)
- **Smoking**: Gray/white regions + medium saturation (<80)
- **Boiling**: High local variance (>100) indicating bubbles/activity
- **Still**: Low activity, default when no other state is strongly detected

## Customization

### Change YOLO Model

Edit the model in `pan_pot_detector.py`:

```python
detector = PanPotDetector(model_name='yolov8s.pt')  # Use small model
# Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large)
```

### Adjust Detection Confidence

```python
detections, img = detector.detect_pan_pot(image_path, conf_threshold=0.5)
```

### Modify Target Size

```python
detector = PanPotDetector(target_size=(1280, 1280))  # Larger images for better detection
```

## Advanced Usage

### Process Single Image

```python
from pan_pot_detector import PanPotDetector

detector = PanPotDetector()
result = detector.process_image('path/to/image.jpg', visualize=True)

# Access results
for cls in result['classifications']:
    print(f"State: {cls['state']}")
    print(f"Scores: {cls['state_scores']}")
```

### Get Cropped Images

```python
result = detector.process_image('path/to/image.jpg', visualize=False)

for cls in result['classifications']:
    cropped_img = cls['cropped_image']
    # Process cropped image further...
```

## Notes

- First run will download YOLOv8 weights (~6MB for nano model)
- The classification is heuristic-based; for production use, consider training a dedicated classifier
- YOLO detects general objects (bowls, cups, etc.) that may represent pans/pots
- Adjust the classification thresholds in `classify_state()` based on your specific images

## Improving Classification Accuracy

For better results, consider:

1. **Training a Custom Classifier**: Collect labeled data and train a CNN for the 4 states
2. **Fine-tuning YOLO**: Train YOLO on your specific pan/pot images
3. **Temporal Analysis**: If you have video, analyze frame sequences for more accurate state detection
4. **Multi-modal Features**: Add sound analysis (sizzling, boiling sounds) if available

## Troubleshooting

- **No objects detected**: Lower `conf_threshold` or use a larger YOLO model
- **Wrong classifications**: Adjust heuristic thresholds in `classify_state()` method
- **Out of memory**: Use a smaller YOLO model (yolov8n.pt) or reduce `target_size`
