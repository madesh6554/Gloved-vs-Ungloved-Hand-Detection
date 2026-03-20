# Part 1: Gloved vs Ungloved Hand Detection

This project implements an object detection pipeline to identify whether workers are wearing gloves (`glove_hand`) or not (`bare_hand`). This is designed for safety compliance systems in industrial environments.

## Dataset
- **Name**: Glove Hand and Bare Hand (v3)
- **Source**: [Roboflow Universe](https://universe.roboflow.com/glove-detection-3vldq/glove-hand-and-bare-hand-zwvif/dataset/3)
- **Classes**: `bare_hand`, `glove_hand`
- **Total Images**: ~200 images in the validation set used for testing.
- **Preprocessing**: 
  - Auto-orientation of pixel data.
  - Resizing to 640x640.
  - Augmentations: Horizontal flip, Rotation (-15° to +15°), Brightness adjustments (-25% to +25%).

## Model
- **Architecture**: YOLOv8n (Ultralytics)
- **Training**: 
  - Fine-tuned for 30 epochs on the custom dataset.
  - Trained on CPU (12th Gen Intel Core i5-1240P).
- **Performance**:
  - Precision: 0.921
  - Recall: 0.831
  - mAP50: 0.914
  - mAP50-95: 0.724

## Script Features
The `detection_script.py` has been enhanced with:
- **Streamlit Interface**: Premium web application with Upload, URL, and Sample Gallery support.
- **Real-time Metrics**: Instant counts of gloved vs bare hands.
- **CLI Arguments**: Custom input/output paths and confidence thresholds.
- **Automated Logging**: Saves per-image detection results in the required JSON format.
- **Visualization**: Generates annotated images with bounding boxes and confidence scores.

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install ultralytics opencv-python
   ```

3. **Run Streamlit Web App (Recommended)**:
   For an interactive experience, run the Streamlit app from the root folder:
   ```bash
   streamlit run submission/Part_1_Glove_Detection/streamlit_app.py
   ```

4. **Run CLI Detection**:
   To run the detection script on all images in the `input_images` folder:
   ```bash
   python submission/Part_1_Glove_Detection/detection_script.py --input submission/Part_1_Glove_Detection/input_images --output submission/Part_1_Glove_Detection/output
   ```

5. **Arguments**:
   - `--input`: Folder containing input `.jpg` images (default: `input_images`).
   - `--output`: Folder to save annotated images (default: `output`).
   - `--logs`: Folder to save JSON logs (default: `logs`).
   - `--weights`: Path to model weights (default: `weights/best.pt`).
   - `--conf`: Confidence threshold (default: `0.25`).

## What Worked and What Didn't
- **Worked**: YOLOv8n proved to be very efficient for this task, achieving high accuracy even with a small dataset and limited training epochs. The model distinguishes between bare hands and various types of gloves effectively.
- **Challenges**: Some images with complex backgrounds or partial occlusions showed slightly lower confidence, but the overall mAP remains robust for a safety compliance baseline.
