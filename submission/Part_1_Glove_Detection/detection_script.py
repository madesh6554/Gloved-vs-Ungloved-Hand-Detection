import os
import cv2
import json
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Glove Hand and Bare Hand Detection")
    parser.add_argument("--input", type=str, default="input_images", help="Folder containing .jpg images")
    parser.add_argument("--output", type=str, default="output", help="Folder to save annotated images")
    parser.add_argument("--logs", type=str, default="logs", help="Folder to save JSON logs")
    
    # Use absolute path relative to the script for weights
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_weights = os.path.join(script_dir, "weights", "best.pt")
    
    parser.add_argument("--weights", type=str, default=default_weights, help="Path to YOLOv8 weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")

    args = parser.parse_args()

    # Load trained model
    if not os.path.exists(args.weights):
        print(f"❌ Error: Weights not found at {args.weights}")
        return

    model = YOLO(args.weights)

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)

    if not os.path.exists(args.input):
        print(f"❌ Error: Input folder not found at {args.input}")
        return

    images = [f for f in os.listdir(args.input) if f.lower().endswith(".jpg")]
    
    if not images:
        print(f"⚠️ No .jpg images found in {args.input}")
        return

    print(f"🚀 Starting detection on {len(images)} images...")

    for img_name in images:
        img_path = os.path.join(args.input, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"⚠️ Could not read image: {img_name}")
            continue

        results = model(image, conf=args.conf)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = model.names[cls_id]

            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "bbox": [x1, y1, x2, y2]
            })

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save output image
        cv2.imwrite(os.path.join(args.output, img_name), image)

        # Save JSON log
        json_data = {
            "filename": img_name,
            "detections": detections
        }

        log_path = os.path.join(args.logs, os.path.splitext(img_name)[0] + ".json")
        with open(log_path, "w") as f:
            json.dump(json_data, f, indent=2)

    print("✅ Detection completed! Results saved to output/ and logs/")

if __name__ == "__main__":
    main()
