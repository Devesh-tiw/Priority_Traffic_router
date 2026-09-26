import cv2
import os
import sys
from ultralytics import YOLO

class SingleImageTrafficAI:
    def __init__(self, model_path='yolov8n.pt'):
        try:
            print("Loading YOLOv8 AI Model...")
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load model. {e}")
            sys.exit(1)

    def process_image(self, image_name='done.jpg'):
        # Get the exact folder where this Python script lives
        current_folder = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_folder, image_name)
        
        # Check if the image exists
        if not os.path.exists(image_path):
            print(f"ERROR: Image not found at {image_path}")
            return

        # Read the image
        frame = cv2.imread(image_path)
        if frame is None:
            print("ERROR: Could not open or read the image file.")
            return

        # Detect vehicles (COCO dataset IDs: 2=car, 3=motorbike, 5=bus, 7=truck)
        results = self.model(frame, verbose=False)
        boxes = results[0].boxes
        
        # Count vehicles and check for emergency overrides (Bus/Truck)
        vehicle_count = len(boxes)
        is_emergency = any(int(box.cls) in [5, 7] for box in boxes)

        # Visual layout logic
        color = (0, 255, 0) if is_emergency else (0, 0, 255)
        status = "GREEN (OVERRIDE)" if is_emergency else "RED"
        text = f"Status: {status} | Count: {vehicle_count} vehicles"
        
        # Draw detections and text overlay
        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Show the result on screen
        cv2.imshow("AI Vision Single Image Test", annotated_frame)
        print("Press any key on the image window to close it...")
        cv2.waitKey(0)  # Wait indefinitely for a key press
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ensure the script runs using the class structure
    analyzer = SingleImageTrafficAI()
    analyzer.process_image('done.jpg')
