import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os  # ADD THIS IMPORT

class SmartIntersectionAI:
    def __init__(self, model_path='yolov8n.pt', video_sources=None):
        try:
            print("Loading YOLOv8 AI Model...")
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load model. {e}")
            sys.exit(1)
            
        # Get the exact folder where this Python script lives
        current_folder = os.path.dirname(os.path.abspath(__file__))
        
        # Build the exact paths to your two videos
        vid_ns = os.path.join(current_folder, "Traffic_north_south.mp4")
        vid_ew = os.path.join(current_folder, "Traffic_east_west.mp4")
        
        # Use those direct paths for the 4 video feeds
        self.video_sources = video_sources or [vid_ns, vid_ns, vid_ew, vid_ew]
        
        self.caps = [cv2.VideoCapture(vid) for vid in self.video_sources]
        self.names = ["NORTH", "SOUTH", "EAST", "WEST"]

# ==========================================
# SYLLABUS: Object-Oriented Programming (OOP)
# ==========================================

        
    def run_controller(self):
        while True:
            frames = []
            counts = []
            emergency_flags = []

            for cap in self.caps:
                ret, frame = cap.read()
                if not ret:
                    break # Break loop if video ends
                
                frame = cv2.resize(frame, (400, 300))
                
                # Detect vehicles (COCO: 2=car, 3=motorbike, 5=bus, 7=truck)
                results = self.model(frame, verbose=False)
                boxes = results[0].boxes
                
                # PRESENTATION DEMO: Treat Buses(5) and Trucks(7) as Emergency
                is_emergency = any(int(box.cls) in [5, 7] for box in boxes)
                
                counts.append(len(boxes))
                emergency_flags.append(is_emergency)
                frames.append(results[0].plot())

            if len(frames) < 4:
                print("End of video stream reached.")
                break

            # ==========================================
            # SYLLABUS: Control Statements & Logic
            # ==========================================
            if any(emergency_flags):
                # Override: Give GREEN to the emergency lane
                green_index = emergency_flags.index(True)
            else:
                # Standard: Give GREEN to the lane with maximum vehicles
                green_index = counts.index(max(counts))

            for i in range(4):
                # SYLLABUS: Tuples (RGB Colors)
                color = (0, 255, 0) if i == green_index else (0, 0, 255)
                status = "GREEN" if i == green_index else "RED"
                warning = " !!! EMERGENCY OVERRIDE !!!" if emergency_flags[i] else ""
                
                text = f"{self.names[i]}: {status} ({counts[i]} vehicles){warning}"
                cv2.putText(frames[i], text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 3)

            # ==========================================
            # SYLLABUS: NumPy Matrix Operations
            # ==========================================
            top_row = np.hstack((frames[0], frames[1]))
            bottom_row = np.hstack((frames[2], frames[3]))
            grid = np.vstack((top_row, bottom_row))

            cv2.imshow("AI Vision Traffic Controller", grid)

            if cv2.waitKey(1) == ord('q'):
                break

        # SYLLABUS: File/Resource Handling (Releasing memory)
        for cap in self.caps:
            cap.release()
        cv2.destroyAllWindows()

# ==========================================
# SYLLABUS: Functions and Modules usage
# ==========================================
if __name__ == "__main__":
    # Instantiate the OOP class and run the main loop
    intersection = SmartIntersectionAI()
    intersection.run_controller()