import cv2
import numpy as np
from ultralytics import YOLO

# Load YOUR custom trained model, not the default one
model = YOLO('runs/detect/smart_traffic_model/weights/best.pt') 

videos = ["north.mp4", "south.mp4", "east.mp4", "west.mp4"]
caps = [cv2.VideoCapture(vid) for vid in videos]
names = ["NORTH", "SOUTH", "EAST", "WEST"]

while True:
    frames, counts, emergency_flags = [], [], []

    for cap in caps:
        ret, frame = cap.read()
        if not ret: 
            break
            
        frame = cv2.resize(frame, (400, 300))
        
        # Run inference using your custom model
        results = model(frame, verbose=False)
        boxes = results[0].boxes
        
        # Check specifically for your custom Class 0 (Ambulance) and Class 1 (Police)
        is_emergency = any(int(box.cls) in [0, 1] for box in boxes)
        
        counts.append(len(boxes))
        emergency_flags.append(is_emergency)
        frames.append(results[0].plot()) 

    if len(frames) < 4: 
        break

    # Priority Logic
    if any(emergency_flags):
        # Emergency override
        green_index = emergency_flags.index(True)
    else:
        # Standard density-based logic
        green_index = counts.index(max(counts))

    # UI and Frame Drawing
    for i in range(4):
        color = (0, 255, 0) if i == green_index else (0, 0, 255)
        status = "GREEN" if i == green_index else "RED"
        warning = " !!! EMERGENCY !!!" if emergency_flags[i] else ""
        
        text = f"{names[i]}: {status} (Count: {counts[i]}){warning}"
        cv2.putText(frames[i], text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 3)

    # 2x2 Grid Assembly
    top_row = np.hstack((frames[0], frames[1]))
    bottom_row = np.hstack((frames[2], frames[3]))
    grid = np.vstack((top_row, bottom_row))

    cv2.imshow("Smart 4-Way Traffic AI (Custom Model)", grid)

    if cv2.waitKey(1) == ord('q'): 
        break

for cap in caps: cap.release()
cv2.destroyAllWindows()