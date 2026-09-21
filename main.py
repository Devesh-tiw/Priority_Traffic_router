import cv2
from ultralytics import YOLO

# Load the lightest YOLOv8 model for maximum FPS
model = YOLO('yolov8n.pt') 
cap = cv2.VideoCapture(0) # Replace "traffic.mp4" with 0 to use your webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break

    # Detect vehicles (COCO classes: 2=car, 3=motorbike, 5=bus, 7=truck)
    results = model(frame, classes=[2, 3, 5, 7], verbose=False)
    vehicle_count = len(results[0].boxes)

    # Simple Priority Logic based on vehicle density
    if vehicle_count > 10:
        priority = f"HIGH PRIORITY: {vehicle_count} Vehicles -> GREEN (60s)"
        color = (0, 255, 0) # Green text
    elif vehicle_count > 0:
        priority = f"NORMAL: {vehicle_count} Vehicles -> GREEN (30s)"
        color = (0, 255, 255) # Yellow text
    else:
        priority = "NO TRAFFIC -> RED LIGHT"
        color = (0, 0, 255) # Red text

    # Draw bounding boxes and text on the frame
    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, priority, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    cv2.imshow("Smart Traffic Controller", annotated_frame)

    # Press 'q' to quit the window
    if cv2.waitKey(1) == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()