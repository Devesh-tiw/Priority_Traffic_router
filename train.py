from ultralytics import YOLO

def main():
    # Load the base pretrained YOLOv8 Nano model
    model = YOLO('yolov8n.pt')

    # Train the model on your custom dataset
    results = model.train(
        data='data.yaml',
        epochs=100,        # Number of training loops
        imgsz=640,         # Image resolution
        batch=16,          # Images per batch (lower if out of memory)
        device=0,          # 0 for GPU, 'cpu' for CPU
        name='smart_traffic_model' # Folder name for saved weights
    )

    print("Training complete! Your custom model is saved in: runs/detect/smart_traffic_model/weights/best.pt")

if __name__ == '__main__':
    main()