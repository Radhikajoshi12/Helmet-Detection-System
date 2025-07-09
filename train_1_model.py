from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (nano version for speed)
model = YOLO('yolov8n.pt')

# Train the model
model.train(
    data='data.yaml',        # Because data.yaml is in same folder
    epochs=20,               # You can adjust this
    imgsz=640,               # Image size
    batch=16,                # Batch size
    project='helmet_detection',  
    name='helmet_model'      # Folder name where model gets saved
)
